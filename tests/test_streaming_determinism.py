"""Byte-level determinism regression tests (streaming_review.md §T2 #1/#2).

The existing suite cannot catch the H1/H2/PY-2 nondeterminism: test_overlap_
matches_serial canonical-SORTS rows (multiset equality, weaker than byte
identity) and runs at max_batch_bytes=1 (forces n_parts=1, the blind spot).
These tests compare actual file BYTES across (a) independent generators in one
process — catches H1a, the per-map-instance ahash tile-iteration order — and
(b) separate processes with different RAYON_NUM_THREADS — catches PY-2/H1b,
where the legacy writer's part count was rayon::current_num_threads().

They exercise the LEGACY generate_parquet_native — the path the converters
actually call (converters/{obj,geojson,xenium}.py) — at a scale that produces
multiple parts per zoom, so tile order + part splitting actually matter.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

try:
    from mudm_tools._rs import StreamingTileGenerator

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not compiled")

WORLD_BOUNDS = (0.0, 0.0, 0.0, 100.0, 200.0, 300.0)

# Shared, RNG-free fixture text so the subprocess driver builds the IDENTICAL
# corpus the in-process test does. 24 spread TINs -> several tiles per zoom.
_FIXTURE = """
def _feats(n=24):
    out = []
    for i in range(n):
        cx = 0.05 + (i % 4) * 0.3
        cy = 0.05 + ((i // 4) % 4) * 0.22
        cz = 0.05 + (i % 3) * 0.3
        d = 0.04
        xy = [cx - d, cy - d, cx + d, cy - d, cx, cy + d]
        z = [cz - d, cz + d, cz]
        m = len(z)
        out.append({
            "geometry": xy, "geometry_z": z, "ring_lengths": [3], "type": 5,
            "tags": {"i": str(i)},
            "minX": min(xy[k * 2] for k in range(m)), "minY": min(xy[k * 2 + 1] for k in range(m)),
            "minZ": min(z),
            "maxX": max(xy[k * 2] for k in range(m)), "maxY": max(xy[k * 2 + 1] for k in range(m)),
            "maxZ": max(z),
        })
    return out
"""
exec(_FIXTURE)  # defines _feats() in this module  # noqa: S102


def _sig(d) -> dict:
    """relpath -> sha256 over every file under dir d (byte-level signature)."""
    d = Path(d)
    return {
        str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(d.rglob("*"))
        if p.is_file()
    }


def _generate(out) -> int:
    gen = StreamingTileGenerator(min_zoom=0, max_zoom=3)
    for f in _feats():
        gen.add_feature(f)
    return gen.generate_parquet_native(str(out), WORLD_BOUNDS, True)


def test_legacy_parquet_byte_identical_run_to_run(tmp_path):
    """Two independent generators -> byte-identical Parquet. Catches H1a (ahash
    tile order is seeded per map instance, so it differs every read pre-fix)."""
    a, b = tmp_path / "a", tmp_path / "b"
    na, nb = _generate(a), _generate(b)
    assert na == nb > 0
    sa, sb = _sig(a), _sig(b)
    # meaningful only if there are multiple parts per zoom (else ordering is moot)
    assert (
        sum(1 for k in sa if k.startswith("zoom=1/")) >= 2
    ), f"too few parts to test ordering: {sorted(sa)}"
    differing = [k for k in sa if k in sb and sa[k] != sb[k]]
    assert sa == sb, (
        f"non-deterministic parquet bytes: only_a={sorted(set(sa) - set(sb))} "
        f"only_b={sorted(set(sb) - set(sa))} differing={differing}"
    )


_DRIVER = (
    "import sys\n"
    "from mudm_tools._rs import StreamingTileGenerator\n"
    + _FIXTURE
    + "g = StreamingTileGenerator(min_zoom=0, max_zoom=3)\n"
    "for f in _feats():\n"
    "    g.add_feature(f)\n"
    "g.generate_parquet_native(sys.argv[1], (0.0, 0.0, 0.0, 100.0, 200.0, 300.0), True)\n"
)


def test_legacy_parquet_byte_identical_across_thread_counts(tmp_path):
    """Part splitting must be content-deterministic, independent of the live
    rayon pool width. Catches PY-2/H1b (n_parts was current_num_threads())."""
    driver = tmp_path / "driver.py"
    driver.write_text(_DRIVER)

    def run(out, threads):
        env = {**os.environ, "RAYON_NUM_THREADS": str(threads)}
        subprocess.run(
            [sys.executable, str(driver), str(out)], env=env, check=True, capture_output=True
        )
        return _sig(out)

    s1 = run(tmp_path / "t1", 1)
    s4 = run(tmp_path / "t4", 4)
    s4b = run(tmp_path / "t4b", 4)
    assert s1, "subprocess produced no output"
    assert s1 == s4, (
        "Parquet part layout depends on RAYON_NUM_THREADS (PY-2 regression): "
        f"n_files 1-thread={len(s1)} 4-thread={len(s4)}; "
        f"only_1={sorted(set(s1) - set(s4))[:4]} only_4={sorted(set(s4) - set(s1))[:4]}"
    )
    assert s4 == s4b, "non-deterministic across runs at a fixed thread count (H1a)"


# ---- 2D (streaming_review.md §T2 #2): H2 deterministic fids + 2D PY-2 ----


def _write_geojson_points(path: Path, points, key="k") -> Path:
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {key: f"p{i}"},
                "geometry": {"type": "Point", "coordinates": [float(x), float(y)]},
            }
            for i, (x, y) in enumerate(points)
        ],
    }
    path.write_text(json.dumps(fc))
    return path


def test_2d_geojson_parquet_byte_identical_run_to_run(tmp_path):
    """2D add_geojson_files (the multi-file, H2-fixed positional-fid path) +
    generate_parquet_native is byte-identical across two independent generators.
    Catches H2 (fids were assigned in rayon scheduling order pre-fix, so the
    feature_id column — and any fid-keyed decimation — varied per run) and the
    2D PY-2 (tile order / n_parts = current_num_threads())."""
    from mudm_tools._rs import StreamingTileGenerator2D

    bounds = (0.0, 0.0, 100.0, 100.0)
    f1 = _write_geojson_points(
        tmp_path / "a.geojson", [(5 + i * 4, 10 + (i % 5) * 15) for i in range(20)]
    )
    f2 = _write_geojson_points(
        tmp_path / "b.geojson", [(50 + i * 2, 30 + (i % 7) * 8) for i in range(20)]
    )

    def run(out: Path):
        tmp = tmp_path / ("frag_" + out.name)
        tmp.mkdir()
        g = StreamingTileGenerator2D(min_zoom=0, max_zoom=4, buffer=0.0, temp_dir=str(tmp))
        g.add_geojson_files([str(f1), str(f2)], bounds)
        return g.generate_parquet_native(str(out), bounds, True)

    a, b = tmp_path / "oa", tmp_path / "ob"
    na, nb = run(a), run(b)
    assert na == nb > 0
    sa, sb = _sig(a), _sig(b)
    assert sa, "no 2D parquet output"
    differing = [k for k in sa if k in sb and sa[k] != sb[k]]
    assert sa == sb, (
        f"non-deterministic 2D parquet (H2/2D-PY-2): only_a={sorted(set(sa) - set(sb))} "
        f"only_b={sorted(set(sb) - set(sa))} differing={differing}"
    )


# ---- G1-2D (streaming_review.md §G): bounded 2D Parquet path ----


def _read_rowset_2d(d: Path):
    """Multiset of (zoom, tile_x, tile_y, feature_id, geom_type, positions, indices)
    across all part files — layout/part-numbering-independent content signature."""
    import pyarrow.parquet as pq

    out = []
    for p in sorted(Path(d).rglob("*.mu.parquet")):
        zoom = int(p.parent.name.split("=")[1])
        cols = pq.read_table(p).to_pydict()
        for i in range(len(cols["feature_id"])):
            out.append(
                (
                    zoom,
                    cols["tile_x"][i],
                    cols["tile_y"][i],
                    cols["feature_id"][i],
                    cols["geom_type"][i],
                    bytes(cols["positions"][i]),
                    bytes(cols["indices"][i]),
                )
            )
    return sorted(out)


def test_2d_partitioned_content_equivalent_to_legacy_and_bounded(tmp_path):
    """The new bounded 2D generate_parquet_native_partitioned emits the SAME row
    set as the unbounded generate_parquet_native (only part-file layout differs),
    AND a tiny max_batch_bytes forces multiple chunks (the bounding is exercised)."""
    from mudm_tools._rs import StreamingTileGenerator2D

    bounds = (0.0, 0.0, 100.0, 100.0)
    f1 = _write_geojson_points(
        tmp_path / "a.geojson", [(5 + i * 4, 10 + (i % 5) * 15) for i in range(30)]
    )
    f2 = _write_geojson_points(
        tmp_path / "b.geojson", [(50 + i * 2, 30 + (i % 7) * 8) for i in range(30)]
    )

    def ingest(name):
        tmp = tmp_path / ("frag_" + name)
        tmp.mkdir()
        g = StreamingTileGenerator2D(min_zoom=0, max_zoom=4, buffer=0.0, temp_dir=str(tmp))
        g.add_geojson_files([str(f1), str(f2)], bounds)
        return g, tmp_path / name

    g1, legacy_out = ingest("legacy")
    n_legacy = g1.generate_parquet_native(str(legacy_out), bounds, simplify=True)
    g2, part_out = ingest("part")
    # max_batch_bytes=1 forces one chunk per shard -> exercises the multi-chunk
    # bounded loop + persistent cross-chunk part numbering.
    n_part = g2.generate_parquet_native_partitioned(str(part_out), bounds, True, "zstd", 1)

    assert n_legacy == n_part > 0
    assert _read_rowset_2d(legacy_out) == _read_rowset_2d(
        part_out
    ), "bounded path changed row content"
    # the bounded path actually chunked (else it's not testing the bounded loop)
    assert len(list(part_out.rglob("*.mu.parquet"))) >= 2


def test_2d_partitioned_content_stable_run_to_run(tmp_path):
    """The bounded 2D path is CONTENT-deterministic run-to-run (identical row
    set). NOTE the trade vs the legacy whole-corpus path: part-file BYTE layout
    is NOT run-to-run reproducible — which tile lands in which part depends on
    the ingest's per-thread shard distribution (2D ingest writes one shard per
    worker thread, unlike 3D's one-shard-per-input-file, so the byte-budget chunk
    composition varies with rayon scheduling). Bounded memory is the trade;
    content — what consumers actually read — is identical."""
    from mudm_tools._rs import StreamingTileGenerator2D

    bounds = (0.0, 0.0, 100.0, 100.0)
    f1 = _write_geojson_points(
        tmp_path / "a.geojson", [(5 + i * 4, 10 + (i % 5) * 15) for i in range(30)]
    )
    f2 = _write_geojson_points(
        tmp_path / "b.geojson", [(50 + i * 2, 30 + (i % 7) * 8) for i in range(30)]
    )

    def run(name):
        tmp = tmp_path / ("frag_" + name)
        tmp.mkdir()
        g = StreamingTileGenerator2D(min_zoom=0, max_zoom=4, buffer=0.0, temp_dir=str(tmp))
        g.add_geojson_files([str(f1), str(f2)], bounds)
        out = tmp_path / name
        g.generate_parquet_native_partitioned(str(out), bounds, True, "zstd", 1)
        return out

    assert _read_rowset_2d(run("a")) == _read_rowset_2d(run("b"))
