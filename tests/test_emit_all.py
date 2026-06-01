"""WS-A Task A.1 — emit-all byte-identity gate (standalone, no driver dependency).

The EMIT-ALL correctness premise (spec §WS-A): each ``generate_*`` opens
``frag_dir`` read-only; only ``Drop`` deletes ``frag_dir`` (``streaming.rs``).
So ONE generator can be ingested ONCE and then have ``generate_3dtiles`` +
``generate_parquet`` called on it, kept alive until the last call, producing
output IDENTICAL to two SEPARATE generators that each ingest the same inputs.

Byte-identity contract:
  - 3dtiles GLB is deterministic post-WS-C (compare ``{relpath: sha256}`` maps).
  - Parquet is identical modulo row order (decode + sort by
    ``(zoom, tile_x, tile_y, tile_d, feature_id)``; compare positions/indices).

This test also proves EMIT-ALL creates exactly ONE ``frag_dir`` while SEPARATE
creates TWO — by counting the per-generator ``microjson_frags_<pid>_<gid>``
temp dirs that exist while each path's generator(s) are alive.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

try:
    from mudm_tools._rs import StreamingTileGenerator, scan_obj_bounds

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

from mudm_tools.tiling3d.parquet_writer import generate_parquet
from mudm_tools.tiling3d.parquet_reader import read_parquet

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not compiled")


# ---------------------------------------------------------------------------
# Fixture: a few tiny OBJ files with multiple features sharing tiles across
# 2+ zooms (mirrors the WS-C gate fixture in test_tiling3d_3dtiles.py, but
# small + deterministic so it is fast and the assertions are exact).
# ---------------------------------------------------------------------------


def _make_obj_fixture(obj_dir: Path) -> list[str]:
    """Write tiny single-triangle .obj files spread across world space.

    Each file is its own shard, so the encode phase must merge many shards'
    fragments into the same coarse-zoom tiles (the structure that makes the
    feature_id sort load-bearing for byte identity). Centers are spread so
    several distinct coarse tiles exist AND some cluster to share tiles —
    giving multiple features per tile across 2+ zooms.
    """
    obj_dir.mkdir(parents=True, exist_ok=True)
    # Deterministic centers: a few tight clusters (share tiles) plus spread
    # ones (distinct tiles), all inside [0, 100]^3.
    centers = [
        (10.0, 10.0, 10.0),
        (10.5, 10.3, 10.2),
        (11.0, 10.1, 9.8),  # cluster A
        (60.0, 60.0, 60.0),
        (60.4, 59.7, 60.3),  # cluster B
        (90.0, 20.0, 80.0),  # lone
        (25.0, 85.0, 35.0),  # lone
        (45.0, 45.0, 55.0),
        (45.6, 44.8, 54.9),  # cluster C
    ]
    paths: list[str] = []
    for i, (cx, cy, cz) in enumerate(centers):
        p = obj_dir / f"mesh_{i:03d}.obj"
        p.write_text(
            f"v {cx} {cy} {cz}\n"
            f"v {cx + 1.0} {cy} {cz}\n"
            f"v {cx + 0.5} {cy + 1.0} {cz + 0.5}\n"
            "f 1 2 3\n"
        )
        paths.append(str(p))
    return paths


def _glb_sha_map(out_dir: Path) -> dict[str, str]:
    return {
        str(p.relative_to(out_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out_dir.rglob("*.glb"))
    }


def _canon_parquet(rows) -> list:
    """Canonicalize parquet rows: sort by the 5-key tile order + feature_id,
    comparing positions+indices bytes (modulo row order)."""
    sig = []
    for r in rows:
        sig.append(
            (
                int(r["zoom"]),
                int(r["tile_x"]),
                int(r["tile_y"]),
                int(r["tile_d"]),
                int(r["feature_id"]),
                int(r["geom_type"]),
                r["positions"].tobytes(),
                r["indices"].tobytes(),
                tuple(sorted(r["tags"].items())),
            )
        )
    return sorted(sig)


def _frag_dirs() -> set[str]:
    """The set of this-process generator temp dirs that currently exist.

    Each ``StreamingTileGenerator`` creates exactly one
    ``microjson_frags_<pid>_<gid>`` dir on construction and removes it on Drop
    (``streaming.rs``). Counting the per-generator suffixes that appear while a
    path is alive proves how many generators (= how many ingests) that path
    used.
    """
    import tempfile

    pid = os.getpid()
    prefix = f"microjson_frags_{pid}_"
    tmp = Path(tempfile.gettempdir())
    return {p.name for p in tmp.glob(f"{prefix}*") if p.is_dir()}


def _new_generator(max_zoom: int = 3):
    return StreamingTileGenerator(min_zoom=0, max_zoom=max_zoom, base_cells=100)


def _ingest(gen, paths: list[str], bounds) -> None:
    """Production ingest: add_obj_files with parallel reads (io_threads),
    bounds via scan_obj_bounds."""
    gen._set_io_threads(0)  # 0 -> global rayon pool (parallel reads)
    tags = [{"idx": str(i)} for i in range(len(paths))]
    gen.add_obj_files(paths, bounds, tags, 0)


def test_emit_all_byte_identical_to_separate(tmp_path):
    """EMIT-ALL (one ingest -> generate_3dtiles + generate_parquet) is
    byte-identical to SEPARATE (two generators each ingesting the same inputs).

    Asserts (a) GLB sha maps identical, (b) parquet identical modulo row order,
    and (c) EMIT-ALL used exactly ONE frag_dir vs TWO for SEPARATE."""
    obj_dir = tmp_path / "objs"
    paths = _make_obj_fixture(obj_dir)
    # Bounds via the production scan (not a hardcoded box).
    bounds = scan_obj_bounds(paths)

    # Snapshot pre-existing frag dirs (other live generators / leftovers) so we
    # only count dirs created by THIS test's generators.
    pre = _frag_dirs()

    # --- Path EMIT-ALL: one generator, one ingest, both formats ---
    dir_a = tmp_path / "emit_all_3dtiles"
    pq_a = tmp_path / "emit_all_parquet"
    gen = _new_generator()
    _ingest(gen, paths, bounds)
    emit_all_frags = _frag_dirs() - pre  # generator(s) alive during EMIT-ALL
    gen.generate_3dtiles(str(dir_a), bounds)
    generate_parquet(gen, pq_a, bounds, partitioned=True)
    glb_a = _glb_sha_map(dir_a)
    rows_a = _canon_parquet(read_parquet(pq_a))
    del gen  # frag_dir removed only here (Drop)

    # --- Path SEPARATE: two generators, each its own ingest ---
    dir_b = tmp_path / "separate_3dtiles"
    pq_b = tmp_path / "separate_parquet"

    pre_sep = _frag_dirs()
    gen3d = _new_generator()
    _ingest(gen3d, paths, bounds)
    gen_pq = _new_generator()
    _ingest(gen_pq, paths, bounds)
    separate_frags = _frag_dirs() - pre_sep  # both generators alive here
    gen3d.generate_3dtiles(str(dir_b), bounds)
    generate_parquet(gen_pq, pq_b, bounds, partitioned=True)
    glb_b = _glb_sha_map(dir_b)
    rows_b = _canon_parquet(read_parquet(pq_b))
    del gen3d
    del gen_pq

    # (c) frag_dir count: EMIT-ALL == 1, SEPARATE == 2.
    assert (
        len(emit_all_frags) == 1
    ), f"EMIT-ALL must create exactly ONE frag_dir; saw {sorted(emit_all_frags)}"
    assert (
        len(separate_frags) == 2
    ), f"SEPARATE must create exactly TWO frag_dirs; saw {sorted(separate_frags)}"

    # (a) GLB byte-identity: same tile set + same sha256 per tile.
    assert glb_a, "no GLB tiles produced"
    assert set(glb_a) == set(glb_b), (
        f"different GLB tile sets: emit_all-only={set(glb_a) - set(glb_b)} "
        f"separate-only={set(glb_b) - set(glb_a)}"
    )
    glb_diffs = {k: (glb_a[k], glb_b[k]) for k in glb_a if glb_a[k] != glb_b[k]}
    assert not glb_diffs, (
        f"{len(glb_diffs)} GLB tiles differ between emit-all and separate; "
        f"reuse leaked cross-call state. e.g. {list(glb_diffs)[:5]}"
    )

    # (b) Parquet byte-identity modulo row order.
    assert rows_a, "no parquet rows produced"
    assert len(rows_a) == len(
        rows_b
    ), f"parquet row count differs: emit_all={len(rows_a)} separate={len(rows_b)}"
    assert rows_a == rows_b, (
        "parquet positions/indices/tags differ between emit-all and separate "
        "(canonicalized by zoom,tile_x,tile_y,tile_d,feature_id)"
    )
    # The fixture must actually exercise multiple tiles across multiple zooms,
    # else byte-identity is trivially true.
    tilekeys = {(s[0], s[1], s[2], s[3]) for s in rows_a}
    assert len({s[0] for s in rows_a}) >= 2, "fixture must span >=2 zooms"
    assert len(tilekeys) > 1, "fixture must span >1 tile"
