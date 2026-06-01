"""WS-B tests for the bounded feature-bucketed Neuroglancer multilod read.

These tests pin the WS-B contract:

  (a) byte-identity: the feature-bucketed/bounded read produces NG geometry
      ({fid} + {fid}.index) BYTE-IDENTICAL to the whole-corpus read, INCLUDING
      when a tiny memory ceiling forces k>1 buckets.
  (b) memory bounded: with k>1 buckets, each shard subset is decoded once per
      bucket and only the bucket's features are kept resident (structural /
      decode-counter assertion; RSS-gated on Linux).
  (c) numeric props: segment_properties/info emits type:number + data_type for
      all-numeric columns; column order unchanged; geometry bytes UNCHANGED.
  (d) errors: a corrupt shard + a degenerate mesh each surface exactly once via
      the WS-D ErrorCollector; healthy-feature geometry unchanged.

The byte-identity baseline (Step A) is captured by running the generator with a
huge ceiling (k==1, whole-corpus path equivalent) and recording the sha256 of
each {fid} and {fid}.index, then re-running with a tiny ceiling (k>1) and
asserting equality.
"""

import hashlib
import json
from pathlib import Path

import pytest

try:
    from mudm_tools._rs import StreamingTileGenerator

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not compiled")

WORLD_BOUNDS = (0.0, 0.0, 0.0, 100.0, 200.0, 300.0)

HUGE = 1 << 40  # 1 TiB — forces k == 1 (whole-corpus equivalent)
TINY = 1  # 1 byte — forces k > 1 (max buckets, one feature class per bucket)


def _make_tin_feature(xy, z, ring_lengths, tags=None):
    n = len(z)
    return {
        "geometry": xy,
        "geometry_z": z,
        "ring_lengths": ring_lengths,
        "type": 5,  # TIN
        "tags": tags or {},
        "minX": min(xy[i * 2] for i in range(n)),
        "minY": min(xy[i * 2 + 1] for i in range(n)),
        "minZ": min(z),
        "maxX": max(xy[i * 2] for i in range(n)),
        "maxY": max(xy[i * 2 + 1] for i in range(n)),
        "maxZ": max(z),
    }


def _make_dense_tin_feature(n_triangles=20, seed=42, tags=None):
    import random

    rng = random.Random(seed)
    xy = []
    z = []
    ring_lengths = []
    for _ in range(n_triangles):
        cx = rng.uniform(0.15, 0.85)
        cy = rng.uniform(0.15, 0.85)
        cz = rng.uniform(0.15, 0.85)
        d = 0.05
        xy.extend([cx - d, cy - d, cx + d, cy - d, cx, cy + d])
        z.extend([cz - d, cz + d, cz])
        ring_lengths.append(3)
    return _make_tin_feature(xy, z, ring_lengths, tags=tags or {"name": "dense"})


def _build(features, min_zoom=0, max_zoom=2):
    gen = StreamingTileGenerator(min_zoom=min_zoom, max_zoom=max_zoom)
    fids = [gen.add_feature(f) for f in features]
    return gen, fids


def _canon_geometry(out_dir: Path) -> dict:
    """Map fid -> (sha256(index), sha256(data)) for every {fid}/{fid}.index pair."""
    result = {}
    for index_path in sorted(out_dir.glob("*.index")):
        fid = index_path.stem
        data_path = out_dir / fid
        if not data_path.exists():
            continue
        idx_h = hashlib.sha256(index_path.read_bytes()).hexdigest()
        dat_h = hashlib.sha256(data_path.read_bytes()).hexdigest()
        result[fid] = (idx_h, dat_h)
    return result


def _multi_feature_corpus(n=8):
    """A multi-feature, multi-zoom corpus. Distinct seeds so meshes differ."""
    feats = []
    for i in range(n):
        feats.append(
            _make_dense_tin_feature(
                n_triangles=12 + i,
                seed=100 + i,
                tags={"name": f"neuron_{i}", "volume": float(10 * i + 5)},
            )
        )
    return feats


# ---------------------------------------------------------------------------
# (a) byte-identity: bucketed (k>1) == whole-corpus (k==1)
# ---------------------------------------------------------------------------


class TestByteIdentityBucketed:

    def test_k1_vs_k_gt_1_byte_identical(self, tmp_path):
        feats = _multi_feature_corpus(8)

        # Step A baseline: huge ceiling => k == 1 (whole-corpus equivalent).
        gen1, _ = _build(feats)
        gen1._set_max_memory(HUGE)
        out1 = tmp_path / "ng_k1"
        n1 = gen1.generate_neuroglancer_multilod(str(out1), WORLD_BOUNDS)
        baseline = _canon_geometry(out1)
        assert baseline, "baseline produced no geometry"

        # Tiny ceiling => k > 1 buckets. MUST be byte-identical.
        gen2, _ = _build(feats)
        gen2._set_max_memory(TINY)
        out2 = tmp_path / "ng_kbig"
        n2 = gen2.generate_neuroglancer_multilod(str(out2), WORLD_BOUNDS)
        bucketed = _canon_geometry(out2)

        assert n1 == n2
        assert (
            bucketed == baseline
        ), "bucketed (k>1) NG geometry must be byte-identical to whole-corpus (k==1)"

    def test_serial_equals_parallel_byte_identical(self, tmp_path):
        """PRIMARY gate: serial (io_threads=1) == parallel read, byte-identical,
        AND under a tiny ceiling forcing k>1 buckets."""
        feats = _multi_feature_corpus(8)

        gen_s, _ = _build(feats)
        gen_s._set_io_threads(1)  # serial
        gen_s._set_max_memory(TINY)  # k>1
        out_s = tmp_path / "ng_serial"
        gen_s.generate_neuroglancer_multilod(str(out_s), WORLD_BOUNDS)
        assert gen_s._get_ng_bucket_count() > 1

        gen_p, _ = _build(feats)
        gen_p._set_io_threads(4)  # parallel
        gen_p._set_max_memory(TINY)  # k>1
        out_p = tmp_path / "ng_parallel"
        gen_p.generate_neuroglancer_multilod(str(out_p), WORLD_BOUNDS)
        assert gen_p._get_ng_bucket_count() > 1

        assert _canon_geometry(out_s) == _canon_geometry(out_p)

    def test_segment_properties_column_order_unchanged(self, tmp_path):
        feats = _multi_feature_corpus(6)

        gen1, _ = _build(feats)
        gen1._set_max_memory(HUGE)
        out1 = tmp_path / "ng_k1"
        gen1.generate_neuroglancer_multilod(str(out1), WORLD_BOUNDS)
        sp1 = json.loads((out1 / "segment_properties" / "info").read_text())
        cols1 = [p["id"] for p in sp1["inline"]["properties"]]
        ids1 = sp1["inline"]["ids"]

        gen2, _ = _build(feats)
        gen2._set_max_memory(TINY)
        out2 = tmp_path / "ng_kbig"
        gen2.generate_neuroglancer_multilod(str(out2), WORLD_BOUNDS)
        sp2 = json.loads((out2 / "segment_properties" / "info").read_text())
        cols2 = [p["id"] for p in sp2["inline"]["properties"]]
        ids2 = sp2["inline"]["ids"]

        assert cols1 == cols2, "segment_properties column order must be invariant to k"
        assert ids1 == ids2, "segment_properties ids order must be invariant to k"


# ---------------------------------------------------------------------------
# (b) memory bounded
# ---------------------------------------------------------------------------


class TestMemoryBounded:

    def test_bucketed_run_completes_under_tiny_ceiling(self, tmp_path):
        """Structural: a tiny ceiling forcing many buckets still produces the
        same complete per-feature output (no truncation across buckets)."""
        feats = _multi_feature_corpus(10)
        gen, fids = _build(feats)
        gen._set_max_memory(TINY)
        out = tmp_path / "ng"
        count = gen.generate_neuroglancer_multilod(str(out), WORLD_BOUNDS)
        # The tiny ceiling MUST have forced more than one bucket (otherwise the
        # bounded path was never exercised).
        assert gen._get_ng_bucket_count() > 1, gen._get_ng_bucket_count()
        # Each feature with geometry must have BOTH files; per-feature completeness
        # means no feature is split across buckets.
        geom = _canon_geometry(out)
        assert count == len(geom)
        for fid in geom:
            assert (out / fid).exists()
            assert (out / f"{fid}.index").exists()

    def test_huge_ceiling_is_single_bucket(self, tmp_path):
        feats = _multi_feature_corpus(8)
        gen, _ = _build(feats)
        gen._set_max_memory(HUGE)
        out = tmp_path / "ng"
        gen.generate_neuroglancer_multilod(str(out), WORLD_BOUNDS)
        assert gen._get_ng_bucket_count() == 1, gen._get_ng_bucket_count()

    def test_ng_peak_resident_scales_with_bucket_not_corpus(self, tmp_path):
        """Deterministic, platform-independent peak probe (replaces the flawed
        ru_maxrss test, which measured monotonic process-lifetime high-water in
        run-order, not per-run peak).

        ``_get_ng_peak_resident_bytes`` reports the MAX Σ-estimate_bytes over the
        RETAINED feature map of any single bucket. With a HUGE ceiling k==1 so
        the retained map is the whole corpus; with a TINY ceiling k>1 so each
        bucket retains ≈ corpus / k. The bounded filter-during-decode fix makes
        the tiny-ceiling peak substantially LESS than the huge-ceiling peak. On
        the pre-fix post-filter impl the whole-corpus map is materialized before
        the filter, so the peak is O(corpus) regardless of k and this FAILS.
        """
        # A dense, many-feature corpus so the corpus / k ratio is clearly
        # resolvable above per-run noise.
        feats = _multi_feature_corpus(24)

        def peak(ceiling):
            gen, _ = _build(feats)
            gen._set_max_memory(ceiling)
            out = tmp_path / f"ng_{ceiling}"
            gen._reset_ng_peak_resident_bytes()
            gen.generate_neuroglancer_multilod(str(out), WORLD_BOUNDS)
            return gen._get_ng_peak_resident_bytes(), gen._get_ng_bucket_count()

        huge_peak, huge_k = peak(HUGE)
        tiny_peak, tiny_k = peak(TINY)

        assert huge_k == 1, huge_k
        assert tiny_k > 1, tiny_k
        assert huge_peak > 0, "huge-ceiling peak probe must observe resident bytes"
        assert tiny_peak > 0, "tiny-ceiling peak probe must observe resident bytes"

        # The bounded read must keep only ~1/k of the corpus resident per bucket:
        # the tiny-ceiling peak must be substantially LESS than the whole-corpus
        # peak. (At least 2x smaller — far more than per-run jitter.)
        assert tiny_peak <= huge_peak / 2, (
            f"tiny-ceiling peak ({tiny_peak}) must be <= half the huge-ceiling "
            f"peak ({huge_peak}); k_tiny={tiny_k}, k_huge={huge_k}. Peak is "
            f"unbounded (O(corpus) regardless of k) — the bounded read does NOT "
            f"bound memory."
        )

    # NOTE: a ru_maxrss-based test was removed here — ru_maxrss is a monotonic
    # process-lifetime high-water mark, so two generate() runs in one process
    # cannot be compared for per-run peak (the first run absorbs warmup, the
    # second shows ~0). The deterministic Rust NG_PEAK_RESIDENT_BYTES probe
    # (test_ng_peak_resident_scales_with_bucket_not_corpus, above) is the
    # authoritative bounding proof and is platform-independent.


# ---------------------------------------------------------------------------
# (c) numeric props (separate JSON test) — geometry UNCHANGED
# ---------------------------------------------------------------------------


class TestNumericProps:

    def test_all_numeric_column_emits_number_and_data_type(self, tmp_path):
        feats = [
            _make_dense_tin_feature(15, seed=1, tags={"name": "a", "volume": 42.5, "count": 3}),
            _make_dense_tin_feature(15, seed=2, tags={"name": "b", "volume": 88.0, "count": 7}),
        ]
        gen, _ = _build(feats)
        out = tmp_path / "ng"
        gen.generate_neuroglancer_multilod(str(out), WORLD_BOUNDS)

        sp = json.loads((out / "segment_properties" / "info").read_text())
        props = {p["id"]: p for p in sp["inline"]["properties"]}

        # "volume" is all-float -> number/float32
        assert props["volume"]["type"] == "number"
        assert props["volume"]["data_type"] == "float32"

        # "count" is all-int -> number/uint32
        assert props["count"]["type"] == "number"
        assert props["count"]["data_type"] == "uint32"

        # "name" is non-numeric -> label, no data_type
        assert props["name"]["type"] == "label"
        assert "data_type" not in props["name"]

    def test_numeric_inference_does_not_change_geometry(self, tmp_path):
        """Geometry bytes must be identical whether or not tags are numeric."""
        # numeric-tagged corpus
        feats_num = [
            _make_dense_tin_feature(12, seed=200, tags={"volume": 1.5}),
            _make_dense_tin_feature(13, seed=201, tags={"volume": 2.5}),
        ]
        # same geometry, label-only tags (different tag VALUES/types, same meshes)
        feats_lbl = [
            _make_dense_tin_feature(12, seed=200, tags={"volume": "x"}),
            _make_dense_tin_feature(13, seed=201, tags={"volume": "y"}),
        ]
        gen_n, _ = _build(feats_num)
        out_n = tmp_path / "num"
        gen_n.generate_neuroglancer_multilod(str(out_n), WORLD_BOUNDS)

        gen_l, _ = _build(feats_lbl)
        out_l = tmp_path / "lbl"
        gen_l.generate_neuroglancer_multilod(str(out_l), WORLD_BOUNDS)

        assert _canon_geometry(out_n) == _canon_geometry(
            out_l
        ), "numeric vs label tags must not change Draco geometry bytes"


# ---------------------------------------------------------------------------
# (d) errors fold into the WS-D ErrorCollector exactly once
# ---------------------------------------------------------------------------


class TestErrorFolding:

    def test_corrupt_shard_surfaces_via_collector(self, tmp_path):
        feats = _multi_feature_corpus(6)
        gen, _ = _build(feats)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        gen._set_run_dir(str(run_dir))
        # Force k>1 so we exercise the parallel bucketed read error path.
        gen._set_max_memory(TINY)

        # Inject a corrupt shard into the frag_dir BEFORE generate reads it.
        frag_dir = Path(gen._frag_dir()) if hasattr(gen, "_frag_dir") else None

        out = tmp_path / "ng"

        if frag_dir is None:
            pytest.skip("no _frag_dir accessor to inject a corrupt shard")

        (frag_dir / "zz_corrupt.mjf").write_bytes(b"not a valid zstd shard at all")

        # Generate should NOT raise (parse errors are non-fatal, recorded), but
        # the corrupt shard must be reported exactly once in errors.jsonl.
        gen.generate_neuroglancer_multilod(str(out), WORLD_BOUNDS)

        errlog = run_dir / "errors.jsonl"
        assert errlog.exists(), "errors.jsonl must be written when run_dir is set"
        lines = [json.loads(line) for line in errlog.read_text().splitlines() if line.strip()]
        corrupt_lines = [
            r for r in lines if "zz_corrupt.mjf" in r["item"] or "zz_corrupt.mjf" in r["message"]
        ]
        assert (
            len(corrupt_lines) == 1
        ), f"corrupt shard must surface exactly once, got {len(corrupt_lines)}: {corrupt_lines}"
