"""Tests for Neuroglancer multilod_draco output.

Covers: info JSON structure, .index binary manifest parsing, fragment data,
Morton ordering, segment properties, and end-to-end pipeline.
"""

import json
import struct
from pathlib import Path

import pytest

try:
    from mudm_tools._rs import StreamingTileGenerator
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not RUST_AVAILABLE, reason="Rust extensions not compiled"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WORLD_BOUNDS = (0.0, 0.0, 0.0, 100.0, 200.0, 300.0)


def _make_tin_feature(xy, z, ring_lengths, tags=None):
    """Build a TIN feature dict in normalized [0,1]³ space."""
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


def _make_dense_tin_feature(n_triangles=20):
    """Build a TIN feature with many triangles spread across [0.1, 0.9]³."""
    import random
    random.seed(42)
    xy = []
    z = []
    ring_lengths = []
    for _ in range(n_triangles):
        cx = random.uniform(0.15, 0.85)
        cy = random.uniform(0.15, 0.85)
        cz = random.uniform(0.15, 0.85)
        d = 0.05
        xy.extend([cx - d, cy - d, cx + d, cy - d, cx, cy + d])
        z.extend([cz - d, cz + d, cz])
        ring_lengths.append(3)
    return _make_tin_feature(xy, z, ring_lengths, tags={"name": "dense"})


def _build_generator_with_features(features, min_zoom=0, max_zoom=2):
    """Create a StreamingTileGenerator, add features, return (gen, fids)."""
    gen = StreamingTileGenerator(min_zoom=min_zoom, max_zoom=max_zoom)
    fids = []
    for feat in features:
        fid = gen.add_feature(feat)
        fids.append(fid)
    return gen, fids


def _parse_index_file(data: bytes) -> dict:
    """Parse a .index binary manifest file."""
    off = 0

    def read_f32(n=1):
        nonlocal off
        vals = struct.unpack_from(f"<{n}f", data, off)
        off += 4 * n
        return vals if n > 1 else vals[0]

    def read_u32(n=1):
        nonlocal off
        vals = struct.unpack_from(f"<{n}I", data, off)
        off += 4 * n
        return vals if n > 1 else vals[0]

    chunk_shape = read_f32(3)
    grid_origin = read_f32(3)
    num_lods = read_u32()

    lod_scales = read_f32(num_lods)
    if num_lods == 1:
        lod_scales = (lod_scales,) if isinstance(lod_scales, float) else lod_scales

    vertex_offsets = []
    for _ in range(num_lods):
        vertex_offsets.append(read_f32(3))

    num_fragments_per_lod = read_u32(num_lods)
    if num_lods == 1:
        num_fragments_per_lod = (num_fragments_per_lod,) if isinstance(num_fragments_per_lod, int) else num_fragments_per_lod

    lods = []
    for lod_idx in range(num_lods):
        nf = num_fragments_per_lod[lod_idx]
        # fragment_positions: [3, nf] column-major
        x_vals = read_u32(nf) if nf > 0 else ()
        y_vals = read_u32(nf) if nf > 0 else ()
        z_vals = read_u32(nf) if nf > 0 else ()
        if nf == 1:
            x_vals = (x_vals,) if isinstance(x_vals, int) else x_vals
            y_vals = (y_vals,) if isinstance(y_vals, int) else y_vals
            z_vals = (z_vals,) if isinstance(z_vals, int) else z_vals
        positions = list(zip(x_vals, y_vals, z_vals))
        # fragment_offsets: [nf]
        frag_offsets = read_u32(nf) if nf > 0 else ()
        if nf == 1:
            frag_offsets = (frag_offsets,) if isinstance(frag_offsets, int) else frag_offsets
        lods.append({
            "positions": positions,
            "offsets": list(frag_offsets),
        })

    return {
        "chunk_shape": chunk_shape,
        "grid_origin": grid_origin,
        "num_lods": num_lods,
        "lod_scales": list(lod_scales),
        "vertex_offsets": vertex_offsets,
        "num_fragments_per_lod": list(num_fragments_per_lod),
        "lods": lods,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInfoJsonType:
    """Test that info JSON has correct @type."""

    def test_info_json_type(self, tmp_path):
        """info file has @type == neuroglancer_multilod_draco."""
        features = [_make_dense_tin_feature(10)]
        gen, _ = _build_generator_with_features(features)
        out = str(tmp_path / "ng_multilod")
        gen.generate_neuroglancer_multilod(out, WORLD_BOUNDS)

        info = json.loads((tmp_path / "ng_multilod" / "info").read_text())
        assert info["@type"] == "neuroglancer_multilod_draco"
        assert "vertex_quantization_bits" in info
        assert info["vertex_quantization_bits"] == 10
        assert "transform" in info
        assert len(info["transform"]) == 12


class TestIndexFileHeader:
    """Test .index binary manifest structure."""

    def test_index_file_header(self, tmp_path):
        """Parse binary manifest, verify chunk_shape, grid_origin, num_lods."""
        features = [_make_dense_tin_feature(20)]
        gen, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=2)
        out = str(tmp_path / "ng_multilod")
        gen.generate_neuroglancer_multilod(out, WORLD_BOUNDS)

        index_data = (tmp_path / "ng_multilod" / "0.index").read_bytes()
        parsed = _parse_index_file(index_data)

        assert parsed["num_lods"] >= 1
        assert parsed["num_lods"] <= 3  # max_zoom=2 → up to 3 LODs
        assert len(parsed["chunk_shape"]) == 3
        assert len(parsed["grid_origin"]) == 3
        assert all(cs > 0 for cs in parsed["chunk_shape"])
        assert len(parsed["lod_scales"]) == parsed["num_lods"]
        assert parsed["lod_scales"][0] == 1.0  # finest LOD scale = 1


class TestFragmentCountPerLod:
    """Coarser LODs have fewer or equal fragments."""

    def test_fragment_count_per_lod(self, tmp_path):
        """Coarser LODs have fewer fragments since tiles are bigger."""
        features = [_make_dense_tin_feature(50)]
        gen, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=2)
        out = str(tmp_path / "ng_multilod")
        gen.generate_neuroglancer_multilod(out, WORLD_BOUNDS)

        index_data = (tmp_path / "ng_multilod" / "0.index").read_bytes()
        parsed = _parse_index_file(index_data)

        frag_counts = parsed["num_fragments_per_lod"]
        # finest LOD (lod 0) should have >= coarser LODs
        for i in range(len(frag_counts) - 1):
            assert frag_counts[i] >= frag_counts[i + 1], (
                f"lod {i} ({frag_counts[i]} frags) < lod {i+1} ({frag_counts[i+1]} frags)"
            )


class TestFragmentDataSize:
    """Fragment data size matches sum of offsets."""

    def test_fragment_data_size(self, tmp_path):
        """Data file size == sum of all fragment_offsets across LODs."""
        features = [_make_dense_tin_feature(20)]
        gen, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=2)
        out = str(tmp_path / "ng_multilod")
        gen.generate_neuroglancer_multilod(out, WORLD_BOUNDS)

        index_data = (tmp_path / "ng_multilod" / "0.index").read_bytes()
        parsed = _parse_index_file(index_data)

        data_path = tmp_path / "ng_multilod" / "0"
        data_size = data_path.stat().st_size

        total_offsets = sum(
            sum(lod["offsets"]) for lod in parsed["lods"]
        )
        assert data_size == total_offsets


class TestDracoMagic:
    """Fragment data contains valid Draco-encoded meshes."""

    def test_draco_magic(self, tmp_path):
        """First 5 bytes of data file are DRACO magic."""
        features = [_make_dense_tin_feature(10)]
        gen, _ = _build_generator_with_features(features)
        out = str(tmp_path / "ng_multilod")
        gen.generate_neuroglancer_multilod(out, WORLD_BOUNDS)

        data = (tmp_path / "ng_multilod" / "0").read_bytes()
        assert len(data) > 5
        assert data[:5] == b"DRACO"


class TestMortonOrder:
    """Fragment positions within each LOD are Z-curve sorted."""

    def test_morton_order(self, tmp_path):
        """Fragment positions are sorted by Morton code within each LOD."""
        features = [_make_dense_tin_feature(50)]
        gen, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=2)
        out = str(tmp_path / "ng_multilod")
        gen.generate_neuroglancer_multilod(out, WORLD_BOUNDS)

        index_data = (tmp_path / "ng_multilod" / "0.index").read_bytes()
        parsed = _parse_index_file(index_data)

        def morton_3d(x, y, z):
            """Simple Morton code for comparison."""
            def spread(v):
                v &= 0x1fffff
                v = (v | (v << 32)) & 0x1f00000000ffff
                v = (v | (v << 16)) & 0x1f0000ff0000ff
                v = (v | (v << 8)) & 0x100f00f00f00f00f
                v = (v | (v << 4)) & 0x10c30c30c30c30c3
                v = (v | (v << 2)) & 0x1249249249249249
                return v
            return spread(x) | (spread(y) << 1) | (spread(z) << 2)

        for lod in parsed["lods"]:
            positions = lod["positions"]
            if len(positions) <= 1:
                continue
            morton_codes = [morton_3d(*p) for p in positions]
            for i in range(len(morton_codes) - 1):
                assert morton_codes[i] <= morton_codes[i + 1], (
                    f"Morton order violation at index {i}: "
                    f"{positions[i]} ({morton_codes[i]}) > "
                    f"{positions[i+1]} ({morton_codes[i+1]})"
                )


class TestSegmentProperties:
    """Segment properties metadata."""

    def test_segment_properties(self, tmp_path):
        """segment_properties/info has correct structure."""
        # Use dense features to avoid Draco encoding issues with tiny meshes
        features = [
            _make_dense_tin_feature(15),
            _make_dense_tin_feature(15),
        ]
        # Override tags manually
        features[0]["tags"] = {"name": "neuron_1", "volume": 42.5}
        features[1]["tags"] = {"name": "neuron_2", "volume": 88.0}

        gen, _ = _build_generator_with_features(features)
        out = str(tmp_path / "ng_multilod")
        gen.generate_neuroglancer_multilod(out, WORLD_BOUNDS)

        sp = json.loads((tmp_path / "ng_multilod" / "segment_properties" / "info").read_text())
        assert sp["@type"] == "neuroglancer_segment_properties"
        assert "inline" in sp
        assert len(sp["inline"]["ids"]) == 2
        assert len(sp["inline"]["properties"]) >= 1


class TestEndToEnd:
    """Full pipeline end-to-end test."""

    def test_end_to_end(self, tmp_path):
        """Full pipeline: add features → generate multilod → verify output structure."""
        features = [
            _make_dense_tin_feature(30),
            _make_tin_feature(
                [0.8, 0.8, 0.9, 0.8, 0.85, 0.9],
                [0.7, 0.8, 0.75],
                [3],
                tags={"name": "simple"},
            ),
        ]
        gen, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=2)
        out = str(tmp_path / "ng_multilod")
        count = gen.generate_neuroglancer_multilod(out, WORLD_BOUNDS)

        assert count == 2  # 2 segments

        out_path = tmp_path / "ng_multilod"
        assert (out_path / "info").exists()
        assert (out_path / "0.index").exists()
        assert (out_path / "0").exists()
        assert (out_path / "1.index").exists()
        assert (out_path / "1").exists()
        assert (out_path / "segment_properties" / "info").exists()

        # Both segments should have valid Draco data
        for seg_id in range(2):
            data = (out_path / str(seg_id)).read_bytes()
            assert len(data) > 0
            assert data[:5] == b"DRACO"

            index_data = (out_path / f"{seg_id}.index").read_bytes()
            parsed = _parse_index_file(index_data)
            assert parsed["num_lods"] >= 1
