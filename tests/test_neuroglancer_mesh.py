"""Tests for Neuroglancer precomputed legacy mesh export.

Covers:
    - MeshInfo model
    - mesh_to_binary / decode_mesh_binary round-trip
    - write_mesh / write_mesh_info directory structure
    - fragments_to_mesh geometry merging
    - End-to-end: StreamingTileGenerator.generate_neuroglancer()
"""

from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mudm_tools.neuroglancer.mesh_models import MeshInfo
from mudm_tools.neuroglancer.mesh_writer import (
    decode_mesh_binary,
    fragments_to_mesh,
    mesh_to_binary,
    write_mesh,
    write_mesh_info,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_triangle():
    """A single triangle mesh."""
    verts = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    indices = np.array([[0, 1, 2]], dtype=np.uint32)
    return verts, indices


@pytest.fixture
def quad_mesh():
    """Two-triangle quad mesh."""
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    return verts, indices


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ---------------------------------------------------------------------------
# TestMeshInfo
# ---------------------------------------------------------------------------


class TestMeshInfo:
    def test_default_info(self):
        info = MeshInfo()
        d = info.to_info_dict()
        assert d["@type"] == "neuroglancer_legacy_mesh"
        assert "segment_properties" not in d

    def test_with_segment_properties(self):
        info = MeshInfo(segment_properties="segment_properties")
        d = info.to_info_dict()
        assert d["@type"] == "neuroglancer_legacy_mesh"
        assert d["segment_properties"] == "segment_properties"


# ---------------------------------------------------------------------------
# TestMeshToBinary
# ---------------------------------------------------------------------------


class TestMeshToBinary:
    def test_header_num_vertices(self, simple_triangle):
        verts, indices = simple_triangle
        data = mesh_to_binary(verts, indices)
        (num_verts,) = struct.unpack_from("<I", data, 0)
        assert num_verts == 3

    def test_vertex_positions(self, simple_triangle):
        verts, indices = simple_triangle
        data = mesh_to_binary(verts, indices)
        # Skip header (4 bytes), read 9 float32s (3 verts * 3 coords)
        positions = struct.unpack_from("<9f", data, 4)
        assert positions[:3] == pytest.approx([0.0, 0.0, 0.0])
        assert positions[3:6] == pytest.approx([1.0, 0.0, 0.0])
        assert positions[6:9] == pytest.approx([0.0, 1.0, 0.0])

    def test_triangle_indices(self, simple_triangle):
        verts, indices = simple_triangle
        data = mesh_to_binary(verts, indices)
        # Header(4) + 3*3*4=36 bytes of positions → offset 40
        idx = struct.unpack_from("<3I", data, 40)
        assert idx == (0, 1, 2)

    def test_total_size(self, simple_triangle):
        verts, indices = simple_triangle
        data = mesh_to_binary(verts, indices)
        # 4 (header) + 3*3*4 (positions) + 1*3*4 (indices)
        expected = 4 + 36 + 12
        assert len(data) == expected

    def test_quad_mesh_size(self, quad_mesh):
        verts, indices = quad_mesh
        data = mesh_to_binary(verts, indices)
        # 4 (header) + 4*3*4=48 (positions) + 2*3*4=24 (indices)
        expected = 4 + 48 + 24
        assert len(data) == expected

    def test_round_trip(self, quad_mesh):
        verts, indices = quad_mesh
        data = mesh_to_binary(verts, indices)
        dec_v, dec_i = decode_mesh_binary(data)
        np.testing.assert_array_almost_equal(dec_v, verts)
        np.testing.assert_array_equal(dec_i, indices)

    def test_empty_mesh(self):
        verts = np.zeros((0, 3), dtype=np.float32)
        indices = np.zeros((0, 3), dtype=np.uint32)
        data = mesh_to_binary(verts, indices)
        (num_verts,) = struct.unpack_from("<I", data, 0)
        assert num_verts == 0
        dec_v, dec_i = decode_mesh_binary(data)
        assert dec_v.shape[0] == 0

    def test_large_mesh_roundtrip(self):
        """Test with a larger mesh to verify no off-by-one errors."""
        rng = np.random.default_rng(42)
        n_verts = 1000
        n_faces = 500
        verts = rng.random((n_verts, 3), dtype=np.float32) * 100
        indices = rng.integers(0, n_verts, size=(n_faces, 3), dtype=np.uint32)
        data = mesh_to_binary(verts, indices)
        dec_v, dec_i = decode_mesh_binary(data)
        np.testing.assert_array_almost_equal(dec_v, verts)
        np.testing.assert_array_equal(dec_i, indices)


# ---------------------------------------------------------------------------
# TestWriteMesh
# ---------------------------------------------------------------------------


class TestWriteMesh:
    def test_creates_directory(self, simple_triangle, tmp_dir):
        verts, indices = simple_triangle
        out = tmp_dir / "meshes"
        write_mesh(out, 42, verts, indices)
        assert out.is_dir()

    def test_binary_file_exists(self, simple_triangle, tmp_dir):
        verts, indices = simple_triangle
        write_mesh(tmp_dir, 42, verts, indices)
        assert (tmp_dir / "42").exists()

    def test_manifest_file_exists(self, simple_triangle, tmp_dir):
        verts, indices = simple_triangle
        write_mesh(tmp_dir, 42, verts, indices)
        manifest_path = tmp_dir / "42:0"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert "fragments" in manifest
        assert "42" in manifest["fragments"]

    def test_binary_content_correct(self, simple_triangle, tmp_dir):
        verts, indices = simple_triangle
        write_mesh(tmp_dir, 1, verts, indices)
        data = (tmp_dir / "1").read_bytes()
        dec_v, dec_i = decode_mesh_binary(data)
        np.testing.assert_array_almost_equal(dec_v, verts)
        np.testing.assert_array_equal(dec_i, indices)


class TestWriteMeshInfo:
    def test_creates_info_file(self, tmp_dir):
        write_mesh_info(tmp_dir)
        info_path = tmp_dir / "info"
        assert info_path.exists()
        info = json.loads(info_path.read_text())
        assert info["@type"] == "neuroglancer_legacy_mesh"

    def test_with_segment_properties(self, tmp_dir):
        write_mesh_info(tmp_dir, segment_properties="segment_properties")
        info = json.loads((tmp_dir / "info").read_text())
        assert info["segment_properties"] == "segment_properties"


# ---------------------------------------------------------------------------
# TestFragmentsToMesh
# ---------------------------------------------------------------------------


class TestFragmentsToMesh:
    def test_single_fragment(self):
        """Single triangle fragment in normalized space."""
        frag = {
            "xy": [0.0, 0.0, 1.0, 0.0, 0.5, 1.0],
            "z": [0.0, 0.0, 0.0],
            "ring_lengths": [3],
            "geom_type": 5,
        }
        bounds = (0.0, 0.0, 0.0, 100.0, 100.0, 100.0)
        verts, indices = fragments_to_mesh([frag], bounds)
        assert verts.shape == (3, 3)
        assert indices.shape == (1, 3)
        # Verify unprojection: 0.0*100=0, 1.0*100=100, 0.5*100=50
        assert verts[0][0] == pytest.approx(0.0, abs=0.01)
        assert verts[1][0] == pytest.approx(100.0, abs=0.01)
        assert verts[2][0] == pytest.approx(50.0, abs=0.01)

    def test_multi_fragment_vertex_dedup(self):
        """Two fragments sharing an edge should deduplicate shared vertices."""
        frag1 = {
            "xy": [0.0, 0.0, 0.5, 0.0, 0.5, 0.5],
            "z": [0.0, 0.0, 0.0],
            "ring_lengths": [3],
            "geom_type": 5,
        }
        frag2 = {
            "xy": [0.5, 0.0, 1.0, 0.0, 0.5, 0.5],
            "z": [0.0, 0.0, 0.0],
            "ring_lengths": [3],
            "geom_type": 5,
        }
        bounds = (0.0, 0.0, 0.0, 10.0, 10.0, 10.0)
        verts, indices = fragments_to_mesh([frag1, frag2], bounds)
        # Two fragments share vertex (0.5,0.0,0.0) and (0.5,0.5,0.0)
        # Total unique verts: 4 (not 6)
        assert verts.shape[0] == 4
        assert indices.shape[0] == 2  # 2 triangles

    def test_ring_length_4_closing_vertex(self):
        """Fragment with closing vertex (ring_length=4) should use only first 3."""
        frag = {
            "xy": [0.0, 0.0, 1.0, 0.0, 0.5, 1.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0, 0.0],
            "ring_lengths": [4],
            "geom_type": 5,
        }
        bounds = (0.0, 0.0, 0.0, 10.0, 10.0, 10.0)
        verts, indices = fragments_to_mesh([frag], bounds)
        assert verts.shape[0] == 3  # Only 3 unique vertices, not 4
        assert indices.shape[0] == 1

    def test_empty_fragments(self):
        verts, indices = fragments_to_mesh([], (0, 0, 0, 1, 1, 1))
        assert verts.shape == (0, 3)
        assert indices.shape == (0, 3)

    def test_world_coordinate_unprojection(self):
        """Verify correct unprojection from [0,1]³ to world space."""
        frag = {
            "xy": [0.5, 0.5, 1.0, 0.0, 0.0, 1.0],
            "z": [0.5, 0.0, 1.0],
            "ring_lengths": [3],
            "geom_type": 5,
        }
        bounds = (10.0, 20.0, 30.0, 110.0, 120.0, 130.0)
        verts, _ = fragments_to_mesh([frag], bounds)
        # v0: (10 + 0.5*100, 20 + 0.5*100, 30 + 0.5*100) = (60, 70, 80)
        np.testing.assert_array_almost_equal(
            verts[0],
            [60.0, 70.0, 80.0],
            decimal=0,
        )


# ---------------------------------------------------------------------------
# TestEndToEnd: StreamingTileGenerator → generate_neuroglancer()
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Integration tests using the Rust StreamingTileGenerator."""

    def _make_tin_feature(self, xy, z, ring_lengths, tags=None):
        """Build a feature dict for add_feature()."""
        xs = [xy[i * 2] for i in range(len(z))]
        ys = [xy[i * 2 + 1] for i in range(len(z))]
        return {
            "geometry": xy,
            "geometry_z": z,
            "type": 5,  # TIN
            "ring_lengths": ring_lengths,
            "minX": min(xs),
            "minY": min(ys),
            "minZ": min(z),
            "maxX": max(xs),
            "maxY": max(ys),
            "maxZ": max(z),
            "tags": tags or {},
        }

    def test_single_feature_produces_segment(self, tmp_dir):
        """A single TIN feature should produce one segment file."""
        from mudm_tools._rs import StreamingTileGenerator

        gen = StreamingTileGenerator(min_zoom=0, max_zoom=1)
        feat = self._make_tin_feature(
            xy=[0.1, 0.1, 0.9, 0.1, 0.5, 0.9, 0.1, 0.1],
            z=[0.5, 0.5, 0.5, 0.5],
            ring_lengths=[4],
            tags={"name": "test_mesh"},
        )
        gen.add_feature(feat)

        out = str(tmp_dir / "ng_out")
        bounds = (0.0, 0.0, 0.0, 100.0, 100.0, 100.0)
        count = gen.generate_neuroglancer(out, bounds)

        assert count >= 1
        # Check info file
        info_path = Path(out) / "info"
        assert info_path.exists()
        info = json.loads(info_path.read_text())
        assert info["@type"] == "neuroglancer_legacy_mesh"

        # Check segment binary exists
        seg_path = Path(out) / "0"
        assert seg_path.exists()
        data = seg_path.read_bytes()
        assert len(data) > 4  # At least header

        # Check manifest
        manifest_path = Path(out) / "0:0"
        assert manifest_path.exists()

    def test_multiple_features_separate_segments(self, tmp_dir):
        """Multiple features should produce separate segment files."""
        from mudm_tools._rs import StreamingTileGenerator

        gen = StreamingTileGenerator(min_zoom=0, max_zoom=1)

        for i in range(3):
            offset = i * 0.25
            feat = self._make_tin_feature(
                xy=[0.1 + offset, 0.1, 0.3 + offset, 0.1, 0.2 + offset, 0.3, 0.1 + offset, 0.1],
                z=[0.5, 0.5, 0.5, 0.5],
                ring_lengths=[4],
                tags={"feature_name": f"feature_{i}"},
            )
            gen.add_feature(feat)

        out = str(tmp_dir / "ng_out")
        bounds = (0.0, 0.0, 0.0, 100.0, 100.0, 100.0)
        count = gen.generate_neuroglancer(out, bounds)

        assert count == 3
        for i in range(3):
            assert (Path(out) / str(i)).exists()
            assert (Path(out) / f"{i}:0").exists()

    def test_segment_properties_written(self, tmp_dir):
        """Tags should be written to segment_properties/info."""
        from mudm_tools._rs import StreamingTileGenerator

        gen = StreamingTileGenerator(min_zoom=0, max_zoom=1)
        feat = self._make_tin_feature(
            xy=[0.1, 0.1, 0.9, 0.1, 0.5, 0.9, 0.1, 0.1],
            z=[0.5, 0.5, 0.5, 0.5],
            ring_lengths=[4],
            tags={"brain_region": "cortex", "body_id": 42},
        )
        gen.add_feature(feat)

        out = str(tmp_dir / "ng_out")
        gen.generate_neuroglancer(out, (0, 0, 0, 100, 100, 100))

        sp_path = Path(out) / "segment_properties" / "info"
        assert sp_path.exists()
        sp = json.loads(sp_path.read_text())
        assert sp["@type"] == "neuroglancer_segment_properties"
        assert "inline" in sp
        assert len(sp["inline"]["ids"]) >= 1

    def test_binary_format_correct(self, tmp_dir):
        """Verify the binary layout matches Neuroglancer spec."""
        from mudm_tools._rs import StreamingTileGenerator

        gen = StreamingTileGenerator(min_zoom=0, max_zoom=1)
        feat = self._make_tin_feature(
            xy=[0.1, 0.1, 0.9, 0.1, 0.5, 0.9, 0.1, 0.1],
            z=[0.5, 0.5, 0.5, 0.5],
            ring_lengths=[4],
        )
        gen.add_feature(feat)

        out = str(tmp_dir / "ng_out")
        gen.generate_neuroglancer(out, (0, 0, 0, 100, 100, 100))

        data = (Path(out) / "0").read_bytes()
        (num_verts,) = struct.unpack_from("<I", data, 0)
        assert num_verts > 0

        # Verify total size is consistent
        # 4 (header) + num_verts*3*4 (positions) + remaining = indices
        pos_bytes = num_verts * 3 * 4
        remaining = len(data) - 4 - pos_bytes
        assert remaining >= 0
        assert remaining % 4 == 0  # indices are uint32

    def test_degenerate_feature_skipped(self, tmp_dir):
        """A point-type feature should not produce a mesh segment."""
        from mudm_tools._rs import StreamingTileGenerator

        gen = StreamingTileGenerator(min_zoom=0, max_zoom=1)
        # Add a point feature (geom_type=1)
        feat = {
            "geometry": [0.5, 0.5],
            "geometry_z": [0.5],
            "type": 1,  # POINT
            "ring_lengths": None,
            "minX": 0.5,
            "minY": 0.5,
            "minZ": 0.5,
            "maxX": 0.5,
            "maxY": 0.5,
            "maxZ": 0.5,
            "tags": {},
        }
        gen.add_feature(feat)

        out = str(tmp_dir / "ng_out")
        count = gen.generate_neuroglancer(out, (0, 0, 0, 100, 100, 100))

        # Points don't produce mesh segments
        assert count == 0
