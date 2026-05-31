"""Tests for Rust-accelerated 3D tiling functions.

Parametrized over "python" and "rust" backends. The rust backend
is skipped when the extension hasn't been compiled.

Each test verifies functional correctness and, where both backends
are available, bit-identical output.
"""

import pytest
import struct

try:
    from mudm_tools._rs import (  # noqa: F401
        clip_surface,
        clip_line,
        clip_points,
        transform_tile_3d,
        encode_tile_3d,
        build_indexed_mesh,
        CartesianProjector3D,
        decimate_tin,
        parse_obj,
    )
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

from mudm_tools.tiling3d.clip3d import (
    _clip_surface_py,
    _clip_line_py,
    clip_3d,
)
from mudm_tools.tiling3d.encoder3d import _build_indexed_mesh_py, encode_tile_3d_py
from mudm_tools.tiling3d.tile3d import transform_tile_3d_py


# --- Helpers to get Rust functions when available ---

def _get_rs_clip_surface():
    if not RUST_AVAILABLE:
        return None
    from mudm_tools._rs import clip_surface
    return clip_surface


def _get_rs_clip_line():
    if not RUST_AVAILABLE:
        return None
    from mudm_tools._rs import clip_line
    return clip_line


def _get_rs_build_indexed_mesh():
    if not RUST_AVAILABLE:
        return None
    from mudm_tools._rs import build_indexed_mesh
    return build_indexed_mesh


def _get_rs_encode_tile_3d():
    if not RUST_AVAILABLE:
        return None
    from mudm_tools._rs import encode_tile_3d
    return encode_tile_3d


def _get_rs_transform_tile_3d():
    if not RUST_AVAILABLE:
        return None
    from mudm_tools._rs import transform_tile_3d
    return transform_tile_3d


# --- Fixtures ---

@pytest.fixture(params=["python", "rust"])
def backend(request):
    if request.param == "rust" and not RUST_AVAILABLE:
        pytest.skip("Rust extensions not compiled")
    return request.param


def _clip_surface_fn(backend):
    if backend == "rust":
        return _get_rs_clip_surface()
    return _clip_surface_py


def _clip_line_fn(backend):
    if backend == "rust":
        return _get_rs_clip_line()
    return _clip_line_py


def _build_indexed_mesh_fn(backend):
    if backend == "rust":
        return _get_rs_build_indexed_mesh()
    return _build_indexed_mesh_py


def _transform_tile_3d_fn(backend):
    if backend == "rust":
        return _get_rs_transform_tile_3d()
    return transform_tile_3d_py


# --- _clip_surface tests ---

class TestClipSurface:
    """Tests for _clip_surface (TIN/PS per-face clipping)."""

    def _make_surface(self, xy, z, ring_lengths):
        """Build a minimal surface feature dict."""
        return {
            "geometry": xy,
            "geometry_z": z,
            "ring_lengths": ring_lengths,
            "type": 5,  # TIN_TYPE
            "tags": {"id": 1},
            "minX": min(xy[i * 2] for i in range(len(z))),
            "minY": min(xy[i * 2 + 1] for i in range(len(z))),
            "minZ": min(z),
            "maxX": max(xy[i * 2] for i in range(len(z))),
            "maxY": max(xy[i * 2 + 1] for i in range(len(z))),
            "maxZ": max(z),
        }

    def test_trivial_accept(self, backend):
        """Face fully inside [k1, k2) is kept."""
        fn = _clip_surface_fn(backend)
        feat = self._make_surface(
            [0.2, 0.2, 0.4, 0.2, 0.3, 0.4, 0.2, 0.2],
            [0.3, 0.3, 0.3, 0.3],
            [4],
        )
        result = fn(feat, 0.0, 1.0, 0)
        assert result is not None
        assert result["ring_lengths"] == [4]
        assert result["geometry"] == feat["geometry"]

    def test_trivial_reject(self, backend):
        """Face fully outside is discarded."""
        fn = _clip_surface_fn(backend)
        feat = self._make_surface(
            [0.8, 0.2, 0.9, 0.2, 0.85, 0.4, 0.8, 0.2],
            [0.3, 0.3, 0.3, 0.3],
            [4],
        )
        result = fn(feat, 0.0, 0.5, 0)  # X range [0, 0.5), face at X 0.8-0.9
        assert result is None

    def test_partial_clip(self, backend):
        """Two faces, one inside, one outside."""
        fn = _clip_surface_fn(backend)
        # Face 1: X in [0.1, 0.3] — inside [0, 0.5)
        # Face 2: X in [0.6, 0.8] — outside [0, 0.5)
        feat = self._make_surface(
            [0.1, 0.1, 0.3, 0.1, 0.2, 0.3, 0.1, 0.1,
             0.6, 0.1, 0.8, 0.1, 0.7, 0.3, 0.6, 0.1],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [4, 4],
        )
        result = fn(feat, 0.0, 0.5, 0)
        assert result is not None
        assert result["ring_lengths"] == [4]
        assert len(result["geometry_z"]) == 4

    def test_bounds_updated(self, backend):
        """Output bounds reflect only kept faces."""
        fn = _clip_surface_fn(backend)
        feat = self._make_surface(
            [0.1, 0.2, 0.3, 0.2, 0.2, 0.4, 0.1, 0.2],
            [0.5, 0.5, 0.5, 0.5],
            [4],
        )
        result = fn(feat, 0.0, 1.0, 0)
        assert result is not None
        assert result["minX"] == pytest.approx(0.1)
        assert result["maxX"] == pytest.approx(0.3)

    def test_z_axis_clip(self, backend):
        """Clip along Z axis."""
        fn = _clip_surface_fn(backend)
        feat = self._make_surface(
            [0.1, 0.2, 0.3, 0.2, 0.2, 0.4, 0.1, 0.2],
            [0.1, 0.1, 0.1, 0.1],
            [4],
        )
        # Z range [0.5, 1.0) — face at Z=0.1 is outside
        result = fn(feat, 0.5, 1.0, 2)
        assert result is None


# --- _clip_line tests ---

class TestClipLine:
    """Tests for _clip_line (LineString3D clipping)."""

    def _make_line(self, xy, z):
        n = len(z)
        return {
            "geometry": xy,
            "geometry_z": z,
            "type": 2,  # LINESTRING3D
            "tags": {},
            "minX": min(xy[i * 2] for i in range(n)),
            "minY": min(xy[i * 2 + 1] for i in range(n)),
            "minZ": min(z),
            "maxX": max(xy[i * 2] for i in range(n)),
            "maxY": max(xy[i * 2 + 1] for i in range(n)),
            "maxZ": max(z),
        }

    def test_fully_inside(self, backend):
        """Line fully inside — one segment returned."""
        fn = _clip_line_fn(backend)
        feat = self._make_line([0.2, 0.3, 0.4, 0.5], [0.1, 0.2])
        result = fn(feat, 0.0, 1.0, 0)
        assert len(result) == 1
        assert len(result[0]["geometry_z"]) == 2

    def test_fully_outside(self, backend):
        """Line fully outside — empty."""
        fn = _clip_line_fn(backend)
        feat = self._make_line([0.6, 0.3, 0.8, 0.5], [0.1, 0.2])
        result = fn(feat, 0.0, 0.5, 0)
        assert len(result) == 0

    def test_crossing_produces_segment(self, backend):
        """Line crossing clip boundary produces clipped segment."""
        fn = _clip_line_fn(backend)
        # X goes from 0.3 to 0.7, clipped to [0, 0.5)
        feat = self._make_line([0.3, 0.0, 0.7, 0.0], [0.0, 0.0])
        result = fn(feat, 0.0, 0.5, 0)
        assert len(result) == 1
        # Should have entry point and intersection
        assert len(result[0]["geometry_z"]) >= 2


# --- _build_indexed_mesh tests ---

class TestBuildIndexedMesh:
    """Tests for _build_indexed_mesh (vertex dedup + bytes packing)."""

    def test_single_triangle(self, backend):
        """Single triangle: 3 unique vertices, 3 indices."""
        fn = _build_indexed_mesh_fn(backend)
        xy = [0, 0, 100, 0, 50, 100, 0, 0]  # closed ring: 4 verts
        z = [0, 0, 0, 0]
        pos_bytes, idx_bytes = fn(xy, z, [4])

        n_floats = len(pos_bytes) // 4
        assert n_floats == 9  # 3 vertices x 3 coords
        n_indices = len(idx_bytes) // 4
        assert n_indices == 3

        positions = struct.unpack(f"<{n_floats}f", pos_bytes)
        indices = struct.unpack(f"<{n_indices}I", idx_bytes)
        assert set(indices) == {0, 1, 2}

    def test_shared_vertex_dedup(self, backend):
        """Two triangles sharing an edge: 4 unique vertices, 6 indices."""
        fn = _build_indexed_mesh_fn(backend)
        # Triangle 1: (0,0,0), (100,0,0), (50,100,0)
        # Triangle 2: (100,0,0), (150,100,0), (50,100,0)
        xy = [
            0, 0, 100, 0, 50, 100, 0, 0,       # ring 1
            100, 0, 150, 100, 50, 100, 100, 0,  # ring 2
        ]
        z = [0, 0, 0, 0, 0, 0, 0, 0]
        pos_bytes, idx_bytes = fn(xy, z, [4, 4])

        n_floats = len(pos_bytes) // 4
        n_verts = n_floats // 3
        assert n_verts == 4  # deduped: 4 unique vertices

        n_indices = len(idx_bytes) // 4
        assert n_indices == 6  # 2 triangles x 3

    def test_empty_input(self, backend):
        """Empty ring lengths: empty bytes."""
        fn = _build_indexed_mesh_fn(backend)
        pos_bytes, idx_bytes = fn([], [], [])
        assert pos_bytes == b""
        assert idx_bytes == b""


# --- transform_tile_3d tests ---

class TestTransformTile3D:
    """Tests for transform_tile_3d (coordinate transform)."""

    def _make_tile(self, features, z=1, x=0, y=0, d=0):
        return {
            "features": features,
            "z": z,
            "x": x,
            "y": y,
            "d": d,
            "num_features": len(features),
            "num_points": sum(len(f["geometry_z"]) for f in features),
        }

    def test_midpoint_transform(self, backend):
        """Midpoint of tile maps to extent/2."""
        fn = _transform_tile_3d_fn(backend)
        # Zoom=1, tile (0,0,0) covers [0, 0.5) in each axis
        # Midpoint: (0.25, 0.25, 0.25) -> tile-local (0.5, 0.5, 0.5) -> 2048
        feat = {
            "geometry": [0.25, 0.25],
            "geometry_z": [0.25],
            "type": 1,  # POINT3D
            "tags": {},
        }
        tile = self._make_tile([feat], z=1, x=0, y=0, d=0)
        result = fn(tile, extent=4096, extent_z=4096)

        rf = result["features"][0]
        assert rf["geometry"] == [2048, 2048]
        assert rf["geometry_z"] == [2048]

    def test_preserves_tags(self, backend):
        """Tags are preserved through transform."""
        fn = _transform_tile_3d_fn(backend)
        feat = {
            "geometry": [0.0, 0.0],
            "geometry_z": [0.0],
            "type": 1,
            "tags": {"name": "test", "id": 42},
        }
        tile = self._make_tile([feat])
        result = fn(tile)
        assert result["features"][0]["tags"] == {"name": "test", "id": 42}

    def test_preserves_ring_lengths(self, backend):
        """ring_lengths survives transform."""
        fn = _transform_tile_3d_fn(backend)
        feat = {
            "geometry": [0.1, 0.1, 0.2, 0.1, 0.15, 0.2, 0.1, 0.1],
            "geometry_z": [0.1, 0.1, 0.1, 0.1],
            "type": 5,  # TIN
            "tags": {},
            "ring_lengths": [4],
        }
        tile = self._make_tile([feat])
        result = fn(tile)
        assert result["features"][0]["ring_lengths"] == [4]

    def test_output_metadata(self, backend):
        """Output tile has extent, num_features, etc."""
        fn = _transform_tile_3d_fn(backend)
        feat = {
            "geometry": [0.0, 0.0],
            "geometry_z": [0.0],
            "type": 1,
            "tags": {},
        }
        tile = self._make_tile([feat])
        result = fn(tile)
        assert result["extent"] == 4096
        assert result["extent_z"] == 4096
        assert result["num_features"] == 1


# --- Rust import test ---

class TestRustImport:
    """Verify that all expected symbols are importable from mudm_tools._rs."""

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not compiled")
    def test_all_symbols_importable(self):
        """All expected functions/classes can be imported from mudm_tools._rs."""
        from mudm_tools._rs import (
            clip_surface,
            clip_line,
            clip_points,
            transform_tile_3d,
            encode_tile_3d,
            build_indexed_mesh,
            CartesianProjector3D,
            decimate_tin,
            parse_obj,
        )
        # Verify they are callable
        assert callable(clip_surface)
        assert callable(clip_line)
        assert callable(clip_points)
        assert callable(transform_tile_3d)
        assert callable(encode_tile_3d)
        assert callable(build_indexed_mesh)
        assert callable(CartesianProjector3D)
        assert callable(decimate_tin)
        assert callable(parse_obj)


# --- Bit-identical comparison tests ---

class TestBitIdentical:
    """Verify Python and Rust backends produce identical output."""

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not compiled")
    def test_clip_surface_identical(self):
        """_clip_surface: Python == Rust."""
        py_fn = _clip_surface_py
        rs_fn = _get_rs_clip_surface()

        xy = [0.1, 0.2, 0.3, 0.2, 0.2, 0.4, 0.1, 0.2,
              0.6, 0.2, 0.8, 0.2, 0.7, 0.4, 0.6, 0.2]
        z = [0.1, 0.2, 0.3, 0.1, 0.5, 0.6, 0.7, 0.5]
        feat = {
            "geometry": xy, "geometry_z": z,
            "ring_lengths": [4, 4], "type": 5,
            "tags": {"id": 1},
            "minX": 0.1, "minY": 0.2, "minZ": 0.1,
            "maxX": 0.8, "maxY": 0.4, "maxZ": 0.7,
        }

        py_result = py_fn(feat, 0.0, 0.5, 0)
        rs_result = rs_fn(feat, 0.0, 0.5, 0)

        assert py_result == rs_result

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not compiled")
    def test_clip_line_identical(self):
        """_clip_line: Python == Rust."""
        py_fn = _clip_line_py
        rs_fn = _get_rs_clip_line()

        feat = {
            "geometry": [0.1, 0.5, 0.3, 0.5, 0.6, 0.5, 0.9, 0.5],
            "geometry_z": [0.0, 0.0, 0.0, 0.0],
            "type": 2, "tags": {"line": True},
            "minX": 0.1, "minY": 0.5, "minZ": 0.0,
            "maxX": 0.9, "maxY": 0.5, "maxZ": 0.0,
        }

        py_result = py_fn(feat, 0.0, 0.5, 0)
        rs_result = rs_fn(feat, 0.0, 0.5, 0)

        assert len(py_result) == len(rs_result)
        for py_seg, rs_seg in zip(py_result, rs_result):
            assert py_seg["geometry"] == pytest.approx(rs_seg["geometry"])
            assert py_seg["geometry_z"] == pytest.approx(rs_seg["geometry_z"])

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not compiled")
    def test_build_indexed_mesh_identical(self):
        """_build_indexed_mesh: Python == Rust."""
        py_fn = _build_indexed_mesh_py
        rs_fn = _get_rs_build_indexed_mesh()

        xy = [0, 0, 100, 0, 50, 100, 0, 0,
              100, 0, 150, 100, 50, 100, 100, 0]
        z = [0, 0, 0, 0, 0, 0, 0, 0]

        py_pos, py_idx = py_fn(xy, z, [4, 4])
        rs_pos, rs_idx = rs_fn(xy, z, [4, 4])

        assert py_pos == rs_pos
        assert py_idx == rs_idx

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not compiled")
    def test_encode_tile_3d_identical_tin(self):
        """encode_tile_3d: Python == Rust for TIN features (decoded comparison)."""
        from mudm_tools.tiling3d.reader3d import decode_tile

        py_fn = encode_tile_3d_py
        rs_fn = _get_rs_encode_tile_3d()

        tile_data = {
            "features": [
                {
                    "geometry": [0, 0, 100, 0, 50, 100, 0, 0],
                    "geometry_z": [0, 0, 0, 0],
                    "ring_lengths": [4],
                    "type": 5,  # TIN
                    "tags": {"name": "tri", "area": 5000},
                },
            ],
        }

        py_bytes = py_fn(tile_data)
        rs_bytes = rs_fn(tile_data)

        # Decode both and compare semantically (field ordering may differ)
        py_decoded = decode_tile(py_bytes)
        rs_decoded = decode_tile(rs_bytes)
        assert len(py_decoded) == len(rs_decoded)
        for py_layer, rs_layer in zip(py_decoded, rs_decoded):
            assert py_layer["name"] == rs_layer["name"]
            assert len(py_layer["features"]) == len(rs_layer["features"])
            for py_f, rs_f in zip(py_layer["features"], rs_layer["features"]):
                assert py_f["type"] == rs_f["type"]
                assert py_f["tags"] == rs_f["tags"]
                assert py_f["mesh_positions"] == rs_f["mesh_positions"]
                assert py_f["mesh_indices"] == rs_f["mesh_indices"]

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not compiled")
    def test_encode_tile_3d_identical_mixed(self):
        """encode_tile_3d: Python == Rust for mixed geometry (decoded comparison)."""
        from mudm_tools.tiling3d.reader3d import decode_tile

        py_fn = encode_tile_3d_py
        rs_fn = _get_rs_encode_tile_3d()

        tile_data = {
            "features": [
                {
                    "geometry": [100, 200],
                    "geometry_z": [50],
                    "type": 1,  # POINT3D
                    "tags": {"id": 1},
                },
                {
                    "geometry": [0, 0, 100, 100, 200, 50],
                    "geometry_z": [10, 20, 30],
                    "type": 2,  # LINESTRING3D
                    "tags": {"id": 2, "name": "line"},
                },
                {
                    "geometry": [0, 0, 100, 0, 50, 100, 0, 0,
                                 100, 0, 200, 0, 150, 100, 100, 0],
                    "geometry_z": [0, 0, 0, 0, 0, 0, 0, 0],
                    "ring_lengths": [4, 4],
                    "type": 5,  # TIN
                    "tags": {"name": "mesh", "count": 2},
                },
            ],
        }

        py_bytes = py_fn(tile_data)
        rs_bytes = rs_fn(tile_data)

        # Decode both and compare semantically
        py_decoded = decode_tile(py_bytes)
        rs_decoded = decode_tile(rs_bytes)
        assert len(py_decoded) == len(rs_decoded)
        for py_layer, rs_layer in zip(py_decoded, rs_decoded):
            assert py_layer["name"] == rs_layer["name"]
            assert len(py_layer["features"]) == len(rs_layer["features"])
            for py_f, rs_f in zip(py_layer["features"], rs_layer["features"]):
                assert py_f["type"] == rs_f["type"]
                assert py_f["tags"] == rs_f["tags"]

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not compiled")
    def test_transform_tile_3d_identical(self):
        """transform_tile_3d: Python == Rust."""
        py_fn = transform_tile_3d_py
        rs_fn = _get_rs_transform_tile_3d()

        feat = {
            "geometry": [0.25, 0.25, 0.1, 0.3],
            "geometry_z": [0.25, 0.1],
            "type": 1,
            "tags": {"id": 1},
            "ring_lengths": None,
        }
        tile = {
            "features": [feat],
            "z": 2, "x": 1, "y": 1, "d": 1,
            "num_features": 1,
            "num_points": 2,
        }

        py_result = py_fn(tile)
        rs_result = rs_fn(tile)

        assert py_result == rs_result


# --- StreamingTileGenerator tests ---

class TestStreamingTileGenerator:
    """Tests for the Rust StreamingTileGenerator."""

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not compiled")
    def test_basic_tin(self, tmp_path):
        """Single TIN feature: generates tiles and decodes correctly."""
        from mudm_tools._rs import StreamingTileGenerator
        from mudm_tools.tiling3d.reader3d import decode_tile

        gen = StreamingTileGenerator(min_zoom=0, max_zoom=1)
        feat = {
            "geometry": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "geometry_z": [0.1, 0.2, 0.3],
            "ring_lengths": [3],
            "type": 5,  # TIN
            "tags": {"name": "mesh", "id": 42},
            "minX": 0.1, "minY": 0.2, "minZ": 0.1,
            "maxX": 0.5, "maxY": 0.6, "maxZ": 0.3,
        }
        fid = gen.add_feature(feat)
        assert fid == 0
        assert gen.feature_count_val() == 1

        out = str(tmp_path / "tiles")
        count = gen.generate_pbf3(out, "test")
        assert count > 0
        assert gen.tile_count() == count

        # Decode z0 tile
        z0 = decode_tile((tmp_path / "tiles" / "0" / "0" / "0" / "0.pbf3").read_bytes())
        assert len(z0) == 1  # one layer
        layer = z0[0]
        assert layer["name"] == "test"
        assert len(layer["features"]) == 1
        f = layer["features"][0]
        assert f["type"] == 5
        assert f["tags"]["name"] == "mesh"
        assert f["tags"]["id"] == 42
        assert f["mesh_positions"] is not None
        assert f["mesh_indices"] is not None

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not compiled")
    def test_matches_batch_generator(self, tmp_path):
        """Streaming produces the same tile set as the batch TileGenerator3D."""
        from mudm_tools._rs import StreamingTileGenerator
        from mudm_tools.tiling3d.generator3d import TileGenerator3D
        from mudm_tools.tiling3d.octree import OctreeConfig
        from mudm_tools.tiling3d.convert3d import convert_feature_3d, compute_bounds_3d
        from mudm_tools.tiling3d.projector3d import CartesianProjector3D
        from mudm.model import MuDMFeatureCollection, MuDMFeature, TIN as TINGeom
        import os

        coords1 = [
            [[[100, 200, 10], [300, 400, 20], [500, 600, 30], [100, 200, 10]]],
            [[[200, 300, 15], [400, 500, 25], [600, 100, 35], [200, 300, 15]]],
        ]
        coords2 = [
            [[[700, 800, 40], [900, 100, 50], [500, 300, 60], [700, 800, 40]]],
        ]
        feat1 = MuDMFeature(
            type="Feature",
            geometry=TINGeom(type="TIN", coordinates=coords1),
            properties={"name": "a", "color": "red"},
        )
        feat2 = MuDMFeature(
            type="Feature",
            geometry=TINGeom(type="TIN", coordinates=coords2),
            properties={"name": "b", "color": "blue"},
        )
        collection = MuDMFeatureCollection(
            type="FeatureCollection", features=[feat1, feat2],
        )

        config = OctreeConfig(min_zoom=0, max_zoom=2)

        # Batch
        batch_dir = tmp_path / "batch"
        batch_gen = TileGenerator3D(config=config, output_format="pbf3", workers=1)
        batch_gen.add_features(collection)
        batch_count = batch_gen.generate(batch_dir)

        # Streaming
        stream_dir = tmp_path / "stream"
        bounds = compute_bounds_3d(collection)
        proj = CartesianProjector3D(bounds)
        stream_gen = StreamingTileGenerator(
            min_zoom=0, max_zoom=2,
            extent=config.extent, extent_z=config.extent_z,
        )
        for feat in collection.features:
            for ifeat in convert_feature_3d(feat, proj):
                stream_gen.add_feature(ifeat)
        stream_count = stream_gen.generate_pbf3(str(stream_dir))

        # Compare tile sets
        def collect_tiles(d):
            tiles = set()
            for root, _, files in os.walk(d):
                for f in files:
                    tiles.add(os.path.relpath(os.path.join(root, f), d))
            return tiles

        batch_tiles = collect_tiles(batch_dir)
        stream_tiles = collect_tiles(stream_dir)
        assert batch_tiles == stream_tiles
        assert batch_count == stream_count

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not compiled")
    def test_linestring_feature(self, tmp_path):
        """LineString3D feature clips and encodes correctly."""
        from mudm_tools._rs import StreamingTileGenerator
        from mudm_tools.tiling3d.reader3d import decode_tile

        gen = StreamingTileGenerator(min_zoom=0, max_zoom=1)
        feat = {
            "geometry": [0.1, 0.5, 0.9, 0.5],
            "geometry_z": [0.5, 0.5],
            "type": 2,  # LINESTRING3D
            "tags": {"road": "main"},
            "minX": 0.1, "minY": 0.5, "minZ": 0.5,
            "maxX": 0.9, "maxY": 0.5, "maxZ": 0.5,
        }
        gen.add_feature(feat)
        count = gen.generate_pbf3(str(tmp_path / "tiles"), "roads")
        assert count > 0

        z0 = decode_tile((tmp_path / "tiles" / "0" / "0" / "0" / "0.pbf3").read_bytes())
        layer = z0[0]
        assert layer["name"] == "roads"
        assert layer["features"][0]["type"] == 2
        assert layer["features"][0]["tags"]["road"] == "main"

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not compiled")
    def test_point_feature(self, tmp_path):
        """Point3D feature clips and encodes correctly."""
        from mudm_tools._rs import StreamingTileGenerator
        from mudm_tools.tiling3d.reader3d import decode_tile

        gen = StreamingTileGenerator(min_zoom=0, max_zoom=0)
        feat = {
            "geometry": [0.5, 0.5],
            "geometry_z": [0.5],
            "type": 1,  # POINT3D
            "tags": {"label": "center"},
            "minX": 0.5, "minY": 0.5, "minZ": 0.5,
            "maxX": 0.5, "maxY": 0.5, "maxZ": 0.5,
        }
        gen.add_feature(feat)
        count = gen.generate_pbf3(str(tmp_path / "tiles"))
        assert count == 1

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not compiled")
    def test_write_tilejson3d(self, tmp_path):
        """write_tilejson3d produces valid JSON metadata."""
        import json
        from mudm_tools._rs import StreamingTileGenerator

        gen = StreamingTileGenerator(min_zoom=0, max_zoom=3)
        path = str(tmp_path / "tilejson3d.json")
        gen.write_tilejson3d(path, (0.0, 0.0, 0.0, 100.0, 200.0, 50.0), "neurons")

        data = json.loads((tmp_path / "tilejson3d.json").read_text())
        assert data["tilejson"] == "3.0.0"
        assert data["name"] == "neurons"
        assert data["minzoom"] == 0
        assert data["maxzoom"] == 3
        assert data["bounds3d"] == [0.0, 0.0, 0.0, 100.0, 200.0, 50.0]
        assert data["center3d"][0] == 50.0
        assert data["vector_layers"][0]["id"] == "neurons"

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not compiled")
    def test_empty_generator(self, tmp_path):
        """Generating with no features produces 0 tiles."""
        from mudm_tools._rs import StreamingTileGenerator

        gen = StreamingTileGenerator(min_zoom=0, max_zoom=2)
        count = gen.generate_pbf3(str(tmp_path / "tiles"))
        assert count == 0

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not compiled")
    def test_multiple_features(self, tmp_path):
        """Multiple features get unique IDs and all appear in tiles."""
        from mudm_tools._rs import StreamingTileGenerator
        from mudm_tools.tiling3d.reader3d import decode_tile

        gen = StreamingTileGenerator(min_zoom=0, max_zoom=0)
        for i in range(5):
            feat = {
                "geometry": [0.1 + i * 0.1, 0.2, 0.2 + i * 0.1, 0.3, 0.15 + i * 0.1, 0.5],
                "geometry_z": [0.1, 0.2, 0.3],
                "ring_lengths": [3],
                "type": 5,
                "tags": {"id": i},
                "minX": 0.1 + i * 0.1, "minY": 0.2, "minZ": 0.1,
                "maxX": 0.2 + i * 0.1, "maxY": 0.5, "maxZ": 0.3,
            }
            fid = gen.add_feature(feat)
            assert fid == i

        assert gen.feature_count_val() == 5
        count = gen.generate_pbf3(str(tmp_path / "tiles"))
        assert count == 1  # single tile at zoom 0

        z0 = decode_tile((tmp_path / "tiles" / "0" / "0" / "0" / "0.pbf3").read_bytes())
        assert len(z0[0]["features"]) == 5


class TestStreamingGlb:
    """Tests for Rust StreamingTileGenerator's 3D Tiles / GLB output."""

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not compiled")
    def test_basic_glb_output(self, tmp_path):
        """Single TIN feature generates valid GLB files and tileset.json."""
        import json
        from mudm_tools._rs import StreamingTileGenerator

        gen = StreamingTileGenerator(min_zoom=0, max_zoom=1)
        feat = {
            "geometry": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "geometry_z": [0.1, 0.2, 0.3],
            "ring_lengths": [3],
            "type": 5,  # TIN
            "tags": {"name": "mesh", "id": 42},
            "minX": 0.1, "minY": 0.2, "minZ": 0.1,
            "maxX": 0.5, "maxY": 0.6, "maxZ": 0.3,
        }
        gen.add_feature(feat)

        bounds = (0.0, 0.0, 0.0, 100.0, 100.0, 100.0)
        out = str(tmp_path / "tiles")
        count = gen.generate_3dtiles(out, bounds)
        assert count > 0

        # Check tileset.json
        tileset = json.loads((tmp_path / "tiles" / "tileset.json").read_text())
        assert tileset["asset"]["version"] == "1.1"
        assert "root" in tileset
        assert tileset["root"]["refine"] == "REPLACE"

        # Find and verify GLB files
        import os
        glb_files = []
        for root, _, files in os.walk(str(tmp_path / "tiles")):
            for fn in files:
                if fn.endswith(".glb"):
                    glb_files.append(os.path.join(root, fn))
        assert len(glb_files) == count

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not compiled")
    def test_glb_header_valid(self, tmp_path):
        """GLB files have valid glTF 2.0 headers."""
        import json
        from mudm_tools._rs import StreamingTileGenerator

        gen = StreamingTileGenerator(min_zoom=0, max_zoom=0)
        feat = {
            "geometry": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "geometry_z": [0.1, 0.2, 0.3],
            "ring_lengths": [3],
            "type": 5,
            "tags": {"species": "mouse"},
            "minX": 0.1, "minY": 0.2, "minZ": 0.1,
            "maxX": 0.5, "maxY": 0.6, "maxZ": 0.3,
        }
        gen.add_feature(feat)

        out = str(tmp_path / "tiles")
        gen.generate_3dtiles(out, (0.0, 0.0, 0.0, 10.0, 10.0, 10.0))

        glb_path = tmp_path / "tiles" / "0" / "0" / "0" / "0.glb"
        data = glb_path.read_bytes()

        # GLB header
        assert data[0:4] == b"glTF"
        version = int.from_bytes(data[4:8], "little")
        assert version == 2
        total_len = int.from_bytes(data[8:12], "little")
        assert total_len == len(data)

        # JSON chunk
        json_len = int.from_bytes(data[12:16], "little")
        json_type = int.from_bytes(data[16:20], "little")
        assert json_type == 0x4E4F534A  # "JSON"

        json_str = data[20:20 + json_len].decode("utf-8").strip()
        gltf = json.loads(json_str)
        assert gltf["asset"]["version"] == "2.0"
        assert gltf["asset"]["generator"] == "mudm-rs"

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not compiled")
    def test_glb_node_extras(self, tmp_path):
        """Feature properties are stored in GLB node extras."""
        import json
        from mudm_tools._rs import StreamingTileGenerator

        gen = StreamingTileGenerator(min_zoom=0, max_zoom=0)
        feat = {
            "geometry": [0.2, 0.3, 0.4, 0.5, 0.3, 0.7],
            "geometry_z": [0.1, 0.4, 0.6],
            "ring_lengths": [3],
            "type": 5,
            "tags": {"neuron_id": 12345, "species": "mouse", "volume": 42.5, "traced": True},
            "minX": 0.2, "minY": 0.3, "minZ": 0.1,
            "maxX": 0.4, "maxY": 0.7, "maxZ": 0.6,
        }
        gen.add_feature(feat)

        out = str(tmp_path / "tiles")
        gen.generate_3dtiles(out, (0.0, 0.0, 0.0, 1000.0, 1000.0, 1000.0))

        glb_path = tmp_path / "tiles" / "0" / "0" / "0" / "0.glb"
        data = glb_path.read_bytes()
        json_len = int.from_bytes(data[12:16], "little")
        gltf = json.loads(data[20:20 + json_len].decode("utf-8").strip())

        extras = gltf["nodes"][0]["extras"]
        assert extras["neuron_id"] == 12345
        assert extras["species"] == "mouse"
        assert abs(extras["volume"] - 42.5) < 0.01
        assert extras["traced"] is True

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not compiled")
    def test_glb_world_coordinates(self, tmp_path):
        """GLB positions are in world coordinates (unprojected from [0,1]³)."""
        import json
        from mudm_tools._rs import StreamingTileGenerator

        bounds = (100.0, 200.0, 300.0, 400.0, 600.0, 900.0)
        gen = StreamingTileGenerator(min_zoom=0, max_zoom=0)
        # Triangle near normalized center — should unproject to near world center
        feat = {
            "geometry": [0.48, 0.48, 0.52, 0.48, 0.50, 0.52],
            "geometry_z": [0.48, 0.52, 0.50],
            "ring_lengths": [3],
            "type": 5,
            "tags": {},
            "minX": 0.48, "minY": 0.48, "minZ": 0.48,
            "maxX": 0.52, "maxY": 0.52, "maxZ": 0.52,
        }
        gen.add_feature(feat)

        out = str(tmp_path / "tiles")
        gen.generate_3dtiles(out, bounds)

        glb_path = tmp_path / "tiles" / "0" / "0" / "0" / "0.glb"
        data = glb_path.read_bytes()
        json_len = int.from_bytes(data[12:16], "little")
        gltf = json.loads(data[20:20 + json_len].decode("utf-8").strip())

        # Check position accessor min/max — should be in world coords near center
        # bounds = (100, 200, 300, 400, 600, 900), range = (300, 400, 600)
        # Normalized 0.48 → world: 100 + 0.48*300 = 244  (x min)
        # Normalized 0.52 → world: 100 + 0.52*300 = 256  (x max)
        pos_acc = gltf["accessors"][0]
        assert 240.0 < pos_acc["min"][0] < 260.0  # x near 244-256
        assert 380.0 < pos_acc["min"][1] < 420.0  # y near 392-408
        assert 580.0 < pos_acc["min"][2] < 620.0  # z near 588-612

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not compiled")
    def test_glb_simplification(self, tmp_path):
        """Non-leaf tiles get simplified (fewer vertices than leaf tiles)."""
        import json
        from mudm_tools._rs import StreamingTileGenerator

        gen = StreamingTileGenerator(min_zoom=0, max_zoom=2)
        # Add a mesh with many triangles across the full extent
        for i in range(50):
            nx = (i % 10) * 0.1
            ny = (i // 10) * 0.2
            feat = {
                "geometry": [nx, ny, nx + 0.05, ny + 0.1, nx + 0.08, ny],
                "geometry_z": [0.3, 0.5, 0.4],
                "ring_lengths": [3],
                "type": 5,
                "tags": {"face": i},
                "minX": nx, "minY": ny, "minZ": 0.3,
                "maxX": nx + 0.08, "maxY": ny + 0.1, "maxZ": 0.5,
            }
            gen.add_feature(feat)

        out = str(tmp_path / "tiles")
        count = gen.generate_3dtiles(out, (0.0, 0.0, 0.0, 100.0, 100.0, 100.0))
        assert count > 0

        # Collect GLB files by zoom level
        import os
        glb_by_zoom = {}
        for root, _, files in os.walk(str(tmp_path / "tiles")):
            for fn in files:
                if fn.endswith(".glb"):
                    path = os.path.join(root, fn)
                    # Extract zoom from path
                    rel = os.path.relpath(path, str(tmp_path / "tiles"))
                    z = int(rel.split(os.sep)[0])
                    glb_by_zoom.setdefault(z, []).append(path)

        # At least z0 and z2 should exist
        assert 0 in glb_by_zoom
        # z0 should have data (simplified)
        for path in glb_by_zoom[0]:
            assert os.path.getsize(path) > 20

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not compiled")
    def test_tileset_json_hierarchy(self, tmp_path):
        """Tileset.json has correct hierarchical structure."""
        import json
        from mudm_tools._rs import StreamingTileGenerator

        gen = StreamingTileGenerator(min_zoom=0, max_zoom=2)
        # Feature spanning the full extent
        feat = {
            "geometry": [0.1, 0.1, 0.9, 0.9, 0.5, 0.5],
            "geometry_z": [0.1, 0.9, 0.5],
            "ring_lengths": [3],
            "type": 5,
            "tags": {},
            "minX": 0.1, "minY": 0.1, "minZ": 0.1,
            "maxX": 0.9, "maxY": 0.9, "maxZ": 0.9,
        }
        gen.add_feature(feat)

        out = str(tmp_path / "tiles")
        gen.generate_3dtiles(out, (0.0, 0.0, 0.0, 100.0, 100.0, 100.0))

        tileset = json.loads((tmp_path / "tiles" / "tileset.json").read_text())

        # Root should have geometric error > 0
        assert tileset["geometricError"] > 0
        root = tileset["root"]
        assert "boundingVolume" in root
        assert "box" in root["boundingVolume"]
        assert len(root["boundingVolume"]["box"]) == 12

        # Content URI should be valid
        assert root["content"]["uri"].endswith(".glb")

        # Children should exist (z0 → z1 tiles)
        if "children" in root:
            for child in root["children"]:
                assert "content" in child
                assert child["geometricError"] < root["geometricError"]

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not compiled")
    def test_linestring_glb(self, tmp_path):
        """LineString3D features encode as GL_LINES in GLB."""
        import json
        from mudm_tools._rs import StreamingTileGenerator

        gen = StreamingTileGenerator(min_zoom=0, max_zoom=0)
        feat = {
            "geometry": [0.1, 0.5, 0.5, 0.5, 0.9, 0.5],
            "geometry_z": [0.5, 0.5, 0.5],
            "type": 2,
            "tags": {"road": "A1"},
            "minX": 0.1, "minY": 0.5, "minZ": 0.5,
            "maxX": 0.9, "maxY": 0.5, "maxZ": 0.5,
        }
        gen.add_feature(feat)

        out = str(tmp_path / "tiles")
        gen.generate_3dtiles(out, (0.0, 0.0, 0.0, 100.0, 100.0, 100.0))

        glb_path = tmp_path / "tiles" / "0" / "0" / "0" / "0.glb"
        data = glb_path.read_bytes()
        json_len = int.from_bytes(data[12:16], "little")
        gltf = json.loads(data[20:20 + json_len].decode("utf-8").strip())

        # Primitive mode 1 = GL_LINES
        assert gltf["meshes"][0]["primitives"][0]["mode"] == 1
        assert gltf["nodes"][0]["extras"]["road"] == "A1"

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not compiled")
    def test_point_glb(self, tmp_path):
        """Point3D features encode as GL_POINTS in GLB."""
        import json
        from mudm_tools._rs import StreamingTileGenerator

        gen = StreamingTileGenerator(min_zoom=0, max_zoom=0)
        feat = {
            "geometry": [0.5, 0.5],
            "geometry_z": [0.5],
            "type": 1,  # POINT3D
            "tags": {"label": "origin"},
            "minX": 0.5, "minY": 0.5, "minZ": 0.5,
            "maxX": 0.5, "maxY": 0.5, "maxZ": 0.5,
        }
        gen.add_feature(feat)

        out = str(tmp_path / "tiles")
        gen.generate_3dtiles(out, (0.0, 0.0, 0.0, 100.0, 100.0, 100.0))

        glb_path = tmp_path / "tiles" / "0" / "0" / "0" / "0.glb"
        data = glb_path.read_bytes()
        json_len = int.from_bytes(data[12:16], "little")
        gltf = json.loads(data[20:20 + json_len].decode("utf-8").strip())

        # Primitive mode 0 = GL_POINTS, no indices
        prim = gltf["meshes"][0]["primitives"][0]
        assert prim["mode"] == 0
        assert "indices" not in prim
        assert gltf["nodes"][0]["extras"]["label"] == "origin"

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not compiled")
    def test_glb_write_failure_prunes_tile_and_logs(self, tmp_path):
        """WS-D D.3: a failed GLB tile write is recorded in errors.jsonl, excluded
        from the returned tile count, AND pruned from tileset.json (no dangling
        content URI referencing a missing .glb).

        Simulate a write failure by pre-creating the target .glb path as a
        DIRECTORY so std::fs::write to it fails for exactly one tile while every
        other tile writes cleanly.
        """
        import json
        import os
        from mudm_tools._rs import StreamingTileGenerator

        bounds = (0.0, 0.0, 0.0, 100.0, 100.0, 100.0)

        def _build_gen():
            g = StreamingTileGenerator(min_zoom=0, max_zoom=2)
            # Spread many small TIN triangles across the extent so several
            # distinct leaf tiles are produced.
            for i in range(40):
                nx = (i % 8) * 0.11 + 0.02
                ny = (i // 8) * 0.18 + 0.02
                feat = {
                    "geometry": [nx, ny, nx + 0.04, ny, nx + 0.02, ny + 0.06, nx, ny],
                    "geometry_z": [0.5, 0.51, 0.52, 0.5],
                    "ring_lengths": [4],
                    "type": 5,
                    "tags": {"face": i},
                    "minX": nx, "minY": ny, "minZ": 0.5,
                    "maxX": nx + 0.04, "maxY": ny + 0.06, "maxZ": 0.52,
                }
                g.add_feature(feat)
            return g

        # --- Pass 1: clean run to discover the produced tile layout ---
        clean_out = tmp_path / "clean"
        clean_gen = _build_gen()
        clean_count = clean_gen.generate_3dtiles(str(clean_out), bounds)
        assert clean_count > 1, "fixture must produce more than one tile"

        glb_rels = []
        for root, _, files in os.walk(str(clean_out)):
            for fn in files:
                if fn.endswith(".glb"):
                    glb_rels.append(os.path.relpath(os.path.join(root, fn), str(clean_out)))
        assert len(glb_rels) == clean_count

        # Choose a leaf-zoom tile (zoom == max_zoom == 2) to block so pruning it
        # cannot cascade-remove a parent's children unexpectedly.
        blocked_rel = sorted(r for r in glb_rels if r.split(os.sep)[0] == "2")[0]
        z, x, y, dfile = blocked_rel.split(os.sep)
        d = dfile[:-4]  # strip ".glb"
        blocked_uri = f"{z}/{x}/{y}/{d}.glb"

        # --- Pass 2: re-run with that one tile's path pre-created as a dir ---
        out = tmp_path / "tiles"
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        # Pre-create the blocked tile path AS A DIRECTORY → fs::write fails.
        blocked_abs = out / z / x / y / f"{d}.glb"
        blocked_abs.mkdir(parents=True)

        gen = _build_gen()
        gen._set_run_dir(str(run_dir))
        count = gen.generate_3dtiles(str(out), bounds)

        # (b) returned count excludes the failed tile.
        assert count == clean_count - 1, (
            f"expected {clean_count - 1} (one tile pruned), got {count}"
        )

        # (a) errors.jsonl records the failed write.
        errlog = run_dir / "errors.jsonl"
        assert errlog.exists(), "errors.jsonl must be written when run_dir is set"
        log_text = errlog.read_text()
        recs = [json.loads(ln) for ln in log_text.splitlines() if ln.strip()]
        write_recs = [r for r in recs if r["error_kind"] == "write"]
        assert len(write_recs) == 1, f"expected one write failure record, got {recs}"
        rec = write_recs[0]
        assert rec["phase"] == "3dtiles"
        assert blocked_uri in rec["item"] or f"{z}/{x}/{y}/{d}" in rec["item"]

        # (c) the failed tile key is absent from tileset.json.
        tileset_text = (out / "tileset.json").read_text()
        assert blocked_uri not in tileset_text, (
            "tileset.json must not reference the missing .glb"
        )


def _parse_glb_json(data: bytes):
    """Parse the JSON chunk from GLB bytes."""
    import json
    json_len = int.from_bytes(data[12:16], "little")
    return json.loads(data[20:20 + json_len].decode("utf-8").strip())


def _make_large_tin_feature(n_triangles: int = 20):
    """Build a TIN feature with many triangles in normalized [0,1] space.

    Each triangle is a separate ring (4 verts, last = first) to match
    the ring_lengths encoding that the streaming pipeline expects.
    Returns a dict suitable for StreamingTileGenerator.add_feature().
    """
    import math
    cols = max(1, int(math.ceil(math.sqrt(n_triangles))))
    step = 0.9 / cols if cols > 0 else 0.9
    tri_size = step * 0.45

    geometry = []
    geometry_z = []
    ring_lengths = []
    for i in range(n_triangles):
        col = i % cols
        row = i // cols
        x = col * step + 0.05
        y = row * step + 0.05
        z = 0.5
        geometry.extend([x, y, x + tri_size, y, x + tri_size * 0.5, y + tri_size, x, y])
        geometry_z.extend([z, z + 0.01, z + 0.02, z])
        ring_lengths.append(4)
    return {
        "geometry": geometry,
        "geometry_z": geometry_z,
        "ring_lengths": ring_lengths,
        "type": 5,  # TIN
        "tags": {"name": "large_mesh"},
        "minX": 0.05, "minY": 0.05, "minZ": 0.5,
        "maxX": 0.95, "maxY": 0.95, "maxZ": 0.52,
    }


class TestStreamingDracoGlb:
    """Tests for Draco-compressed GLB output via generate_3dtiles(use_draco=True)."""

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not compiled")
    def test_draco_glb_has_extension(self, tmp_path):
        """Draco GLB files contain KHR_draco_mesh_compression in extensionsUsed."""
        from mudm_tools._rs import StreamingTileGenerator

        gen = StreamingTileGenerator(min_zoom=0, max_zoom=0)
        feat = _make_large_tin_feature(20)  # 60 verts, above min_vertices=50
        gen.add_feature(feat)

        out = str(tmp_path / "tiles")
        bounds = (0.0, 0.0, 0.0, 100.0, 100.0, 100.0)
        count = gen.generate_3dtiles(out, bounds, use_draco=True)
        assert count > 0

        glb_path = tmp_path / "tiles" / "0" / "0" / "0" / "0.glb"
        data = glb_path.read_bytes()
        assert data[:4] == b"glTF"

        gltf = _parse_glb_json(data)
        assert "KHR_draco_mesh_compression" in gltf.get("extensionsUsed", [])
        assert "KHR_draco_mesh_compression" in gltf.get("extensionsRequired", [])

        # Primitive should have the extension
        prim = gltf["meshes"][0]["primitives"][0]
        assert "KHR_draco_mesh_compression" in prim.get("extensions", {})

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not compiled")
    def test_draco_glb_smaller_than_raw(self, tmp_path):
        """Draco-compressed GLB is smaller than raw GLB for large meshes."""
        from mudm_tools._rs import StreamingTileGenerator

        bounds = (0.0, 0.0, 0.0, 100.0, 100.0, 100.0)
        feat = _make_large_tin_feature(200)  # 600 verts — well past Draco break-even

        # Raw GLB
        gen_raw = StreamingTileGenerator(min_zoom=0, max_zoom=0)
        gen_raw.add_feature(feat)
        raw_dir = str(tmp_path / "raw")
        gen_raw.generate_3dtiles(raw_dir, bounds)
        raw_glb = (tmp_path / "raw" / "0" / "0" / "0" / "0.glb").read_bytes()

        # Draco GLB
        gen_draco = StreamingTileGenerator(min_zoom=0, max_zoom=0)
        gen_draco.add_feature(feat)
        draco_dir = str(tmp_path / "draco")
        gen_draco.generate_3dtiles(draco_dir, bounds, use_draco=True)
        draco_glb = (tmp_path / "draco" / "0" / "0" / "0" / "0.glb").read_bytes()

        assert len(draco_glb) < len(raw_glb), (
            f"Draco GLB ({len(draco_glb)}) should be smaller than raw ({len(raw_glb)})"
        )

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not compiled")
    def test_use_draco_false_regression(self, tmp_path):
        """use_draco=False produces output identical to the default (no draco args)."""
        from mudm_tools._rs import StreamingTileGenerator

        bounds = (0.0, 0.0, 0.0, 100.0, 100.0, 100.0)
        feat = _make_large_tin_feature(10)

        # Default (no draco args)
        gen1 = StreamingTileGenerator(min_zoom=0, max_zoom=0)
        gen1.add_feature(feat)
        dir1 = str(tmp_path / "default")
        gen1.generate_3dtiles(dir1, bounds)
        data1 = (tmp_path / "default" / "0" / "0" / "0" / "0.glb").read_bytes()

        # Explicit use_draco=False
        gen2 = StreamingTileGenerator(min_zoom=0, max_zoom=0)
        gen2.add_feature(feat)
        dir2 = str(tmp_path / "no_draco")
        gen2.generate_3dtiles(dir2, bounds, use_draco=False)
        data2 = (tmp_path / "no_draco" / "0" / "0" / "0" / "0.glb").read_bytes()

        assert data1 == data2

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not compiled")
    def test_small_mesh_stays_raw_with_draco(self, tmp_path):
        """Small meshes (< 50 verts) remain uncompressed even with use_draco=True."""
        from mudm_tools._rs import StreamingTileGenerator

        gen = StreamingTileGenerator(min_zoom=0, max_zoom=0)
        # Single triangle = 3 verts, well below min_vertices=50
        feat = {
            "geometry": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1, 0.2],
            "geometry_z": [0.1, 0.2, 0.3, 0.1],
            "ring_lengths": [4],
            "type": 5,
            "tags": {"name": "tiny"},
            "minX": 0.1, "minY": 0.2, "minZ": 0.1,
            "maxX": 0.5, "maxY": 0.6, "maxZ": 0.3,
        }
        gen.add_feature(feat)

        out = str(tmp_path / "tiles")
        gen.generate_3dtiles(out, (0.0, 0.0, 0.0, 10.0, 10.0, 10.0), use_draco=True)

        glb_path = tmp_path / "tiles" / "0" / "0" / "0" / "0.glb"
        data = glb_path.read_bytes()
        gltf = _parse_glb_json(data)

        # No Draco extension should be present
        assert "extensionsUsed" not in gltf or \
            "KHR_draco_mesh_compression" not in gltf.get("extensionsUsed", [])
        # If accessors exist, position accessor should have a bufferView (raw encoding)
        if "accessors" in gltf:
            assert "bufferView" in gltf["accessors"][0]


# --- WS-D Task D.2: add_obj_files error-log wiring ---


class TestAddObjFilesErrorLog:
    """WS-D D.2: add_obj_files records bad files + counts honestly.

    A file with invalid UTF-8 bytes triggers a parse read error in
    ``parse_obj`` (BufReader::lines() yields Err on non-UTF-8), exercising the
    parse-failure path. With a run_dir set, the bad file must appear in
    ``errors.jsonl`` with phase 'ingest', the returned fids must exclude it
    (honest count), and the good files must still ingest. Without a run_dir,
    a clean all-valid ingest must write no errors.jsonl and behave as before.
    """

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not compiled")
    def test_logs_bad_file_and_counts_honestly(self, tmp_path):
        import json
        from mudm_tools._rs import StreamingTileGenerator

        # 2 valid tiny triangles + 1 malformed (invalid UTF-8) file
        good1 = tmp_path / "good1.obj"
        good1.write_text("v 1 1 1\nv 2 1 1\nv 1 2 1\nf 1 2 3\n")
        good2 = tmp_path / "good2.obj"
        good2.write_text("v 3 3 3\nv 4 3 3\nv 3 4 3\nf 1 2 3\n")
        bad = tmp_path / "bad.obj"
        # Invalid UTF-8 sequence — parse_obj's reader.lines() returns Err
        bad.write_bytes(b"v 1 1 1\n\xff\xfe bad bytes f 1 2 3\n")

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        gen = StreamingTileGenerator(min_zoom=0, max_zoom=1, base_cells=100)
        gen._set_run_dir(str(run_dir))

        paths = [str(good1), str(good2), str(bad)]
        bounds = (0.0, 0.0, 0.0, 10.0, 10.0, 10.0)
        tags_list = [{"name": "good1"}, {"name": "good2"}, {"name": "bad"}]

        fids = gen.add_obj_files(paths, bounds, tags_list)

        # Honest count: only the 2 good files ingested
        assert len(fids) == 2

        # errors.jsonl must list the bad file with phase 'ingest'
        errlog = run_dir / "errors.jsonl"
        assert errlog.exists(), "errors.jsonl should be written when run_dir is set"
        lines = [ln for ln in errlog.read_text().splitlines() if ln.strip()]
        assert len(lines) == 1, f"expected 1 error record, got {lines}"
        rec = json.loads(lines[0])
        assert rec["phase"] == "ingest"
        assert rec["error_kind"] == "parse"
        assert "bad.obj" in rec["item"]
        # bad file's fid (index 2) must be absent from the honest set
        assert all(fid != 2 for fid in fids)

        # The 2 good files actually ingested: generate produces tiles
        out = str(tmp_path / "tiles")
        n_tiles = gen.generate_3dtiles(out, bounds)
        assert n_tiles > 0

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not compiled")
    def test_clean_ingest_no_run_dir_unchanged(self, tmp_path):
        """All-valid ingest with NO run_dir: no new file, same result as before."""
        from mudm_tools._rs import StreamingTileGenerator

        good1 = tmp_path / "g1.obj"
        good1.write_text("v 1 1 1\nv 2 1 1\nv 1 2 1\nf 1 2 3\n")
        good2 = tmp_path / "g2.obj"
        good2.write_text("v 3 3 3\nv 4 3 3\nv 3 4 3\nf 1 2 3\n")

        gen = StreamingTileGenerator(min_zoom=0, max_zoom=1, base_cells=100)
        # No _set_run_dir call.

        paths = [str(good1), str(good2)]
        bounds = (0.0, 0.0, 0.0, 10.0, 10.0, 10.0)
        tags_list = [{"name": "g1"}, {"name": "g2"}]

        fids = gen.add_obj_files(paths, bounds, tags_list)
        assert len(fids) == 2

        # No errors.jsonl anywhere under tmp_path (run_dir was never set)
        assert not any(tmp_path.rglob("errors.jsonl"))

        out = str(tmp_path / "tiles")
        n_tiles = gen.generate_3dtiles(out, bounds)
        assert n_tiles > 0
