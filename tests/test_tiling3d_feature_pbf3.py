"""Tests for per-feature PBF3 output (feature-centric dual-index architecture).

Covers: generate_feature_pbf3(), manifest.json structure, decode roundtrip,
world coordinates, vertex count parity with Neuroglancer, bbox accuracy,
point features, and empty features.
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

def _make_tin_feature(xy, z, ring_lengths, tags=None, feature_id=None):
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


def _make_point_feature(x, y, z, tags=None):
    """Build a Point3D feature dict in normalized [0,1]³ space."""
    return {
        "geometry": [x, y],
        "geometry_z": [z],
        "type": 1,  # POINT3D
        "tags": tags or {},
        "minX": x, "minY": y, "minZ": z,
        "maxX": x, "maxY": y, "maxZ": z,
    }


def _make_line_feature(xy, z, tags=None):
    """Build a LineString3D feature dict in normalized [0,1]³ space."""
    n = len(z)
    return {
        "geometry": xy,
        "geometry_z": z,
        "type": 2,  # LINESTRING3D
        "tags": tags or {},
        "minX": min(xy[i * 2] for i in range(n)),
        "minY": min(xy[i * 2 + 1] for i in range(n)),
        "minZ": min(z),
        "maxX": max(xy[i * 2] for i in range(n)),
        "maxY": max(xy[i * 2 + 1] for i in range(n)),
        "maxZ": max(z),
    }


WORLD_BOUNDS = (0.0, 0.0, 0.0, 100.0, 200.0, 300.0)


def _build_generator_with_features(features, min_zoom=0, max_zoom=2):
    """Create a StreamingTileGenerator, add features, return (gen, fids)."""
    gen = StreamingTileGenerator(min_zoom=min_zoom, max_zoom=max_zoom)
    fids = []
    for feat in features:
        fid = gen.add_feature(feat)
        fids.append(fid)
    return gen, fids


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFeaturePbf3ProducesFiles:
    """Test that generate_feature_pbf3 creates the expected files."""

    def test_feature_pbf3_produces_files(self, tmp_path):
        """N .pbf3 files created, one per feature."""
        features = [
            _make_tin_feature(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [0.1, 0.2, 0.3],
                [3],
                tags={"name": "mesh_a"},
            ),
            _make_tin_feature(
                [0.6, 0.7, 0.8, 0.8, 0.7, 0.9],
                [0.4, 0.5, 0.6],
                [3],
                tags={"name": "mesh_b"},
            ),
        ]
        gen, fids = _build_generator_with_features(features)
        out = str(tmp_path / "features")
        count = gen.generate_feature_pbf3(out, WORLD_BOUNDS)

        assert count == 2
        assert (tmp_path / "features" / "0.pbf3").exists()
        assert (tmp_path / "features" / "1.pbf3").exists()
        assert (tmp_path / "features" / "manifest.json").exists()


class TestFeaturePbf3Manifest:
    """Test manifest.json structure and contents."""

    def test_feature_pbf3_manifest_structure(self, tmp_path):
        """manifest.json has correct format/version/features."""
        features = [
            _make_tin_feature(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [0.1, 0.2, 0.3],
                [3],
                tags={"name": "neuron_1", "volume": 42.5},
            ),
            _make_tin_feature(
                [0.5, 0.5, 0.7, 0.6, 0.6, 0.8],
                [0.3, 0.4, 0.5],
                [3],
                tags={"name": "neuron_2", "volume": 88.0},
            ),
        ]
        gen, _ = _build_generator_with_features(features)
        out = str(tmp_path / "features")
        gen.generate_feature_pbf3(out, WORLD_BOUNDS)

        manifest = json.loads((tmp_path / "features" / "manifest.json").read_text())
        assert manifest["format"] == "mudm_feature_pbf3"
        assert manifest["version"] == 2  # multilod=True is new default
        assert manifest["feature_count"] == 2
        assert manifest["world_bounds"] == list(WORLD_BOUNDS)
        assert manifest["multilod"] is True
        assert "0" in manifest["features"]
        assert "1" in manifest["features"]

        # Each feature entry has bbox, tags, and lod_count
        f0 = manifest["features"]["0"]
        assert "bbox" in f0
        assert len(f0["bbox"]) == 6
        assert "tags" in f0
        assert f0["tags"]["name"] == "neuron_1"
        assert f0["tags"]["volume"] == 42.5
        assert "lod_count" in f0
        assert f0["lod_count"] >= 1


class TestFeaturePbf3DecodeRoundtrip:
    """Test that decode_tile() works on per-feature .pbf3 files."""

    def test_feature_pbf3_decode_roundtrip(self, tmp_path):
        """decode_tile() works on per-feature files, correct tags."""
        from mudm_tools.tiling3d.reader3d import decode_tile

        features = [
            _make_tin_feature(
                [0.2, 0.3, 0.4, 0.5, 0.3, 0.7],
                [0.1, 0.4, 0.6],
                [3],
                tags={"species": "mouse", "neuron_id": 12345},
            ),
        ]
        gen, _ = _build_generator_with_features(features)
        out = str(tmp_path / "features")
        gen.generate_feature_pbf3(out, WORLD_BOUNDS)

        # Decode the per-feature PBF3 (multilod=True default, so multiple layers)
        data = (tmp_path / "features" / "0.pbf3").read_bytes()
        layers = decode_tile(data)
        assert len(layers) >= 1  # at least lod_0

        # lod_0 (finest) is first layer
        layer = layers[0]
        assert layer["name"] == "lod_0"
        assert len(layer["features"]) == 1

        feat = layer["features"][0]
        assert feat["type"] == 5  # TIN
        assert feat["tags"]["species"] == "mouse"
        assert feat["tags"]["neuron_id"] == 12345
        assert feat["mesh_positions"] is not None
        assert len(feat["mesh_positions"]) > 0
        assert feat["mesh_indices"] is not None
        assert len(feat["mesh_indices"]) > 0


class TestFeaturePbf3WorldCoordinates:
    """Test that mesh positions are in world coordinate range."""

    def test_feature_pbf3_world_coordinates(self, tmp_path):
        """Mesh positions are in world coordinate range."""
        # Feature near center of [0,1]³ → world coords near center of bounds
        features = [
            _make_tin_feature(
                [0.48, 0.48, 0.52, 0.48, 0.50, 0.52],
                [0.48, 0.52, 0.50],
                [3],
                tags={},
            ),
        ]
        gen, _ = _build_generator_with_features(features)
        out = str(tmp_path / "features")
        gen.generate_feature_pbf3(out, WORLD_BOUNDS)

        data = (tmp_path / "features" / "0.pbf3").read_bytes()
        from mudm_tools.tiling3d.reader3d import decode_tile
        layers = decode_tile(data)
        feat = layers[0]["features"][0]

        # Extract float32 positions from mesh_positions bytes
        pos_bytes = feat["mesh_positions"]
        n_floats = len(pos_bytes) // 4
        positions = struct.unpack(f"<{n_floats}f", pos_bytes)

        # Group into xyz triples
        n_verts = n_floats // 3
        assert n_verts >= 3

        for i in range(n_verts):
            x = positions[i * 3]
            y = positions[i * 3 + 1]
            z = positions[i * 3 + 2]
            # bounds = (0, 0, 0, 100, 200, 300), feature at ~0.5 normalized
            # x ≈ 48-52, y ≈ 96-104, z ≈ 144-156
            assert 40.0 < x < 60.0, f"x={x} out of range"
            assert 80.0 < y < 120.0, f"y={y} out of range"
            assert 130.0 < z < 170.0, f"z={z} out of range"


class TestFeaturePbf3VertexCountMatchesNeuroglancer:
    """Test vertex count parity between feature PBF3 and Neuroglancer."""

    def test_feature_pbf3_vertex_count_matches_neuroglancer(self, tmp_path):
        """Same geometry from both outputs."""
        features = [
            _make_tin_feature(
                [0.1, 0.1, 0.3, 0.1, 0.2, 0.3, 0.1, 0.1,
                 0.4, 0.4, 0.6, 0.4, 0.5, 0.6, 0.4, 0.4],
                [0.1, 0.2, 0.3, 0.1, 0.4, 0.5, 0.6, 0.4],
                [4, 4],
                tags={"name": "test"},
            ),
        ]

        # Generate feature PBF3
        gen1, _ = _build_generator_with_features(features)
        feat_dir = str(tmp_path / "features")
        gen1.generate_feature_pbf3(feat_dir, WORLD_BOUNDS)

        # Generate Neuroglancer
        gen2, _ = _build_generator_with_features(features)
        ng_dir = str(tmp_path / "neuroglancer")
        gen2.generate_neuroglancer(ng_dir, WORLD_BOUNDS)

        # Decode feature PBF3
        from mudm_tools.tiling3d.reader3d import decode_tile
        pbf3_data = (tmp_path / "features" / "0.pbf3").read_bytes()
        layers = decode_tile(pbf3_data)
        pbf3_feat = layers[0]["features"][0]
        pbf3_pos = pbf3_feat["mesh_positions"]
        pbf3_n_verts = len(pbf3_pos) // 12  # 3 floats × 4 bytes

        # Decode Neuroglancer
        ng_data = (tmp_path / "neuroglancer" / "0").read_bytes()
        (ng_n_verts,) = struct.unpack_from("<I", ng_data, 0)

        # Both should have the same vertex count (same dedup logic)
        assert pbf3_n_verts == ng_n_verts


class TestFeaturePbf3BboxCorrect:
    """Test that manifest bbox matches actual geometry bounds."""

    def test_feature_pbf3_bbox_correct(self, tmp_path):
        """Manifest bbox matches actual geometry bounds."""
        features = [
            _make_tin_feature(
                [0.2, 0.3, 0.4, 0.5, 0.3, 0.7],
                [0.1, 0.4, 0.6],
                [3],
                tags={},
            ),
        ]
        gen, _ = _build_generator_with_features(features)
        out = str(tmp_path / "features")
        gen.generate_feature_pbf3(out, WORLD_BOUNDS)

        manifest = json.loads((tmp_path / "features" / "manifest.json").read_text())
        bbox = manifest["features"]["0"]["bbox"]

        # Extract actual positions from PBF3 file
        from mudm_tools.tiling3d.reader3d import decode_tile
        data = (tmp_path / "features" / "0.pbf3").read_bytes()
        layers = decode_tile(data)
        feat = layers[0]["features"][0]
        pos_bytes = feat["mesh_positions"]
        n_floats = len(pos_bytes) // 4
        positions = struct.unpack(f"<{n_floats}f", pos_bytes)

        n_verts = n_floats // 3
        xs = [positions[i * 3] for i in range(n_verts)]
        ys = [positions[i * 3 + 1] for i in range(n_verts)]
        zs = [positions[i * 3 + 2] for i in range(n_verts)]

        # Bbox should contain all actual positions
        assert bbox[0] <= min(xs) + 0.1  # xmin
        assert bbox[1] <= min(ys) + 0.1  # ymin
        assert bbox[2] <= min(zs) + 0.1  # zmin
        assert bbox[3] >= max(xs) - 0.1  # xmax
        assert bbox[4] >= max(ys) - 0.1  # ymax
        assert bbox[5] >= max(zs) - 0.1  # zmax


class TestFeaturePbf3PointFeatures:
    """Test non-mesh geometry types."""

    def test_feature_pbf3_point_features(self, tmp_path):
        """Point features are encoded and decodable."""
        from mudm_tools.tiling3d.reader3d import decode_tile

        features = [
            _make_point_feature(0.5, 0.5, 0.5, tags={"label": "center"}),
        ]
        gen, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=0)
        out = str(tmp_path / "features")
        count = gen.generate_feature_pbf3(out, WORLD_BOUNDS)

        assert count == 1
        data = (tmp_path / "features" / "0.pbf3").read_bytes()
        layers = decode_tile(data)
        feat = layers[0]["features"][0]
        assert feat["type"] == 1  # POINT3D
        assert feat["tags"]["label"] == "center"

    def test_feature_pbf3_line_features(self, tmp_path):
        """LineString features are encoded and decodable."""
        from mudm_tools.tiling3d.reader3d import decode_tile

        features = [
            _make_line_feature(
                [0.1, 0.5, 0.9, 0.5],
                [0.5, 0.5],
                tags={"road": "A1"},
            ),
        ]
        gen, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=0)
        out = str(tmp_path / "features")
        count = gen.generate_feature_pbf3(out, WORLD_BOUNDS)

        assert count == 1
        data = (tmp_path / "features" / "0.pbf3").read_bytes()
        layers = decode_tile(data)
        feat = layers[0]["features"][0]
        assert feat["type"] == 2  # LINESTRING3D
        assert feat["tags"]["road"] == "A1"


class TestFeaturePbf3EmptyFeature:
    """Test features with no max_zoom fragments."""

    def test_feature_pbf3_empty_feature(self, tmp_path):
        """Features with no max_zoom fragments produce no file."""
        # Create a feature that only appears at zoom 0 but max_zoom is 2
        # A point at exactly (0, 0, 0) should still clip to tiles at all zooms,
        # so use a generator with only zoom 0 fragments by making min_zoom=max_zoom=0
        # and then a separate generator with max_zoom=2 to force the point to be
        # at a non-max zoom
        #
        # Actually, the octree clips at all levels [min_zoom, max_zoom],
        # so a point will always have max_zoom fragments.
        # Instead test with zero features: should produce 0 files.
        gen = StreamingTileGenerator(min_zoom=0, max_zoom=2)
        out = str(tmp_path / "features")
        count = gen.generate_feature_pbf3(out, WORLD_BOUNDS)

        assert count == 0
        manifest = json.loads((tmp_path / "features" / "manifest.json").read_text())
        assert manifest["feature_count"] == 0
        assert manifest["features"] == {}


class TestFeaturePbf3MultipleFeatures:
    """Test multiple features with different tags."""

    def test_multiple_features_separate_files(self, tmp_path):
        """Each feature gets its own .pbf3 with correct tags."""
        from mudm_tools.tiling3d.reader3d import decode_tile

        features = [
            _make_tin_feature(
                [0.1, 0.1, 0.2, 0.1, 0.15, 0.2],
                [0.1, 0.2, 0.15],
                [3],
                tags={"name": "alpha", "id": 0},
            ),
            _make_tin_feature(
                [0.5, 0.5, 0.6, 0.5, 0.55, 0.6],
                [0.4, 0.5, 0.45],
                [3],
                tags={"name": "beta", "id": 1},
            ),
            _make_tin_feature(
                [0.8, 0.8, 0.9, 0.8, 0.85, 0.9],
                [0.7, 0.8, 0.75],
                [3],
                tags={"name": "gamma", "id": 2},
            ),
        ]
        gen, _ = _build_generator_with_features(features)
        out = str(tmp_path / "features")
        count = gen.generate_feature_pbf3(out, WORLD_BOUNDS)

        assert count == 3

        for fid in range(3):
            data = (tmp_path / "features" / f"{fid}.pbf3").read_bytes()
            layers = decode_tile(data)
            feat = layers[0]["features"][0]
            assert feat["type"] == 5
            assert feat["tags"]["id"] == fid


# ---------------------------------------------------------------------------
# Multi-LOD tests
# ---------------------------------------------------------------------------

def _make_dense_tin_feature(n_triangles=20):
    """Build a TIN feature with many triangles spread across [0.1, 0.9]³."""
    import random
    random.seed(42)
    xy = []
    z = []
    ring_lengths = []
    for _ in range(n_triangles):
        cx, cy, cz = random.uniform(0.15, 0.85), random.uniform(0.15, 0.85), random.uniform(0.15, 0.85)
        d = 0.05
        xy.extend([cx - d, cy - d, cx + d, cy - d, cx, cy + d])
        z.extend([cz - d, cz + d, cz])
        ring_lengths.append(3)
    n = len(z)
    return _make_tin_feature(xy, z, ring_lengths, tags={"name": "dense"})


class TestMultilodLayerCount:
    """Multi-LOD produces one layer per zoom level."""

    def test_multilod_layer_count(self, tmp_path):
        """Number of layers == number of zoom levels with fragments."""
        max_zoom = 2
        features = [_make_dense_tin_feature(20)]
        gen, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=max_zoom)
        out = str(tmp_path / "features")
        gen.generate_feature_pbf3(out, WORLD_BOUNDS)

        from mudm_tools.tiling3d.reader3d import decode_tile
        data = (tmp_path / "features" / "0.pbf3").read_bytes()
        layers = decode_tile(data)
        # Should have up to max_zoom + 1 layers (zoom 0, 1, 2)
        assert len(layers) == max_zoom + 1


class TestMultilodLayerNames:
    """Layer names follow lod_0, lod_1, ... convention."""

    def test_multilod_layer_names(self, tmp_path):
        """Layers are named lod_0 (finest) through lod_N (coarsest)."""
        max_zoom = 2
        features = [_make_dense_tin_feature(20)]
        gen, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=max_zoom)
        out = str(tmp_path / "features")
        gen.generate_feature_pbf3(out, WORLD_BOUNDS)

        from mudm_tools.tiling3d.reader3d import decode_tile
        data = (tmp_path / "features" / "0.pbf3").read_bytes()
        layers = decode_tile(data)
        names = [l["name"] for l in layers]
        assert names == ["lod_0", "lod_1", "lod_2"]


class TestMultilodVertexReduction:
    """Coarser LODs have fewer or equal vertices than finer LODs."""

    def test_multilod_vertex_reduction(self, tmp_path):
        """Coarser layers have fewer vertices due to grid clustering."""
        features = [_make_dense_tin_feature(50)]
        gen, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=2)
        out = str(tmp_path / "features")
        gen.generate_feature_pbf3(out, WORLD_BOUNDS)

        from mudm_tools.tiling3d.reader3d import decode_tile
        data = (tmp_path / "features" / "0.pbf3").read_bytes()
        layers = decode_tile(data)

        # Count unique vertices per LOD
        vertex_counts = []
        for layer in layers:
            feat = layer["features"][0]
            pos_bytes = feat["mesh_positions"]
            n_verts = len(pos_bytes) // 12  # 3 floats * 4 bytes
            vertex_counts.append(n_verts)

        # lod_0 (finest) should have >= lod_1 >= lod_2 (coarsest) vertices
        for i in range(len(vertex_counts) - 1):
            assert vertex_counts[i] >= vertex_counts[i + 1], (
                f"lod_{i} ({vertex_counts[i]} verts) < lod_{i+1} ({vertex_counts[i+1]} verts)"
            )
        # Coarsest should have strictly fewer than finest (with 50 triangles)
        assert vertex_counts[-1] < vertex_counts[0], (
            f"Expected coarsest ({vertex_counts[-1]}) < finest ({vertex_counts[0]})"
        )


class TestMultilodWorldCoords:
    """All LODs produce world-coordinate meshes."""

    def test_multilod_world_coords(self, tmp_path):
        """Mesh positions at all LODs are in world coordinate range."""
        features = [_make_dense_tin_feature(20)]
        gen, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=2)
        out = str(tmp_path / "features")
        gen.generate_feature_pbf3(out, WORLD_BOUNDS)

        from mudm_tools.tiling3d.reader3d import decode_tile
        data = (tmp_path / "features" / "0.pbf3").read_bytes()
        layers = decode_tile(data)

        for layer in layers:
            feat = layer["features"][0]
            pos_bytes = feat["mesh_positions"]
            if not pos_bytes:
                continue
            n_floats = len(pos_bytes) // 4
            positions = struct.unpack(f"<{n_floats}f", pos_bytes)
            n_verts = n_floats // 3

            for i in range(n_verts):
                x = positions[i * 3]
                y = positions[i * 3 + 1]
                z = positions[i * 3 + 2]
                # bounds = (0, 0, 0, 100, 200, 300)
                assert -5.0 < x < 105.0, f"x={x} out of range in {layer['name']}"
                assert -5.0 < y < 205.0, f"y={y} out of range in {layer['name']}"
                assert -5.0 < z < 305.0, f"z={z} out of range in {layer['name']}"


class TestMultilodFalseBackwardCompat:
    """multilod=False gives single-layer output (backward compat)."""

    def test_multilod_false_backward_compat(self, tmp_path):
        """Setting multilod=False produces single-layer PBF3 like version 1."""
        features = [_make_dense_tin_feature(20)]
        gen, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=2)
        out = str(tmp_path / "features")
        gen.generate_feature_pbf3(out, WORLD_BOUNDS, multilod=False)

        from mudm_tools.tiling3d.reader3d import decode_tile
        data = (tmp_path / "features" / "0.pbf3").read_bytes()
        layers = decode_tile(data)
        assert len(layers) == 1
        assert layers[0]["name"] == "default"

        manifest = json.loads((tmp_path / "features" / "manifest.json").read_text())
        assert manifest["version"] == 1
        assert "multilod" not in manifest


class TestMultilodManifest:
    """Manifest includes multilod-specific fields."""

    def test_multilod_manifest(self, tmp_path):
        """Manifest has version 2, multilod flag, and per-feature lod_count."""
        features = [
            _make_dense_tin_feature(20),
            _make_tin_feature(
                [0.8, 0.8, 0.9, 0.8, 0.85, 0.9],
                [0.7, 0.8, 0.75],
                [3],
                tags={"name": "simple"},
            ),
        ]
        gen, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=2)
        out = str(tmp_path / "features")
        gen.generate_feature_pbf3(out, WORLD_BOUNDS)

        manifest = json.loads((tmp_path / "features" / "manifest.json").read_text())
        assert manifest["version"] == 2
        assert manifest["multilod"] is True

        for fid_str in manifest["features"]:
            feat = manifest["features"][fid_str]
            assert "lod_count" in feat
            assert feat["lod_count"] >= 1
            assert feat["lod_count"] <= 3  # max_zoom=2 → up to 3 levels
