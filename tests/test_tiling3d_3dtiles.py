"""Tests for OGC 3D Tiles output format (tileset.json + .glb tiles).

Covers: glTF tile encoding, tileset.json structure, geometric error,
3D Tiles reader, and end-to-end round-trip comparison with .pbf3.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from geojson_pydantic import LineString, Point, Polygon

from mudm.model import MuDMFeature, MuDMFeatureCollection, TIN
from mudm_tools.tiling3d.generator3d import TileGenerator3D
from mudm_tools.tiling3d.octree import OctreeConfig
from mudm_tools.tiling3d.reader_3dtiles import TileReader3DTiles
from mudm_tools.tiling3d.projector3d import CartesianProjector3D
from mudm_tools.tiling3d.gltf_encoder3d import tile_to_glb, _unproject_coords
from mudm_tools.tiling3d.tileset_json import (
    generate_tileset_json,
    _box_volume,
    _geometric_error,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _point_feature(x: float, y: float, z: float, **props) -> MuDMFeature:
    return MuDMFeature(
        type="Feature",
        geometry=Point(type="Point", coordinates=[x, y, z]),
        properties=props if props else {},
    )


def _line_feature(coords: list[list[float]], **props) -> MuDMFeature:
    return MuDMFeature(
        type="Feature",
        geometry=LineString(type="LineString", coordinates=coords),
        properties=props if props else {},
    )


def _tin_feature(**props) -> MuDMFeature:
    tin = TIN(
        type="TIN",
        coordinates=[
            [[[0, 0, 0], [5, 0, 1], [2.5, 5, 2], [0, 0, 0]]],
            [[[5, 0, 1], [10, 0, 0], [7.5, 5, 3], [5, 0, 1]]],
        ],
    )
    return MuDMFeature(
        type="Feature",
        geometry=tin,
        properties=props if props else {},
    )


def _collection(*features: MuDMFeature) -> MuDMFeatureCollection:
    return MuDMFeatureCollection(
        type="FeatureCollection",
        features=list(features),
    )


# ===========================================================================
# GLB tile encoder tests
# ===========================================================================


class TestGltfEncoder3D:
    """Test tile_to_glb conversion from intermediate features."""

    def test_point_tile_produces_glb(self):
        """A tile with point features produces valid GLB bytes."""
        # Create a minimal tile dict in normalized space
        tile = {
            "features": [{
                "geometry": [0.5, 0.5],
                "geometry_z": [0.5],
                "type": 1,  # POINT3D
                "tags": {"name": "test"},
                "minX": 0.5, "minY": 0.5, "minZ": 0.5,
                "maxX": 0.5, "maxY": 0.5, "maxZ": 0.5,
            }],
            "z": 0, "x": 0, "y": 0, "d": 0,
        }
        proj = CartesianProjector3D((0, 0, 0, 10, 10, 10))
        data = tile_to_glb(tile, proj)
        assert isinstance(data, bytes)
        assert len(data) > 0
        # GLB magic number
        assert data[:4] == b"glTF"

    def test_line_tile_produces_glb(self):
        """A tile with line features produces valid GLB."""
        tile = {
            "features": [{
                "geometry": [0.0, 0.0, 1.0, 1.0],
                "geometry_z": [0.0, 1.0],
                "type": 2,  # LINESTRING3D
                "tags": {},
                "minX": 0.0, "minY": 0.0, "minZ": 0.0,
                "maxX": 1.0, "maxY": 1.0, "maxZ": 1.0,
            }],
            "z": 0, "x": 0, "y": 0, "d": 0,
        }
        proj = CartesianProjector3D((0, 0, 0, 100, 100, 100))
        data = tile_to_glb(tile, proj)
        assert data[:4] == b"glTF"

    def test_tin_tile_produces_glb(self):
        """A tile with TIN features produces valid GLB."""
        tile = {
            "features": [{
                "geometry": [0.0, 0.0, 0.5, 0.0, 0.25, 0.5, 0.0, 0.0],
                "geometry_z": [0.0, 0.1, 0.2, 0.0],
                "type": 5,  # TIN_TYPE
                "ring_lengths": [4],
                "tags": {"type": "mesh"},
                "minX": 0.0, "minY": 0.0, "minZ": 0.0,
                "maxX": 0.5, "maxY": 0.5, "maxZ": 0.2,
            }],
            "z": 0, "x": 0, "y": 0, "d": 0,
        }
        proj = CartesianProjector3D((0, 0, 0, 10, 10, 10))
        data = tile_to_glb(tile, proj)
        assert data[:4] == b"glTF"

    def test_empty_tile(self):
        """An empty tile still produces valid GLB."""
        tile = {"features": [], "z": 0, "x": 0, "y": 0, "d": 0}
        proj = CartesianProjector3D((0, 0, 0, 1, 1, 1))
        data = tile_to_glb(tile, proj)
        assert data[:4] == b"glTF"

    def test_unproject_coords(self):
        """Coordinates are correctly unprojected to world space."""
        proj = CartesianProjector3D((10, 20, 30, 110, 120, 130))
        feat = {
            "geometry": [0.0, 0.0, 1.0, 1.0, 0.5, 0.5],
            "geometry_z": [0.0, 1.0, 0.5],
        }
        coords = _unproject_coords(feat, proj)
        assert len(coords) == 3
        assert coords[0] == pytest.approx([10, 20, 30], abs=0.01)
        assert coords[1] == pytest.approx([110, 120, 130], abs=0.01)
        assert coords[2] == pytest.approx([60, 70, 80], abs=0.01)


# ===========================================================================
# Tileset JSON tests
# ===========================================================================


class TestTilesetJson:
    """Test tileset.json generation."""

    def test_box_volume_axis_aligned(self):
        """Box volume produces correct axis-aligned format."""
        box = _box_volume(0, 0, 0, 10, 20, 30)
        assert box[0] == 5      # cx
        assert box[1] == 10     # cy
        assert box[2] == 15     # cz
        assert box[3] == 5      # halfX
        assert box[4] == 0
        assert box[5] == 0
        assert box[6] == 0
        assert box[7] == 10     # halfY
        assert box[8] == 0
        assert box[9] == 0
        assert box[10] == 0
        assert box[11] == 15    # halfZ

    def test_geometric_error_at_max_zoom(self):
        """Geometric error at max zoom is 0."""
        bounds = (0, 0, 0, 100, 100, 100)
        assert _geometric_error(bounds, 3, 3) == 0.0

    def test_geometric_error_decreases_with_zoom(self):
        """Geometric error decreases as zoom increases."""
        bounds = (0, 0, 0, 100, 100, 100)
        e0 = _geometric_error(bounds, 0, 3)
        e1 = _geometric_error(bounds, 1, 3)
        e2 = _geometric_error(bounds, 2, 3)
        e3 = _geometric_error(bounds, 3, 3)
        assert e0 > e1 > e2 > e3

    def test_generate_tileset_structure(self):
        """Generated tileset has correct OGC 3D Tiles structure."""
        # Create a minimal octree-like tile dict
        tiles = {
            (0, 0, 0, 0): {"features": [{"geometry": [0.5, 0.5], "geometry_z": [0.5], "type": 1}]},
            (1, 0, 0, 0): {"features": [{"geometry": [0.25, 0.25], "geometry_z": [0.25], "type": 1}]},
            (1, 1, 1, 1): {"features": [{"geometry": [0.75, 0.75], "geometry_z": [0.75], "type": 1}]},
        }
        bounds = (0, 0, 0, 10, 10, 10)
        proj = CartesianProjector3D(bounds)

        tileset = generate_tileset_json(tiles, bounds, proj, min_zoom=0, max_zoom=1)

        assert tileset["asset"]["version"] == "1.1"
        assert "geometricError" in tileset
        assert tileset["geometricError"] > 0
        root = tileset["root"]
        assert "boundingVolume" in root
        assert "box" in root["boundingVolume"]
        assert root["refine"] == "REPLACE"

    def test_tileset_has_content_uris(self):
        """Each tile node has a content URI pointing to .glb."""
        tiles = {
            (0, 0, 0, 0): {"features": [{"geometry": [0.5, 0.5], "geometry_z": [0.5], "type": 1}]},
        }
        bounds = (0, 0, 0, 10, 10, 10)
        proj = CartesianProjector3D(bounds)

        tileset = generate_tileset_json(tiles, bounds, proj, min_zoom=0, max_zoom=0)
        root = tileset["root"]
        assert root["content"]["uri"] == "0/0/0/0.glb"

    def test_tileset_hierarchical(self):
        """Children at zoom 1 are nested under zoom 0 root."""
        tiles = {
            (0, 0, 0, 0): {"features": [{"geometry": [0.5, 0.5], "geometry_z": [0.5], "type": 1}]},
            (1, 0, 0, 0): {"features": [{"geometry": [0.25, 0.25], "geometry_z": [0.25], "type": 1}]},
            (1, 1, 1, 1): {"features": [{"geometry": [0.75, 0.75], "geometry_z": [0.75], "type": 1}]},
        }
        bounds = (0, 0, 0, 10, 10, 10)
        proj = CartesianProjector3D(bounds)

        tileset = generate_tileset_json(tiles, bounds, proj, min_zoom=0, max_zoom=1)
        root = tileset["root"]
        assert "children" in root
        assert len(root["children"]) == 2

        child_uris = {c["content"]["uri"] for c in root["children"]}
        assert "1/0/0/0.glb" in child_uris
        assert "1/1/1/1.glb" in child_uris


# ===========================================================================
# End-to-end 3D Tiles generator tests
# ===========================================================================


class TestGenerator3DTiles:
    """Test TileGenerator3D with output_format='3dtiles'."""

    def test_generates_glb_files(self, tmp_path):
        """Generator creates .glb files, not .pbf3."""
        coll = _collection(
            _point_feature(1, 2, 3),
            _point_feature(8, 9, 7),
        )
        gen = TileGenerator3D(OctreeConfig(max_zoom=1), output_format="3dtiles")
        gen.add_features(coll)
        count = gen.generate(tmp_path)
        assert count > 0

        # Check .glb files exist
        glb_files = list(tmp_path.rglob("*.glb"))
        assert len(glb_files) == count
        # No .pbf3 files
        mvt_files = list(tmp_path.rglob("*.pbf3"))
        assert len(mvt_files) == 0

    def test_glb_files_are_valid(self, tmp_path):
        """Generated .glb files have correct magic number."""
        coll = _collection(_point_feature(5, 5, 5))
        gen = TileGenerator3D(OctreeConfig(max_zoom=0), output_format="3dtiles")
        gen.add_features(coll)
        gen.generate(tmp_path)

        glb_files = list(tmp_path.rglob("*.glb"))
        assert len(glb_files) >= 1
        for f in glb_files:
            data = f.read_bytes()
            assert data[:4] == b"glTF"

    def test_tileset_json_written(self, tmp_path):
        """write_tileset_json produces a valid tileset.json."""
        coll = _collection(
            _point_feature(0, 0, 0),
            _point_feature(10, 10, 10),
        )
        gen = TileGenerator3D(OctreeConfig(max_zoom=1), output_format="3dtiles")
        gen.add_features(coll)
        gen.generate(tmp_path)
        gen.write_tileset_json(tmp_path / "tileset.json")

        assert (tmp_path / "tileset.json").exists()
        tileset = json.loads((tmp_path / "tileset.json").read_text())
        assert tileset["asset"]["version"] == "1.1"
        assert "geometricError" in tileset
        assert "root" in tileset

    def test_write_metadata_dispatches(self, tmp_path):
        """write_metadata writes tileset.json for 3dtiles format."""
        coll = _collection(_point_feature(5, 5, 5))
        gen = TileGenerator3D(OctreeConfig(max_zoom=0), output_format="3dtiles")
        gen.add_features(coll)
        gen.generate(tmp_path)
        gen.write_metadata(tmp_path)

        assert (tmp_path / "tileset.json").exists()
        assert not (tmp_path / "tilejson3d.json").exists()

    def test_write_metadata_dispatches_pbf3(self, tmp_path):
        """write_metadata writes tilejson3d.json for pbf3 format."""
        coll = _collection(_point_feature(5, 5, 5))
        gen = TileGenerator3D(OctreeConfig(max_zoom=0), output_format="pbf3")
        gen.add_features(coll)
        gen.generate(tmp_path)
        gen.write_metadata(tmp_path)

        assert (tmp_path / "tilejson3d.json").exists()
        assert not (tmp_path / "tileset.json").exists()

    def test_line_features_3dtiles(self, tmp_path):
        """Line features produce valid GLB tiles."""
        coll = _collection(
            _line_feature([[0, 0, 0], [5, 5, 5], [10, 0, 10]]),
        )
        gen = TileGenerator3D(OctreeConfig(max_zoom=0), output_format="3dtiles")
        gen.add_features(coll)
        count = gen.generate(tmp_path)
        assert count >= 1
        for f in tmp_path.rglob("*.glb"):
            assert f.read_bytes()[:4] == b"glTF"

    def test_tin_features_3dtiles(self, tmp_path):
        """TIN features produce valid GLB tiles."""
        coll = _collection(_tin_feature(material="bone"))
        gen = TileGenerator3D(OctreeConfig(max_zoom=0), output_format="3dtiles")
        gen.add_features(coll)
        count = gen.generate(tmp_path)
        assert count >= 1

    def test_mixed_geometry_3dtiles(self, tmp_path):
        """Mixed geometry types produce valid GLB tiles."""
        coll = _collection(
            _point_feature(1, 1, 1, kind="point"),
            _line_feature([[2, 2, 2], [8, 8, 8]], kind="line"),
            _tin_feature(kind="tin"),
        )
        gen = TileGenerator3D(OctreeConfig(max_zoom=0), output_format="3dtiles")
        gen.add_features(coll)
        count = gen.generate(tmp_path)
        assert count >= 1


# ===========================================================================
# 3D Tiles reader tests
# ===========================================================================


class TestReader3DTiles:
    """Test TileReader3DTiles."""

    def test_reader_loads_metadata(self, tmp_path):
        """Reader correctly loads tileset.json metadata."""
        coll = _collection(
            _point_feature(0, 0, 0),
            _point_feature(10, 10, 10),
        )
        gen = TileGenerator3D(OctreeConfig(max_zoom=1), output_format="3dtiles")
        gen.add_features(coll)
        gen.generate(tmp_path)
        gen.write_tileset_json(tmp_path / "tileset.json")

        reader = TileReader3DTiles(tmp_path / "tileset.json")
        assert reader.asset["version"] == "1.1"
        assert reader.geometric_error > 0

    def test_reader_tile_count(self, tmp_path):
        """Reader counts tiles correctly."""
        coll = _collection(
            _point_feature(2, 2, 2),
            _point_feature(8, 8, 8),
        )
        gen = TileGenerator3D(OctreeConfig(max_zoom=1), output_format="3dtiles")
        gen.add_features(coll)
        count = gen.generate(tmp_path)
        gen.write_tileset_json(tmp_path / "tileset.json")

        reader = TileReader3DTiles(tmp_path / "tileset.json")
        assert reader.tile_count() == count

    def test_reader_read_tile(self, tmp_path):
        """Reader can read individual GLB tiles."""
        coll = _collection(_point_feature(5, 5, 5))
        gen = TileGenerator3D(OctreeConfig(max_zoom=0), output_format="3dtiles")
        gen.add_features(coll)
        gen.generate(tmp_path)
        gen.write_tileset_json(tmp_path / "tileset.json")

        reader = TileReader3DTiles(tmp_path / "tileset.json")
        tiles = reader.all_tiles()
        assert len(tiles) >= 1
        data = reader.read_tile(tiles[0]["uri"])
        assert data is not None
        assert data[:4] == b"glTF"

    def test_reader_max_depth(self, tmp_path):
        """Reader reports correct max depth."""
        coll = _collection(
            _point_feature(0, 0, 0),
            _point_feature(10, 10, 10),
        )
        gen = TileGenerator3D(OctreeConfig(max_zoom=2), output_format="3dtiles")
        gen.add_features(coll)
        gen.generate(tmp_path)
        gen.write_tileset_json(tmp_path / "tileset.json")

        reader = TileReader3DTiles(tmp_path / "tileset.json")
        assert reader.max_depth() >= 1

    def test_reader_tiles_at_depth(self, tmp_path):
        """Reader can filter tiles by depth."""
        coll = _collection(
            _point_feature(2, 2, 2),
            _point_feature(8, 8, 8),
        )
        gen = TileGenerator3D(OctreeConfig(max_zoom=1), output_format="3dtiles")
        gen.add_features(coll)
        gen.generate(tmp_path)
        gen.write_tileset_json(tmp_path / "tileset.json")

        reader = TileReader3DTiles(tmp_path / "tileset.json")
        depth0 = reader.tiles_at_depth(0)
        depth1 = reader.tiles_at_depth(1)
        assert len(depth0) >= 1
        assert len(depth1) >= 1

    def test_reader_nonexistent_tile(self, tmp_path):
        """Reading a non-existent tile returns None."""
        coll = _collection(_point_feature(5, 5, 5))
        gen = TileGenerator3D(OctreeConfig(max_zoom=0), output_format="3dtiles")
        gen.add_features(coll)
        gen.generate(tmp_path)
        gen.write_tileset_json(tmp_path / "tileset.json")

        reader = TileReader3DTiles(tmp_path / "tileset.json")
        assert reader.read_tile("99/99/99/99.glb") is None


# ===========================================================================
# Format comparison tests (same input → both formats)
# ===========================================================================


class TestFormatComparison:
    """Compare pbf3 and 3dtiles output from the same input."""

    def test_same_tile_count(self, tmp_path):
        """Both formats produce the same number of tiles."""
        coll = _collection(
            _point_feature(1, 1, 1),
            _point_feature(9, 9, 9),
        )
        pbf3_dir = tmp_path / "pbf3"
        tiles3d_dir = tmp_path / "3dtiles"

        gen_pbf3 = TileGenerator3D(OctreeConfig(max_zoom=1), output_format="pbf3")
        gen_pbf3.add_features(coll)
        count_pbf3 = gen_pbf3.generate(pbf3_dir)

        gen_3dt = TileGenerator3D(OctreeConfig(max_zoom=1), output_format="3dtiles")
        gen_3dt.add_features(coll)
        count_3dt = gen_3dt.generate(tiles3d_dir)

        assert count_pbf3 == count_3dt

    def test_default_format_is_pbf3(self, tmp_path):
        """Default output format is pbf3."""
        coll = _collection(_point_feature(5, 5, 5))
        gen = TileGenerator3D(OctreeConfig(max_zoom=0))
        gen.add_features(coll)
        gen.generate(tmp_path)

        assert len(list(tmp_path.rglob("*.pbf3"))) >= 1
        assert len(list(tmp_path.rglob("*.glb"))) == 0
