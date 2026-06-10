"""Tests for OGC 3D Tiles output format (tileset.json + .glb tiles).

Covers: glTF tile encoding, tileset.json structure, geometric error,
3D Tiles reader, and end-to-end round-trip comparison with .pbf3.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from geojson_pydantic import LineString, Point

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
            "features": [
                {
                    "geometry": [0.5, 0.5],
                    "geometry_z": [0.5],
                    "type": 1,  # POINT3D
                    "tags": {"name": "test"},
                    "minX": 0.5,
                    "minY": 0.5,
                    "minZ": 0.5,
                    "maxX": 0.5,
                    "maxY": 0.5,
                    "maxZ": 0.5,
                }
            ],
            "z": 0,
            "x": 0,
            "y": 0,
            "d": 0,
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
            "features": [
                {
                    "geometry": [0.0, 0.0, 1.0, 1.0],
                    "geometry_z": [0.0, 1.0],
                    "type": 2,  # LINESTRING3D
                    "tags": {},
                    "minX": 0.0,
                    "minY": 0.0,
                    "minZ": 0.0,
                    "maxX": 1.0,
                    "maxY": 1.0,
                    "maxZ": 1.0,
                }
            ],
            "z": 0,
            "x": 0,
            "y": 0,
            "d": 0,
        }
        proj = CartesianProjector3D((0, 0, 0, 100, 100, 100))
        data = tile_to_glb(tile, proj)
        assert data[:4] == b"glTF"

    def test_tin_tile_produces_glb(self):
        """A tile with TIN features produces valid GLB."""
        tile = {
            "features": [
                {
                    "geometry": [0.0, 0.0, 0.5, 0.0, 0.25, 0.5, 0.0, 0.0],
                    "geometry_z": [0.0, 0.1, 0.2, 0.0],
                    "type": 5,  # TIN_TYPE
                    "ring_lengths": [4],
                    "tags": {"type": "mesh"},
                    "minX": 0.0,
                    "minY": 0.0,
                    "minZ": 0.0,
                    "maxX": 0.5,
                    "maxY": 0.5,
                    "maxZ": 0.2,
                }
            ],
            "z": 0,
            "x": 0,
            "y": 0,
            "d": 0,
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
        assert box[0] == 5  # cx
        assert box[1] == 10  # cy
        assert box[2] == 15  # cz
        assert box[3] == 5  # halfX
        assert box[4] == 0
        assert box[5] == 0
        assert box[6] == 0
        assert box[7] == 10  # halfY
        assert box[8] == 0
        assert box[9] == 0
        assert box[10] == 0
        assert box[11] == 15  # halfZ

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
            (1, 0, 0, 0): {
                "features": [{"geometry": [0.25, 0.25], "geometry_z": [0.25], "type": 1}]
            },
            (1, 1, 1, 1): {
                "features": [{"geometry": [0.75, 0.75], "geometry_z": [0.75], "type": 1}]
            },
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
            (1, 0, 0, 0): {
                "features": [{"geometry": [0.25, 0.25], "geometry_z": [0.25], "type": 1}]
            },
            (1, 1, 1, 1): {
                "features": [{"geometry": [0.75, 0.75], "geometry_z": [0.75], "type": 1}]
            },
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


# ===========================================================================
# GLB extras — parentId round-trip into node.extras._parent_id
# ===========================================================================


def test_parentid_extras_in_glb(tmp_path):
    """MuDMFeature.parentId must appear as _parent_id in GLB node extras.

    Task 2 promotes top-level muDM fields (parentId/ref/id/featureClass) into
    the tags map as reserved `_`-prefixed keys. Task 4 verifies those tags
    land on each GLB node's `extras` object unchanged.
    """
    import struct

    pytest.importorskip("mudm_tools._rs")
    from mudm_tools._rs import StreamingTileGenerator

    gen = StreamingTileGenerator(min_zoom=0, max_zoom=0)
    # Triangle in normalized [0,1]^3 space (same pattern as test_tiling3d_rust.py).
    feat = {
        "geometry": [0.2, 0.3, 0.4, 0.5, 0.3, 0.7],
        "geometry_z": [0.1, 0.4, 0.6],
        "ring_lengths": [3],
        "type": 5,  # TIN
        "tags": {"compartment": "axon"},
        "parentId": "neuron_17",
        "minX": 0.2,
        "minY": 0.3,
        "minZ": 0.1,
        "maxX": 0.4,
        "maxY": 0.7,
        "maxZ": 0.6,
    }
    gen.add_feature(feat)

    out = str(tmp_path / "tiles")
    gen.generate_3dtiles(out, (0.0, 0.0, 0.0, 100.0, 100.0, 100.0))

    glbs = list(Path(tmp_path).rglob("*.glb"))
    assert glbs, "no GLB produced"

    glb = glbs[0].read_bytes()
    assert glb[:4] == b"glTF"

    # Parse GLB: 12-byte header, then JSON chunk (length u32, type u32, payload).
    json_len = struct.unpack("<I", glb[12:16])[0]
    json_bytes = glb[20 : 20 + json_len]
    gltf = json.loads(json_bytes)
    nodes = gltf.get("nodes", [])
    assert nodes, f"no nodes in gltf; got {gltf}"
    assert any(
        n.get("extras", {}).get("_parent_id") == "neuron_17" for n in nodes
    ), f"no node carries _parent_id; got nodes={nodes}"
    assert any(
        n.get("extras", {}).get("compartment") == "axon" for n in nodes
    ), f"no node carries compartment tag; got nodes={nodes}"


# ===========================================================================
# WS-C C.1 + C.4 — GLB byte-identity gate (serial == parallel, canonical order)
# ===========================================================================


def _make_obj_shards(obj_dir: Path, n: int = 60, *, seed: int = 1) -> list[str]:
    """Write `n` tiny single-triangle .obj files clustered in world space.

    add_obj_files writes ONE `.mjf` shard per input file, so the encode phase
    must merge many shards' fragments into the same coarse-zoom tiles. The
    within-tile merge order (DashMap) is non-deterministic across reads; this
    is the structure that makes the feature_id sort load-bearing for byte
    identity.
    """
    import random

    obj_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    paths: list[str] = []
    for i in range(n):
        cx = rng.uniform(5.0, 15.0)
        cy = rng.uniform(5.0, 15.0)
        cz = rng.uniform(5.0, 15.0)
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
    import hashlib

    return {
        str(p.relative_to(out_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out_dir.rglob("*.glb"))
    }


def _generate_3dtiles_from_shards(
    tmp_path: Path, out_name: str, *, io_threads: int
) -> dict[str, str]:
    """Ingest the shared OBJ shard fixture and emit GLB tiles, returning the
    {glb_relpath: sha256} map. `io_threads=1` is serial, `0` uses the global
    rayon pool (parallel reads)."""
    from mudm_tools._rs import StreamingTileGenerator

    obj_dir = tmp_path / "objs"
    if not obj_dir.exists():
        _make_obj_shards(obj_dir)
    paths = sorted(str(p) for p in obj_dir.glob("*.obj"))
    bounds = (0.0, 0.0, 0.0, 100.0, 100.0, 100.0)

    gen = StreamingTileGenerator(min_zoom=0, max_zoom=3, base_cells=100)
    gen._set_io_threads(io_threads)
    tags = [{"idx": str(i)} for i in range(len(paths))]
    gen.add_obj_files(paths, bounds, tags, 0)

    out_dir = tmp_path / out_name
    gen.generate_3dtiles(str(out_dir), bounds)
    return _glb_sha_map(out_dir)


def test_glb_serial_equals_parallel_byte_identity(tmp_path):
    """WS-C C.1+C.4 gate: with the per-tile feature_id sort canonicalizing
    within-tile node order, a SERIAL run (io_threads=1) and a PARALLEL run
    (global pool) must produce BYTE-IDENTICAL .glb files.

    Pre-sort this FAILS because the DashMap merge of many shards into a tile is
    non-deterministic (verified: serial run-to-run output itself varies). The
    sort canonicalizes node order so serial == parallel == stable.
    """
    pytest.importorskip("mudm_tools._rs")

    serial = _generate_3dtiles_from_shards(tmp_path, "serial", io_threads=1)
    parallel = _generate_3dtiles_from_shards(tmp_path, "parallel", io_threads=0)

    assert serial, "no GLB tiles produced"
    assert set(serial) == set(parallel), (
        f"different tile sets: serial-only={set(serial) - set(parallel)} "
        f"parallel-only={set(parallel) - set(serial)}"
    )
    diffs = {k: (serial[k], parallel[k]) for k in serial if serial[k] != parallel[k]}
    assert not diffs, (
        f"{len(diffs)} GLB tiles differ between serial and parallel runs "
        f"(non-canonical within-tile order); e.g. {list(diffs)[:5]}"
    )


def test_glb_serial_runs_are_stable_byte_identity(tmp_path):
    """WS-C C.1+C.4 gate: two independent serial ingests of the same multi-shard
    fixture must produce byte-identical GLB. Pre-sort this FAILS (DashMap merge
    order is non-deterministic even serially); the feature_id sort fixes it."""
    pytest.importorskip("mudm_tools._rs")

    run_a = _generate_3dtiles_from_shards(tmp_path, "serial_a", io_threads=1)
    run_b = _generate_3dtiles_from_shards(tmp_path, "serial_b", io_threads=1)

    assert run_a, "no GLB tiles produced"
    diffs = {k: (run_a[k], run_b[k]) for k in run_a if run_a[k] != run_b[k]}
    assert not diffs, (
        f"{len(diffs)} GLB tiles differ between two serial runs "
        f"(non-deterministic merge order); e.g. {list(diffs)[:5]}"
    )


# ===========================================================================
# WS-C C.2 — n_batches back-pressure against the ceiling + Step 2b floor
# ===========================================================================


def _make_spread_obj_shards(obj_dir: Path, n: int = 80, *, seed: int = 7) -> list[str]:
    """Write `n` tiny single-triangle .obj files SPREAD across world space so
    they fall into many distinct coarse-zoom tile keys.

    Unlike `_make_obj_shards` (which clusters into one coarse tile), spreading
    across tile keys lets the C.2 hash-batcher actually split the resident set
    across batches — exercising the re-split path. Each .obj is its own shard,
    so the encode phase still merges shards into tiles per the C.4 sort.
    """
    import random

    obj_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    paths: list[str] = []
    for i in range(n):
        # Spread across the full [2, 98] cube so coarse zooms get distinct tiles.
        cx = rng.uniform(2.0, 98.0)
        cy = rng.uniform(2.0, 98.0)
        cz = rng.uniform(2.0, 98.0)
        p = obj_dir / f"spread_{i:03d}.obj"
        p.write_text(
            f"v {cx} {cy} {cz}\n"
            f"v {cx + 0.5} {cy} {cz}\n"
            f"v {cx + 0.25} {cy + 0.5} {cz + 0.25}\n"
            "f 1 2 3\n"
        )
        paths.append(str(p))
    return paths


def _generate_3dtiles_spread(
    tmp_path: Path,
    out_name: str,
    *,
    io_threads: int,
    max_memory_bytes: int | None = None,
    probe: bool = False,
) -> tuple[dict[str, str], int]:
    """Ingest the spread OBJ fixture and emit GLB tiles. Returns
    ({glb_relpath: sha256}, peak_resident_bytes). `peak` is 0 unless
    `probe=True` (resets + reads the per-batch peak-resident probe)."""
    from mudm_tools._rs import StreamingTileGenerator

    obj_dir = tmp_path / "spread_objs"
    if not obj_dir.exists():
        _make_spread_obj_shards(obj_dir)
    paths = sorted(str(p) for p in obj_dir.glob("*.obj"))
    bounds = (0.0, 0.0, 0.0, 100.0, 100.0, 100.0)

    gen = StreamingTileGenerator(min_zoom=0, max_zoom=3, base_cells=100)
    gen._set_io_threads(io_threads)
    if max_memory_bytes is not None:
        gen._set_max_memory(max_memory_bytes)
    tags = [{"idx": str(i)} for i in range(len(paths))]
    gen.add_obj_files(paths, bounds, tags, 0)

    if probe:
        gen._reset_peak_resident_bytes()

    out_dir = tmp_path / out_name
    gen.generate_3dtiles(str(out_dir), bounds)
    peak = gen._get_peak_resident_bytes() if probe else 0
    return _glb_sha_map(out_dir), peak


def test_glb_back_pressure_keeps_peak_within_budget(tmp_path):
    """WS-C gate: a tiny ceiling forces the hash-partition spill to split the
    corpus into K>1 partitions (K = ceil(disk*3 / 0.4*budget)), keeping per-
    partition resident decoded bytes within budget — AND the GLB output stays
    byte-identical to the default-budget (single-partition) run (the feature_id
    sort is preserved, and the spill routes fragments unchanged)."""
    pytest.importorskip("mudm_tools._rs")

    # Baseline: default (large) ceiling — the byte-identity reference (K==1).
    baseline, _ = _generate_3dtiles_spread(tmp_path, "bp_baseline", io_threads=1)
    assert baseline, "no GLB tiles produced"

    # Tiny ceiling forces the corpus to be hash-partitioned into K>1 partitions,
    # each read and encoded in isolation. 64 KiB is far below the whole corpus's
    # resident set for this fixture but above any single tile's floor.
    tiny = 64 * 1024
    bounded, peak = _generate_3dtiles_spread(
        tmp_path, "bp_bounded", io_threads=1, max_memory_bytes=tiny, probe=True
    )

    # Output unchanged vs the C.1 baseline (back-pressure changes batching only).
    assert set(baseline) == set(bounded), (
        f"different tile sets: baseline-only={set(baseline) - set(bounded)} "
        f"bounded-only={set(bounded) - set(baseline)}"
    )
    diffs = {k: (baseline[k], bounded[k]) for k in baseline if baseline[k] != bounded[k]}
    assert not diffs, (
        f"{len(diffs)} GLB tiles differ between default-budget and bounded runs; "
        f"e.g. {list(diffs)[:5]}"
    )

    # Peak resident decoded bytes per batch stays within the derived budget.
    # The function derives effective_budget = 0.8 * ceiling; the probe records
    # the actual resident bytes after the (possibly re-split) batch is built.
    assert peak > 0, "peak-resident probe did not record any batch"
    assert peak <= tiny, (
        f"peak resident bytes {peak} exceeded ceiling {tiny} — back-pressure "
        f"re-split did not bound the batch"
    )


def test_glb_partition_count_reflects_ceiling(tmp_path):
    """WS-C spill gate: the hash-partition-spill encode picks K from the on-disk
    estimate vs the memory ceiling. A large (default) ceiling fits the corpus in
    ONE partition (k==1, no spill); a tiny ceiling forces k>1 (spill engaged).
    Proven via the `_get_tiles_partition_count` probe (mirrors the NG
    `_get_ng_bucket_count` gate). The byte-identity of both paths is covered by
    `test_glb_back_pressure_keeps_peak_within_budget`."""
    pytest.importorskip("mudm_tools._rs")
    from mudm_tools._rs import StreamingTileGenerator

    obj_dir = tmp_path / "spread_objs"
    paths = _make_spread_obj_shards(obj_dir)
    bounds = (0.0, 0.0, 0.0, 100.0, 100.0, 100.0)
    tags = [{"idx": str(i)} for i in range(len(paths))]

    # Default (large) ceiling → corpus fits one partition, no spill.
    g_big = StreamingTileGenerator(min_zoom=0, max_zoom=3, base_cells=100)
    g_big._set_io_threads(1)
    g_big.add_obj_files(paths, bounds, tags, 0)
    g_big.generate_3dtiles(str(tmp_path / "big"), bounds)
    assert g_big._get_tiles_partition_count() == 1, (
        f"large ceiling should use 1 partition, got {g_big._get_tiles_partition_count()}"
    )

    # Tiny ceiling → hash-partition spill into >1 partition.
    g_tiny = StreamingTileGenerator(min_zoom=0, max_zoom=3, base_cells=100)
    g_tiny._set_io_threads(1)
    g_tiny._set_max_memory(64 * 1024)
    g_tiny.add_obj_files(paths, bounds, tags, 0)
    g_tiny.generate_3dtiles(str(tmp_path / "tiny"), bounds)
    assert g_tiny._get_tiles_partition_count() > 1, (
        "tiny ceiling should spill into >1 partition, got "
        f"{g_tiny._get_tiles_partition_count()}"
    )


def _make_one_tile_obj_shards(obj_dir: Path, n: int = 200, *, seed: int = 3) -> list[str]:
    """Write `n` tiny .obj files tightly clustered so ALL fragments land in the
    SAME single coarse-zoom tile key — an irreducible completeness unit that
    cannot be split across batches. Used to exercise the Step 2b hard floor."""
    import random

    obj_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    paths: list[str] = []
    for i in range(n):
        # All within a tiny neighborhood → one tile at every zoom.
        cx = 50.0 + rng.uniform(-0.05, 0.05)
        cy = 50.0 + rng.uniform(-0.05, 0.05)
        cz = 50.0 + rng.uniform(-0.05, 0.05)
        p = obj_dir / f"onetile_{i:04d}.obj"
        p.write_text(
            f"v {cx} {cy} {cz}\n"
            f"v {cx + 0.01} {cy} {cz}\n"
            f"v {cx + 0.005} {cy + 0.01} {cz + 0.005}\n"
            "f 1 2 3\n"
        )
        paths.append(str(p))
    return paths


def test_glb_floor_violation_raises(tmp_path):
    """WS-C Step 2b (hard-cap floor): a single tile whose resident fragments
    exceed a tiny ceiling cannot be split (it is written whole via fs::write),
    so the generator must record a Fatal ceiling_floor failure and raise the
    typed error — fail-fast, rather than silently OOMing."""
    pytest.importorskip("mudm_tools._rs")
    from mudm_tools._rs import StreamingTileGenerator

    obj_dir = tmp_path / "onetile_objs"
    paths = _make_one_tile_obj_shards(obj_dir)
    bounds = (0.0, 0.0, 0.0, 100.0, 100.0, 100.0)

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    gen = StreamingTileGenerator(min_zoom=0, max_zoom=3, base_cells=100)
    gen._set_io_threads(1)
    gen._set_run_dir(str(run_dir))
    # A ceiling below one tile's resident fragment bytes → irreducible floor.
    gen._set_max_memory(256)
    tags = [{"idx": str(i)} for i in range(len(paths))]
    gen.add_obj_files(paths, bounds, tags, 0)

    out_dir = tmp_path / "floor_tiles"
    with pytest.raises(Exception) as exc:
        gen.generate_3dtiles(str(out_dir), bounds)
    msg = str(exc.value).lower()
    assert (
        "ceiling" in msg or "floor" in msg
    ), f"floor-violation error should name the ceiling/floor; got: {exc.value}"

    # The Fatal record must be in errors.jsonl with kind ceiling_floor.
    log = (run_dir / "errors.jsonl").read_text()
    assert "ceiling_floor" in log, f"no ceiling_floor record in errors.jsonl; got: {log}"
    assert '"severity":"fatal"' in log, f"floor violation must be fatal; got: {log}"


# ===========================================================================
# WS-C C.5 — GLB peak-RSS test (Linux-gated)
# ===========================================================================


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="ru_maxrss semantics differ per-OS (KB on Linux, bytes on macOS); "
    "the peak-RSS budget assertion is calibrated for Linux. Skips on macOS.",
)
def test_glb_peak_rss_within_budget_linux(tmp_path):
    """WS-C C.5 gate (Linux-only): under a tiny `_set_max_memory` ceiling that
    forces the per-zoom resident set to be split into >1 batch, the process's
    peak RSS growth across `generate_3dtiles` must stay within the ceiling plus
    a generous slack (encode staging, Python/arrow allocator headroom, glibc
    arena retention). This is the end-to-end RSS proof that complements the
    cheap per-batch resident-bytes probe asserted by
    `test_glb_back_pressure_keeps_peak_within_budget`.

    Decode-counter follow-up: a per-shard "decoded <= once per zoom" counter is
    NOT currently exposed by the Rust extension (only the process-global
    PEAK_RESIDENT_BYTES probe via `_get/_reset_peak_resident_bytes`). The
    per-batch resident-bytes probe below stands in as the bounded-memory proof;
    a dedicated decode-counter is a follow-up if cross-zoom re-decode accounting
    is needed (see WS-C C.5 Step 1).
    """
    import resource

    pytest.importorskip("mudm_tools._rs")

    # Tiny ceiling forces >1 batch (well below one zoom's resident set for the
    # spread fixture, but above any single tile's irreducible floor).
    tiny = 64 * 1024

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    shas, peak = _generate_3dtiles_spread(
        tmp_path, "rss_bounded", io_threads=1, max_memory_bytes=tiny, probe=True
    )
    assert shas, "no GLB tiles produced"

    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is KB on Linux; convert the delta to bytes.
    rss_delta_bytes = (rss_after - rss_before) * 1024

    # The cheap per-batch resident probe must show the bounded batch.
    assert peak > 0, "peak-resident probe did not record any batch"
    assert peak <= tiny, (
        f"per-batch resident bytes {peak} exceeded ceiling {tiny} — back-pressure "
        f"re-split did not bound the batch"
    )

    # End-to-end RSS growth must stay within the ceiling + slack. ru_maxrss is a
    # high-water mark for the whole process (never shrinks), so a large slack
    # absorbs the Rust/Python/arrow allocator footprint that is unrelated to the
    # decoded-fragment residency the ceiling governs. The point is that growth is
    # bounded (does NOT scale with the full corpus), not byte-exact.
    slack = 256 * 1024 * 1024  # 256 MiB
    assert rss_delta_bytes < tiny + slack, (
        f"peak RSS grew by {rss_delta_bytes} bytes across generate_3dtiles, "
        f"exceeding ceiling {tiny} + slack {slack} — memory is not bounded"
    )
