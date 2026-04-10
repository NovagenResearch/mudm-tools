"""Tests for the 2D PBF (Mapbox Vector Tile) pipeline."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mudm_tools._rs import StreamingTileGenerator2D
from mudm_tools.tiling2d.pbf_writer import generate_pbf
from mudm_tools.tiling2d.pbf_reader import read_pbf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _make_point_feature(x: float, y: float, tags: dict | None = None):
    return {
        "xy": [x, y],
        "geom_type": 1,
        "ring_lengths": [],
        "min_x": x, "min_y": y,
        "max_x": x, "max_y": y,
        "tags": tags or {},
    }


def _make_line_feature(coords: list[tuple[float, float]], tags: dict | None = None):
    xy = []
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    for cx, cy in coords:
        xy.extend([cx, cy])
        min_x = min(min_x, cx)
        min_y = min(min_y, cy)
        max_x = max(max_x, cx)
        max_y = max(max_y, cy)
    return {
        "xy": xy,
        "geom_type": 2,
        "ring_lengths": [],
        "min_x": min_x, "min_y": min_y,
        "max_x": max_x, "max_y": max_y,
        "tags": tags or {},
    }


def _make_polygon_feature(rings: list[list[tuple[float, float]]], tags: dict | None = None):
    xy = []
    ring_lengths = []
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    for ring in rings:
        ring_lengths.append(len(ring))
        for cx, cy in ring:
            xy.extend([cx, cy])
            min_x = min(min_x, cx)
            min_y = min(min_y, cy)
            max_x = max(max_x, cx)
            max_y = max(max_y, cy)
    return {
        "xy": xy,
        "geom_type": 3,
        "ring_lengths": ring_lengths,
        "min_x": min_x, "min_y": min_y,
        "max_x": max_x, "max_y": max_y,
        "tags": tags or {},
    }


# ---------------------------------------------------------------------------
# Point round-trip
# ---------------------------------------------------------------------------

class TestPointRoundtrip:
    def test_single_point(self, tmp_dir):
        gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=0)
        feat = _make_point_feature(0.5, 0.5, tags={"name": "center"})
        gen.add_feature(feat)

        out = tmp_dir / "pbf_point"
        n = generate_pbf(gen, out, (0.0, 0.0, 100.0, 100.0))
        assert n > 0

        rows = read_pbf(out, (0.0, 0.0, 100.0, 100.0))
        assert len(rows) > 0
        r = rows[0]
        assert r["geom_type"] == 1
        assert r["positions"].shape[1] == 2
        # World coords: normalized 0.5 → world 50
        np.testing.assert_allclose(r["positions"][0, 0], 50.0, atol=1.0)
        np.testing.assert_allclose(r["positions"][0, 1], 50.0, atol=1.0)

    def test_point_tags(self, tmp_dir):
        gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=0)
        feat = _make_point_feature(0.5, 0.5, tags={"color": "red", "size": "10"})
        gen.add_feature(feat)

        out = tmp_dir / "pbf_tags"
        generate_pbf(gen, out, (0.0, 0.0, 1.0, 1.0))

        rows = read_pbf(out, (0.0, 0.0, 1.0, 1.0))
        assert len(rows) > 0
        assert rows[0]["tags"]["color"] == "red"
        assert rows[0]["tags"]["size"] == "10"


# ---------------------------------------------------------------------------
# LineString round-trip
# ---------------------------------------------------------------------------

class TestLineStringRoundtrip:
    def test_linestring(self, tmp_dir):
        gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=0)
        feat = _make_line_feature([(0.1, 0.1), (0.5, 0.5), (0.9, 0.1)])
        gen.add_feature(feat)

        out = tmp_dir / "pbf_line"
        n = generate_pbf(gen, out, (0.0, 0.0, 1.0, 1.0))
        assert n > 0

        rows = read_pbf(out, (0.0, 0.0, 1.0, 1.0))
        assert len(rows) > 0
        r = rows[0]
        assert r["geom_type"] == 2
        assert r["positions"].shape[0] == 3
        # Check coords are approximately correct
        np.testing.assert_allclose(r["positions"][0, 0], 0.1, atol=0.01)
        np.testing.assert_allclose(r["positions"][1, 0], 0.5, atol=0.01)
        np.testing.assert_allclose(r["positions"][2, 0], 0.9, atol=0.01)


# ---------------------------------------------------------------------------
# Polygon round-trip
# ---------------------------------------------------------------------------

class TestPolygonRoundtrip:
    def test_polygon(self, tmp_dir):
        gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=0)
        outer = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]
        feat = _make_polygon_feature([outer])
        gen.add_feature(feat)

        out = tmp_dir / "pbf_poly"
        n = generate_pbf(gen, out, (0.0, 0.0, 1.0, 1.0))
        assert n > 0

        rows = read_pbf(out, (0.0, 0.0, 1.0, 1.0))
        assert len(rows) > 0
        r = rows[0]
        assert r["geom_type"] == 3
        assert len(r["ring_lengths"]) >= 1
        # Polygon should have at least 4 vertices (closed ring)
        assert r["positions"].shape[0] >= 4

    def test_polygon_with_hole(self, tmp_dir):
        gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=0)
        outer = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        hole = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
        feat = _make_polygon_feature([outer, hole])
        gen.add_feature(feat)

        out = tmp_dir / "pbf_poly_hole"
        generate_pbf(gen, out, (0.0, 0.0, 1.0, 1.0))

        rows = read_pbf(out, (0.0, 0.0, 1.0, 1.0))
        assert len(rows) >= 1
        r = rows[0]
        assert r["geom_type"] == 3
        assert len(r["ring_lengths"]) == 2


# ---------------------------------------------------------------------------
# Directory structure
# ---------------------------------------------------------------------------

class TestDirectoryStructure:
    def test_tile_paths(self, tmp_dir):
        gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=1)
        feat = _make_point_feature(0.5, 0.5)
        gen.add_feature(feat)

        out = tmp_dir / "pbf_dirs"
        generate_pbf(gen, out, (0.0, 0.0, 1.0, 1.0))

        # Should have z/x/y.pbf files
        pbf_files = list(out.rglob("*.pbf"))
        assert len(pbf_files) > 0

        # Check path structure
        for p in pbf_files:
            parts = p.relative_to(out).parts
            assert len(parts) == 3  # z/x/y.pbf
            int(parts[0])  # z must be numeric
            int(parts[1])  # x must be numeric
            assert parts[2].endswith(".pbf")

    def test_metadata_json(self, tmp_dir):
        gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=2)
        feat = _make_point_feature(0.5, 0.5)
        gen.add_feature(feat)

        out = tmp_dir / "pbf_meta"
        generate_pbf(gen, out, (0.0, 0.0, 100.0, 100.0))

        meta_path = out / "metadata.json"
        assert meta_path.exists()

        meta = json.loads(meta_path.read_text())
        assert meta["tilejson"] == "3.0.0"
        assert meta["minzoom"] == 0
        assert meta["maxzoom"] == 2
        assert meta["tile_count"] > 0
        assert len(meta["bounds"]) == 4
        assert len(meta["vector_layers"]) == 1
        assert meta["vector_layers"][0]["id"] == "geojsonLayer"


# ---------------------------------------------------------------------------
# Multi-tile tiling
# ---------------------------------------------------------------------------

class TestMultiTile:
    def test_polygon_spans_tiles(self, tmp_dir):
        gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=1)
        # Large polygon covering all of [0,1]²
        outer = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        feat = _make_polygon_feature([outer])
        gen.add_feature(feat)

        out = tmp_dir / "pbf_multi"
        n = generate_pbf(gen, out, (0.0, 0.0, 1.0, 1.0))

        # At zoom 1, should have tiles in all 4 quadrants
        rows_z1 = read_pbf(out, (0.0, 0.0, 1.0, 1.0), zoom=1)
        tiles = set((r["tile_x"], r["tile_y"]) for r in rows_z1)
        assert len(tiles) == 4


# ---------------------------------------------------------------------------
# Zoom / tile filtering
# ---------------------------------------------------------------------------

class TestFiltering:
    def test_zoom_filter(self, tmp_dir):
        gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=2)
        feat = _make_point_feature(0.5, 0.5)
        gen.add_feature(feat)

        out = tmp_dir / "pbf_filter"
        generate_pbf(gen, out, (0.0, 0.0, 1.0, 1.0))

        rows_all = read_pbf(out, (0.0, 0.0, 1.0, 1.0))
        rows_z0 = read_pbf(out, (0.0, 0.0, 1.0, 1.0), zoom=0)
        rows_z2 = read_pbf(out, (0.0, 0.0, 1.0, 1.0), zoom=2)

        assert len(rows_all) > len(rows_z0)
        assert all(r["zoom"] == 0 for r in rows_z0)
        assert all(r["zoom"] == 2 for r in rows_z2)

    def test_tile_filter(self, tmp_dir):
        gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=1)
        outer = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        feat = _make_polygon_feature([outer])
        gen.add_feature(feat)

        out = tmp_dir / "pbf_tile_filter"
        generate_pbf(gen, out, (0.0, 0.0, 1.0, 1.0))

        rows = read_pbf(out, (0.0, 0.0, 1.0, 1.0), zoom=1, tile_x=0, tile_y=0)
        assert len(rows) >= 1
        assert all(r["tile_x"] == 0 and r["tile_y"] == 0 for r in rows)


# ---------------------------------------------------------------------------
# Simplification
# ---------------------------------------------------------------------------

class TestSimplify:
    def _make_jagged_polygon(self):
        n = 60
        ring = []
        for i in range(n):
            angle = 2 * math.pi * i / n
            r = 0.3 + 0.02 * (i % 3)
            ring.append((0.5 + r * math.cos(angle), 0.5 + r * math.sin(angle)))
        return _make_polygon_feature([ring])

    def test_simplify_reduces_vertices(self, tmp_dir):
        gen_yes = StreamingTileGenerator2D(min_zoom=0, max_zoom=4)
        gen_yes.add_feature(self._make_jagged_polygon())

        gen_no = StreamingTileGenerator2D(min_zoom=0, max_zoom=4)
        gen_no.add_feature(self._make_jagged_polygon())

        out_yes = tmp_dir / "pbf_simp_yes"
        generate_pbf(gen_yes, out_yes, (0.0, 0.0, 1.0, 1.0), simplify=True)

        out_no = tmp_dir / "pbf_simp_no"
        generate_pbf(gen_no, out_no, (0.0, 0.0, 1.0, 1.0), simplify=False)

        rows_yes = read_pbf(out_yes, (0.0, 0.0, 1.0, 1.0), zoom=0)
        rows_no = read_pbf(out_no, (0.0, 0.0, 1.0, 1.0), zoom=0)

        verts_yes = sum(r["positions"].shape[0] for r in rows_yes)
        verts_no = sum(r["positions"].shape[0] for r in rows_no)

        assert verts_no >= verts_yes


# ---------------------------------------------------------------------------
# GeoJSON ingestion + PBF output
# ---------------------------------------------------------------------------

class TestGeoJsonIngestion:
    def test_geojson_to_pbf(self, tmp_dir):
        gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=1)
        geojson = json.dumps({
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [50.0, 50.0]},
                    "properties": {"label": "center"},
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]],
                    },
                    "properties": {"label": "box"},
                },
            ],
        })
        fids = gen.add_geojson(geojson, (0.0, 0.0, 100.0, 100.0))
        assert len(fids) == 2

        out = tmp_dir / "pbf_geojson"
        n = generate_pbf(gen, out, (0.0, 0.0, 100.0, 100.0))
        assert n > 0

        rows = read_pbf(out, (0.0, 0.0, 100.0, 100.0), zoom=0)
        assert len(rows) >= 2

        # Check tags survived
        labels = {r["tags"].get("label") for r in rows}
        assert "center" in labels
        assert "box" in labels


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_generator(self, tmp_dir):
        gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=2)
        out = tmp_dir / "pbf_empty"
        n = generate_pbf(gen, out, (0.0, 0.0, 1.0, 1.0))
        assert n == 0

        rows = read_pbf(out, (0.0, 0.0, 1.0, 1.0))
        assert len(rows) == 0

    def test_nonexistent_dir_read(self, tmp_dir):
        rows = read_pbf(str(tmp_dir / "nonexistent"), (0.0, 0.0, 1.0, 1.0))
        assert len(rows) == 0
