"""Tests for the 2D tiled Parquet pipeline."""

from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mudm_tools._rs import CartesianProjector2D, StreamingTileGenerator2D
from mudm_tools.tiling2d.parquet_writer import generate_parquet, _parquet_schema
from mudm_tools.tiling2d.parquet_reader import read_parquet
from mudm_tools.tiling2d.parquet_prime import prime_parquet, deprime_parquet, repartition_parquet


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _make_point_feature(x: float, y: float, tags: dict | None = None):
    """Create a projected point feature dict for add_feature()."""
    return {
        "xy": [x, y],
        "geom_type": 1,
        "ring_lengths": [],
        "min_x": x, "min_y": y,
        "max_x": x, "max_y": y,
        "tags": tags or {},
    }


def _make_line_feature(coords: list[tuple[float, float]], tags: dict | None = None):
    """Create a projected line feature dict for add_feature()."""
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
    """Create a projected polygon feature dict for add_feature()."""
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
# Schema validation
# ---------------------------------------------------------------------------

class TestSchema:
    def test_schema_columns(self):
        schema = _parquet_schema()
        names = [f.name for f in schema]
        assert names == [
            "zoom", "tile_x", "tile_y", "feature_id",
            "geom_type", "positions", "indices",
            "ring_lengths", "tags",
        ]

    def test_schema_types(self):
        schema = _parquet_schema()
        assert schema.field("zoom").type == pa.uint8()
        assert schema.field("tile_x").type == pa.uint16()
        assert schema.field("tile_y").type == pa.uint16()
        assert schema.field("feature_id").type == pa.uint32()
        assert schema.field("geom_type").type == pa.uint8()
        assert schema.field("positions").type == pa.large_binary()
        assert schema.field("indices").type == pa.large_binary()
        assert schema.field("ring_lengths").type == pa.list_(pa.uint32())
        assert schema.field("tags").type == pa.map_(pa.utf8(), pa.utf8())


# ---------------------------------------------------------------------------
# Point encoding
# ---------------------------------------------------------------------------

class TestPointEncoding:
    def test_single_point(self, tmp_dir):
        gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=1)
        feat = _make_point_feature(0.25, 0.75, tags={"name": "p1"})
        gen.add_feature(feat)

        out = tmp_dir / "points.parquet"
        n = generate_parquet(gen, out, (0.0, 0.0, 1.0, 1.0))
        assert n > 0

        rows = read_parquet(out)
        assert len(rows) > 0

        # All rows should have geom_type=1 (Point)
        for r in rows:
            assert r["geom_type"] == 1
            assert r["positions"].shape[1] == 2
            assert len(r["indices"]) == 0  # Points have no indices
            assert r["ring_lengths"] == []

    def test_point_world_coords(self, tmp_dir):
        """Check that positions are in world coordinates."""
        gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=0)
        # Point at normalized (0.5, 0.5) → world (50, 50)
        feat = _make_point_feature(0.5, 0.5)
        gen.add_feature(feat)

        out = tmp_dir / "pt_world.parquet"
        generate_parquet(gen, out, (0.0, 0.0, 100.0, 100.0))

        rows = read_parquet(out)
        assert len(rows) == 1
        pos = rows[0]["positions"]
        assert pos.shape == (1, 2)
        np.testing.assert_allclose(pos[0, 0], 50.0, atol=0.1)
        np.testing.assert_allclose(pos[0, 1], 50.0, atol=0.1)

    def test_point_tags(self, tmp_dir):
        gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=0)
        feat = _make_point_feature(0.5, 0.5, tags={"color": "red", "size": "10"})
        gen.add_feature(feat)

        out = tmp_dir / "pt_tags.parquet"
        generate_parquet(gen, out, (0.0, 0.0, 1.0, 1.0))

        rows = read_parquet(out)
        assert rows[0]["tags"]["color"] == "red"
        assert rows[0]["tags"]["size"] == "10"


# ---------------------------------------------------------------------------
# LineString encoding
# ---------------------------------------------------------------------------

class TestLineStringEncoding:
    def test_linestring_indices(self, tmp_dir):
        gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=0)
        feat = _make_line_feature([(0.1, 0.1), (0.5, 0.5), (0.9, 0.1)])
        gen.add_feature(feat)

        out = tmp_dir / "line.parquet"
        generate_parquet(gen, out, (0.0, 0.0, 1.0, 1.0))

        rows = read_parquet(out)
        assert len(rows) == 1
        r = rows[0]
        assert r["geom_type"] == 2
        assert r["positions"].shape == (3, 2)
        # Line segment indices: [0,1, 1,2]
        assert len(r["indices"]) == 4
        assert list(r["indices"]) == [0, 1, 1, 2]
        assert r["ring_lengths"] == []


# ---------------------------------------------------------------------------
# Polygon encoding
# ---------------------------------------------------------------------------

class TestPolygonEncoding:
    def test_polygon_ring_lengths(self, tmp_dir):
        gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=0)
        # Square polygon
        outer = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]
        feat = _make_polygon_feature([outer])
        gen.add_feature(feat)

        out = tmp_dir / "poly.parquet"
        generate_parquet(gen, out, (0.0, 0.0, 1.0, 1.0))

        rows = read_parquet(out)
        assert len(rows) >= 1
        r = rows[0]
        assert r["geom_type"] == 3
        assert len(r["indices"]) == 0  # Polygons have no indices
        assert len(r["ring_lengths"]) >= 1  # At least exterior ring

    def test_polygon_with_hole(self, tmp_dir):
        gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=0)
        outer = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        hole = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
        feat = _make_polygon_feature([outer, hole])
        gen.add_feature(feat)

        out = tmp_dir / "poly_hole.parquet"
        generate_parquet(gen, out, (0.0, 0.0, 1.0, 1.0))

        rows = read_parquet(out)
        assert len(rows) >= 1
        # At zoom 0, the full polygon should have 2 rings
        r = rows[0]
        assert r["geom_type"] == 3
        # ring_lengths should have 2 entries (exterior + hole)
        # (may have more vertices due to Sutherland-Hodgman clipping)
        assert len(r["ring_lengths"]) == 2


# ---------------------------------------------------------------------------
# Quadtree tiling correctness
# ---------------------------------------------------------------------------

class TestQuadtreeTiling:
    def test_point_correct_tile(self, tmp_dir):
        gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=2)
        # Point at (0.25, 0.75) → at zoom 2: tile (1, 3)
        feat = _make_point_feature(0.25, 0.75)
        gen.add_feature(feat)

        out = tmp_dir / "qt.parquet"
        generate_parquet(gen, out, (0.0, 0.0, 1.0, 1.0))

        rows = read_parquet(out, zoom=2)
        assert len(rows) >= 1
        # At zoom 2, the point is in tile x=1 (0.25 * 4 = 1), y=3 (0.75 * 4 = 3)
        tiles = [(r["tile_x"], r["tile_y"]) for r in rows]
        assert (1, 3) in tiles

    def test_polygon_spans_multiple_tiles(self, tmp_dir):
        gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=1)
        # Large polygon covering all of [0,1]²
        outer = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        feat = _make_polygon_feature([outer])
        gen.add_feature(feat)

        out = tmp_dir / "qt_poly.parquet"
        generate_parquet(gen, out, (0.0, 0.0, 1.0, 1.0))

        # At zoom 1, should have fragments in all 4 tiles
        rows_z1 = read_parquet(out, zoom=1)
        tiles = set((r["tile_x"], r["tile_y"]) for r in rows_z1)
        assert len(tiles) == 4


# ---------------------------------------------------------------------------
# World-coordinate round-trip
# ---------------------------------------------------------------------------

class TestWorldCoordRoundtrip:
    def test_round_trip(self, tmp_dir):
        bounds = (10.0, 20.0, 110.0, 220.0)
        gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=0)

        proj = CartesianProjector2D(bounds)
        nx, ny = proj.project(60.0, 120.0)
        feat = _make_point_feature(nx, ny)
        gen.add_feature(feat)

        out = tmp_dir / "roundtrip.parquet"
        generate_parquet(gen, out, bounds)

        rows = read_parquet(out)
        assert len(rows) == 1
        pos = rows[0]["positions"]
        np.testing.assert_allclose(pos[0, 0], 60.0, atol=0.1)
        np.testing.assert_allclose(pos[0, 1], 120.0, atol=0.1)


# ---------------------------------------------------------------------------
# Streaming API
# ---------------------------------------------------------------------------

class TestStreamingAPI:
    def test_streaming_single_file(self, tmp_dir):
        gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=2)
        for i in range(10):
            x = (i + 0.5) / 10.0
            y = (i + 0.5) / 10.0
            feat = _make_point_feature(x, y, tags={"idx": str(i)})
            gen.add_feature(feat)

        out = tmp_dir / "streaming.parquet"
        n = generate_parquet(gen, out, (0.0, 0.0, 1.0, 1.0), batch_size=5)
        assert n > 0

        rows = read_parquet(out)
        assert len(rows) == n

    def test_streaming_partitioned(self, tmp_dir):
        gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=2)
        for i in range(10):
            x = (i + 0.5) / 10.0
            y = (i + 0.5) / 10.0
            feat = _make_point_feature(x, y)
            gen.add_feature(feat)

        out = tmp_dir / "partitioned"
        n = generate_parquet(
            gen, out, (0.0, 0.0, 1.0, 1.0),
            partitioned=True, batch_size=5,
        )
        assert n > 0

        # Check directory structure: zoom=N/part_NNN.parquet
        zoom_dirs = sorted(out.glob("zoom=*"))
        assert len(zoom_dirs) > 0

        rows = read_parquet(out)
        assert len(rows) == n


# ---------------------------------------------------------------------------
# Predicate pushdown
# ---------------------------------------------------------------------------

class TestPredicatePushdown:
    def test_filter_zoom(self, tmp_dir):
        gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=2)
        feat = _make_point_feature(0.5, 0.5)
        gen.add_feature(feat)

        out = tmp_dir / "pushdown.parquet"
        generate_parquet(gen, out, (0.0, 0.0, 1.0, 1.0))

        rows_all = read_parquet(out)
        rows_z0 = read_parquet(out, zoom=0)
        rows_z2 = read_parquet(out, zoom=2)

        assert len(rows_all) > len(rows_z0)
        assert all(r["zoom"] == 0 for r in rows_z0)
        assert all(r["zoom"] == 2 for r in rows_z2)

    def test_filter_feature_id(self, tmp_dir):
        gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=0)
        gen.add_feature(_make_point_feature(0.3, 0.3, tags={"id": "a"}))
        gen.add_feature(_make_point_feature(0.7, 0.7, tags={"id": "b"}))

        out = tmp_dir / "fid.parquet"
        generate_parquet(gen, out, (0.0, 0.0, 1.0, 1.0))

        rows_0 = read_parquet(out, feature_id=0)
        rows_1 = read_parquet(out, feature_id=1)
        assert all(r["feature_id"] == 0 for r in rows_0)
        assert all(r["feature_id"] == 1 for r in rows_1)


# ---------------------------------------------------------------------------
# GeoJSON ingestion
# ---------------------------------------------------------------------------

class TestGeoJsonIngestion:
    def test_add_geojson_point(self, tmp_dir):
        gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=1)
        geojson = json.dumps({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [50.0, 50.0]},
            "properties": {"label": "center"},
        })
        fids = gen.add_geojson(geojson, (0.0, 0.0, 100.0, 100.0))
        assert len(fids) == 1

        out = tmp_dir / "geojson_pt.parquet"
        n = generate_parquet(gen, out, (0.0, 0.0, 100.0, 100.0))
        assert n > 0

        rows = read_parquet(out, zoom=0)
        assert len(rows) >= 1
        assert rows[0]["tags"]["label"] == "center"

    def test_add_geojson_feature_collection(self, tmp_dir):
        gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=0)
        geojson = json.dumps({
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [25.0, 25.0]},
                    "properties": {"name": "a"},
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": [[0.0, 0.0], [100.0, 100.0]]},
                    "properties": {"name": "b"},
                },
            ],
        })
        fids = gen.add_geojson(geojson, (0.0, 0.0, 100.0, 100.0))
        assert len(fids) == 2

        out = tmp_dir / "fc.parquet"
        n = generate_parquet(gen, out, (0.0, 0.0, 100.0, 100.0))
        assert n >= 2

    def test_add_geojson_polygon(self, tmp_dir):
        gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=0)
        geojson = json.dumps({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]],
            },
            "properties": {},
        })
        fids = gen.add_geojson(geojson, (0.0, 0.0, 100.0, 100.0))
        assert len(fids) == 1

        out = tmp_dir / "geojson_poly.parquet"
        generate_parquet(gen, out, (0.0, 0.0, 100.0, 100.0))

        rows = read_parquet(out)
        assert len(rows) >= 1
        assert rows[0]["geom_type"] == 3

    def test_add_geojson_multipolygon(self, tmp_dir):
        gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=0)
        geojson = json.dumps({
            "type": "Feature",
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [
                    [[[0.0, 0.0], [50.0, 0.0], [50.0, 50.0], [0.0, 50.0]]],
                    [[[50.0, 50.0], [100.0, 50.0], [100.0, 100.0], [50.0, 100.0]]],
                ],
            },
            "properties": {"name": "multi"},
        })
        fids = gen.add_geojson(geojson, (0.0, 0.0, 100.0, 100.0))
        assert len(fids) == 1

        out = tmp_dir / "multipoly.parquet"
        generate_parquet(gen, out, (0.0, 0.0, 100.0, 100.0))

        rows = read_parquet(out)
        assert len(rows) >= 1
        # MultiPolygon should have ring_lengths with entries for both sub-polygons
        total_ring_count = sum(len(r["ring_lengths"]) for r in rows)
        assert total_ring_count >= 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_generator(self, tmp_dir):
        gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=2)
        out = tmp_dir / "empty.parquet"
        n = generate_parquet(gen, out, (0.0, 0.0, 1.0, 1.0))
        assert n == 0
        # File should still be readable
        rows = read_parquet(out)
        assert len(rows) == 0

    def test_degenerate_bounds(self, tmp_dir):
        gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=0)
        feat = _make_point_feature(0.0, 0.0)
        gen.add_feature(feat)

        out = tmp_dir / "degen.parquet"
        n = generate_parquet(gen, out, (5.0, 5.0, 5.0, 5.0))
        assert n >= 1

    def test_feature_count(self):
        gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=0)
        assert gen.feature_count_val() == 0
        gen.add_feature(_make_point_feature(0.5, 0.5))
        assert gen.feature_count_val() == 1
        gen.add_feature(_make_point_feature(0.3, 0.3))
        assert gen.feature_count_val() == 2


# ---------------------------------------------------------------------------
# Prime / deprime / repartition
# ---------------------------------------------------------------------------

class TestPrime:
    def test_prime_deprime(self, tmp_dir):
        gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=2)
        for i in range(5):
            gen.add_feature(_make_point_feature((i + 0.5) / 5, 0.5))

        out = tmp_dir / "primed"
        generate_parquet(gen, out, (0.0, 0.0, 1.0, 1.0), partitioned=True)

        n_primed = prime_parquet(out)
        assert n_primed > 0

        # Arrow IPC files should exist
        arrow_files = list(out.glob("zoom=*/*.arrow"))
        assert len(arrow_files) > 0

        # Read should still work
        rows = read_parquet(out)
        assert len(rows) > 0

        n_deprimed = deprime_parquet(out)
        assert n_deprimed == n_primed

        # Arrow files gone
        arrow_files = list(out.glob("zoom=*/*.arrow"))
        assert len(arrow_files) == 0

    def test_repartition(self, tmp_dir):
        gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=2)
        for i in range(20):
            gen.add_feature(_make_point_feature((i + 0.5) / 20, 0.5))

        out = tmp_dir / "repart"
        generate_parquet(gen, out, (0.0, 0.0, 1.0, 1.0), partitioned=True)

        result = repartition_parquet(out, max_file_bytes=100)
        assert len(result) > 0
        # All values should be positive
        assert all(v > 0 for v in result.values())


# ---------------------------------------------------------------------------
# CartesianProjector2D
# ---------------------------------------------------------------------------

class TestProjector2D:
    def test_project_unproject(self):
        proj = CartesianProjector2D((10.0, 20.0, 110.0, 220.0))
        nx, ny = proj.project(60.0, 120.0)
        assert abs(nx - 0.5) < 1e-10
        assert abs(ny - 0.5) < 1e-10

        wx, wy = proj.unproject(nx, ny)
        assert abs(wx - 60.0) < 1e-10
        assert abs(wy - 120.0) < 1e-10


# ---------------------------------------------------------------------------
# Integration: load example.json
# ---------------------------------------------------------------------------

class TestIntegration:
    @pytest.fixture
    def example_json_path(self):
        p = Path(__file__).parent.parent / "src" / "mudm" / "examples" / "example.json"
        if not p.exists():
            pytest.skip("example.json not found")
        return p

    def test_end_to_end(self, tmp_dir, example_json_path):
        """Load example.json → tile → write Parquet → read back → verify."""
        geojson_str = example_json_path.read_text()

        # Parse to find bounds
        data = json.loads(geojson_str)
        all_coords = []
        for feat in data.get("features", [data] if "geometry" in data else []):
            geom = feat.get("geometry", feat)
            _collect_coords(geom.get("coordinates", []), all_coords)

        if not all_coords:
            pytest.skip("No coordinates in example.json")

        xs = [c[0] for c in all_coords]
        ys = [c[1] for c in all_coords]
        bounds = (min(xs), min(ys), max(xs), max(ys))

        # Ensure non-degenerate bounds
        if bounds[0] == bounds[2]:
            bounds = (bounds[0] - 1, bounds[1], bounds[2] + 1, bounds[3])
        if bounds[1] == bounds[3]:
            bounds = (bounds[0], bounds[1] - 1, bounds[2], bounds[3] + 1)

        gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=2)
        fids = gen.add_geojson(geojson_str, bounds)
        assert len(fids) > 0

        out = tmp_dir / "integration.parquet"
        n = generate_parquet(gen, out, bounds)
        assert n > 0

        rows = read_parquet(out)
        assert len(rows) == n

        # Verify all positions are within bounds (with f32 tolerance)
        for r in rows:
            pos = r["positions"]
            assert pos.shape[1] == 2
            for i in range(pos.shape[0]):
                x, y = pos[i]
                assert bounds[0] - 1 <= x <= bounds[2] + 1, f"x={x} out of bounds"
                assert bounds[1] - 1 <= y <= bounds[3] + 1, f"y={y} out of bounds"


class TestSimplify:
    """Test that simplify=True/False is respected."""

    def _make_jagged_polygon(self):
        """Create a polygon with many vertices that simplification can reduce."""
        import math
        # Jagged circle — lots of small deviations from a smooth curve
        n = 60
        ring = []
        for i in range(n):
            angle = 2 * math.pi * i / n
            r = 0.3 + 0.02 * (i % 3)  # small jag
            ring.append((0.5 + r * math.cos(angle), 0.5 + r * math.sin(angle)))
        return _make_polygon_feature([ring])

    def test_simplify_false_preserves_vertices(self, tmp_dir):
        """With simplify=False, coarse zoom should keep all polygon vertices."""
        gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=4)
        gen.add_feature(self._make_jagged_polygon())

        out_no = tmp_dir / "no_simplify.parquet"
        generate_parquet(gen, out_no, (0.0, 0.0, 1.0, 1.0), simplify=False)

        # Read zoom=0 rows (coarsest — would be simplified if simplify=True)
        rows_no = read_parquet(out_no, zoom=0)
        assert len(rows_no) > 0
        total_verts_no = sum(r["positions"].shape[0] for r in rows_no)

        gen2 = StreamingTileGenerator2D(min_zoom=0, max_zoom=4)
        gen2.add_feature(self._make_jagged_polygon())

        out_yes = tmp_dir / "yes_simplify.parquet"
        generate_parquet(gen2, out_yes, (0.0, 0.0, 1.0, 1.0), simplify=True)

        rows_yes = read_parquet(out_yes, zoom=0)
        assert len(rows_yes) > 0
        total_verts_yes = sum(r["positions"].shape[0] for r in rows_yes)

        # simplify=False should preserve more (or equal) vertices than simplify=True
        assert total_verts_no >= total_verts_yes

    def test_simplify_false_streaming(self, tmp_dir):
        """simplify=False works with the streaming partitioned path too."""
        gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=3)
        gen.add_feature(self._make_jagged_polygon())

        out = tmp_dir / "stream_no_simplify"
        n = generate_parquet(
            gen, out, (0.0, 0.0, 1.0, 1.0),
            partitioned=True, simplify=False,
        )
        assert n > 0

        rows = read_parquet(out, zoom=0)
        assert len(rows) > 0
        # At zoom=0 with simplify=False, all vertices should be preserved
        for r in rows:
            assert r["positions"].shape[0] >= 3  # at least a triangle


def _collect_coords(coords, out):
    """Recursively collect [x, y] coordinate pairs from nested arrays."""
    if not isinstance(coords, list) or len(coords) == 0:
        return
    if isinstance(coords[0], (int, float)):
        out.append(coords)
    else:
        for c in coords:
            _collect_coords(c, out)
