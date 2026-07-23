"""PY-3 (streaming_review.md §G): geojson auto-bounds must reflect real extent,
including GeometryCollection features (whose coordinates live under `geometries`,
not `coordinates`). Pre-fix, `_compute_bounds` read only `coordinates`, so a
GeometryCollection contributed nothing and auto-bounds fell back to (0,0,1,1)."""

import json
from pathlib import Path

from mudm_tools.converters.geojson import GeoJsonConverter


def _write(p: Path, fc: dict) -> Path:
    p.write_text(json.dumps(fc))
    return p


def test_compute_bounds_geometrycollection(tmp_path):
    # Only feature is a GeometryCollection — real extent x[10,40], y[5,60].
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "GeometryCollection",
                    "geometries": [
                        {"type": "Point", "coordinates": [10.0, 20.0]},
                        {"type": "LineString", "coordinates": [[30.0, 5.0], [40.0, 60.0]]},
                    ],
                },
            }
        ],
    }
    f = _write(tmp_path / "gc.geojson", fc)
    b = GeoJsonConverter()._compute_bounds([f])
    assert b == (10.0, 5.0, 40.0, 60.0), f"GeometryCollection extent ignored: {b}"


def test_compute_bounds_plain_polygon(tmp_path):
    # Regression: the already-fixed plain-geometry path still works.
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 0.0]]],
                },
            }
        ],
    }
    f = _write(tmp_path / "poly.geojson", fc)
    b = GeoJsonConverter()._compute_bounds([f])
    assert b == (0.0, 0.0, 100.0, 50.0)


def test_compute_bounds_nested_geometrycollection(tmp_path):
    # Defensive: GeometryCollection nested inside a GeometryCollection.
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "GeometryCollection",
                    "geometries": [
                        {
                            "type": "GeometryCollection",
                            "geometries": [{"type": "Point", "coordinates": [-5.0, 7.0]}],
                        },
                        {"type": "Point", "coordinates": [3.0, -2.0]},
                    ],
                },
            }
        ],
    }
    f = _write(tmp_path / "nested.geojson", fc)
    b = GeoJsonConverter()._compute_bounds([f])
    assert b == (-5.0, -2.0, 3.0, 7.0), f"nested GeometryCollection extent wrong: {b}"
