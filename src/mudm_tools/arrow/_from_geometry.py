"""Shapely → MuDM/GeoJSON geometry conversion (reverse of _geometry.py)."""

from __future__ import annotations

from typing import Any

import shapely
from shapely.geometry import (
    GeometryCollection as ShapelyGeometryCollection,
    LineString as ShapelyLineString,
    MultiLineString as ShapelyMultiLineString,
    MultiPoint as ShapelyMultiPoint,
    MultiPolygon as ShapelyMultiPolygon,
    Point as ShapelyPoint,
    Polygon as ShapelyPolygon,
)

from geojson_pydantic import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

from mudm.model import (
    TIN,
)


def _pos(coord: tuple) -> tuple:
    """Convert a Shapely coordinate tuple to a GeoJSON position."""
    return tuple(float(c) for c in coord)


def _is_tin(geom: ShapelyMultiPolygon) -> bool:
    """Check if a MultiPolygon is a TIN (all polygons are 3D triangles)."""
    if not geom.has_z:
        return False
    for poly in geom.geoms:
        if len(poly.interiors) != 0:
            return False
        coords = list(poly.exterior.coords)
        if len(coords) != 4:  # closed triangle = 4 coords
            return False
    return True


def _multipolygon_to_tin(geom: ShapelyMultiPolygon) -> TIN:
    """Convert a triangle-only 3D MultiPolygon to a TIN."""
    faces = []
    for poly in geom.geoms:
        ring = [_pos(c) for c in poly.exterior.coords]
        faces.append([ring])
    return TIN(type="TIN", coordinates=faces)


def _ring_coords(ring) -> list[tuple]:
    """Extract ring coordinates from a Shapely LinearRing or coord sequence."""
    return [_pos(c) for c in ring.coords]


def shapely_to_microjson(geom: Any) -> Any:
    """Convert a Shapely geometry to a GeoJSON-pydantic model.

    Returns None for None input.
    """
    if geom is None:
        return None

    if isinstance(geom, ShapelyPoint):
        return Point(type="Point", coordinates=_pos(geom.coords[0]))

    if isinstance(geom, ShapelyMultiPoint):
        return MultiPoint(
            type="MultiPoint",
            coordinates=[_pos(p.coords[0]) for p in geom.geoms],
        )

    if isinstance(geom, ShapelyLineString):
        return LineString(
            type="LineString",
            coordinates=[_pos(c) for c in geom.coords],
        )

    if isinstance(geom, ShapelyMultiLineString):
        return MultiLineString(
            type="MultiLineString",
            coordinates=[
                [_pos(c) for c in line.coords] for line in geom.geoms
            ],
        )

    if isinstance(geom, ShapelyPolygon):
        rings = [_ring_coords(geom.exterior)]
        for interior in geom.interiors:
            rings.append(_ring_coords(interior))
        return Polygon(type="Polygon", coordinates=rings)

    if isinstance(geom, ShapelyMultiPolygon):
        if _is_tin(geom):
            return _multipolygon_to_tin(geom)
        polys = []
        for poly in geom.geoms:
            rings = [_ring_coords(poly.exterior)]
            for interior in poly.interiors:
                rings.append(_ring_coords(interior))
            polys.append(rings)
        return MultiPolygon(type="MultiPolygon", coordinates=polys)

    if isinstance(geom, ShapelyGeometryCollection):
        parts = [shapely_to_microjson(g) for g in geom.geoms]
        return GeometryCollection(type="GeometryCollection", geometries=parts)

    raise TypeError(f"Unsupported Shapely geometry type: {type(geom)}")


