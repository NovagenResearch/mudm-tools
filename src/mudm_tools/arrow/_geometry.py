"""MuDM geometry → Shapely → WKB conversion."""

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
    PolyhedralSurface,
    TIN,
)


def _coords(pos: Any) -> tuple:
    """Convert a Position (list/tuple) to a plain tuple."""
    return tuple(pos)


def _ring(coords: list) -> list[tuple]:
    """Convert a ring (list of positions) to list of tuples."""
    return [_coords(p) for p in coords]


def point_to_shapely(geom: Point) -> ShapelyPoint:
    return ShapelyPoint(*_coords(geom.coordinates))


def multipoint_to_shapely(geom: MultiPoint) -> ShapelyMultiPoint:
    return ShapelyMultiPoint([_coords(p) for p in geom.coordinates])


def linestring_to_shapely(geom: LineString) -> ShapelyLineString:
    return ShapelyLineString([_coords(p) for p in geom.coordinates])


def multilinestring_to_shapely(geom: MultiLineString) -> ShapelyMultiLineString:
    return ShapelyMultiLineString(
        [[_coords(p) for p in line] for line in geom.coordinates]
    )


def polygon_to_shapely(geom: Polygon) -> ShapelyPolygon:
    exterior = _ring(geom.coordinates[0])
    holes = [_ring(h) for h in geom.coordinates[1:]]
    return ShapelyPolygon(exterior, holes)


def multipolygon_to_shapely(geom: MultiPolygon) -> ShapelyMultiPolygon:
    polys = []
    for poly_coords in geom.coordinates:
        exterior = _ring(poly_coords[0])
        holes = [_ring(h) for h in poly_coords[1:]]
        polys.append(ShapelyPolygon(exterior, holes))
    return ShapelyMultiPolygon(polys)


def geometry_collection_to_shapely(
    geom: GeometryCollection,
) -> ShapelyGeometryCollection:
    parts = [to_shapely(g) for g in geom.geometries]
    return ShapelyGeometryCollection(parts)


def tin_to_shapely(geom: TIN) -> ShapelyMultiPolygon:
    """Convert TIN to MultiPolygon3D (each triangle face → polygon)."""
    polys = []
    for face in geom.coordinates:
        exterior = _ring(face[0])
        polys.append(ShapelyPolygon(exterior))
    return ShapelyMultiPolygon(polys)


def polyhedral_to_shapely(geom: PolyhedralSurface) -> ShapelyMultiPolygon:
    """Convert PolyhedralSurface to MultiPolygon3D (each face → polygon)."""
    polys = []
    for face in geom.coordinates:
        exterior = _ring(face[0])
        holes = [_ring(h) for h in face[1:]]
        polys.append(ShapelyPolygon(exterior, holes))
    return ShapelyMultiPolygon(polys)


def to_shapely(geom: Any) -> Any:
    """Convert any MuDM/GeoJSON geometry to a Shapely geometry.

    Returns None for None geometry.
    """
    if geom is None:
        return None
    if isinstance(geom, Point):
        return point_to_shapely(geom)
    if isinstance(geom, MultiPoint):
        return multipoint_to_shapely(geom)
    if isinstance(geom, LineString):
        return linestring_to_shapely(geom)
    if isinstance(geom, MultiLineString):
        return multilinestring_to_shapely(geom)
    if isinstance(geom, Polygon):
        return polygon_to_shapely(geom)
    if isinstance(geom, MultiPolygon):
        return multipolygon_to_shapely(geom)
    if isinstance(geom, GeometryCollection):
        return geometry_collection_to_shapely(geom)
    if isinstance(geom, TIN):
        return tin_to_shapely(geom)
    if isinstance(geom, PolyhedralSurface):
        return polyhedral_to_shapely(geom)
    raise TypeError(f"Unsupported geometry type: {type(geom)}")


def to_wkb(shapely_geom: Any) -> bytes | None:
    """Convert a Shapely geometry to WKB bytes, or None for null geometry."""
    if shapely_geom is None:
        return None
    return shapely.to_wkb(shapely_geom, include_srid=False)


def geometry_type_name(shapely_geom: Any) -> str | None:
    """Return the GeoParquet geometry type name (e.g. 'Point Z', 'MultiLineString Z')."""
    if shapely_geom is None:
        return None
    name = shapely_geom.geom_type
    if shapely.get_coordinate_dimension(shapely_geom) == 3:
        name += " Z"
    return name
