"""Convert MuDM features to intermediate flat-array format for 3D tiling.

Output format per feature:
  - geometry: [x0, y0, x1, y1, ...] (flat XY in normalized [0,1] space)
  - geometry_z: [z0, z1, ...]       (parallel Z in normalized [0,1] space)
  - type: int (GeomType enum value)
  - tags: dict of properties
  - min/max bounding box in normalized space
  - radii: optional per-vertex radii
"""

from __future__ import annotations

from typing import Any

from mudm.layout import geometry_bounds
from mudm.model import (
    MuDMFeature,
    MuDMFeatureCollection,
    PolyhedralSurface,
    TIN,
)
from .projector3d import CartesianProjector3D

# GeomType enum values matching the protobuf
POINT3D = 1
LINESTRING3D = 2
POLYGON3D = 3
POLYHEDRALSURFACE = 4
TIN_TYPE = 5


def _flatten_positions(
    coords: Any,
) -> tuple[list[float], list[float]]:
    """Recursively flatten nested coordinate arrays to (xy_flat, z_flat).

    Handles: single position [x, y, z], ring [[x,y,z],...],
    polygon [ring,...], multi [polygon,...].
    """
    xy: list[float] = []
    z: list[float] = []
    if not coords:
        return xy, z
    if isinstance(coords[0], (int, float)):
        # Single position
        xy.append(float(coords[0]))
        xy.append(float(coords[1]) if len(coords) > 1 else 0.0)
        z.append(float(coords[2]) if len(coords) > 2 else 0.0)
    else:
        for item in coords:
            sxy, sz = _flatten_positions(item)
            xy.extend(sxy)
            z.extend(sz)
    return xy, z


def _project_flat(
    xy: list[float], z: list[float], proj: CartesianProjector3D,
) -> tuple[list[float], list[float]]:
    """Project flat arrays through the 3D projector."""
    n = len(z)
    out_xy: list[float] = [0.0] * (n * 2)
    out_z: list[float] = [0.0] * n
    for i in range(n):
        px, py, pz = proj.project(xy[i * 2], xy[i * 2 + 1], z[i])
        out_xy[i * 2] = px
        out_xy[i * 2 + 1] = py
        out_z[i] = pz
    return out_xy, out_z


def _geom_type_for(geom: Any) -> int:
    """Map geometry to our GeomType enum."""
    t = geom.type
    if t == "TIN":
        return TIN_TYPE
    if t == "PolyhedralSurface":
        return POLYHEDRALSURFACE
    if t in ("Point", "MultiPoint"):
        return POINT3D
    if t in ("LineString", "MultiLineString"):
        return LINESTRING3D
    if t in ("Polygon", "MultiPolygon"):
        return POLYGON3D
    return 0  # UNKNOWN


def _convert_point(
    geom: Any, proj: CartesianProjector3D,
) -> list[dict]:
    """Convert Point or MultiPoint."""
    results = []
    if geom.type == "Point":
        coords_list = [geom.coordinates]
    else:
        coords_list = geom.coordinates

    for coord in coords_list:
        x = float(coord[0])
        y = float(coord[1]) if len(coord) > 1 else 0.0
        z_val = float(coord[2]) if len(coord) > 2 else 0.0
        px, py, pz = proj.project(x, y, z_val)
        results.append({
            "geometry": [px, py],
            "geometry_z": [pz],
            "type": POINT3D,
            "minX": px, "minY": py, "minZ": pz,
            "maxX": px, "maxY": py, "maxZ": pz,
        })
    return results


def _convert_line(
    geom: Any, proj: CartesianProjector3D,
) -> list[dict]:
    """Convert LineString or MultiLineString to flat arrays."""
    if geom.type == "LineString":
        rings = [geom.coordinates]
    else:
        rings = geom.coordinates

    results = []
    for ring in rings:
        xy, z = _flatten_positions(ring)
        xy, z = _project_flat(xy, z, proj)
        n = len(z)
        if n == 0:
            continue
        min_x = min(xy[i * 2] for i in range(n))
        max_x = max(xy[i * 2] for i in range(n))
        min_y = min(xy[i * 2 + 1] for i in range(n))
        max_y = max(xy[i * 2 + 1] for i in range(n))
        min_z = min(z)
        max_z = max(z)
        results.append({
            "geometry": xy,
            "geometry_z": z,
            "type": LINESTRING3D,
            "minX": min_x, "minY": min_y, "minZ": min_z,
            "maxX": max_x, "maxY": max_y, "maxZ": max_z,
        })
    return results


def _convert_polygon(
    geom: Any, proj: CartesianProjector3D,
) -> list[dict]:
    """Convert Polygon or MultiPolygon to flat arrays.

    Each ring is stored as a separate segment in the flat array,
    separated by ring-length markers (same pattern as 2D MVT).
    """
    if geom.type == "Polygon":
        polys = [geom.coordinates]
    else:
        polys = geom.coordinates

    results = []
    for poly in polys:
        all_xy: list[float] = []
        all_z: list[float] = []
        ring_lengths: list[int] = []
        for ring in poly:
            xy, z = _flatten_positions(ring)
            xy, z = _project_flat(xy, z, proj)
            ring_lengths.append(len(z))
            all_xy.extend(xy)
            all_z.extend(z)
        n = len(all_z)
        if n == 0:
            continue
        min_x = min(all_xy[i * 2] for i in range(n))
        max_x = max(all_xy[i * 2] for i in range(n))
        min_y = min(all_xy[i * 2 + 1] for i in range(n))
        max_y = max(all_xy[i * 2 + 1] for i in range(n))
        min_z = min(all_z)
        max_z = max(all_z)
        results.append({
            "geometry": all_xy,
            "geometry_z": all_z,
            "ring_lengths": ring_lengths,
            "type": POLYGON3D,
            "minX": min_x, "minY": min_y, "minZ": min_z,
            "maxX": max_x, "maxY": max_y, "maxZ": max_z,
        })
    return results


def _convert_surface(
    geom: Any, proj: CartesianProjector3D, geom_type: int,
) -> list[dict]:
    """Convert TIN or PolyhedralSurface — each face is a polygon ring."""
    all_xy: list[float] = []
    all_z: list[float] = []
    ring_lengths: list[int] = []

    for face in geom.coordinates:
        # Each face is [ring, ...]; typically one outer ring
        for ring in face:
            xy, z = _flatten_positions(ring)
            xy, z = _project_flat(xy, z, proj)
            ring_lengths.append(len(z))
            all_xy.extend(xy)
            all_z.extend(z)

    n = len(all_z)
    if n == 0:
        return []

    min_x = min(all_xy[i * 2] for i in range(n))
    max_x = max(all_xy[i * 2] for i in range(n))
    min_y = min(all_xy[i * 2 + 1] for i in range(n))
    max_y = max(all_xy[i * 2 + 1] for i in range(n))
    min_z = min(all_z)
    max_z = max(all_z)

    return [{
        "geometry": all_xy,
        "geometry_z": all_z,
        "ring_lengths": ring_lengths,
        "type": geom_type,
        "minX": min_x, "minY": min_y, "minZ": min_z,
        "maxX": max_x, "maxY": max_y, "maxZ": max_z,
    }]


def convert_feature_3d(
    feature: MuDMFeature,
    proj: CartesianProjector3D,
) -> list[dict]:
    """Convert a single MuDMFeature to intermediate 3D tile features.

    Returns a list (possibly >1 for Multi* types) of intermediate dicts.
    """
    geom = feature.geometry
    if geom is None:
        return []

    tags = dict(feature.properties) if feature.properties else {}

    gt = _geom_type_for(geom)

    if gt == POINT3D:
        items = _convert_point(geom, proj)
    elif gt == LINESTRING3D:
        items = _convert_line(geom, proj)
    elif gt == POLYGON3D:
        items = _convert_polygon(geom, proj)
    elif gt in (TIN_TYPE, POLYHEDRALSURFACE):
        items = _convert_surface(geom, proj, gt)
    else:
        return []

    for item in items:
        item["tags"] = tags

    return items


def convert_collection_3d(
    collection: MuDMFeatureCollection,
    proj: CartesianProjector3D,
) -> list[dict]:
    """Convert an entire MuDMFeatureCollection to intermediate format."""
    results: list[dict] = []
    for feat in collection.features:
        results.extend(convert_feature_3d(feat, proj))
    return results


def compute_bounds_3d(
    collection: MuDMFeatureCollection,
) -> tuple[float, float, float, float, float, float]:
    """Compute overall 3D bounding box for a feature collection.

    Returns (xmin, ymin, zmin, xmax, ymax, zmax).
    """
    xs_min: list[float] = []
    xs_max: list[float] = []
    ys_min: list[float] = []
    ys_max: list[float] = []
    zs_min: list[float] = []
    zs_max: list[float] = []

    for feat in collection.features:
        if feat.geometry is None:
            continue
        b = geometry_bounds(feat.geometry)
        if b is None:
            continue
        xs_min.append(b[0])
        ys_min.append(b[1])
        zs_min.append(b[2])
        xs_max.append(b[3])
        ys_max.append(b[4])
        zs_max.append(b[5])

    if not xs_min:
        return (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)

    return (
        min(xs_min), min(ys_min), min(zs_min),
        max(xs_max), max(ys_max), max(zs_max),
    )
