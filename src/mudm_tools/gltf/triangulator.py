"""Triangulate Shapely polygons into vertex/index arrays for glTF meshes.

Uses ``shapely.ops.triangulate`` (Delaunay) constrained to the polygon
boundary.  Handles polygons with holes by filtering triangles whose
centroids fall outside the original polygon.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import triangulate


def polygon_to_mesh(
    polygon: Polygon,
    z: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a Shapely Polygon into a triangle mesh.

    Args:
        polygon: A Shapely Polygon (may contain holes).
        z: Z coordinate to assign when polygon coords are 2D.

    Returns:
        ``(vertices, indices)`` where vertices is Nx3 float64 and
        indices is Mx3 uint32 (triangle face indices).
    """
    if polygon.is_empty:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.uint32)

    # Collect all boundary points (exterior + holes)
    triangles = triangulate(polygon, edges=False)

    # Filter: keep only triangles whose centroid is inside the polygon
    kept = []
    for tri in triangles:
        centroid = tri.centroid
        if polygon.contains(centroid):
            kept.append(tri)

    if not kept:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.uint32)

    # Build unique vertex list and index buffer
    vertex_map: dict[tuple[float, float], int] = {}
    vertices: list[list[float]] = []
    indices: list[list[int]] = []

    for tri in kept:
        coords = list(tri.exterior.coords)[:-1]  # 3 vertices, drop closing
        face = []
        for c in coords:
            key = (round(c[0], 12), round(c[1], 12))
            if key not in vertex_map:
                vertex_map[key] = len(vertices)
                vz = c[2] if len(c) > 2 else z
                vertices.append([c[0], c[1], vz])
            face.append(vertex_map[key])
        indices.append(face)

    return (
        np.array(vertices, dtype=np.float64),
        np.array(indices, dtype=np.uint32),
    )


def multipolygon_to_mesh(
    multipoly: MultiPolygon,
    z: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a MultiPolygon into a single combined mesh.

    Args:
        multipoly: A Shapely MultiPolygon.
        z: Default Z coordinate for 2D polygons.

    Returns:
        ``(vertices, indices)`` — combined mesh from all sub-polygons.
    """
    all_verts: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []
    offset = 0

    for poly in multipoly.geoms:
        verts, idxs = polygon_to_mesh(poly, z=z)
        if verts.shape[0] == 0:
            continue
        all_verts.append(verts)
        all_indices.append(idxs + offset)
        offset += verts.shape[0]

    if not all_verts:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.uint32)

    return (
        np.concatenate(all_verts, axis=0),
        np.concatenate(all_indices, axis=0),
    )
