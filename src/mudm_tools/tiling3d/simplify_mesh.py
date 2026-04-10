"""Vertex-clustering mesh decimation for LOD tile generation.

Reduces triangle count by snapping vertices to a uniform 3D grid,
merging coincident vertices, and removing degenerate faces. This is
fast, dependency-free, and well-suited for spatial LOD where distant
geometry needs progressively less detail.

The grid is aligned to the world origin so that shared boundary
vertices in adjacent tiles snap to the same cell — preventing cracks.
"""

from __future__ import annotations

import numpy as np


# Base grid resolution at zoom 0. Surface cells ≈ 6 * N^2, so:
#   z0: 10 cells → ~600 surface cells → ~1.7% of 35K faces
#   z1: 20 cells → ~2400 → ~6.9%
#   z2: 40 cells → ~9600 → ~27%
_BASE_CELLS = 10


def decimate_tin(
    coordinates: list,
    target_ratio: float,
    world_bounds: tuple[float, float, float, float, float, float] | None = None,
    zoom: int = 0,
) -> list:
    """Decimate TIN coordinates using vertex clustering.

    Parameters
    ----------
    coordinates : list
        TIN coordinates in MuDM format:
        [[[ring0_pos0, ring0_pos1, ...]], [[ring1_pos0, ...]], ...]
        Each face is [[p0, p1, p2, p0]] (closed triangle ring).
    target_ratio : float
        Target fraction of faces to keep (0.0–1.0). Values >= 1.0
        return the input unchanged.
    world_bounds : tuple, optional
        (xmin, ymin, zmin, xmax, ymax, zmax) in world coordinates.
        When provided, the grid is aligned to world origin so that
        adjacent tiles share consistent cell boundaries (no cracks).
    zoom : int
        Current zoom level. Used with world_bounds to compute a
        fixed grid resolution per zoom level.

    Returns
    -------
    list
        Simplified TIN coordinates in the same format.
    """
    if target_ratio >= 1.0 or len(coordinates) <= 4:
        return coordinates

    # --- Extract vertices and faces ---
    n_faces = len(coordinates)

    # Collect all triangle vertices (3 per face, skip closing vertex)
    all_verts = []
    for face in coordinates:
        ring = face[0]  # TIN: single ring per face
        for v in ring[:3]:  # first 3 vertices (skip closing duplicate)
            all_verts.append(v)

    verts = np.array(all_verts, dtype=np.float64)  # (n_faces*3, 3)
    faces = np.arange(n_faces * 3, dtype=np.int32).reshape(n_faces, 3)

    # --- Compute grid (world-aligned if bounds provided) ---
    if world_bounds is not None:
        grid_origin = np.array(world_bounds[:3], dtype=np.float64)
        world_size = np.array(world_bounds[3:], dtype=np.float64) - grid_origin
        world_size = np.maximum(world_size, 1e-10)

        # Fixed grid: cells_per_axis doubles each zoom level
        cells_per_axis = _BASE_CELLS * (1 << zoom)
        cell_size = world_size / cells_per_axis
    else:
        # Fallback: mesh-local grid (no cross-tile consistency)
        grid_origin = verts.min(axis=0)
        bbox_size = verts.max(axis=0) - grid_origin
        bbox_size = np.maximum(bbox_size, 1e-10)
        current_res = max(np.cbrt(len(verts)), 2.0)
        target_res = max(current_res * np.sqrt(target_ratio), 2.0)
        cell_size = bbox_size / target_res

    # --- Quantize vertices to grid cells ---
    quantized = np.floor((verts - grid_origin) / cell_size).astype(np.int32)

    # Map each vertex to a unique cell ID
    cell_map: dict[tuple, int] = {}
    new_verts_accum: list[list[float]] = []  # [sum_x, sum_y, sum_z, count]
    vert_remap = np.empty(len(verts), dtype=np.int32)

    for i in range(len(verts)):
        key = (quantized[i, 0], quantized[i, 1], quantized[i, 2])
        if key not in cell_map:
            cell_map[key] = len(new_verts_accum)
            new_verts_accum.append([verts[i, 0], verts[i, 1], verts[i, 2], 1.0])
        else:
            idx = cell_map[key]
            new_verts_accum[idx][0] += verts[i, 0]
            new_verts_accum[idx][1] += verts[i, 1]
            new_verts_accum[idx][2] += verts[i, 2]
            new_verts_accum[idx][3] += 1.0
        vert_remap[i] = cell_map[key]

    # Compute cell centroids
    new_verts = np.array(new_verts_accum, dtype=np.float64)
    new_verts[:, :3] /= new_verts[:, 3:4]
    new_verts = new_verts[:, :3]

    # --- Remap faces and remove degenerates ---
    new_faces = vert_remap[faces]  # (n_faces, 3)

    # A face is degenerate if any two vertex indices are the same
    non_degenerate = (
        (new_faces[:, 0] != new_faces[:, 1])
        & (new_faces[:, 1] != new_faces[:, 2])
        & (new_faces[:, 0] != new_faces[:, 2])
    )
    new_faces = new_faces[non_degenerate]

    if len(new_faces) == 0:
        # Degenerated completely — return a minimal subset
        return coordinates[:4] if len(coordinates) >= 4 else coordinates

    # --- Reconstruct TIN coordinates ---
    result = []
    for f in new_faces:
        v0 = new_verts[f[0]].tolist()
        v1 = new_verts[f[1]].tolist()
        v2 = new_verts[f[2]].tolist()
        result.append([[v0, v1, v2, v0]])  # closed ring

    return result
