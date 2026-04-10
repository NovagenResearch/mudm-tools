"""Ramer-Douglas-Peucker simplification in 3D.

Operates on flat coordinate arrays: geometry=[x0,y0,x1,y1,...],
geometry_z=[z0,z1,...] (parallel arrays).
"""

from __future__ import annotations

import math


def _sq_seg_dist_3d(
    px: float, py: float, pz: float,
    ax: float, ay: float, az: float,
    bx: float, by: float, bz: float,
) -> float:
    """Squared distance from point P to line segment AB in 3D."""
    dx = bx - ax
    dy = by - ay
    dz = bz - az

    if dx != 0.0 or dy != 0.0 or dz != 0.0:
        t = ((px - ax) * dx + (py - ay) * dy + (pz - az) * dz) / (
            dx * dx + dy * dy + dz * dz
        )
        if t > 1.0:
            ax, ay, az = bx, by, bz
        elif t > 0.0:
            ax += dx * t
            ay += dy * t
            az += dz * t

    dx = px - ax
    dy = py - ay
    dz = pz - az
    return dx * dx + dy * dy + dz * dz


def simplify_3d(
    coords_xy: list[float],
    coords_z: list[float],
    sq_tolerance: float,
    min_vertices: int = 3,
) -> tuple[list[float], list[float]]:
    """Simplify a 3D polyline using Ramer-Douglas-Peucker.

    Parameters
    ----------
    coords_xy : list[float]
        Flat array [x0, y0, x1, y1, ...].
    coords_z : list[float]
        Parallel Z array [z0, z1, ...].
    sq_tolerance : float
        Squared distance tolerance.
    min_vertices : int
        Minimum number of vertices to keep.

    Returns
    -------
    (new_xy, new_z) : tuple of simplified coordinate arrays.
    """
    n = len(coords_z)
    if n <= 2:
        return coords_xy[:], coords_z[:]

    # Mark which vertices to keep
    keep = [False] * n
    keep[0] = True
    keep[n - 1] = True

    _rdp_3d(coords_xy, coords_z, sq_tolerance, 0, n - 1, keep)

    kept_count = sum(keep)
    tol = sq_tolerance
    max_iter = 50
    while kept_count < min_vertices and max_iter > 0:
        tol *= 0.5
        keep = [False] * n
        keep[0] = True
        keep[n - 1] = True
        _rdp_3d(coords_xy, coords_z, tol, 0, n - 1, keep)
        kept_count = sum(keep)
        max_iter -= 1

    # Build output
    out_xy: list[float] = []
    out_z: list[float] = []
    for i in range(n):
        if keep[i]:
            out_xy.append(coords_xy[i * 2])
            out_xy.append(coords_xy[i * 2 + 1])
            out_z.append(coords_z[i])
    return out_xy, out_z


def _rdp_3d(
    xy: list[float],
    z: list[float],
    sq_tol: float,
    first: int,
    last: int,
    keep: list[bool],
) -> None:
    """Recursive RDP in 3D."""
    max_sq_dist = sq_tol
    index = 0

    ax = xy[first * 2]
    ay = xy[first * 2 + 1]
    az = z[first]
    bx = xy[last * 2]
    by = xy[last * 2 + 1]
    bz = z[last]

    for i in range(first + 1, last):
        sq_dist = _sq_seg_dist_3d(
            xy[i * 2], xy[i * 2 + 1], z[i],
            ax, ay, az,
            bx, by, bz,
        )
        if sq_dist > max_sq_dist:
            index = i
            max_sq_dist = sq_dist

    if index != 0:
        keep[index] = True
        if first + 1 < index:
            _rdp_3d(xy, z, sq_tol, first, index, keep)
        if index + 1 < last:
            _rdp_3d(xy, z, sq_tol, index, last, keep)
