"""Generic mesh generation utilities (tubes, icospheres, splines).

Neuron-specific functions live in ``mudm.swc``.
"""

from __future__ import annotations

import math

import numpy as np

def _catmull_rom(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    n: int,
) -> list[np.ndarray]:
    """Evaluate Catmull-Rom spline between p1 and p2.

    Returns *n* evenly spaced points **excluding** p1 and **including** p2.
    """
    pts: list[np.ndarray] = []
    for i in range(1, n + 1):
        t = i / n
        t2 = t * t
        t3 = t2 * t
        v = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )
        pts.append(v)
    return pts


def smooth_path(
    points: list[np.ndarray],
    radii: list[float],
    subdivisions: int,
) -> tuple[list[np.ndarray], list[float]]:
    """Smooth a polyline path using Catmull-Rom spline interpolation.

    Inserts *subdivisions* intermediate samples between each pair of
    original points.  Radii are linearly interpolated.  The output always
    passes through the original control points.

    Args:
        points: Original 3D positions (length >= 2).
        radii: Radius at each original position.
        subdivisions: Number of points to insert per segment (0 = no-op).

    Returns:
        ``(smooth_points, smooth_radii)``
    """
    if subdivisions <= 0 or len(points) < 2:
        return points, radii

    n = len(points)
    out_pts: list[np.ndarray] = [points[0]]
    out_r: list[float] = [radii[0]]

    for i in range(n - 1):
        # Catmull-Rom needs 4 control points; clamp at boundaries
        p0 = points[max(i - 1, 0)]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[min(i + 2, n - 1)]

        interp = _catmull_rom(p0, p1, p2, p3, subdivisions + 1)
        for j, pt in enumerate(interp):
            t = (j + 1) / (subdivisions + 1)
            r = radii[i] * (1 - t) + radii[i + 1] * t
            out_pts.append(pt)
            out_r.append(r)

    return out_pts, out_r


def thin_path(
    points: list[np.ndarray],
    radii: list[float],
    keep_fraction: float,
) -> tuple[list[np.ndarray], list[float]]:
    """Uniformly subsample a path to reduce vertex count.

    Intended to run *after* :func:`smooth_path` so the dense spline is
    thinned while the remaining points still lie on the smooth curve.

    Args:
        points: Path positions (length >= 2).
        radii: Radius at each position.
        keep_fraction: Fraction of points to retain (0.0-1.0).
            ``1.0`` keeps all points.

    Returns:
        ``(thinned_points, thinned_radii)``
    """
    n = len(points)
    if keep_fraction >= 1.0 or n <= 2:
        return points, radii

    target = max(2, round(n * keep_fraction))
    if target >= n:
        return points, radii

    indices = np.round(np.linspace(0, n - 1, target)).astype(int)
    return [points[i] for i in indices], [radii[i] for i in indices]


def _find_perp(tangent: np.ndarray) -> np.ndarray:
    """Find an arbitrary unit vector perpendicular to *tangent*."""
    if abs(tangent[0]) < 0.9:
        perp = np.cross(tangent, np.array([1.0, 0.0, 0.0]))
    else:
        perp = np.cross(tangent, np.array([0.0, 1.0, 0.0]))
    return perp / np.linalg.norm(perp)


def _tube_along_path(
    points: list[np.ndarray],
    radii: list[float],
    segments: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a continuous tube mesh along a polyline.

    Uses parallel transport frame for smooth, consistent cross-section
    orientation.  Ring vertices are shared at intermediate nodes,
    eliminating gaps between consecutive segments.

    Args:
        points: Ordered 3D positions along the path (length >= 2).
        radii: Radius at each point.
        segments: Number of sides in the cross-section polygon.

    Returns:
        ``(vertices, normals, indices)`` -- Nx3, Nx3, Mx3 arrays.
    """
    n = len(points)
    if n < 2:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.uint32),
        )

    # Compute tangent at each point
    tangents = []
    for i in range(n):
        if i == 0:
            t = points[1] - points[0]
        elif i == n - 1:
            t = points[-1] - points[-2]
        else:
            # Average of incoming and outgoing directions
            t = points[i + 1] - points[i - 1]
        length = np.linalg.norm(t)
        if length < 1e-12:
            t = tangents[-1] if tangents else np.array([0.0, 0.0, 1.0])
        else:
            t = t / length
        tangents.append(t)

    # Build rings using parallel transport frame
    angles = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    all_verts = np.empty((n * segments, 3), dtype=np.float64)
    all_normals = np.empty((n * segments, 3), dtype=np.float64)

    perp = _find_perp(tangents[0])

    for i in range(n):
        if i > 0:
            # Parallel transport: project previous perp onto the plane
            # perpendicular to the new tangent
            perp = perp - np.dot(perp, tangents[i]) * tangents[i]
            norm = np.linalg.norm(perp)
            if norm < 1e-10:
                perp = _find_perp(tangents[i])
            else:
                perp = perp / norm

        perp2 = np.cross(tangents[i], perp)
        base = i * segments

        for j in range(segments):
            direction = perp * cos_a[j] + perp2 * sin_a[j]
            all_verts[base + j] = points[i] + direction * radii[i]
            all_normals[base + j] = direction

    # Connect consecutive rings with triangle strips
    indices = []
    for ring_i in range(n - 1):
        base0 = ring_i * segments
        base1 = (ring_i + 1) * segments
        for j in range(segments):
            j_next = (j + 1) % segments
            indices.append([base0 + j, base1 + j, base1 + j_next])
            indices.append([base0 + j, base1 + j_next, base0 + j_next])

    return all_verts, all_normals, np.array(indices, dtype=np.uint32)


def _icosphere(
    center: np.ndarray,
    radius: float,
    subdivisions: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate an icosphere mesh.

    Args:
        center: Center point (3,).
        radius: Sphere radius.
        subdivisions: Number of subdivision iterations (0=icosahedron).

    Returns:
        ``(vertices, normals, indices)`` -- Nx3, Nx3, Mx3 arrays.
    """
    if radius < 1e-12:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.uint32),
        )

    # Start with icosahedron
    t = (1.0 + math.sqrt(5.0)) / 2.0
    raw_verts = [
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ]
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]

    verts_list = []
    for v in raw_verts:
        n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        verts_list.append([v[0] / n, v[1] / n, v[2] / n])

    for _ in range(subdivisions):
        midpoint_cache: dict[tuple[int, int], int] = {}
        new_faces = []
        for tri in faces:
            mids = []
            for edge in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
                key = (min(edge), max(edge))
                if key not in midpoint_cache:
                    v0 = verts_list[edge[0]]
                    v1 = verts_list[edge[1]]
                    mid = [(v0[i] + v1[i]) / 2 for i in range(3)]
                    n = math.sqrt(mid[0] ** 2 + mid[1] ** 2 + mid[2] ** 2)
                    mid = [mid[i] / n for i in range(3)]
                    midpoint_cache[key] = len(verts_list)
                    verts_list.append(mid)
                mids.append(midpoint_cache[key])
            new_faces.append([tri[0], mids[0], mids[2]])
            new_faces.append([tri[1], mids[1], mids[0]])
            new_faces.append([tri[2], mids[2], mids[1]])
            new_faces.append([mids[0], mids[1], mids[2]])
        faces = new_faces

    verts_arr = np.array(verts_list, dtype=np.float64)
    normals_arr = verts_arr.copy()  # On unit sphere, vertex == normal
    verts_arr = verts_arr * radius + center

    return verts_arr, normals_arr, np.array(faces, dtype=np.uint32)
