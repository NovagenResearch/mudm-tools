# Original code from geojson2vt
# Copyright (c) 2015, Mapbox
# ISC License terms apply; see LICENSE file for details.

# Modifications by PolusAI, 2024

"""2D Douglas-Peucker polygon simplification.

This is the Python counterpart of ``rust/src/simplify2d.rs`` and follows the
same semantics so the pure-Python tiler and the Rust streaming tiler produce
the same simplified geometry (the Rust path is only faster). The key property
is *reduce-to-floor*: a ring is reduced toward a minimum vertex count rather
than reverting to full detail when the tolerance would drop it below the floor.
"""

import math


def _dp_importance(pts, imp):
    """Assign each vertex its Douglas-Peucker "importance": the perpendicular
    deviation at which it becomes a split point. Endpoints are left at whatever
    the caller pre-set (infinity). Mirrors ``dp_importance`` in simplify2d.rs;
    an explicit stack replaces recursion so deep rings cannot overflow, and the
    per-vertex value is independent of traversal order so the result is
    identical to the recursive Rust version.

    ``pts`` is a sequence of (x, y) pairs; ``imp`` is a list of length len(pts).
    """
    n = len(pts)
    stack = [(0, n - 1)]
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        ax = pts[start][0]
        ay = pts[start][1]
        bx = pts[end][0]
        by = pts[end][1]
        dx = bx - ax
        dy = by - ay
        len_sq = dx * dx + dy * dy

        max_dist = 0.0
        max_idx = start + 1
        if len_sq < 1e-30:
            # Start and end coincident (e.g. a closed ring's shared vertex) —
            # use distance to the start point.
            for i in range(start + 1, end):
                ex = pts[i][0] - ax
                ey = pts[i][1] - ay
                dist = math.sqrt(ex * ex + ey * ey)
                if dist > max_dist:
                    max_dist = dist
                    max_idx = i
        else:
            seg_len = math.sqrt(len_sq)
            for i in range(start + 1, end):
                px = pts[i][0]
                py = pts[i][1]
                # Perpendicular distance from the point to the segment line.
                dist = abs((py - ay) * dx - (px - ax) * dy) / seg_len
                if dist > max_dist:
                    max_dist = dist
                    max_idx = i

        imp[max_idx] = max_dist
        stack.append((start, max_idx))
        stack.append((max_idx, end))


def douglas_peucker_floor(pts, epsilon, min_verts):
    """Douglas-Peucker simplification with a MINIMUM-vertex floor.

    Keeps every vertex whose deviation exceeds ``epsilon``, but never reduces
    below ``min_verts`` vertices: if the epsilon threshold would drop below the
    floor, the most-important remaining vertices are added back until
    ``min_verts`` are kept. Unlike a plain DP + revert-to-original guard, this
    reduces a ring *to* the floor rather than snapping back to full detail.
    Mirrors ``douglas_peucker_floor`` in simplify2d.rs.

    ``pts`` is a sequence of (x, y) pairs; returns a list of [x, y] pairs.
    """
    n = len(pts)
    floor = max(min_verts, 2)
    if n <= floor:
        return [[p[0], p[1]] for p in pts]

    imp = [0.0] * n
    imp[0] = math.inf
    imp[n - 1] = math.inf
    _dp_importance(pts, imp)

    keep = [d > epsilon for d in imp]
    kept = sum(1 for k in keep if k)
    if kept < floor:
        # Add back the most-important not-yet-kept vertices to reach the floor.
        # sort() is stable and preserves ascending index order among ties,
        # matching the Rust stable sort.
        order = [i for i in range(n) if not keep[i]]
        order.sort(key=lambda i: imp[i], reverse=True)
        for i in order:
            if kept >= floor:
                break
            keep[i] = True
            kept += 1

    return [[pts[i][0], pts[i][1]] for i in range(n) if keep[i]]


def simplify_polygon_ring(ring, epsilon, min_verts):
    """Simplify a single polygon ring using reduce-to-floor Douglas-Peucker.

    A valid closed ring needs >= 4 vertices (3 unique + closing), so the floor
    is ``max(min_verts, 4)``. Rings already at/below the floor are kept as-is.
    ``ring`` is a sequence of (x, y) pairs; returns a list of [x, y] pairs.
    Mirrors the per-ring logic of ``simplify_polygon_rings`` in simplify2d.rs.
    """
    floor = max(min_verts, 4)
    if len(ring) <= floor:
        return [[p[0], p[1]] for p in ring]
    return douglas_peucker_floor(ring, epsilon, floor)


def simplify_polygon_rings(rings, epsilon, min_verts):
    """Simplify a list of polygon rings; floor = max(min_verts, 4) per ring.

    Mirrors ``simplify_polygon_rings`` in simplify2d.rs. ``rings`` is a list of
    rings, each a sequence of (x, y) pairs.
    """
    return [simplify_polygon_ring(r, epsilon, min_verts) for r in rings]


def compute_tolerance(zoom, max_zoom):
    """Douglas-Peucker tolerance for a zoom level, in normalized [0, 1] space.

    At ``max_zoom`` the tolerance is 0 (no simplification). At coarser levels
    it doubles per zoom level out. Mirrors ``compute_tolerance`` in
    simplify2d.rs.
    """
    if zoom >= max_zoom:
        return 0.0
    zoom_diff = max_zoom - zoom
    base = 1.0 / (1 << max_zoom)
    return base * (1 << zoom_diff)


def simplify(coords, sq_tolerance, min_vertices=3):
    """Backwards-compatible wrapper for the historical Ramer-Douglas-Peucker
    entry point. The old implementation took a SQUARED tolerance and reverted
    to the full ring when it could not keep ``min_vertices``. It now delegates
    to the reduce-to-floor importance-DP (parity with the Rust tiler):
    ``epsilon = sqrt(sq_tolerance)``, floor = ``min_vertices``, and it never
    reverts to full detail.

    ``coords`` is a sequence of (x, y) pairs; returns a list of [x, y] pairs.
    """
    if sq_tolerance <= 0:
        return [[c[0], c[1]] for c in coords]
    epsilon = math.sqrt(sq_tolerance)
    return douglas_peucker_floor(coords, epsilon, min_vertices)
