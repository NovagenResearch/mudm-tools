"""3D axis-parallel clipping for octree tiling.

Clips features against an axis-aligned plane at position k,
for axis 0 (X), 1 (Y), or 2 (Z).  Follows the same pattern as
the 2D ``mudm2vt/clip.py`` but extended to three axes.
"""

from __future__ import annotations

from .convert3d import POINT3D, LINESTRING3D, POLYGON3D, TIN_TYPE, POLYHEDRALSURFACE


def clip_3d(
    features: list[dict],
    k1: float,
    k2: float,
    axis: int,
) -> list[dict]:
    """Clip features to the range [k1, k2] along the given axis.

    Parameters
    ----------
    features : list[dict]
        Intermediate features with geometry/geometry_z/type/min*/max* keys.
    k1, k2 : float
        Clip bounds in normalized [0, 1] space.
    axis : int
        0 = X, 1 = Y, 2 = Z.

    Returns
    -------
    Clipped features list (features fully outside are discarded).
    """
    clipped: list[dict] = []

    for feat in features:
        ft = feat["type"]

        # Get min/max along the clip axis
        if axis == 0:
            a_min, a_max = feat["minX"], feat["maxX"]
        elif axis == 1:
            a_min, a_max = feat["minY"], feat["maxY"]
        else:
            a_min, a_max = feat["minZ"], feat["maxZ"]

        # Trivial reject — half-open interval [k1, k2)
        if a_min >= k2 or a_max < k1:
            continue

        # Trivial accept — fully within [k1, k2)
        if a_min >= k1 and a_max < k2:
            clipped.append(feat)
            continue

        # Surface types: clip per-face (keep only faces that overlap)
        if ft in (TIN_TYPE, POLYHEDRALSURFACE):
            result = _clip_surface(feat, k1, k2, axis)
            if result is not None:
                clipped.append(result)
            continue

        if ft == POINT3D:
            clipped.extend(_clip_points(feat, k1, k2, axis))
        elif ft == LINESTRING3D:
            clipped.extend(_clip_line(feat, k1, k2, axis))
        elif ft == POLYGON3D:
            result = _clip_polygon(feat, k1, k2, axis)
            if result is not None:
                clipped.append(result)

    return clipped


def _get_axis_val(feat: dict, idx: int, axis: int) -> float:
    """Get the value along axis for vertex at index idx."""
    if axis == 0:
        return feat["geometry"][idx * 2]
    elif axis == 1:
        return feat["geometry"][idx * 2 + 1]
    else:
        return feat["geometry_z"][idx]


def _clip_points(feat: dict, k1: float, k2: float, axis: int) -> list[dict]:
    """Clip point features — keep points within [k1, k2]."""
    xy = feat["geometry"]
    z = feat["geometry_z"]
    n = len(z)
    results = []
    for i in range(n):
        val = _get_axis_val(feat, i, axis)
        if k1 <= val < k2:
            px, py = xy[i * 2], xy[i * 2 + 1]
            pz = z[i]
            results.append(
                {
                    "geometry": [px, py],
                    "geometry_z": [pz],
                    "type": POINT3D,
                    "tags": feat.get("tags", {}),
                    "minX": px,
                    "minY": py,
                    "minZ": pz,
                    "maxX": px,
                    "maxY": py,
                    "maxZ": pz,
                }
            )
    return results


def _intersect(
    xy: list[float],
    z: list[float],
    i: int,
    j: int,
    k: float,
    axis: int,
) -> tuple[float, float, float]:
    """Interpolate intersection point where the line (i→j) crosses axis=k."""
    ax, ay, az = xy[i * 2], xy[i * 2 + 1], z[i]
    bx, by, bz = xy[j * 2], xy[j * 2 + 1], z[j]

    if axis == 0:
        d = bx - ax
        t = (k - ax) / d if d != 0.0 else 0.0
    elif axis == 1:
        d = by - ay
        t = (k - ay) / d if d != 0.0 else 0.0
    else:
        d = bz - az
        t = (k - az) / d if d != 0.0 else 0.0

    return (
        ax + (bx - ax) * t,
        ay + (by - ay) * t,
        az + (bz - az) * t,
    )


def _clip_line(feat: dict, k1: float, k2: float, axis: int) -> list[dict]:
    """Clip a LineString3D to [k1, k2) along axis.

    Returns zero or more clipped line segments.  Fixes two issues from the
    original single-result implementation:

    1. Intersection ordering — when a segment crosses both k1 and k2,
       intersections are now emitted in parameter-order along the segment.
    2. Exit/re-enter splitting — a line that leaves and re-enters the clip
       range produces separate line features instead of one spurious polyline.
    """
    xy = feat["geometry"]
    z = feat["geometry_z"]
    n = len(z)
    if n < 2:
        return []

    segments: list[dict] = []
    out_xy: list[float] = []
    out_z: list[float] = []

    def _flush() -> None:
        nonlocal out_xy, out_z
        if len(out_z) >= 2:
            nn = len(out_z)
            segments.append(
                {
                    "geometry": out_xy,
                    "geometry_z": out_z,
                    "type": LINESTRING3D,
                    "tags": feat.get("tags", {}),
                    "ring_lengths": feat.get("ring_lengths"),
                    "minX": min(out_xy[j * 2] for j in range(nn)),
                    "minY": min(out_xy[j * 2 + 1] for j in range(nn)),
                    "minZ": min(out_z),
                    "maxX": max(out_xy[j * 2] for j in range(nn)),
                    "maxY": max(out_xy[j * 2 + 1] for j in range(nn)),
                    "maxZ": max(out_z),
                }
            )
        out_xy = []
        out_z = []

    for i in range(n - 1):
        a_val = _get_axis_val(feat, i, axis)
        b_val = _get_axis_val(feat, i + 1, axis)
        a_in = k1 <= a_val < k2

        if a_in:
            out_xy.extend([xy[i * 2], xy[i * 2 + 1]])
            out_z.append(z[i])

        # Detect boundary crossings
        cross_k1 = (a_val < k1 and b_val > k1) or (b_val < k1 and a_val > k1)
        # Use >= for k2 so vertices exactly on the boundary (from prior-zoom
        # clipping) are detected as exit/entry crossings rather than silently
        # dropped by the half-open interval.
        cross_k2 = (a_val < k2 and b_val >= k2) or (b_val < k2 and a_val >= k2)

        # Collect crossings with parameter t and enter/exit flag
        crossings: list[tuple[float, float, float, float, bool]] = []
        if cross_k1:
            ix, iy, iz = _intersect(xy, z, i, i + 1, k1, axis)
            d = b_val - a_val
            t = (k1 - a_val) / d if d != 0.0 else 0.0
            # Entering = crossing k1 in the direction that goes into [k1, k2)
            entering = b_val > a_val  # upward → entering at k1
            crossings.append((t, ix, iy, iz, entering))
        if cross_k2:
            ix, iy, iz = _intersect(xy, z, i, i + 1, k2, axis)
            d = b_val - a_val
            t = (k2 - a_val) / d if d != 0.0 else 0.0
            # Entering = crossing k2 in the direction that goes into [k1, k2)
            entering = b_val < a_val  # downward → entering at k2
            crossings.append((t, ix, iy, iz, entering))

        # Emit in segment order
        crossings.sort(key=lambda c: c[0])
        for _, ix, iy, iz, entering in crossings:
            if entering:
                _flush()  # close previous segment
                out_xy.extend([ix, iy])
                out_z.append(iz)
            else:
                out_xy.extend([ix, iy])
                out_z.append(iz)
                _flush()  # close current segment at exit

    # Last point
    if n > 0:
        last_val = _get_axis_val(feat, n - 1, axis)
        if k1 <= last_val < k2:
            out_xy.extend([xy[(n - 1) * 2], xy[(n - 1) * 2 + 1]])
            out_z.append(z[n - 1])

    _flush()
    return segments


def _clip_surface(feat: dict, k1: float, k2: float, axis: int) -> dict | None:
    """Clip a TIN/PolyhedralSurface per-face — keep only faces that overlap [k1, k2).

    Each face is delineated by ring_lengths. A face is kept if any of its
    vertices fall within [k1, k2) or its bounding box along the axis
    straddles the range.
    """
    xy = feat["geometry"]
    z = feat["geometry_z"]
    ring_lengths = feat.get("ring_lengths") or [len(z)]

    out_xy: list[float] = []
    out_z: list[float] = []
    out_ring_lengths: list[int] = []

    offset = 0
    for rl in ring_lengths:
        # Compute face min/max along clip axis
        if axis == 0:
            face_vals = [xy[(offset + j) * 2] for j in range(rl)]
        elif axis == 1:
            face_vals = [xy[(offset + j) * 2 + 1] for j in range(rl)]
        else:
            face_vals = [z[offset + j] for j in range(rl)]

        f_min = min(face_vals)
        f_max = max(face_vals)

        # Half-open interval [k1, k2): reject if fully outside
        if f_min >= k2 or f_max < k1:
            offset += rl
            continue

        # Face overlaps — keep it whole
        out_xy.extend(xy[offset * 2 : (offset + rl) * 2])
        out_z.extend(z[offset : offset + rl])
        out_ring_lengths.append(rl)
        offset += rl

    if not out_ring_lengths:
        return None

    n = len(out_z)
    min_x = min(out_xy[j * 2] for j in range(n))
    max_x = max(out_xy[j * 2] for j in range(n))
    min_y = min(out_xy[j * 2 + 1] for j in range(n))
    max_y = max(out_xy[j * 2 + 1] for j in range(n))
    min_z = min(out_z)
    max_z = max(out_z)

    return {
        "geometry": out_xy,
        "geometry_z": out_z,
        "ring_lengths": out_ring_lengths,
        "type": feat["type"],
        "tags": feat.get("tags", {}),
        "minX": min_x,
        "minY": min_y,
        "minZ": min_z,
        "maxX": max_x,
        "maxY": max_y,
        "maxZ": max_z,
    }


def _clip_polygon(feat: dict, k1: float, k2: float, axis: int) -> dict | None:
    """Clip a Polygon3D — Sutherland-Hodgman style per ring.

    For simplicity, uses the same include-whole-face approach as surfaces
    when the polygon straddles the boundary. This avoids complex polygon
    clipping while still producing correct octree splits.
    """
    # Include the whole polygon if any part overlaps
    return feat


# ---------------------------------------------------------------------------
# Rust dispatch: import compiled versions from the Rust extension module.
# Falls back to pure Python if the Rust extension is not available.
# ---------------------------------------------------------------------------
_clip_surface_py = _clip_surface
_clip_line_py = _clip_line

try:
    from mudm_tools._rs import clip_surface as _clip_surface_rs  # noqa: F401
    from mudm_tools._rs import clip_line as _clip_line_rs  # noqa: F401

    def _clip_surface(feat: dict, k1: float, k2: float, axis: int) -> dict | None:  # noqa: F811
        return _clip_surface_rs(feat, k1, k2, axis)

    def _clip_line(feat: dict, k1: float, k2: float, axis: int) -> list[dict]:  # noqa: F811
        return _clip_line_rs(feat, k1, k2, axis)

except ImportError:
    pass
