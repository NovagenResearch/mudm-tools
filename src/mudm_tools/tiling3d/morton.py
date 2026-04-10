"""3D Morton code (Z-order curve) for octree tile addressing.

Provides bit-interleaving encode/decode for (x, y, d) -> single integer,
plus a unique tile_id combining zoom level with Morton code.
"""

from __future__ import annotations


def _part1by2(n: int) -> int:
    """Spread bits of a 10-bit integer to every 3rd bit position.

    E.g. 0b111 -> 0b001001001
    """
    n &= 0x000003FF  # 10 bits max
    n = (n ^ (n << 16)) & 0x030000FF
    n = (n ^ (n << 8)) & 0x0300F00F
    n = (n ^ (n << 4)) & 0x030C30C3
    n = (n ^ (n << 2)) & 0x09249249
    return n


def _compact1by2(n: int) -> int:
    """Inverse of _part1by2: extract every 3rd bit."""
    n &= 0x09249249
    n = (n ^ (n >> 2)) & 0x030C30C3
    n = (n ^ (n >> 4)) & 0x0300F00F
    n = (n ^ (n >> 8)) & 0x030000FF
    n = (n ^ (n >> 16)) & 0x000003FF
    return n


def morton_encode_3d(x: int, y: int, d: int) -> int:
    """Encode (x, y, d) into a 3D Morton code via bit interleaving.

    Each coordinate should be a non-negative integer (max 1023 for 10-bit).
    Result interleaves bits as: ...d2y2x2 d1y1x1 d0y0x0
    """
    return _part1by2(x) | (_part1by2(y) << 1) | (_part1by2(d) << 2)


def morton_decode_3d(code: int) -> tuple[int, int, int]:
    """Decode a 3D Morton code back to (x, y, d)."""
    return (
        _compact1by2(code),
        _compact1by2(code >> 1),
        _compact1by2(code >> 2),
    )


def tile_id_3d(z: int, x: int, y: int, d: int) -> int:
    """Compute a unique tile ID combining zoom level with Morton code.

    Layout: upper bits = zoom, lower 30 bits = Morton code.
    Supports zoom 0-31 and coordinates up to 1023.
    """
    return (z << 30) | morton_encode_3d(x, y, d)
