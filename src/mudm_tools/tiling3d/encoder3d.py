"""3D protobuf encoder for vector tiles.

Encodes intermediate tile data into the mudm_3d_tile protobuf
format with delta-encoded XY, zigzag Z, and dictionary-encoded tags.
"""

from __future__ import annotations

import struct
from typing import Any, Union

from .proto import mudm_3d_tile_pb2 as pb


# GeomType mapping
_GEOM_TYPE_MAP = {
    1: pb.Tile.POINT3D,
    2: pb.Tile.LINESTRING3D,
    3: pb.Tile.POLYGON3D,
    4: pb.Tile.POLYHEDRALSURFACE,
    5: pb.Tile.TIN,
}


def _zigzag(n: int) -> int:
    """Zigzag encode a signed integer."""
    return (n << 1) ^ (n >> 31)


def _command(cmd: int, count: int) -> int:
    """Encode an MVT-style geometry command."""
    return (count << 3) | (cmd & 0x7)


def _encode_point_geometry(xy: list[int]) -> list[int]:
    """Encode point geometry (one or more points)."""
    n = len(xy) // 2
    result = [_command(1, n)]  # MoveTo with count
    x = 0
    y = 0
    for i in range(n):
        dx = xy[i * 2] - x
        dy = xy[i * 2 + 1] - y
        result.append(_zigzag(dx))
        result.append(_zigzag(dy))
        x += dx
        y += dy
    return result


def _encode_line_geometry(xy: list[int]) -> list[int]:
    """Encode linestring geometry."""
    n = len(xy) // 2
    if n < 2:
        return []

    result = []
    x = 0
    y = 0

    # MoveTo first point
    result.append(_command(1, 1))
    dx = xy[0] - x
    dy = xy[1] - y
    result.append(_zigzag(dx))
    result.append(_zigzag(dy))
    x += dx
    y += dy

    # LineTo remaining
    result.append(_command(2, n - 1))
    for i in range(1, n):
        dx = xy[i * 2] - x
        dy = xy[i * 2 + 1] - y
        result.append(_zigzag(dx))
        result.append(_zigzag(dy))
        x += dx
        y += dy

    return result


def _encode_polygon_geometry(
    xy: list[int], ring_lengths: list[int] | None,
) -> list[int]:
    """Encode polygon geometry with rings."""
    result = []
    x = 0
    y = 0

    if ring_lengths is None:
        ring_lengths = [len(xy) // 2]

    offset = 0
    for ring_len in ring_lengths:
        if ring_len < 3:
            offset += ring_len
            continue

        # MoveTo first point
        result.append(_command(1, 1))
        dx = xy[offset * 2] - x
        dy = xy[offset * 2 + 1] - y
        result.append(_zigzag(dx))
        result.append(_zigzag(dy))
        x += dx
        y += dy

        # LineTo (ring_len - 2 points, excluding first and last which is same as first)
        line_count = ring_len - 2
        if line_count > 0:
            result.append(_command(2, line_count))
            for i in range(1, ring_len - 1):
                idx = offset + i
                dx = xy[idx * 2] - x
                dy = xy[idx * 2 + 1] - y
                result.append(_zigzag(dx))
                result.append(_zigzag(dy))
                x += dx
                y += dy

        # ClosePath
        result.append(_command(7, 1))
        offset += ring_len

    return result


def _encode_z(z_coords: list[int]) -> list[int]:
    """Delta-encode Z coordinates (already zigzag via sint32 in proto)."""
    if not z_coords:
        return []
    result = [z_coords[0]]
    for i in range(1, len(z_coords)):
        result.append(z_coords[i] - z_coords[i - 1])
    return result


def _build_indexed_mesh(
    xy: list[int],
    z_coords: list[int],
    ring_lengths: list[int] | None,
) -> tuple[bytes, bytes]:
    """Build indexed triangle mesh from ring-based face data.

    Deduplicates shared vertices and returns raw bytes for positions
    and indices (float32 LE / uint32 LE).

    Parameters
    ----------
    xy : list[int]
        Flat XY integer coords [x0,y0, x1,y1, ...].
    z_coords : list[int]
        Z integer coords, one per vertex.
    ring_lengths : list[int] | None
        Number of vertices per closed ring (triangle = 4: 3 + closing).

    Returns
    -------
    (positions_bytes, indices_bytes) where positions_bytes is raw
    float32 LE [x,y,z, ...] and indices_bytes is raw uint32 LE [i0,i1,i2, ...].
    """
    if ring_lengths is None:
        ring_lengths = [len(xy) // 2]

    vertex_map: dict[tuple[int, int, int], int] = {}
    positions: list[float] = []
    indices: list[int] = []

    offset = 0
    for ring_len in ring_lengths:
        # Each closed ring is a triangle: 4 verts (last == first), take first 3
        n_verts = min(ring_len - 1, 3) if ring_len >= 4 else ring_len
        tri_indices: list[int] = []
        for i in range(n_verts):
            vi = offset + i
            x = xy[vi * 2]
            y = xy[vi * 2 + 1]
            z = z_coords[vi] if vi < len(z_coords) else 0
            key = (x, y, z)
            if key not in vertex_map:
                vertex_map[key] = len(vertex_map)
                positions.extend([float(x), float(y), float(z)])
            tri_indices.append(vertex_map[key])
        if len(tri_indices) == 3:
            indices.extend(tri_indices)
        offset += ring_len

    pos_bytes = struct.pack(f"<{len(positions)}f", *positions) if positions else b""
    idx_bytes = struct.pack(f"<{len(indices)}I", *indices) if indices else b""
    return pos_bytes, idx_bytes


def _write_value(pb_value: Any, value: Union[bool, str, int, float]) -> None:
    """Write a tag value to a protobuf Value message."""
    if isinstance(value, bool):
        pb_value.bool_value = value
    elif isinstance(value, str):
        pb_value.string_value = value
    elif isinstance(value, float):
        pb_value.double_value = value
    elif isinstance(value, int):
        if value < 0:
            pb_value.sint_value = value
        else:
            pb_value.uint_value = value


def encode_tile_3d(
    tile_data: dict,
    layer_name: str = "default",
    extent: int = 4096,
    extent_z: int = 4096,
) -> bytes:
    """Encode a transformed tile to protobuf bytes.

    Parameters
    ----------
    tile_data : dict
        Tile with integer-coordinate features (from transform_tile_3d).
    layer_name : str
        Layer name for the tile.
    extent : int
        XY extent.
    extent_z : int
        Z extent.

    Returns
    -------
    Serialized protobuf bytes.
    """
    tile_pb = pb.Tile()
    layer = tile_pb.layers.add()
    layer.name = layer_name
    layer.version = 3
    layer.extent = extent
    layer.extent_z = extent_z

    key_indices: dict[str, int] = {}
    value_indices: dict[Any, int] = {}

    for feat_idx, feat in enumerate(tile_data.get("features", [])):
        pb_feat = layer.features.add()
        pb_feat.id = feat_idx

        gt = feat.get("type", 0)
        pb_feat.type = _GEOM_TYPE_MAP.get(gt, pb.Tile.UNKNOWN)

        # Encode tags
        tags = feat.get("tags", {})
        for k, v in tags.items():
            if v is None:
                continue
            if k not in key_indices:
                key_indices[k] = len(key_indices)
                layer.keys.append(k)
            pb_feat.tags.append(key_indices[k])

            if v not in value_indices:
                value_indices[v] = len(value_indices)
                _write_value(layer.values.add(), v)
            pb_feat.tags.append(value_indices[v])

        # Encode geometry
        xy = feat.get("geometry", [])
        z_coords = feat.get("geometry_z", [])
        ring_lengths = feat.get("ring_lengths")

        if gt == 1:  # POINT3D
            pb_feat.geometry.extend(_encode_point_geometry(xy))
            pb_feat.geometry_z.extend(_encode_z(z_coords))
        elif gt == 2:  # LINESTRING3D
            pb_feat.geometry.extend(_encode_line_geometry(xy))
            pb_feat.geometry_z.extend(_encode_z(z_coords))
        elif gt in (4, 5):  # POLYHEDRALSURFACE, TIN — indexed mesh
            pos_bytes, idx_bytes = _build_indexed_mesh(xy, z_coords, ring_lengths)
            pb_feat.mesh_positions = pos_bytes
            pb_feat.mesh_indices = idx_bytes
        elif gt == 3:  # POLYGON3D — ring-based
            pb_feat.geometry.extend(_encode_polygon_geometry(xy, ring_lengths))
            pb_feat.geometry_z.extend(_encode_z(z_coords))
        else:
            pb_feat.geometry.extend(_encode_line_geometry(xy))
            pb_feat.geometry_z.extend(_encode_z(z_coords))

        # Per-vertex radii
        radii = feat.get("radii")
        if radii:
            pb_feat.radii.extend(radii)

    return tile_pb.SerializeToString()


# ---------------------------------------------------------------------------
# Rust dispatch: import compiled versions from the Rust extension module.
# Note: The Rust encoder uses hand-rolled protobuf (no protobuf lib dep),
# so output is wire-compatible but may have different field ordering.
# For now, keep using the Python protobuf encoder as the default to
# ensure bit-identical output with existing tests. The Rust encoder
# will be used for the streaming pipeline (Phase 6).
# ---------------------------------------------------------------------------
_build_indexed_mesh_py = _build_indexed_mesh
encode_tile_3d_py = encode_tile_3d

try:
    from mudm_tools._rs import build_indexed_mesh as _build_indexed_mesh  # noqa: F811
    from mudm_tools._rs import encode_tile_3d as _encode_tile_3d_rs  # noqa: F401
except ImportError:
    pass
