"""Neuroglancer precomputed legacy mesh writer.

Converts triangle mesh data to the Neuroglancer precomputed legacy mesh
binary format and writes the on-disk directory structure.

Binary layout per segment (little-endian):
    uint32  num_vertices
    float32 vertices[num_vertices * 3]   (x, y, z interleaved)
    uint32  indices[num_triangles * 3]   (triangle vertex indices)

Directory structure:
    {output_dir}/info                   — JSON info file
    {output_dir}/{segment_id}           — binary mesh data
    {output_dir}/{segment_id}:0         — JSON fragment manifest
    {output_dir}/segment_properties/info — (optional) segment properties

Reference: https://github.com/google/neuroglancer/blob/master/
    src/neuroglancer/datasource/precomputed/meshes.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ._binary import pack_float32_array, pack_uint32, pack_uint32_array
from .mesh_models import MeshInfo


def mesh_to_binary(
    vertices: np.ndarray,
    indices: np.ndarray,
) -> bytes:
    """Encode mesh data as Neuroglancer legacy mesh binary.

    Args:
        vertices: Nx3 float32 array of vertex positions.
        indices: Mx3 uint32 array of triangle face indices (or flat 1D).

    Returns:
        Raw bytes in Neuroglancer legacy mesh binary format.
    """
    verts = np.ascontiguousarray(vertices, dtype=np.float32).ravel()
    idx = np.ascontiguousarray(indices, dtype=np.uint32).ravel()
    num_vertices = len(verts) // 3

    buf = bytearray()
    buf += pack_uint32(num_vertices)
    buf += pack_float32_array(verts.tolist())
    buf += pack_uint32_array(idx.tolist())
    return bytes(buf)


def decode_mesh_binary(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Decode Neuroglancer legacy mesh binary back to arrays.

    Args:
        data: Raw bytes in Neuroglancer legacy mesh binary format.

    Returns:
        (vertices, indices) — Nx3 float32 and Mx3 uint32 arrays.
    """
    import struct

    offset = 0
    (num_vertices,) = struct.unpack_from("<I", data, offset)
    offset += 4

    n_floats = num_vertices * 3
    verts = np.frombuffer(data, dtype=np.float32, count=n_floats, offset=offset)
    offset += n_floats * 4

    remaining = len(data) - offset
    n_indices = remaining // 4
    indices = np.frombuffer(data, dtype=np.uint32, count=n_indices, offset=offset)

    return verts.reshape(-1, 3).copy(), indices.reshape(-1, 3).copy()


def write_mesh_info(
    output_dir: str | Path,
    segment_properties: Optional[str] = None,
) -> Path:
    """Write the Neuroglancer mesh info JSON file.

    Creates ``{output_dir}/info`` with the legacy mesh type.

    Args:
        output_dir: Directory to write to (created if needed).
        segment_properties: Optional relative path to segment_properties dir.

    Returns:
        Path to the output directory.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    info = MeshInfo(segment_properties=segment_properties)
    (out / "info").write_text(json.dumps(info.to_info_dict(), indent=2))
    return out


def write_mesh(
    output_dir: str | Path,
    segment_id: int,
    vertices: np.ndarray,
    indices: np.ndarray,
) -> Path:
    """Write a single mesh segment to the precomputed directory.

    Creates:
        {output_dir}/{segment_id}    — binary mesh data
        {output_dir}/{segment_id}:0  — JSON fragment manifest

    Args:
        output_dir: Directory to write to (created if needed).
        segment_id: Numeric ID for this segment.
        vertices: Nx3 float32 vertex positions.
        indices: Mx3 uint32 triangle face indices.

    Returns:
        Path to the binary mesh file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Write binary mesh
    binary = mesh_to_binary(vertices, indices)
    seg_path = out / str(segment_id)
    seg_path.write_bytes(binary)

    # Write fragment manifest JSON
    # The ":0" file tells Neuroglancer which fragment files to load.
    # For single-resolution legacy mesh, there's one fragment per segment.
    manifest = {"fragments": [str(segment_id)]}
    manifest_path = out / f"{segment_id}:0"
    manifest_path.write_text(json.dumps(manifest))

    return seg_path


def fragments_to_mesh(
    fragments: Sequence[dict],
    world_bounds: tuple[float, float, float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Merge clipped fragments into a single mesh with world coordinates.

    Takes fragment dicts (from the streaming pipeline) with keys:
    ``xy``, ``z``, ``ring_lengths``, ``geom_type`` — all in normalized
    [0,1]^3 space — and unprojections them to world coordinates,
    performing vertex deduplication.

    Args:
        fragments: Sequence of fragment dicts with normalized geometry.
        world_bounds: (xmin, ymin, zmin, xmax, ymax, zmax) world bounds.

    Returns:
        (vertices, indices) — Nx3 float32 positions and Mx3 uint32 indices.
    """
    xmin, ymin, zmin, xmax, ymax, zmax = world_bounds
    dx = xmax - xmin if xmax != xmin else 1.0
    dy = ymax - ymin if ymax != ymin else 1.0
    dz = zmax - zmin if zmax != zmin else 1.0

    vertex_map: dict[tuple[float, float, float], int] = {}
    positions: list[list[float]] = []
    all_indices: list[int] = []

    for frag in fragments:
        xy = frag["xy"]
        z = frag["z"]
        ring_lengths = frag.get("ring_lengths", [])
        n_verts = len(z)

        if not ring_lengths:
            ring_lengths = [n_verts]

        offset = 0
        for rl in ring_lengths:
            nv = 3 if rl >= 4 else rl
            if nv < 3 or offset + 2 >= n_verts:
                offset += rl
                continue

            tri_indices = []
            for vi_off in range(3):
                vi = offset + vi_off
                # Unproject to world coordinates
                wx = xmin + xy[vi * 2] * dx
                wy = ymin + xy[vi * 2 + 1] * dy
                wz = zmin + z[vi] * dz

                # Round to float32 precision for dedup
                wx = float(np.float32(wx))
                wy = float(np.float32(wy))
                wz = float(np.float32(wz))

                key = (wx, wy, wz)
                if key not in vertex_map:
                    vertex_map[key] = len(positions)
                    positions.append([wx, wy, wz])
                tri_indices.append(vertex_map[key])

            # Skip degenerate triangles
            if len(set(tri_indices)) == 3:
                all_indices.extend(tri_indices)

            offset += rl

    if not positions:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint32)

    verts = np.array(positions, dtype=np.float32)
    idx = np.array(all_indices, dtype=np.uint32).reshape(-1, 3)
    return verts, idx
