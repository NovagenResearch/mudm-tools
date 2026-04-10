"""Low-level helpers for building glTF accessors, buffer views, and buffers.

Packs numpy arrays into the binary layout expected by pygltflib's GLTF2 object.
"""

from __future__ import annotations

import struct
from typing import Literal

import numpy as np
from pygltflib import (
    ARRAY_BUFFER,
    ELEMENT_ARRAY_BUFFER,
    FLOAT,
    UNSIGNED_INT,
    Accessor,
    BufferView,
    GLTF2,
)


def pack_vertices(positions: np.ndarray) -> bytes:
    """Pack an Nx3 float32 position array to little-endian bytes."""
    arr = np.ascontiguousarray(positions, dtype="<f4")
    return arr.tobytes()


def pack_indices(indices: np.ndarray) -> bytes:
    """Pack a flat uint32 index array to little-endian bytes."""
    arr = np.ascontiguousarray(indices.ravel(), dtype="<u4")
    return arr.tobytes()


def _append_buffer_data(gltf: GLTF2, data: bytes) -> int:
    """Append raw bytes to the first buffer, return byte offset."""
    if not gltf.buffers:
        from pygltflib import Buffer

        gltf.buffers.append(Buffer(byteLength=0))

    buf = gltf.buffers[0]
    if gltf._glb_data is None:
        gltf._glb_data = b""
    offset = len(gltf._glb_data)

    # Pad to 4-byte alignment
    padding = (4 - (len(data) % 4)) % 4
    gltf._glb_data += data + b"\x00" * padding

    buf.byteLength = len(gltf._glb_data)
    return offset


def create_accessor(
    gltf: GLTF2,
    data: np.ndarray,
    component_type: int,
    accessor_type: Literal["SCALAR", "VEC2", "VEC3", "VEC4", "MAT4"],
    target: int | None = None,
) -> int:
    """Pack *data* into the GLTF2 object and return the accessor index.

    Creates a BufferView and Accessor pointing at the newly appended data.

    Args:
        gltf: The pygltflib GLTF2 object to mutate.
        data: Numpy array to pack (float32 or uint32).
        component_type: GL constant (e.g. ``FLOAT``, ``UNSIGNED_INT``).
        accessor_type: glTF accessor type string.
        target: Buffer view target (``ARRAY_BUFFER`` or ``ELEMENT_ARRAY_BUFFER``).

    Returns:
        Index of the newly created accessor.
    """
    if component_type == FLOAT:
        raw = pack_vertices(data) if data.ndim > 1 else np.ascontiguousarray(data, dtype="<f4").tobytes()
    elif component_type == UNSIGNED_INT:
        raw = pack_indices(data)
    else:
        raise ValueError(f"Unsupported component_type: {component_type}")

    offset = _append_buffer_data(gltf, raw)

    # Determine element count
    if accessor_type == "SCALAR":
        count = int(data.size)
    elif accessor_type == "VEC3":
        count = int(data.shape[0])
    elif accessor_type == "VEC2":
        count = int(data.shape[0])
    elif accessor_type == "VEC4":
        count = int(data.shape[0])
    else:
        count = int(data.shape[0])

    # BufferView
    bv_idx = len(gltf.bufferViews)
    bv = BufferView(
        buffer=0,
        byteOffset=offset,
        byteLength=len(raw),
    )
    if target is not None:
        bv.target = target
    gltf.bufferViews.append(bv)

    # Accessor (compute min/max for positions)
    accessor_kwargs: dict = {
        "bufferView": bv_idx,
        "byteOffset": 0,
        "componentType": component_type,
        "count": count,
        "type": accessor_type,
    }

    if accessor_type == "VEC3" and component_type == FLOAT and target == ARRAY_BUFFER:
        flat = data.reshape(-1, 3).astype(float)
        accessor_kwargs["max"] = flat.max(axis=0).tolist()
        accessor_kwargs["min"] = flat.min(axis=0).tolist()

    acc_idx = len(gltf.accessors)
    gltf.accessors.append(Accessor(**accessor_kwargs))
    return acc_idx
