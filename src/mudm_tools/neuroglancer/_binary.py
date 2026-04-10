"""Binary packing primitives for Neuroglancer precomputed formats.

All values are little-endian, matching the Neuroglancer spec.
"""

from __future__ import annotations

import struct


def pack_uint32(value: int) -> bytes:
    """Pack a single uint32 (little-endian)."""
    return struct.pack("<I", value)


def pack_uint64(value: int) -> bytes:
    """Pack a single uint64 (little-endian)."""
    return struct.pack("<Q", value)


def pack_float32(value: float) -> bytes:
    """Pack a single float32 (little-endian)."""
    return struct.pack("<f", value)


def pack_float32_array(values: list[float]) -> bytes:
    """Pack an array of float32 values (little-endian)."""
    return struct.pack(f"<{len(values)}f", *values)


def pack_uint32_array(values: list[int]) -> bytes:
    """Pack an array of uint32 values (little-endian)."""
    return struct.pack(f"<{len(values)}I", *values)


def pack_uint64_array(values: list[int]) -> bytes:
    """Pack an array of uint64 values (little-endian)."""
    return struct.pack(f"<{len(values)}Q", *values)
