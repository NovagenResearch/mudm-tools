"""Tests for Neuroglancer binary packing primitives."""

import struct


from mudm_tools.neuroglancer._binary import (
    pack_float32,
    pack_float32_array,
    pack_uint32,
    pack_uint32_array,
    pack_uint64,
    pack_uint64_array,
)


class TestPackUint32:
    def test_zero(self):
        assert pack_uint32(0) == b"\x00\x00\x00\x00"

    def test_one(self):
        assert pack_uint32(1) == b"\x01\x00\x00\x00"

    def test_roundtrip(self):
        for val in [0, 1, 42, 255, 65535, 2**32 - 1]:
            assert struct.unpack("<I", pack_uint32(val))[0] == val

    def test_little_endian(self):
        # 0x01020304 in LE = 04 03 02 01
        packed = pack_uint32(0x01020304)
        assert packed == b"\x04\x03\x02\x01"


class TestPackUint64:
    def test_zero(self):
        assert pack_uint64(0) == b"\x00" * 8

    def test_roundtrip(self):
        for val in [0, 1, 2**32, 2**64 - 1]:
            assert struct.unpack("<Q", pack_uint64(val))[0] == val


class TestPackFloat32:
    def test_zero(self):
        assert pack_float32(0.0) == b"\x00\x00\x00\x00"

    def test_roundtrip(self):
        for val in [0.0, 1.0, -1.0, 3.14]:
            unpacked = struct.unpack("<f", pack_float32(val))[0]
            assert abs(unpacked - val) < 1e-6


class TestPackFloat32Array:
    def test_empty(self):
        assert pack_float32_array([]) == b""

    def test_single(self):
        assert pack_float32_array([1.0]) == pack_float32(1.0)

    def test_roundtrip(self):
        values = [1.0, 2.0, 3.0, -0.5]
        packed = pack_float32_array(values)
        assert len(packed) == 4 * len(values)
        unpacked = struct.unpack(f"<{len(values)}f", packed)
        for a, b in zip(values, unpacked):
            assert abs(a - b) < 1e-6


class TestPackUint32Array:
    def test_empty(self):
        assert pack_uint32_array([]) == b""

    def test_roundtrip(self):
        values = [0, 1, 100, 2**32 - 1]
        packed = pack_uint32_array(values)
        assert len(packed) == 4 * len(values)
        unpacked = struct.unpack(f"<{len(values)}I", packed)
        assert list(unpacked) == values


class TestPackUint64Array:
    def test_empty(self):
        assert pack_uint64_array([]) == b""

    def test_roundtrip(self):
        values = [0, 1, 2**32, 2**64 - 1]
        packed = pack_uint64_array(values)
        assert len(packed) == 8 * len(values)
        unpacked = struct.unpack(f"<{len(values)}Q", packed)
        assert list(unpacked) == values
