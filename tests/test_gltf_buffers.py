"""Tests for glTF buffer packing utilities."""

import numpy as np
from pygltflib import ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER, FLOAT, UNSIGNED_INT, GLTF2

from mudm_tools.gltf._buffers import (
    create_accessor,
    pack_indices,
    pack_vertices,
)


class TestPackVertices:
    def test_roundtrip_simple(self):
        positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        raw = pack_vertices(positions)
        recovered = np.frombuffer(raw, dtype="<f4").reshape(-1, 3)
        np.testing.assert_array_equal(recovered, positions)

    def test_empty(self):
        positions = np.empty((0, 3), dtype=np.float32)
        raw = pack_vertices(positions)
        assert raw == b""

    def test_float64_converted(self):
        positions = np.array([[1.5, 2.5, 3.5]], dtype=np.float64)
        raw = pack_vertices(positions)
        recovered = np.frombuffer(raw, dtype="<f4").reshape(-1, 3)
        np.testing.assert_allclose(recovered, positions, atol=1e-6)


class TestPackIndices:
    def test_roundtrip_simple(self):
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        raw = pack_indices(indices)
        recovered = np.frombuffer(raw, dtype="<u4")
        np.testing.assert_array_equal(recovered, indices)

    def test_2d_flattened(self):
        indices = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)
        raw = pack_indices(indices)
        recovered = np.frombuffer(raw, dtype="<u4")
        np.testing.assert_array_equal(recovered, [0, 1, 2, 3, 4, 5])


class TestCreateAccessor:
    def _make_gltf(self) -> GLTF2:
        gltf = GLTF2()
        gltf._glb_data = None
        return gltf

    def test_vec3_positions(self):
        gltf = self._make_gltf()
        data = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
        acc_idx = create_accessor(gltf, data, FLOAT, "VEC3", ARRAY_BUFFER)

        assert acc_idx == 0
        assert len(gltf.accessors) == 1
        assert gltf.accessors[0].count == 2
        assert gltf.accessors[0].type == "VEC3"
        assert gltf.accessors[0].componentType == FLOAT
        assert gltf.accessors[0].max == [3.0, 4.0, 5.0]
        assert gltf.accessors[0].min == [0.0, 1.0, 2.0]

    def test_scalar_indices(self):
        gltf = self._make_gltf()
        data = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        acc_idx = create_accessor(gltf, data, UNSIGNED_INT, "SCALAR", ELEMENT_ARRAY_BUFFER)

        assert acc_idx == 0
        assert gltf.accessors[0].count == 6
        assert gltf.accessors[0].type == "SCALAR"
        assert gltf.accessors[0].componentType == UNSIGNED_INT

    def test_multiple_accessors(self):
        gltf = self._make_gltf()
        pos = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        idx = np.array([0], dtype=np.uint32)

        pos_acc = create_accessor(gltf, pos, FLOAT, "VEC3", ARRAY_BUFFER)
        idx_acc = create_accessor(gltf, idx, UNSIGNED_INT, "SCALAR", ELEMENT_ARRAY_BUFFER)

        assert pos_acc == 0
        assert idx_acc == 1
        assert len(gltf.accessors) == 2
        assert len(gltf.bufferViews) == 2
        # Both point to same buffer
        assert gltf.bufferViews[0].buffer == 0
        assert gltf.bufferViews[1].buffer == 0
        # Offsets differ
        assert gltf.bufferViews[1].byteOffset > 0

    def test_buffer_alignment(self):
        gltf = self._make_gltf()
        # 3 floats = 12 bytes → already 4-byte aligned
        data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        create_accessor(gltf, data, FLOAT, "VEC3", ARRAY_BUFFER)
        assert len(gltf._glb_data) % 4 == 0
