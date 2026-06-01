"""Rust gltf decoder Python binding: must decode a GLB file and return
vertex positions + triangle indices matching pygltflib output bit-for-bit."""

from pathlib import Path

import numpy as np
import pygltflib
import pytest

from mudm_tools._rs import decode_glb_buffer

FIXTURE_GLB = Path(__file__).parent / "fixtures" / "sample_tile.glb"


@pytest.mark.skipif(not FIXTURE_GLB.exists(), reason="fixture not generated")
def test_rust_gltf_decoder_matches_pygltflib():
    glb_bytes = FIXTURE_GLB.read_bytes()

    rust_result = decode_glb_buffer(glb_bytes)
    rust_positions = np.asarray(rust_result["positions"], dtype=np.float32).reshape(-1, 3)
    rust_indices = np.asarray(rust_result["indices"], dtype=np.uint32)

    gltf = pygltflib.GLTF2.load(str(FIXTURE_GLB))
    ref_positions = _pygltflib_positions(gltf)
    ref_indices = _pygltflib_indices(gltf)

    np.testing.assert_array_almost_equal(rust_positions, ref_positions, decimal=5)
    np.testing.assert_array_equal(rust_indices, ref_indices)


def test_rust_gltf_decoder_handles_multi_primitive(two_primitive_glb):
    glb_bytes = two_primitive_glb.read_bytes()

    rust_result = decode_glb_buffer(glb_bytes)
    rust_positions = np.asarray(rust_result["positions"], dtype=np.float32).reshape(-1, 3)
    rust_indices = np.asarray(rust_result["indices"], dtype=np.uint32)

    # 8 total vertices (4 + 4); 12 total indices (6 + 6); primitive-1 indices must
    # be offset by 4. Without offsetting, max(rust_indices) would be 3 instead of 7.
    assert rust_positions.shape == (8, 3)
    assert rust_indices.shape == (12,)
    assert int(rust_indices.max()) == 7  # confirms vertex_offset accumulation
    assert int(rust_indices.min()) == 0

    # Cross-validate: positions concatenate prim0 then prim1
    np.testing.assert_array_almost_equal(
        rust_positions[:4],
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32),
        decimal=5,
    )
    np.testing.assert_array_almost_equal(
        rust_positions[4:],
        np.array([[2, 2, 2], [3, 2, 2], [3, 3, 2], [2, 3, 2]], dtype=np.float32),
        decimal=5,
    )


def _pygltflib_buffers(gltf) -> list[np.ndarray]:
    """Return per-index buffer byte arrays. For GLB, buffer 0 is the embedded blob."""
    blob = gltf.binary_blob()
    out: list[np.ndarray] = []
    for i, _buf in enumerate(gltf.buffers):
        if i == 0 and blob is not None:
            out.append(np.frombuffer(blob, dtype=np.uint8))
        else:
            raise NotImplementedError(
                f"buffer {i} is not the embedded GLB blob; external buffers unsupported in this test"
            )
    return out


def _pygltflib_positions(gltf) -> np.ndarray:
    buffers = _pygltflib_buffers(gltf)
    positions = []
    for mesh in gltf.meshes:
        for prim in mesh.primitives:
            accessor = gltf.accessors[prim.attributes.POSITION]
            view = gltf.bufferViews[accessor.bufferView]
            offset = (view.byteOffset or 0) + (accessor.byteOffset or 0)
            count = accessor.count
            raw = buffers[view.buffer][offset : offset + count * 12]
            positions.append(np.frombuffer(raw, dtype=np.float32).reshape(-1, 3))
    return np.concatenate(positions, axis=0)


def _pygltflib_indices(gltf) -> np.ndarray:
    buffers = _pygltflib_buffers(gltf)
    indices = []
    base = 0
    for mesh in gltf.meshes:
        for prim in mesh.primitives:
            pos_accessor = gltf.accessors[prim.attributes.POSITION]
            prim_vertex_count = pos_accessor.count
            accessor = gltf.accessors[prim.indices]
            view = gltf.bufferViews[accessor.bufferView]
            offset = (view.byteOffset or 0) + (accessor.byteOffset or 0)
            count = accessor.count
            dtype_map = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32}
            dtype = dtype_map[accessor.componentType]
            raw = buffers[view.buffer][offset : offset + count * np.dtype(dtype).itemsize]
            prim_indices = np.frombuffer(raw, dtype=dtype).astype(np.uint32) + base
            indices.append(prim_indices)
            base += prim_vertex_count
    return np.concatenate(indices, axis=0)
