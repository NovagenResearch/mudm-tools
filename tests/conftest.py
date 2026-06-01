"""Shared pytest fixtures for the mudm-tools test suite."""

from pathlib import Path

import numpy as np
import pygltflib
import pytest


def _build_two_primitive_glb(out_path: Path) -> None:
    """Build a minimal valid GLB with two primitives in a single mesh.

    Primitive 0: 4 vertices forming a quad as 2 triangles.
    Primitive 1: 4 vertices forming a different quad as 2 triangles.

    Indices in each primitive are local (0..3); a correct decoder must offset
    primitive-1 indices by primitive-0's vertex count when concatenating."""
    prim0_pos = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    prim0_idx = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

    prim1_pos = np.array([[2, 2, 2], [3, 2, 2], [3, 3, 2], [2, 3, 2]], dtype=np.float32)
    prim1_idx = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

    p0p_bytes = prim0_pos.tobytes()
    p0i_bytes = prim0_idx.tobytes()
    p1p_bytes = prim1_pos.tobytes()
    p1i_bytes = prim1_idx.tobytes()

    blob = bytearray()
    offsets: dict[str, tuple[int, int]] = {}
    for name, b in (
        ("p0p", p0p_bytes),
        ("p0i", p0i_bytes),
        ("p1p", p1p_bytes),
        ("p1i", p1i_bytes),
    ):
        # 4-byte align each section
        while len(blob) % 4 != 0:
            blob.append(0)
        offsets[name] = (len(blob), len(b))
        blob.extend(b)

    gltf = pygltflib.GLTF2(
        asset=pygltflib.Asset(version="2.0"),
        buffers=[pygltflib.Buffer(byteLength=len(blob))],
        bufferViews=[
            pygltflib.BufferView(
                buffer=0,
                byteOffset=offsets["p0p"][0],
                byteLength=offsets["p0p"][1],
                target=34962,
            ),
            pygltflib.BufferView(
                buffer=0,
                byteOffset=offsets["p0i"][0],
                byteLength=offsets["p0i"][1],
                target=34963,
            ),
            pygltflib.BufferView(
                buffer=0,
                byteOffset=offsets["p1p"][0],
                byteLength=offsets["p1p"][1],
                target=34962,
            ),
            pygltflib.BufferView(
                buffer=0,
                byteOffset=offsets["p1i"][0],
                byteLength=offsets["p1i"][1],
                target=34963,
            ),
        ],
        accessors=[
            pygltflib.Accessor(
                bufferView=0,
                componentType=5126,
                count=4,
                type="VEC3",
                min=prim0_pos.min(axis=0).tolist(),
                max=prim0_pos.max(axis=0).tolist(),
            ),
            pygltflib.Accessor(
                bufferView=1,
                componentType=5125,
                count=6,
                type="SCALAR",
            ),
            pygltflib.Accessor(
                bufferView=2,
                componentType=5126,
                count=4,
                type="VEC3",
                min=prim1_pos.min(axis=0).tolist(),
                max=prim1_pos.max(axis=0).tolist(),
            ),
            pygltflib.Accessor(
                bufferView=3,
                componentType=5125,
                count=6,
                type="SCALAR",
            ),
        ],
        meshes=[
            pygltflib.Mesh(
                primitives=[
                    pygltflib.Primitive(
                        attributes=pygltflib.Attributes(POSITION=0),
                        indices=1,
                    ),
                    pygltflib.Primitive(
                        attributes=pygltflib.Attributes(POSITION=2),
                        indices=3,
                    ),
                ]
            )
        ],
    )
    gltf.set_binary_blob(bytes(blob))
    gltf.save_binary(str(out_path))


@pytest.fixture
def two_primitive_glb(tmp_path: Path) -> Path:
    out = tmp_path / "two_primitive.glb"
    _build_two_primitive_glb(out)
    return out
