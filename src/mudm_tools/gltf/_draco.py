"""Draco mesh compression for glTF (KHR_draco_mesh_compression).

Encodes triangle mesh data with Google Draco and wires the compressed
buffer into a pygltflib GLTF2 object with the correct extension metadata.

Requires the optional ``DracoPy`` package::

    pip install DracoPy
"""

from __future__ import annotations

import numpy as np
from pygltflib import (
    FLOAT,
    TRIANGLES,
    UNSIGNED_INT,
    Accessor,
    Attributes,
    BufferView,
    GLTF2,
    Mesh,
    Primitive,
)

from ._buffers import _append_buffer_data
from .models import GltfConfig

DRACO_EXTENSION = "KHR_draco_mesh_compression"


def _require_dracopy():
    """Lazily import DracoPy, raising a helpful error if missing."""
    try:
        import DracoPy
    except ImportError:
        raise ImportError(
            "DracoPy is required for Draco compression. "
            "Install it with: pip install DracoPy"
        ) from None
    return DracoPy


def encode_draco(
    vertices: np.ndarray,
    indices: np.ndarray,
    normals: np.ndarray | None,
    config: GltfConfig,
) -> bytes:
    """Encode mesh data to Draco compressed bytes.

    Args:
        vertices: Nx3 float array of vertex positions.
        indices: Mx3 uint32 array of triangle face indices.
        normals: Optional Nx3 float array of vertex normals.
        config: glTF config with Draco quantization/compression settings.

    Returns:
        Draco-encoded bytes.
    """
    DracoPy = _require_dracopy()

    points = np.ascontiguousarray(vertices, dtype=np.float32)
    faces = np.ascontiguousarray(indices, dtype=np.uint32)

    kwargs: dict = {
        "quantization_bits": config.draco_quantization_position,
        "compression_level": config.draco_compression_level,
    }
    if normals is not None:
        # DracoPy requires normals as float64
        kwargs["normals"] = np.ascontiguousarray(normals, dtype=np.float64)

    return DracoPy.encode(points, faces=faces, **kwargs)


def add_draco_triangle_mesh(
    gltf: GLTF2,
    vertices: np.ndarray,
    indices: np.ndarray,
    normals: np.ndarray | None,
    config: GltfConfig,
    material_idx: int = 0,
) -> int:
    """Add a Draco-compressed triangle mesh to the glTF object.

    Creates stub accessors (no bufferView), a Draco-compressed BufferView,
    and a Primitive with the ``KHR_draco_mesh_compression`` extension.

    Args:
        gltf: The pygltflib GLTF2 object to mutate.
        vertices: Nx3 float positions (already Y-up transformed).
        indices: Mx3 uint32 triangle face indices.
        normals: Optional Nx3 float normals (already Y-up transformed).
        config: glTF config with Draco settings.
        material_idx: Material index for the primitive.

    Returns:
        Index of the newly created mesh.
    """
    # Encode with Draco
    draco_bytes = encode_draco(vertices, indices, normals, config)

    # Append compressed data to the GLB buffer
    offset = _append_buffer_data(gltf, draco_bytes)

    # Create Draco BufferView (no target — Draco data is not raw GL buffer)
    draco_bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(
        BufferView(
            buffer=0,
            byteOffset=offset,
            byteLength=len(draco_bytes),
        )
    )

    # DracoPy assigns unique IDs in this order:
    #   - normals  → unique_id 0 (if present)
    #   - position → unique_id 1 (if normals present), else 0
    if normals is not None:
        draco_pos_id = 1
        draco_norm_id = 0
    else:
        draco_pos_id = 0

    # Stub POSITION accessor — no bufferView or byteOffset (Draco spec requirement)
    verts_f32 = vertices.astype(np.float32)
    pos_acc_idx = len(gltf.accessors)
    gltf.accessors.append(
        Accessor(
            componentType=FLOAT,
            count=int(vertices.shape[0]),
            type="VEC3",
            max=verts_f32.max(axis=0).tolist(),
            min=verts_f32.min(axis=0).tolist(),
            byteOffset=None,
        )
    )

    draco_attributes = {"POSITION": draco_pos_id}
    attrs = Attributes(POSITION=pos_acc_idx)

    # Stub NORMAL accessor — if normals present
    if normals is not None:
        norm_acc_idx = len(gltf.accessors)
        gltf.accessors.append(
            Accessor(
                componentType=FLOAT,
                count=int(normals.shape[0]),
                type="VEC3",
                byteOffset=None,
            )
        )
        attrs.NORMAL = norm_acc_idx
        draco_attributes["NORMAL"] = draco_norm_id

    # Stub indices accessor — no bufferView or byteOffset
    idx_acc_idx = len(gltf.accessors)
    gltf.accessors.append(
        Accessor(
            componentType=UNSIGNED_INT,
            count=int(indices.size),
            type="SCALAR",
            byteOffset=None,
        )
    )

    # Build Primitive with Draco extension
    primitive = Primitive(
        attributes=attrs,
        indices=idx_acc_idx,
        material=material_idx,
        mode=TRIANGLES,
    )
    primitive.extensions = {
        DRACO_EXTENSION: {
            "bufferView": draco_bv_idx,
            "attributes": draco_attributes,
        }
    }

    mesh_idx = len(gltf.meshes)
    gltf.meshes.append(Mesh(primitives=[primitive]))
    return mesh_idx


def ensure_draco_extensions(gltf: GLTF2) -> None:
    """Add Draco extension to extensionsUsed and extensionsRequired."""
    if not gltf.extensionsUsed:
        gltf.extensionsUsed = []
    if DRACO_EXTENSION not in gltf.extensionsUsed:
        gltf.extensionsUsed.append(DRACO_EXTENSION)

    if not gltf.extensionsRequired:
        gltf.extensionsRequired = []
    if DRACO_EXTENSION not in gltf.extensionsRequired:
        gltf.extensionsRequired.append(DRACO_EXTENSION)
