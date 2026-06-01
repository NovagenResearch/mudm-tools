"""Neuroglancer precomputed skeleton writer.

Converts MuDM NeuronMorphology to the Neuroglancer precomputed
skeleton binary format.

Binary layout per segment (little-endian):
    uint32  num_vertices
    uint32  num_edges
    float32 vertices[num_vertices × 3]   (x, y, z interleaved)
    uint32  edges[num_edges × 2]          (source, target)
    float32 radii[num_vertices]           (vertex attribute, optional)
    float32 types[num_vertices]           (vertex attribute, optional)

Reference: https://github.com/google/neuroglancer/blob/master/
    src/neuroglancer/datasource/precomputed/skeleton.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..swc import NeuronMorphology
from mudm.transforms import AffineTransform
from ._binary import pack_float32_array, pack_uint32, pack_uint32_array
from .models import SkeletonInfo, VertexAttributeInfo


def affine_to_ng_transform(transform: AffineTransform) -> list[float]:
    """Convert a 4×4 row-major affine to Neuroglancer's 12-element format.

    Neuroglancer expects 12 floats representing the upper 3×4 sub-matrix
    in row-major order (3 rows of 4 values each):
        [r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz]

    Internally, Neuroglancer loads these into a column-major mat4 and
    transposes, which produces the correct 4×4 affine.
    """
    m = transform.matrix
    result: list[float] = []
    for row in range(3):
        for col in range(4):
            result.append(m[row][col])
    return result


def neuron_to_skeleton_binary(
    morphology: NeuronMorphology,
    *,
    include_radius: bool = True,
    include_type: bool = True,
) -> bytes:
    """Encode a NeuronMorphology as Neuroglancer precomputed skeleton binary.

    Args:
        morphology: The neuron morphology to encode.
        include_radius: Include per-vertex radius attribute.
        include_type: Include per-vertex SWC type attribute.

    Returns:
        Raw bytes in Neuroglancer skeleton binary format.
    """
    tree = morphology.tree
    num_verts = len(tree)

    # Build vertex positions (x, y, z interleaved)
    positions: list[float] = []
    for sample in tree:
        positions.extend([sample.x, sample.y, sample.z])

    # Build edge list — map sample.id → index, then parent edges
    id_to_idx = {sample.id: i for i, sample in enumerate(tree)}
    edges: list[int] = []
    for sample in tree:
        if sample.parent != -1 and sample.parent in id_to_idx:
            edges.extend([id_to_idx[sample.parent], id_to_idx[sample.id]])
    num_edges = len(edges) // 2

    # Pack binary
    buf = bytearray()
    buf += pack_uint32(num_verts)
    buf += pack_uint32(num_edges)
    buf += pack_float32_array(positions)
    buf += pack_uint32_array(edges)

    if include_radius:
        radii = [sample.r for sample in tree]
        buf += pack_float32_array(radii)

    if include_type:
        types = [float(sample.type) for sample in tree]
        buf += pack_float32_array(types)

    return bytes(buf)


def build_skeleton_info(
    *,
    transform: Optional[AffineTransform] = None,
    scale_um_to_nm: bool = False,
    include_radius: bool = True,
    include_type: bool = True,
    segment_properties: Optional[str] = None,
) -> SkeletonInfo:
    """Build a SkeletonInfo config for the info JSON file.

    Args:
        transform: Optional affine transform to embed.
        scale_um_to_nm: If True and no explicit transform, embed a
            1000× scaling transform (micrometers → nanometers). Useful
            for SWC files whose coordinates are in µm.
        include_radius: Include radius vertex attribute.
        include_type: Include type vertex attribute.
        segment_properties: Optional relative path to segment_properties dir.

    Returns:
        A SkeletonInfo model ready to serialize.
    """
    vertex_attributes: list[VertexAttributeInfo] = []
    if include_radius:
        vertex_attributes.append(VertexAttributeInfo(id="radius", data_type="float32"))
    if include_type:
        vertex_attributes.append(VertexAttributeInfo(id="type", data_type="float32"))

    ng_transform: Optional[list[float]] = None
    if transform is not None:
        ng_transform = affine_to_ng_transform(transform)
    elif scale_um_to_nm:
        # Row-major 3×4: scale XYZ by 1000 (µm→nm), no translation
        ng_transform = [1000, 0, 0, 0, 0, 1000, 0, 0, 0, 0, 1000, 0]

    return SkeletonInfo(
        transform=ng_transform,
        vertex_attributes=vertex_attributes,
        segment_properties=segment_properties,
    )


def write_skeleton(
    output_dir: str | Path,
    segment_id: int,
    morphology: NeuronMorphology,
    *,
    transform: Optional[AffineTransform] = None,
    include_radius: bool = True,
    include_type: bool = True,
    segment_properties: Optional[str] = None,
) -> Path:
    """Write a single skeleton segment to the precomputed directory.

    Creates:
        {output_dir}/info          — JSON info file
        {output_dir}/{segment_id}  — binary skeleton data

    Args:
        output_dir: Directory to write to (created if needed).
        segment_id: Numeric ID for this segment.
        morphology: The neuron morphology data.
        transform: Optional affine transform.
        include_radius: Include radius attribute.
        include_type: Include type attribute.
        segment_properties: Optional path to segment_properties.

    Returns:
        Path to the output directory.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Write info JSON
    info = build_skeleton_info(
        transform=transform,
        include_radius=include_radius,
        include_type=include_type,
        segment_properties=segment_properties,
    )
    info_path = out / "info"
    info_path.write_text(json.dumps(info.to_info_dict(), indent=2))

    # Write binary skeleton
    binary = neuron_to_skeleton_binary(
        morphology,
        include_radius=include_radius,
        include_type=include_type,
    )
    seg_path = out / str(segment_id)
    seg_path.write_bytes(binary)

    return out
