"""SWC ↔ MuDM conversion utilities.

SWC is the standard format for neuron morphology data. This module is the
sole owner of NeuronMorphology/SWCSample types and all neuron-specific mesh
generation logic. Generic mesh utilities (smooth_path, thin_path,
_tube_along_path, _icosphere) are imported from ``gltf.mesh_builder``.

Public API:
    swc_to_microjson(swc_path) -> MuDMFeature(geometry=TIN)
    swc_to_tin(swc_path, ...) -> MuDMFeature(geometry=TIN)
    swc_to_linestring3d(swc_path) -> MultiLineString
    microjson_to_swc(morphology) -> str
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np
from geojson_pydantic import MultiLineString
from pydantic import BaseModel, field_validator

from mudm.model import MuDMFeature, MuDMFeatureCollection, TIN


# ---------------------------------------------------------------------------
# SWC constants
# ---------------------------------------------------------------------------

SWC_UNDEFINED = 0
SWC_SOMA = 1
SWC_AXON = 2
SWC_BASAL_DENDRITE = 3
SWC_APICAL_DENDRITE = 4
SWC_FORK_POINT = 5
SWC_END_POINT = 6
SWC_CUSTOM = 7

SWC_TYPE_NAMES: dict[int, str] = {
    SWC_UNDEFINED: "undefined",
    SWC_SOMA: "soma",
    SWC_AXON: "axon",
    SWC_BASAL_DENDRITE: "basal_dendrite",
    SWC_APICAL_DENDRITE: "apical_dendrite",
    SWC_FORK_POINT: "fork_point",
    SWC_END_POINT: "end_point",
    SWC_CUSTOM: "custom",
}


# ---------------------------------------------------------------------------
# SWCSample / NeuronMorphology models (canonical definitions)
# ---------------------------------------------------------------------------

class SWCSample(BaseModel):
    """A single sample point in an SWC neuron morphology tree."""

    id: int
    type: int
    x: float
    y: float
    z: float
    r: float
    parent: int


class NeuronMorphology(BaseModel):
    """A neuron morphology represented as an SWC tree.

    Each node has a 3D position, radius, type code, and parent reference.
    This type is NOT part of the GeometryType union — it cannot be stored
    directly in a MuDMFeature. Use ``swc_to_microjson()`` or
    ``swc_to_tin()`` to get a MuDMFeature with TIN geometry.
    """

    type: Literal["NeuronMorphology"]
    tree: List[SWCSample]

    @field_validator("tree")
    @classmethod
    def _validate_tree(cls, v: List[SWCSample]) -> List[SWCSample]:
        if len(v) == 0:
            raise ValueError(
                "NeuronMorphology tree must have at least one node"
            )
        ids = {s.id for s in v}
        if not any(s.parent == -1 for s in v):
            raise ValueError(
                "NeuronMorphology tree must have at least one root node "
                "(parent == -1)"
            )
        for s in v:
            if s.parent != -1 and s.parent not in ids:
                raise ValueError(
                    f"Node {s.id} has parent {s.parent} which does not exist "
                    f"in the tree"
                )
        return v

    def bbox3d(self) -> Tuple[float, float, float, float, float, float]:
        xs = [s.x for s in self.tree]
        ys = [s.y for s in self.tree]
        zs = [s.z for s in self.tree]
        return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))

    def centroid3d(self) -> Tuple[float, float, float]:
        n = len(self.tree)
        return (
            sum(s.x for s in self.tree) / n,
            sum(s.y for s in self.tree) / n,
            sum(s.z for s in self.tree) / n,
        )


# ---------------------------------------------------------------------------
# Internal SWC parser (returns NeuronMorphology — not for MuDMFeature)
# ---------------------------------------------------------------------------

def _parse_swc(swc_path: str) -> NeuronMorphology:
    """Parse an SWC file into a NeuronMorphology (internal use only)."""
    samples: list[SWCSample] = []
    with open(swc_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            samples.append(
                SWCSample(
                    id=int(float(parts[0])),
                    type=int(float(parts[1])),
                    x=float(parts[2]),
                    y=float(parts[3]),
                    z=float(parts[4]),
                    r=float(parts[5]),
                    parent=int(float(parts[6])),
                )
            )
    return NeuronMorphology(type="NeuronMorphology", tree=samples)


# ---------------------------------------------------------------------------
# Neuron-specific mesh generation
# ---------------------------------------------------------------------------

def _extract_paths(
    tree: list,
    id_to_sample: dict,
) -> list[list[int]]:
    """Extract linear branch paths from the neuron tree."""
    children: dict[int, list[int]] = defaultdict(list)
    roots: list[int] = []

    for s in tree:
        if s.parent == -1:
            roots.append(s.id)
        elif s.parent in id_to_sample:
            children[s.parent].append(s.id)

    paths: list[list[int]] = []
    visited_edges: set[tuple[int, int]] = set()
    stack = list(roots)

    while stack:
        node_id = stack.pop()
        kids = children[node_id]

        for kid in kids:
            if (node_id, kid) in visited_edges:
                continue
            chain = [node_id, kid]
            visited_edges.add((node_id, kid))
            current = kid

            while len(children[current]) == 1:
                child = children[current][0]
                visited_edges.add((current, child))
                chain.append(child)
                current = child

            paths.append(chain)
            if len(children[current]) > 1:
                stack.append(current)

    return paths


def _split_path_by_type(
    path_ids: list[int],
    id_to_sample: dict,
) -> list[tuple[int, list[int]]]:
    """Split a path into sub-paths of consecutive same-type edges."""
    if len(path_ids) < 2:
        return []

    result: list[tuple[int, list[int]]] = []
    current_type = id_to_sample[path_ids[1]].type
    current_sub: list[int] = [path_ids[0], path_ids[1]]

    for i in range(2, len(path_ids)):
        child_type = id_to_sample[path_ids[i]].type
        if child_type != current_type:
            result.append((current_type, current_sub))
            current_sub = [path_ids[i - 1], path_ids[i]]
            current_type = child_type
        else:
            current_sub.append(path_ids[i])

    result.append((current_type, current_sub))
    return result


_MeshTuple = tuple[np.ndarray, np.ndarray, np.ndarray]


def neuron_to_tube_mesh(
    morphology: NeuronMorphology,
    segments: int = 8,
    min_radius: float = 0.1,
    smooth_subdivisions: int = 0,
    mesh_quality: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a NeuronMorphology into a tube mesh with sphere soma."""
    from .gltf.mesh_builder import (
        _icosphere,
        _tube_along_path,
        smooth_path,
        thin_path,
    )

    tree = morphology.tree
    id_to_sample = {s.id: s for s in tree}

    children: dict[int, list[int]] = defaultdict(list)
    for s in tree:
        if s.parent != -1 and s.parent in id_to_sample:
            children[s.parent].append(s.id)

    all_verts: list[np.ndarray] = []
    all_normals: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []
    vertex_offset = 0

    for sample in tree:
        if sample.type == 1 and sample.parent == -1:
            center = np.array([sample.x, sample.y, sample.z])
            r = max(sample.r, min_radius)
            v, n, idx = _icosphere(center, r, subdivisions=2)
            if v.shape[0] > 0:
                all_verts.append(v)
                all_normals.append(n)
                all_indices.append(idx + vertex_offset)
                vertex_offset += v.shape[0]

    for sample in tree:
        if sample.type == 1 and sample.parent == -1:
            continue
        if len(children[sample.id]) > 1:
            center = np.array([sample.x, sample.y, sample.z])
            r = max(sample.r, min_radius)
            v, n, idx = _icosphere(center, r, subdivisions=1)
            if v.shape[0] > 0:
                all_verts.append(v)
                all_normals.append(n)
                all_indices.append(idx + vertex_offset)
                vertex_offset += v.shape[0]

    paths = _extract_paths(tree, id_to_sample)
    for path_ids in paths:
        points = []
        radii = []
        for sid in path_ids:
            s = id_to_sample[sid]
            points.append(np.array([s.x, s.y, s.z]))
            radii.append(max(s.r, min_radius))

        if smooth_subdivisions > 0:
            points, radii = smooth_path(points, radii, smooth_subdivisions)
        if mesh_quality < 1.0:
            points, radii = thin_path(points, radii, mesh_quality)

        v, n, idx = _tube_along_path(points, radii, segments)
        if v.shape[0] > 0:
            all_verts.append(v)
            all_normals.append(n)
            all_indices.append(idx + vertex_offset)
            vertex_offset += v.shape[0]

    if not all_verts:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.uint32),
        )

    return (
        np.concatenate(all_verts),
        np.concatenate(all_normals),
        np.concatenate(all_indices),
    )


def neuron_to_typed_meshes(
    morphology: NeuronMorphology,
    segments: int = 8,
    min_radius: float = 0.1,
    smooth_subdivisions: int = 0,
    mesh_quality: float = 1.0,
) -> dict[int, _MeshTuple]:
    """Convert a NeuronMorphology into per-SWC-type meshes."""
    from .gltf.mesh_builder import (
        _icosphere,
        _tube_along_path,
        smooth_path,
        thin_path,
    )

    tree = morphology.tree
    id_to_sample = {s.id: s for s in tree}

    children: dict[int, list[int]] = defaultdict(list)
    for s in tree:
        if s.parent != -1 and s.parent in id_to_sample:
            children[s.parent].append(s.id)

    typed_verts: dict[int, list[np.ndarray]] = defaultdict(list)
    typed_normals: dict[int, list[np.ndarray]] = defaultdict(list)
    typed_indices: dict[int, list[np.ndarray]] = defaultdict(list)
    typed_offsets: dict[int, int] = defaultdict(int)

    def _append(
        swc_type: int, v: np.ndarray, n: np.ndarray, idx: np.ndarray,
    ) -> None:
        if v.shape[0] == 0:
            return
        typed_verts[swc_type].append(v)
        typed_normals[swc_type].append(n)
        typed_indices[swc_type].append(idx + typed_offsets[swc_type])
        typed_offsets[swc_type] += v.shape[0]

    for sample in tree:
        if sample.type == 1 and sample.parent == -1:
            center = np.array([sample.x, sample.y, sample.z])
            r = max(sample.r, min_radius)
            v, n, idx = _icosphere(center, r, subdivisions=2)
            _append(1, v, n, idx)

    for sample in tree:
        if sample.type == 1 and sample.parent == -1:
            continue
        if len(children[sample.id]) > 1:
            center = np.array([sample.x, sample.y, sample.z])
            r = max(sample.r, min_radius)
            v, n, idx = _icosphere(center, r, subdivisions=1)
            _append(sample.type, v, n, idx)

    paths = _extract_paths(tree, id_to_sample)
    for path_ids in paths:
        typed_subs = _split_path_by_type(path_ids, id_to_sample)
        for swc_type, sub_ids in typed_subs:
            points = [
                np.array([
                    id_to_sample[s].x,
                    id_to_sample[s].y,
                    id_to_sample[s].z,
                ])
                for s in sub_ids
            ]
            radii = [
                max(id_to_sample[s].r, min_radius)
                for s in sub_ids
            ]
            if smooth_subdivisions > 0:
                points, radii = smooth_path(points, radii, smooth_subdivisions)
            if mesh_quality < 1.0:
                points, radii = thin_path(points, radii, mesh_quality)
            v, n, idx = _tube_along_path(points, radii, segments)
            _append(swc_type, v, n, idx)

    result: dict[int, _MeshTuple] = {}
    for swc_type in typed_verts:
        result[swc_type] = (
            np.concatenate(typed_verts[swc_type]),
            np.concatenate(typed_normals[swc_type]),
            np.concatenate(typed_indices[swc_type]),
        )
    return result


# ---------------------------------------------------------------------------
# TIN conversion helpers
# ---------------------------------------------------------------------------

def _mesh_to_tin(vertices: np.ndarray, indices: np.ndarray) -> TIN:
    """Convert a triangle mesh (Nx3 vertices, Mx3 face indices) to a TIN."""
    faces: list[list[list[list[float]]]] = []
    for tri in indices:
        v0 = vertices[tri[0]]
        v1 = vertices[tri[1]]
        v2 = vertices[tri[2]]
        ring: list[list[float]] = [
            [float(v0[0]), float(v0[1]), float(v0[2])],
            [float(v1[0]), float(v1[1]), float(v1[2])],
            [float(v2[0]), float(v2[1]), float(v2[2])],
            [float(v0[0]), float(v0[1]), float(v0[2])],
        ]
        faces.append([ring])
    return TIN(type="TIN", coordinates=faces)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def swc_to_microjson(
    swc_path: str,
    *,
    segments: int = 8,
    min_radius: float = 0.1,
    smooth_subdivisions: int = 3,
    mesh_quality: float = 1.0,
) -> MuDMFeature:
    """Parse an SWC file and return a MuDMFeature with TIN geometry.

    The neuron tree is parsed, converted to a tube mesh, and stored as
    a TIN — standard exporters handle TIN without neuron-specific code.

    Args:
        swc_path: Path to the SWC file.
        segments: Cross-section polygon sides for tubes.
        min_radius: Minimum radius clamp.
        smooth_subdivisions: Catmull-Rom subdivisions per segment (0 = off).
        mesh_quality: Fraction of smoothed path points to keep (0.0-1.0).

    Returns:
        A MuDMFeature wrapping a TIN geometry.
    """
    return swc_to_tin(
        swc_path,
        segments=segments,
        min_radius=min_radius,
        smooth_subdivisions=smooth_subdivisions,
        mesh_quality=mesh_quality,
    )


def swc_to_tin(
    swc_path: str,
    *,
    segments: int = 8,
    min_radius: float = 0.1,
    smooth_subdivisions: int = 3,
    mesh_quality: float = 1.0,
) -> MuDMFeature:
    """Parse an SWC file and return a MuDMFeature with TIN geometry.

    Args:
        swc_path: Path to the SWC file.
        segments: Cross-section polygon sides for tubes.
        min_radius: Minimum radius clamp.
        smooth_subdivisions: Catmull-Rom subdivisions per segment (0 = off).
        mesh_quality: Fraction of smoothed path points to keep (0.0-1.0).

    Returns:
        A MuDMFeature wrapping a TIN geometry.
    """
    morphology = _parse_swc(swc_path)

    verts, _, indices = neuron_to_tube_mesh(
        morphology,
        segments=segments,
        min_radius=min_radius,
        smooth_subdivisions=smooth_subdivisions,
        mesh_quality=mesh_quality,
    )

    tin = _mesh_to_tin(verts, indices)
    return MuDMFeature(type="Feature", geometry=tin, properties={})


def microjson_to_swc(morphology: NeuronMorphology) -> str:
    """Convert a NeuronMorphology to SWC text.

    Args:
        morphology: A NeuronMorphology instance (from _parse_swc or similar).

    Returns:
        SWC-formatted string.
    """
    lines = ["# SWC file generated by MuDM"]
    lines.append("# id type x y z r parent")
    for s in morphology.tree:
        lines.append(f"{s.id} {s.type} {s.x} {s.y} {s.z} {s.r} {s.parent}")
    return "\n".join(lines) + "\n"


def swc_to_linestring3d(swc_path: str) -> MultiLineString:
    """Convert an SWC file to a MultiLineString for skeleton rendering.

    Each edge in the SWC tree becomes a 2-point 3D line segment.
    """
    morphology = _parse_swc(swc_path)
    by_id = {s.id: s for s in morphology.tree}

    lines: list[list[list[float]]] = []
    for s in morphology.tree:
        if s.parent == -1:
            continue
        parent = by_id[s.parent]
        lines.append([
            [parent.x, parent.y, parent.z],
            [s.x, s.y, s.z],
        ])

    return MultiLineString(type="MultiLineString", coordinates=lines)


def _neuron_name_from_path(swc_path: str) -> str:
    """Derive a neuron name from the file path.

    Strips common NeuroMorpho suffixes like ``.CNG.swc`` and falls back
    to the bare stem.
    """
    stem = Path(swc_path).stem
    # NeuroMorpho convention: name.CNG.swc
    stem = re.sub(r"\.CNG$", "", stem, flags=re.IGNORECASE)
    return stem


def swc_to_feature_collection(
    swc_path: str,
    *,
    name: str | None = None,
    segments: int = 8,
    min_radius: float = 0.1,
    smooth_subdivisions: int = 3,
    mesh_quality: float = 1.0,
) -> MuDMFeatureCollection:
    """Parse an SWC file and return one TIN Feature per compartment type.

    Each feature has ``properties={"compartment": "<name>"}`` where the name
    is derived from the SWC type code (e.g. "soma", "axon", "basal_dendrite").

    The collection itself carries ``properties={"name": "<neuron>"}`` where
    the neuron name is derived from the filename (stripping ``.CNG.swc``
    NeuroMorpho suffixes) or set explicitly via *name*.

    Args:
        swc_path: Path to the SWC file.
        name: Neuron name for collection properties.  Derived from
            the filename when ``None``.
        segments: Cross-section polygon sides for tubes.
        min_radius: Minimum radius clamp.
        smooth_subdivisions: Catmull-Rom subdivisions per segment (0 = off).
        mesh_quality: Fraction of smoothed path points to keep (0.0-1.0).

    Returns:
        A MuDMFeatureCollection with one Feature per SWC compartment type.
    """
    neuron_name = name if name is not None else _neuron_name_from_path(swc_path)

    morphology = _parse_swc(swc_path)
    typed_meshes = neuron_to_typed_meshes(
        morphology,
        segments=segments,
        min_radius=min_radius,
        smooth_subdivisions=smooth_subdivisions,
        mesh_quality=mesh_quality,
    )

    features: list[MuDMFeature] = []
    for swc_type, (verts, _normals, indices) in sorted(typed_meshes.items()):
        compartment = SWC_TYPE_NAMES.get(swc_type, f"type_{swc_type}")
        tin = _mesh_to_tin(verts, indices)
        features.append(
            MuDMFeature(
                type="Feature",
                geometry=tin,
                properties={"compartment": compartment},
                featureClass=compartment,
            )
        )

    return MuDMFeatureCollection(
        type="FeatureCollection",
        features=features,
        properties={"name": neuron_name},
    )
