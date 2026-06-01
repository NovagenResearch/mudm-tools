"""Assemble MuDM geometries into a glTF scene graph.

Dispatches each geometry type to the appropriate mesh/primitive builder
and wires them into pygltflib's GLTF2 object.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pygltflib import (
    ARRAY_BUFFER,
    ELEMENT_ARRAY_BUFFER,
    FLOAT,
    LINES,
    POINTS,
    TRIANGLES,
    UNSIGNED_INT,
    Asset,
    Attributes,
    GLTF2,
    Material,
    Mesh,
    Node,
    PbrMetallicRoughness,
    Primitive,
    Scene,
)

from geojson_pydantic import (
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
    Polygon,
    MultiPolygon,
)

from mudm.model import (
    MuDMFeature,
    MuDMFeatureCollection,
    PolyhedralSurface,
    TIN,
)
from mudm.layout import apply_layout
from ._buffers import create_accessor
from .models import GltfConfig
from .triangulator import multipolygon_to_mesh, polygon_to_mesh

# Z-up → Y-up rotation matrix (rotate -90° around X)
_Z_UP_TO_Y_UP = np.array(
    [1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1],
    dtype=np.float64,
).reshape(4, 4)


def _apply_y_up(positions: np.ndarray) -> np.ndarray:
    """Rotate Nx3 positions from Z-up to Y-up (swap Y↔Z, negate new Z)."""
    out = np.empty_like(positions)
    out[:, 0] = positions[:, 0]
    out[:, 1] = positions[:, 2]
    out[:, 2] = -positions[:, 1]
    return out


def _init_gltf(
    config: GltfConfig,
) -> tuple[GLTF2, dict[str, int]]:
    """Create a fresh GLTF2 object with materials.

    Returns:
        A ``(gltf, value_to_material)`` tuple.  ``value_to_material`` maps
        ``color_map`` keys to material indices when ``color_map`` is set;
        otherwise it is empty.  Material 0 is always the default color.
    """
    gltf = GLTF2()
    gltf._glb_data = None
    gltf.asset = Asset(version="2.0", generator="mudm-gltf")
    gltf.scenes = [Scene(nodes=[])]
    gltf.scene = 0

    def _make_material(rgba: tuple[float, float, float, float]) -> Material:
        r, g, b, a = rgba
        return Material(
            pbrMetallicRoughness=PbrMetallicRoughness(
                baseColorFactor=[r, g, b, a],
                metallicFactor=0.1,
                roughnessFactor=0.8,
            ),
            doubleSided=True,
        )

    gltf.materials = [_make_material(config.default_color)]

    value_to_material: dict[str, int] = {}
    if config.color_map:
        for value, rgba in config.color_map.items():
            idx = len(gltf.materials)
            gltf.materials.append(_make_material(rgba))
            value_to_material[value] = idx

    if config.draco:
        from ._draco import ensure_draco_extensions

        ensure_draco_extensions(gltf)

    return gltf, value_to_material


def _add_triangle_mesh(
    gltf: GLTF2,
    vertices: np.ndarray,
    indices: np.ndarray,
    normals: np.ndarray | None,
    config: GltfConfig,
    material_idx: int = 0,
) -> int:
    """Add a triangle mesh and return the mesh index."""
    if config.y_up:
        vertices = _apply_y_up(vertices)
        if normals is not None:
            normals = _apply_y_up(normals)

    if config.draco:
        from ._draco import add_draco_triangle_mesh

        return add_draco_triangle_mesh(
            gltf,
            vertices,
            indices,
            normals,
            config,
            material_idx,
        )

    verts_f32 = vertices.astype(np.float32)
    pos_acc = create_accessor(gltf, verts_f32, FLOAT, "VEC3", ARRAY_BUFFER)

    attrs = Attributes(POSITION=pos_acc)
    if normals is not None:
        norms_f32 = normals.astype(np.float32)
        norm_acc = create_accessor(gltf, norms_f32, FLOAT, "VEC3", ARRAY_BUFFER)
        attrs.NORMAL = norm_acc

    idx_flat = indices.ravel().astype(np.uint32)
    idx_acc = create_accessor(gltf, idx_flat, UNSIGNED_INT, "SCALAR", ELEMENT_ARRAY_BUFFER)

    mesh_idx = len(gltf.meshes)
    gltf.meshes.append(
        Mesh(
            primitives=[
                Primitive(attributes=attrs, indices=idx_acc, material=material_idx, mode=TRIANGLES)
            ]
        )
    )
    return mesh_idx


def _add_line_primitive(
    gltf: GLTF2,
    vertices: np.ndarray,
    indices: np.ndarray,
    config: GltfConfig,
    material_idx: int = 0,
) -> int:
    """Add a line primitive and return the mesh index."""
    if config.y_up:
        vertices = _apply_y_up(vertices)

    verts_f32 = vertices.astype(np.float32)
    pos_acc = create_accessor(gltf, verts_f32, FLOAT, "VEC3", ARRAY_BUFFER)
    idx_flat = indices.ravel().astype(np.uint32)
    idx_acc = create_accessor(gltf, idx_flat, UNSIGNED_INT, "SCALAR", ELEMENT_ARRAY_BUFFER)

    mesh_idx = len(gltf.meshes)
    gltf.meshes.append(
        Mesh(
            primitives=[
                Primitive(
                    attributes=Attributes(POSITION=pos_acc),
                    indices=idx_acc,
                    material=material_idx,
                    mode=LINES,
                )
            ]
        )
    )
    return mesh_idx


def _add_point_primitive(
    gltf: GLTF2,
    vertices: np.ndarray,
    config: GltfConfig,
    material_idx: int = 0,
) -> int:
    """Add a point primitive and return the mesh index."""
    if config.y_up:
        vertices = _apply_y_up(vertices)

    verts_f32 = vertices.astype(np.float32)
    pos_acc = create_accessor(gltf, verts_f32, FLOAT, "VEC3", ARRAY_BUFFER)

    mesh_idx = len(gltf.meshes)
    gltf.meshes.append(
        Mesh(
            primitives=[
                Primitive(
                    attributes=Attributes(POSITION=pos_acc), material=material_idx, mode=POINTS
                )
            ]
        )
    )
    return mesh_idx


def _add_node(
    gltf: GLTF2,
    mesh_idx: int,
    name: str | None = None,
    extras: dict | None = None,
) -> int:
    """Add a node referencing a mesh and return its index."""
    node = Node(mesh=mesh_idx)
    if name:
        node.name = name
    if extras:
        node.extras = extras
    node_idx = len(gltf.nodes)
    gltf.nodes.append(node)
    gltf.scenes[0].nodes.append(node_idx)
    return node_idx


# ---------------------------------------------------------------------------
# Geometry dispatchers
# ---------------------------------------------------------------------------


def _convert_tin(
    gltf: GLTF2,
    geom: TIN,
    config: GltfConfig,
    material_idx: int = 0,
) -> int:
    all_verts = []
    all_indices = []
    offset = 0
    for face_coords in geom.coordinates:
        ring = face_coords[0]  # TIN: exactly 1 ring per face, 4 positions (closed triangle)
        tri_verts = [list(p) for p in ring[:3]]  # drop closing vertex
        # Pad to 3D if needed
        for v in tri_verts:
            while len(v) < 3:
                v.append(0.0)
        all_verts.extend(tri_verts)
        all_indices.append([offset, offset + 1, offset + 2])
        offset += 3

    if not all_verts:
        return -1
    verts = np.array(all_verts, dtype=np.float64)
    indices = np.array(all_indices, dtype=np.uint32)
    return _add_triangle_mesh(gltf, verts, indices, None, config, material_idx)


def _convert_polyhedral_surface(
    gltf: GLTF2,
    geom: PolyhedralSurface,
    config: GltfConfig,
    material_idx: int = 0,
) -> int:
    all_verts: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []
    offset = 0

    for face_coords in geom.coordinates:
        # Each face is a list of rings (outer + optional holes)
        from shapely.geometry import Polygon as ShapelyPolygon

        outer = [(p[0], p[1]) for p in face_coords[0]]
        holes = [[(p[0], p[1]) for p in ring] for ring in face_coords[1:]]
        z = face_coords[0][0][2] if len(face_coords[0][0]) > 2 else 0.0

        poly = ShapelyPolygon(outer, holes)
        verts, idxs = polygon_to_mesh(poly, z=z)
        if verts.shape[0] == 0:
            continue

        # If original coords had 3D, use their Z values
        if len(face_coords[0][0]) > 2:
            # Map unique 2D positions back to 3D from original coords
            all_pts_3d = []
            for ring in face_coords:
                for p in ring:
                    all_pts_3d.append(list(p) + [0.0] * (3 - len(p)))
            # For each vertex in the triangulated output, find nearest original Z
            for i in range(verts.shape[0]):
                vx, vy = verts[i, 0], verts[i, 1]
                best_z = z
                best_dist = float("inf")
                for pt in all_pts_3d:
                    d = (pt[0] - vx) ** 2 + (pt[1] - vy) ** 2
                    if d < best_dist:
                        best_dist = d
                        best_z = pt[2]
                verts[i, 2] = best_z

        all_verts.append(verts)
        all_indices.append(idxs + offset)
        offset += verts.shape[0]

    if not all_verts:
        return -1
    return _add_triangle_mesh(
        gltf,
        np.concatenate(all_verts),
        np.concatenate(all_indices),
        None,
        config,
        material_idx,
    )


def _convert_polygon(
    gltf: GLTF2,
    geom: Polygon,
    config: GltfConfig,
    material_idx: int = 0,
) -> int:
    from shapely.geometry import Polygon as ShapelyPolygon

    coords = geom.coordinates
    outer = [(p[0], p[1]) for p in coords[0]]
    holes = [[(p[0], p[1]) for p in ring] for ring in coords[1:]]
    z = coords[0][0][2] if len(coords[0][0]) > 2 else 0.0

    poly = ShapelyPolygon(outer, holes)
    verts, indices = polygon_to_mesh(poly, z=z)
    if verts.shape[0] == 0:
        return -1
    return _add_triangle_mesh(gltf, verts, indices, None, config, material_idx)


def _convert_multipolygon(
    gltf: GLTF2,
    geom: MultiPolygon,
    config: GltfConfig,
    material_idx: int = 0,
) -> int:
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.geometry import MultiPolygon as ShapelyMultiPolygon

    polys = []
    for poly_coords in geom.coordinates:
        outer = [(p[0], p[1]) for p in poly_coords[0]]
        holes = [[(p[0], p[1]) for p in ring] for ring in poly_coords[1:]]
        polys.append(ShapelyPolygon(outer, holes))

    mp = ShapelyMultiPolygon(polys)
    z = geom.coordinates[0][0][0][2] if len(geom.coordinates[0][0][0]) > 2 else 0.0
    verts, indices = multipolygon_to_mesh(mp, z=z)
    if verts.shape[0] == 0:
        return -1
    return _add_triangle_mesh(gltf, verts, indices, None, config, material_idx)


def _convert_linestring(
    gltf: GLTF2,
    geom: LineString,
    config: GltfConfig,
    material_idx: int = 0,
) -> int:
    coords = geom.coordinates
    verts = []
    for p in coords:
        verts.append([p[0], p[1], p[2] if len(p) > 2 else 0.0])
    if len(verts) < 2:
        return -1
    verts_arr = np.array(verts, dtype=np.float64)

    # Build line segment indices (pairs of consecutive vertices)
    indices = []
    for i in range(len(verts) - 1):
        indices.extend([i, i + 1])
    indices_arr = np.array(indices, dtype=np.uint32)

    return _add_line_primitive(gltf, verts_arr, indices_arr, config, material_idx)


def _convert_multilinestring(
    gltf: GLTF2,
    geom: MultiLineString,
    config: GltfConfig,
    material_idx: int = 0,
) -> int:
    all_verts = []
    all_indices = []
    offset = 0
    for line_coords in geom.coordinates:
        verts = [[p[0], p[1], p[2] if len(p) > 2 else 0.0] for p in line_coords]
        if len(verts) < 2:
            continue
        for i in range(len(verts) - 1):
            all_indices.extend([offset + i, offset + i + 1])
        all_verts.extend(verts)
        offset += len(verts)

    if not all_verts:
        return -1
    return _add_line_primitive(
        gltf,
        np.array(all_verts, dtype=np.float64),
        np.array(all_indices, dtype=np.uint32),
        config,
        material_idx,
    )


def _convert_point(
    gltf: GLTF2,
    geom: Point,
    config: GltfConfig,
    material_idx: int = 0,
) -> int:
    c = geom.coordinates
    z = c[2] if len(c) > 2 else 0.0
    verts = np.array([[c[0], c[1], z]], dtype=np.float64)
    return _add_point_primitive(gltf, verts, config, material_idx)


def _convert_multipoint(
    gltf: GLTF2,
    geom: MultiPoint,
    config: GltfConfig,
    material_idx: int = 0,
) -> int:
    verts = []
    for c in geom.coordinates:
        z = c[2] if len(c) > 2 else 0.0
        verts.append([c[0], c[1], z])
    if not verts:
        return -1
    return _add_point_primitive(gltf, np.array(verts, dtype=np.float64), config, material_idx)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _convert_geometry(
    gltf: GLTF2,
    geom: Any,
    config: GltfConfig,
    material_idx: int = 0,
) -> int | list[int]:
    """Dispatch a geometry to its converter. Returns mesh index(es) or -1."""
    if isinstance(geom, TIN):
        return _convert_tin(gltf, geom, config, material_idx)
    elif isinstance(geom, PolyhedralSurface):
        return _convert_polyhedral_surface(gltf, geom, config, material_idx)
    elif isinstance(geom, Polygon):
        return _convert_polygon(gltf, geom, config, material_idx)
    elif isinstance(geom, MultiPolygon):
        return _convert_multipolygon(gltf, geom, config, material_idx)
    elif isinstance(geom, LineString):
        return _convert_linestring(gltf, geom, config, material_idx)
    elif isinstance(geom, MultiLineString):
        return _convert_multilinestring(gltf, geom, config, material_idx)
    elif isinstance(geom, Point):
        return _convert_point(gltf, geom, config, material_idx)
    elif isinstance(geom, MultiPoint):
        return _convert_multipoint(gltf, geom, config, material_idx)
    return -1


def _add_nodes_for_mesh(
    gltf: GLTF2,
    mesh_result: int | list[int],
    name: str | None = None,
    extras: dict | None = None,
) -> None:
    """Add node(s) for one or more meshes returned by a converter."""
    if isinstance(mesh_result, list):
        for j, midx in enumerate(mesh_result):
            node_name = f"{name}_{j}" if name else None
            _add_node(gltf, midx, name=node_name, extras=extras if j == 0 else None)
    elif mesh_result >= 0:
        _add_node(gltf, mesh_result, name=name, extras=extras)


def _resolve_material(
    feature: MuDMFeature,
    config: GltfConfig,
    value_to_material: dict[str, int],
) -> int:
    """Look up the material index for a feature based on color_by/color_map."""
    if config.color_by and value_to_material and feature.properties:
        value = feature.properties.get(config.color_by)
        if value is not None:
            return value_to_material.get(str(value), 0)
    return 0


def feature_to_gltf(
    feature: MuDMFeature,
    config: GltfConfig | None = None,
) -> GLTF2:
    """Convert a single MuDMFeature to a glTF scene.

    Args:
        feature: The MuDM feature to convert.
        config: Export configuration (uses defaults if None).

    Returns:
        A pygltflib GLTF2 object.
    """
    if config is None:
        config = GltfConfig()

    gltf, value_to_material = _init_gltf(config)

    geom = feature.geometry
    if geom is None:
        return gltf

    material_idx = _resolve_material(feature, config, value_to_material)
    mesh_result = _convert_geometry(gltf, geom, config, material_idx)

    extras = None
    if config.include_metadata and feature.properties:
        extras = dict(feature.properties)

    _add_nodes_for_mesh(gltf, mesh_result, extras=extras)
    return gltf


def collection_to_gltf(
    collection: MuDMFeatureCollection,
    config: GltfConfig | None = None,
) -> GLTF2:
    """Convert a MuDMFeatureCollection to a glTF scene.

    Layout (spacing/grid) is applied to the MuDM coordinates before
    mesh generation, so the glTF nodes contain no translation offsets.

    Args:
        collection: The feature collection.
        config: Export configuration.

    Returns:
        A pygltflib GLTF2 object.
    """
    if config is None:
        config = GltfConfig()

    gltf, value_to_material = _init_gltf(config)

    # Apply layout to geometry coordinates before mesh generation.
    # This raises ValueError early if grid capacity is exceeded.
    laid_out = apply_layout(
        collection,
        spacing=config.feature_spacing,
        grid_max_x=config.grid_max_x,
        grid_max_y=config.grid_max_y,
        grid_max_z=config.grid_max_z,
    )

    for i, feature in enumerate(laid_out.features):
        geom = feature.geometry
        if geom is None:
            continue

        material_idx = _resolve_material(feature, config, value_to_material)
        mesh_result = _convert_geometry(gltf, geom, config, material_idx)

        extras = None
        if config.include_metadata and feature.properties:
            extras = dict(feature.properties)

        _add_nodes_for_mesh(gltf, mesh_result, name=f"feature_{i}", extras=extras)

    # Store collection-level metadata in scene extras
    if config.include_metadata and collection.properties:
        gltf.scenes[0].extras = dict(collection.properties)

    return gltf
