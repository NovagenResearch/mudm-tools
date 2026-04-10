"""Generate OGC 3D Tiles ``tileset.json`` from an octree.

Produces a hierarchical tileset descriptor with axis-aligned bounding
volumes and geometric error that decreases with zoom level.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .projector3d import CartesianProjector3D


def _box_volume(
    xmin: float, ymin: float, zmin: float,
    xmax: float, ymax: float, zmax: float,
) -> list[float]:
    """Build an OGC 3D Tiles oriented bounding box (12 floats).

    Format: [cx, cy, cz, xHalf_x, xHalf_y, xHalf_z,
             yHalf_x, yHalf_y, yHalf_z, zHalf_x, zHalf_y, zHalf_z]
    For axis-aligned: [cx, cy, cz, halfX, 0, 0, 0, halfY, 0, 0, 0, halfZ]
    """
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    cz = (zmin + zmax) / 2
    hx = (xmax - xmin) / 2
    hy = (ymax - ymin) / 2
    hz = (zmax - zmin) / 2
    return [cx, cy, cz, hx, 0, 0, 0, hy, 0, 0, 0, hz]


def _geometric_error(
    world_bounds: tuple[float, float, float, float, float, float],
    zoom: int,
    max_zoom: int,
) -> float:
    """Compute geometric error for a zoom level.

    At max_zoom the error is 0. At each higher level the error
    doubles, representing the spatial resolution loss.
    """
    if zoom >= max_zoom:
        return 0.0
    dx = world_bounds[3] - world_bounds[0]
    dy = world_bounds[4] - world_bounds[1]
    dz = world_bounds[5] - world_bounds[2]
    diagonal = (dx**2 + dy**2 + dz**2) ** 0.5
    # Error at max_zoom is 0; each zoom level up doubles the error
    return diagonal / (1 << max_zoom) * (1 << (max_zoom - zoom))


def _tile_bounds_world(
    z: int, x: int, y: int, d: int,
    proj: CartesianProjector3D,
) -> tuple[float, float, float, float, float, float]:
    """Compute world-space bounding box for a tile address."""
    n = 1 << z
    xmin_n = x / n
    ymin_n = y / n
    zmin_n = d / n
    xmax_n = (x + 1) / n
    ymax_n = (y + 1) / n
    zmax_n = (d + 1) / n
    wxmin, wymin, wzmin = proj.unproject(xmin_n, ymin_n, zmin_n)
    wxmax, wymax, wzmax = proj.unproject(xmax_n, ymax_n, zmax_n)
    return (wxmin, wymin, wzmin, wxmax, wymax, wzmax)


def generate_tileset_json(
    all_tiles: dict[tuple[int, int, int, int], dict],
    world_bounds: tuple[float, float, float, float, float, float],
    proj: CartesianProjector3D,
    min_zoom: int = 0,
    max_zoom: int = 4,
) -> dict[str, Any]:
    """Build an OGC 3D Tiles 1.1 tileset.json structure.

    Parameters
    ----------
    all_tiles : dict
        Tile dict keyed by (z, x, y, d) from the octree.
    world_bounds : tuple
        (xmin, ymin, zmin, xmax, ymax, zmax) in world coordinates.
    proj : CartesianProjector3D
        Projector for coordinate conversion.
    min_zoom, max_zoom : int
        Zoom range.

    Returns
    -------
    dict
        tileset.json structure ready for json.dumps().
    """
    # Group tiles by zoom
    tiles_by_zoom: dict[int, list[tuple[int, int, int, int]]] = {}
    for key in all_tiles:
        z = key[0]
        if z >= min_zoom:
            tiles_by_zoom.setdefault(z, []).append(key)

    root_error = _geometric_error(world_bounds, min_zoom, max_zoom)
    root_box = _box_volume(*world_bounds)

    def _build_node(
        z: int, x: int, y: int, d: int,
    ) -> dict[str, Any] | None:
        """Recursively build a tileset node."""
        if (z, x, y, d) not in all_tiles:
            return None

        tb = _tile_bounds_world(z, x, y, d, proj)
        node: dict[str, Any] = {
            "boundingVolume": {"box": _box_volume(*tb)},
            "geometricError": _geometric_error(world_bounds, z, max_zoom),
            "content": {"uri": f"{z}/{x}/{y}/{d}.glb"},
        }

        # Find children at next zoom level
        if z < max_zoom:
            children: list[dict] = []
            nz = z + 1
            for cx in (x * 2, x * 2 + 1):
                for cy in (y * 2, y * 2 + 1):
                    for cd in (d * 2, d * 2 + 1):
                        child = _build_node(nz, cx, cy, cd)
                        if child is not None:
                            children.append(child)
            if children:
                node["children"] = children

        return node

    # Build root node(s)
    root_children: list[dict] = []
    for key in tiles_by_zoom.get(min_zoom, []):
        node = _build_node(*key)
        if node is not None:
            root_children.append(node)

    # If only one root tile, use it directly; otherwise wrap
    if len(root_children) == 1:
        root = root_children[0]
        root["refine"] = "REPLACE"
        root["geometricError"] = root_error
        root["boundingVolume"] = {"box": root_box}
    else:
        root = {
            "boundingVolume": {"box": root_box},
            "geometricError": root_error,
            "refine": "REPLACE",
            "children": root_children,
        }

    return {
        "asset": {"version": "1.1"},
        "geometricError": root_error,
        "root": root,
    }


def write_tileset_json(
    tileset: dict[str, Any],
    path: Path | str,
) -> None:
    """Write tileset.json to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tileset, indent=2))
