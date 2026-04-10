#!/usr/bin/env python3
"""Validate 3D vector tiles with matplotlib 3D visualization.

Usage::

    # Generate random data → tile → plot
    .venv/bin/python scripts/validate_tiles3d.py --demo

    # Generate with specific zoom level
    .venv/bin/python scripts/validate_tiles3d.py --demo --zoom 2

    # Validate existing TileJSON
    .venv/bin/python scripts/validate_tiles3d.py tiles/tilejson3d.json -z 1

    # Save to file instead of displaying
    .venv/bin/python scripts/validate_tiles3d.py --demo --save out.png
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

# Ensure mudm is importable from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mudm_tools.tiling3d import TileGenerator3D, TileReader3D, OctreeConfig
from mudm_tools.tiling3d.projector3d import CartesianProjector3D


# ---------- wireframe cube drawing ----------


_CUBE_EDGES = [
    (0, 1), (1, 3), (3, 2), (2, 0),  # bottom face
    (4, 5), (5, 7), (7, 6), (6, 4),  # top face
    (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
]


def _cube_lines(
    xmin: float, ymin: float, zmin: float,
    xmax: float, ymax: float, zmax: float,
) -> list[list[tuple[float, float, float]]]:
    """Return 12 line segments for a wireframe cube."""
    corners = [
        (xmin, ymin, zmin), (xmax, ymin, zmin),
        (xmin, ymax, zmin), (xmax, ymax, zmin),
        (xmin, ymin, zmax), (xmax, ymin, zmax),
        (xmin, ymax, zmax), (xmax, ymax, zmax),
    ]
    return [[corners[a], corners[b]] for a, b in _CUBE_EDGES]


# ---------- feature → matplotlib primitives ----------

_GEOM_TYPE_POINT = 1
_GEOM_TYPE_LINE = 2


def _tile_feature_to_world(
    feat: dict,
    tx: int, ty: int, td: int,
    extent: int, extent_z: int,
    n: int,
    proj: CartesianProjector3D,
) -> list[tuple[float, float, float]]:
    """Convert a decoded tile feature to world coordinates (ring-based)."""
    xy = feat["xy"]
    z_vals = feat["z"]
    world = []
    for i, (ix, iy) in enumerate(xy):
        nx = (ix / extent + tx) / n
        ny = (iy / extent + ty) / n
        nz = (z_vals[i] / extent_z + td) / n if i < len(z_vals) else 0.0
        wx, wy, wz = proj.unproject(nx, ny, nz)
        world.append((wx, wy, wz))
    return world


def _mesh_feature_to_triangles(
    feat: dict,
    tx: int, ty: int, td: int,
    extent: int, extent_z: int,
    n: int,
    proj: CartesianProjector3D,
) -> list[list[tuple[float, float, float]]]:
    """Convert an indexed mesh feature to a list of world-space triangles.

    Unpacks mesh_positions (float32 LE xyz triples) and mesh_indices
    (uint32 LE triangle vertex indices) into individual triangles.
    """
    import struct

    mesh_pos = feat.get("mesh_positions", b"")
    mesh_idx = feat.get("mesh_indices", b"")
    if not mesh_pos:
        return []

    n_verts = len(mesh_pos) // 12  # 3 floats × 4 bytes
    floats = struct.unpack(f"<{n_verts * 3}f", mesh_pos)

    # Convert tile-local float coords → world coords
    world_verts: list[tuple[float, float, float]] = []
    for i in range(n_verts):
        ix, iy, iz = floats[i * 3], floats[i * 3 + 1], floats[i * 3 + 2]
        nx = (ix / extent + tx) / n
        ny = (iy / extent + ty) / n
        nz = (iz / extent_z + td) / n
        wx, wy, wz = proj.unproject(nx, ny, nz)
        world_verts.append((wx, wy, wz))

    # Unpack triangle indices
    if mesh_idx:
        n_indices = len(mesh_idx) // 4
        indices = struct.unpack(f"<{n_indices}I", mesh_idx)
    else:
        # No indices — sequential triangles
        indices = tuple(range(n_verts))

    # Build triangles (3 indices per triangle)
    triangles: list[list[tuple[float, float, float]]] = []
    for i in range(0, len(indices) - 2, 3):
        i0, i1, i2 = indices[i], indices[i + 1], indices[i + 2]
        if i0 < n_verts and i1 < n_verts and i2 < n_verts:
            triangles.append([world_verts[i0], world_verts[i1], world_verts[i2]])

    return triangles


# ---------- main plotting ----------

# Consistent colors per geometry type so features look the same across zooms
_TYPE_STYLE = {
    _GEOM_TYPE_POINT: {"color": "red", "label": "Point"},
    _GEOM_TYPE_LINE: {"color": "dodgerblue", "label": "Line"},
    3: {"color": "limegreen", "label": "Polygon"},  # POLYGON3D
    4: {"color": "orange", "label": "PolyhedralSurface"},
    5: {"color": "mediumorchid", "label": "TIN"},
}


def plot_tiles(
    reader: TileReader3D,
    zoom: int,
    save_path: str | None = None,
) -> None:
    """Plot all tiles at *zoom* with wireframe octree grid + feature geometry."""
    meta = reader.metadata
    bounds3d = meta.get("bounds3d", [0, 0, 0, 1, 1, 1])
    proj = CartesianProjector3D(tuple(bounds3d))

    tiles = reader.tiles_at_zoom(zoom)
    n = 1 << zoom

    if not tiles:
        print(f"No tiles at zoom {zoom}")
        return

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Draw octree wireframe cubes — lighter at higher zoom
    grid_alpha = max(0.08, 0.3 / (zoom + 1))
    grid_lw = max(0.2, 0.5 / (zoom + 1))
    all_cube_lines: list[list[tuple[float, float, float]]] = []
    for tz, tx, ty, td, layers in tiles:
        nxmin, nymin, nzmin = tx / n, ty / n, td / n
        nxmax, nymax, nzmax = (tx + 1) / n, (ty + 1) / n, (td + 1) / n
        wxmin, wymin, wzmin = proj.unproject(nxmin, nymin, nzmin)
        wxmax, wymax, wzmax = proj.unproject(nxmax, nymax, nzmax)
        all_cube_lines.extend(
            _cube_lines(wxmin, wymin, wzmin, wxmax, wymax, wzmax)
        )

    if all_cube_lines:
        ax.add_collection3d(Line3DCollection(
            all_cube_lines, colors="lightgray", linewidths=grid_lw,
            alpha=grid_alpha,
        ))

    # Draw features — colored by geometry type for consistency across zooms
    total_feats = 0
    total_tris = 0
    legend_added: set[int] = set()
    for tile_idx, (tz, tx, ty, td, layers) in enumerate(tiles):
        for layer in layers:
            extent = layer.get("extent", 4096)
            extent_z = layer.get("extent_z", 4096)
            for feat in layer.get("features", []):
                gtype = feat.get("type", 0)
                style = _TYPE_STYLE.get(gtype, {"color": "gray", "label": "?"})
                color = style["color"]
                lbl = style["label"] if gtype not in legend_added else None

                has_mesh = bool(feat.get("mesh_positions", b""))

                if has_mesh and gtype in (4, 5):
                    # Indexed mesh (TIN / PolyhedralSurface) — render individual triangles
                    triangles = _mesh_feature_to_triangles(
                        feat, tx, ty, td, extent, extent_z, n, proj,
                    )
                    if not triangles:
                        continue
                    total_feats += 1
                    total_tris += len(triangles)

                    poly = Poly3DCollection(
                        triangles, alpha=0.35,
                        facecolor=color, edgecolor=color,
                        linewidths=0.15,
                    )
                    ax.add_collection3d(poly)
                    if lbl:
                        ax.scatter([], [], [], c=color, label=lbl)
                    legend_added.add(gtype)
                else:
                    # Ring-based features (Point, Line, Polygon)
                    world = _tile_feature_to_world(
                        feat, tx, ty, td, extent, extent_z, n, proj,
                    )
                    if not world:
                        continue
                    total_feats += 1

                    xs = [p[0] for p in world]
                    ys = [p[1] for p in world]
                    zs = [p[2] for p in world]

                    if gtype == _GEOM_TYPE_POINT:
                        ax.scatter(
                            xs, ys, zs, c=color, s=40,
                            depthshade=True, label=lbl, zorder=5,
                        )
                    elif gtype == _GEOM_TYPE_LINE:
                        ax.plot(
                            xs, ys, zs, color=color, linewidth=1.8,
                            alpha=0.85, label=lbl,
                        )
                    else:
                        if len(world) >= 3:
                            verts = [list(world)]
                            poly = Poly3DCollection(
                                verts, alpha=0.4,
                                facecolor=color, edgecolor=color,
                                linewidths=0.5,
                            )
                            ax.add_collection3d(poly)
                            if lbl:
                                ax.scatter([], [], [], c=color, label=lbl)

                    legend_added.add(gtype)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper left", fontsize=9)
    tri_info = f", {total_tris:,} triangles" if total_tris else ""
    ax.set_title(
        f"3D Tiles — zoom {zoom}  |  "
        f"{len(tiles)} tiles, {total_feats} features{tri_info}"
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


# ---------- demo mode ----------


def run_demo(zoom: int, save_path: str | None = None) -> None:
    """Generate random 3D data, tile it, then plot."""
    from mudm_tools.polygen3d import generate_3d_collection

    print("Generating random 3D collection …")
    coll = generate_3d_collection(
        n_tins=15, n_points=10, n_lines=8,
        bounds=(0, 0, 0, 100, 100, 100),
        triangles_per_tin=6,
        seed=42,
    )
    print(f"  {len(coll.features)} features")

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        gen = TileGenerator3D(OctreeConfig(max_zoom=zoom))
        gen.add_features(coll)
        n_tiles = gen.generate(out)
        tj_path = out / "tilejson3d.json"
        gen.write_tilejson(tj_path)
        print(f"  {n_tiles} tiles at max_zoom={zoom}")

        reader = TileReader3D(tj_path)
        plot_tiles(reader, zoom, save_path=save_path)


# ---------- CLI ----------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate 3D vector tiles with matplotlib visualization",
    )
    parser.add_argument(
        "tilejson", nargs="?", default=None,
        help="Path to tilejson3d.json (omit for --demo mode)",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Generate random data, tile, and plot",
    )
    parser.add_argument(
        "-z", "--zoom", type=int, default=1,
        help="Zoom level to visualize (default: 1)",
    )
    parser.add_argument(
        "--save", default=None,
        help="Save plot to file instead of displaying",
    )
    args = parser.parse_args()

    if args.demo:
        run_demo(args.zoom, args.save)
    elif args.tilejson:
        reader = TileReader3D(args.tilejson)
        plot_tiles(reader, args.zoom, save_path=args.save)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
