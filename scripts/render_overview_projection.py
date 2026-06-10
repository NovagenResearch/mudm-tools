#!/usr/bin/env python3
"""Render static XY / XZ / YZ orthographic projection posters for a 3D pyramid's
**overview panel**.

This is the scalable replacement for the OverviewPanel's live z0-GLB render: the
overview shows three pre-rendered images instead of loading + re-rendering all
neuron geometry, so its cost is constant (3 PNGs) whether the dataset has 250 or
250,000 neurons.

Like the catalogue thumbnail (`render_3d_thumb.py`) it vertex-scatters the OBJ
meshes coloured by `features.json`, BUT each plane is framed **exactly to the
octree-root bounds from `tileset.json`** with axes that fill the figure (no
equal-aspect letterboxing). That makes image<->world linear, so the viewer can
drop each poster onto a world-space quad spanning the bounds and the crosshair /
click-navigation stays pixel-aligned.

Output (under <pyramid_dir>/overview/):
    xy.png  xz.png  yz.png        # one per projection plane
    overview.json                 # {planes, bounds:[xmin,ymin,zmin,xmax,ymax,zmax], image_px}

Usage:
    render_overview_projection.py <meshes_dir> <pyramid_dir> [--max-neurons N] [--cap-verts N] [--target-px N]

<pyramid_dir> must contain tileset.json + features.json. OBJ stems must match a
feature body_id (connectomes) or feature id (HRA organs).

Orientation contract (must match viewer js/OverviewPanel.js AXIS_CFG):
    plane  horizontal(h)  vertical(v)   image row0(top)=v_max, col0(left)=h_min
    xy     x              y
    xz     x              z
    yz     y              z
Axes are NOT inverted (world orientation; top = max). Three.js texture flipY
(default true) then puts image-top at the quad's +v edge, matching each panel
camera's up vector (+y for xy, +z for xz/yz).
"""
import argparse
import glob
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BG = "#080810"

# (horizontal-axis-index, vertical-axis-index) per plane. 0=x 1=y 2=z.
PLANES = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}


def _read_one(args):
    """Read + subsample vertices of one OBJ. Returns (Nx3 float32, color)."""
    path, color, cap = args
    xs = []
    with open(path) as f:
        for line in f:
            if line[:2] == "v ":
                p = line.split()
                xs.append((float(p[1]), float(p[2]), float(p[3])))
            elif line[:2] == "f ":
                break
    v = np.asarray(xs, dtype=np.float32)
    if cap and len(v) > cap:
        v = v[np.random.default_rng(0).choice(len(v), cap, replace=False)]
    return v, color


def _bounds_from_tileset(pyramid_dir):
    """Read the octree-root boundingVolume.box -> [xmin,ymin,zmin,xmax,ymax,zmax].

    3D Tiles `box` = [cx,cy,cz, hx,0,0, 0,hy,0, 0,0,hz] (center + half-axis vectors).
    """
    ts = json.load(open(Path(pyramid_dir) / "tileset.json"))
    box = ts["root"]["boundingVolume"]["box"]
    cx, cy, cz = box[0:3]
    hx, hy, hz = abs(box[3]), abs(box[7]), abs(box[11])
    return [cx - hx, cy - hy, cz - hz, cx + hx, cy + hy, cz + hz]


def _load_colors(pyramid_dir):
    feats = json.load(open(Path(pyramid_dir) / "features.json")).get("features", [])
    col = {}
    for f in feats:
        c = f["properties"].get("color", "#888888")
        bid = f["properties"].get("body_id")
        if bid is not None:
            col[str(bid)] = c
        if f.get("id"):
            col[str(f["id"])] = c
    return col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("meshes_dir")
    ap.add_argument("pyramid_dir")
    ap.add_argument("--max-neurons", type=int, default=8000,
                    help="sample at most this many neurons (scales to huge datasets)")
    ap.add_argument("--cap-verts", type=int, default=1500,
                    help="subsample at most this many vertices per neuron")
    ap.add_argument("--target-px", type=int, default=1024,
                    help="longest poster edge in pixels")
    args = ap.parse_args()

    pyramid_dir = Path(args.pyramid_dir)
    out_dir = pyramid_dir / "overview"
    out_dir.mkdir(parents=True, exist_ok=True)

    bounds = _bounds_from_tileset(pyramid_dir)
    col = _load_colors(pyramid_dir)

    objs = [p for p in sorted(glob.glob(args.meshes_dir + "/*.obj")) if Path(p).stem in col]
    if not objs:
        objs = sorted(glob.glob(args.meshes_dir + "/*.obj"))
    if len(objs) > args.max_neurons:
        objs = random.Random(0).sample(objs, args.max_neurons)

    tasks = [(p, col.get(Path(p).stem, "#888888"), args.cap_verts) for p in objs]
    pts, cols = [], []
    with ProcessPoolExecutor() as ex:
        for v, color in ex.map(_read_one, tasks, chunksize=8):
            if len(v):
                pts.append(v)
                cols.extend([color] * len(v))
    P = np.vstack(pts)
    cols = np.array(cols)
    order = np.random.default_rng(0).permutation(len(P))  # avoid draw-order colour bias
    P, cols = P[order], cols[order]
    print(f"  {len(objs)} neurons, {len(P):,} points; bounds={[round(b, 3) for b in bounds]}")

    xmin, ymin, zmin, xmax, ymax, zmax = bounds
    lims = {0: (xmin, xmax), 1: (ymin, ymax), 2: (zmin, zmax)}
    image_px = {}
    for plane, (h, v) in PLANES.items():
        hmin, hmax = lims[h]
        vmin, vmax = lims[v]
        ext_h = max(hmax - hmin, 1e-6)
        ext_v = max(vmax - vmin, 1e-6)
        # Figure sized to the bounds aspect so world->pixel is linear + isotropic.
        scale = args.target_px / max(ext_h, ext_v)
        w_px, h_px = max(1, round(ext_h * scale)), max(1, round(ext_v * scale))
        dpi = 100
        fig = plt.figure(figsize=(w_px / dpi, h_px / dpi), facecolor=BG)
        ax = fig.add_axes([0, 0, 1, 1])  # axes fill the whole figure
        ax.set_facecolor(BG)
        ax.scatter(P[:, h], P[:, v], c=cols, s=2.0, alpha=0.5,
                   marker=".", linewidths=0, rasterized=True)
        ax.set_xlim(hmin, hmax)   # left=h_min, right=h_max
        ax.set_ylim(vmin, vmax)   # bottom=v_min, top=v_max  (NOT inverted)
        ax.axis("off")
        out = out_dir / f"{plane}.png"
        fig.savefig(out, dpi=dpi, facecolor=BG)
        plt.close(fig)
        image_px[plane] = [w_px, h_px]
        print(f"  wrote {out} ({w_px}x{h_px})")

    meta = {
        "planes": {p: f"{p}.png" for p in PLANES},
        "bounds": [round(b, 3) for b in bounds],
        "image_px": image_px,
    }
    (out_dir / "overview.json").write_text(json.dumps(meta, indent=2))
    print(f"  wrote {out_dir / 'overview.json'}")


if __name__ == "__main__":
    main()
