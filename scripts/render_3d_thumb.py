#!/usr/bin/env python3
"""Render an XY + XZ projection thumbnail for a 3D pyramid (neuron/mesh dataset).

Reads per-feature colors from features.json and vertices from the OBJ meshes
(in parallel), projects to XY (top) and XZ (side) orthographic views side by
side, and saves a dark-background PNG matching the viewer's per-neuron coloring.

    render_3d_thumb.py <meshes_dir> <features.json> <out.png> [max_neurons]
"""
import sys
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
CAP_VERTS = 1500  # subsample per neuron


def _read_one(args):
    path, color = args
    xs = []
    with open(path) as f:
        for line in f:
            if line[:2] == "v ":
                p = line.split()
                xs.append((float(p[1]), float(p[2]), float(p[3])))
            elif line[:2] == "f ":
                break
    v = np.asarray(xs, dtype=np.float32)
    if len(v) > CAP_VERTS:
        v = v[np.random.default_rng(0).choice(len(v), CAP_VERTS, replace=False)]
    return v, color


def main():
    meshes_dir, features_json, out_png = sys.argv[1:4]
    max_neurons = int(sys.argv[4]) if len(sys.argv) > 4 else 1000

    feats = json.load(open(features_json)).get("features", [])
    # key colors by body_id (connectomes) AND feature id (HRA organs = OBJ stem)
    col = {}
    for f in feats:
        c = f["properties"].get("color", "#888888")
        bid = f["properties"].get("body_id")
        if bid is not None:
            col[str(bid)] = c
        if f.get("id"):
            col[str(f["id"])] = c

    objs = [p for p in sorted(glob.glob(meshes_dir + "/*.obj")) if Path(p).stem in col]
    if not objs:
        objs = sorted(glob.glob(meshes_dir + "/*.obj"))
    if len(objs) > max_neurons:
        objs = random.Random(0).sample(objs, max_neurons)

    tasks = [(p, col.get(Path(p).stem, "#888888")) for p in objs]
    pts, cols = [], []
    with ProcessPoolExecutor() as ex:
        for v, color in ex.map(_read_one, tasks, chunksize=8):
            if len(v):
                pts.append(v)
                cols.extend([color] * len(v))
    P = np.vstack(pts)
    cols = np.array(cols)
    print(f"  {len(objs)} neurons, {len(P):,} points")

    fig, axs = plt.subplots(1, 2, figsize=(8, 4), facecolor=BG)
    for ax, (i, j) in zip(axs, ((0, 1), (0, 2))):  # XY, XZ
        order = np.random.default_rng(0).permutation(len(P))  # avoid draw-order color bias
        ax.scatter(P[order, i], P[order, j], c=cols[order], s=1.1, alpha=0.6,
                   marker=".", linewidths=0, rasterized=True)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_facecolor(BG)
        ax.invert_yaxis()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.015)
    fig.savefig(out_png, dpi=110, facecolor=BG)
    print(f"  wrote {out_png}")


if __name__ == "__main__":
    main()
