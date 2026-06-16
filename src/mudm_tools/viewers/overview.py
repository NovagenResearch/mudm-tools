#!/usr/bin/env python3
"""Render static XY / XZ / YZ orthographic projection posters for a 3D pyramid's
**overview panel** — the scalable, constant-cost replacement for the viewer
loading + live-rendering all neuron geometry in the overview.

The viewer (``OverviewPanel.loadPosters``) drops each poster onto a world-space
quad spanning the octree-root bounds, so the world<->pixel mapping must be linear
and isotropic with a fixed orientation:

    plane  horizontal(h)  vertical(v)
    xy     x              y
    xz     x              z
    yz     y              z

Column 0 (left) = h_min, row 0 (top) = v_max — axes are NOT inverted (world
orientation; top = max). Three.js texture ``flipY`` (default true) then puts the
image top at the quad's +v edge, matching each overview camera's up vector.

Output (under ``<pyramid_dir>/overview/``):
    xy.png  xz.png  yz.png        one per projection plane
    overview.json                 {planes, bounds:[xmin,ymin,zmin,xmax,ymax,zmax], image_px}

OBJ stems must match a feature ``body_id`` or feature id from ``features.json``
(for per-feature colour); unmatched meshes fall back to a neutral grey.

Implemented with numpy + Pillow only (no matplotlib): each plane is a vectorised
point-splat into a uint8 buffer, optionally supersampled and Lanczos-downscaled
for anti-aliasing.
"""
from __future__ import annotations

import argparse
import json
import random
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

# (horizontal-axis-index, vertical-axis-index) per plane. 0=x 1=y 2=z.
PLANES = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
DEFAULT_COLOR = "#888888"
BG_RGB = np.array([8, 8, 16], dtype=np.uint8)  # "#080810"


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = (h or "").lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        return (136, 136, 136)
    try:
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    except ValueError:
        return (136, 136, 136)


def _read_one(args):
    """Read + subsample vertices of one OBJ. Returns (Nx3 float32, color_hex)."""
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
        # Per-path seed: deterministic but NOT correlated across neurons (a single
        # fixed seed would keep the same vertex-index pattern in every mesh).
        seed = zlib.crc32(str(path).encode()) & 0xFFFFFFFF
        v = v[np.random.default_rng(seed).choice(len(v), cap, replace=False)]
    return v, color


def _find(pyramid_dir: Path, name: str):
    """Locate a pyramid file that may live at the root OR under 3dtiles/ (layout varies)."""
    for cand in (Path(pyramid_dir) / name, Path(pyramid_dir) / "3dtiles" / name):
        if cand.is_file():
            return cand
    return None


def _load_bounds(pyramid_dir: Path) -> list[float]:
    """[xmin,ymin,zmin,xmax,ymax,zmax] = the octree-root box the viewer spans posters
    over. OverviewPanel._planeQuad spans tileManager.root.box3 (= the tileset root box),
    so the poster MUST be framed to that box; prefer tileset.json's root box and only
    fall back to tilejson3d's bounds3d (the data AABB, which can differ if the root is a cube).

    3D Tiles box = [cx,cy,cz, hx,0,0, 0,hy,0, 0,0,hz] (center + half-axis vectors).
    """
    ts = _find(pyramid_dir, "tileset.json")
    if ts:
        box = json.loads(ts.read_text())["root"]["boundingVolume"]["box"]
        cx, cy, cz = box[0:3]
        hx, hy, hz = abs(box[3]), abs(box[7]), abs(box[11])
        return [cx - hx, cy - hy, cz - hz, cx + hx, cy + hy, cz + hz]
    tj = _find(pyramid_dir, "tilejson3d.json")
    if tj:
        b = json.loads(tj.read_text()).get("bounds3d")
        if b and len(b) == 6:
            return [float(x) for x in b]
    raise FileNotFoundError(f"no tileset.json or tilejson3d.json under {pyramid_dir}")


def _load_colors(pyramid_dir: Path) -> dict[str, str]:
    """feature id / body_id -> colour, from features.json (dict OR MicroJSON-list)."""
    fj = _find(pyramid_dir, "features.json")
    if fj is None:
        raise FileNotFoundError(f"no features.json under {pyramid_dir}")
    data = json.loads(fj.read_text())
    feats = data.get("features", {})
    col: dict[str, str] = {}
    if isinstance(feats, dict):  # legacy: name -> properties (flywire uses this)
        for name, p in feats.items():
            p = p or {}
            c = p.get("color", DEFAULT_COLOR)
            col[str(name)] = c
            bid = p.get("body_id")
            if bid is not None:
                col[str(bid)] = c
    else:  # MicroJSON: [{id, properties}]
        for f in feats:
            p = f.get("properties", {}) or {}
            c = p.get("color", DEFAULT_COLOR)
            bid = p.get("body_id")
            if bid is not None:
                col[str(bid)] = c
            if f.get("id"):
                col[str(f["id"])] = c
    return col


def _rasterize(P, C, bounds, h, v, target_px, ss):
    """Splat coloured points onto one plane. Returns (HxWx3 uint8, [W, H])."""
    lims = {0: (bounds[0], bounds[3]), 1: (bounds[1], bounds[4]), 2: (bounds[2], bounds[5])}
    hmin, hmax = lims[h]
    vmin, vmax = lims[v]
    ext_h = max(hmax - hmin, 1e-6)
    ext_v = max(vmax - vmin, 1e-6)
    scale = target_px / max(ext_h, ext_v)        # isotropic world->pixel
    W = max(1, round(ext_h * scale))
    H = max(1, round(ext_v * scale))
    Ws, Hs = W * ss, H * ss

    buf = np.empty((Hs, Ws, 3), dtype=np.uint8)
    buf[:] = BG_RGB
    if len(P):
        cols = np.clip(np.floor((P[:, h] - hmin) / ext_h * (Ws - 1)).astype(np.int64), 0, Ws - 1)
        rows = np.clip(np.floor((vmax - P[:, v]) / ext_v * (Hs - 1)).astype(np.int64), 0, Hs - 1)
        buf[rows, cols] = C   # shuffled beforehand -> duplicate-pixel winner is unbiased

    if ss > 1:
        buf = np.asarray(Image.fromarray(buf, "RGB").resize((W, H), Image.Resampling.LANCZOS))
    return buf, [W, H]


def render_overview(meshes_dir, pyramid_dir, *, max_neurons=8000, cap_verts=1500,
                    target_px=1024, supersample=2, workers=None, bounds=None) -> dict:
    """Render the three overview posters + overview.json under <pyramid_dir>/overview/.

    Args:
        meshes_dir: directory of source ``*.obj`` meshes (stems = feature id/body_id).
        pyramid_dir: directory containing tileset.json (for bounds, unless ``bounds``
            is given) and features.json (for per-feature colour).
        bounds: optional [xmin,ymin,zmin,xmax,ymax,zmax]; if None, read from tileset.json.
        workers: process count for OBJ reads; <=1 reads serially (no subprocesses).
    Returns the written ``overview.json`` meta dict.
    """
    meshes_dir = Path(meshes_dir)
    pyramid_dir = Path(pyramid_dir)
    out_dir = pyramid_dir / "overview"
    out_dir.mkdir(parents=True, exist_ok=True)

    bounds = [float(b) for b in bounds] if bounds is not None else _load_bounds(pyramid_dir)
    col = _load_colors(pyramid_dir)

    objs = [p for p in sorted(meshes_dir.glob("*.obj")) if p.stem in col]
    if not objs:
        objs = sorted(meshes_dir.glob("*.obj"))
    if len(objs) > max_neurons:
        objs = random.Random(0).sample(objs, max_neurons)

    tasks = [(str(p), col.get(p.stem, DEFAULT_COLOR), cap_verts) for p in objs]
    if workers is not None and workers <= 1:
        results = [_read_one(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_read_one, tasks, chunksize=8))

    pts, cols = [], []
    for verts, hexc in results:
        if len(verts):
            pts.append(verts)
            cols.append(np.tile(_hex_to_rgb(hexc), (len(verts), 1)))
    if pts:
        P = np.vstack(pts).astype(np.float32)
        C = np.vstack(cols).astype(np.uint8)
        order = np.random.default_rng(0).permutation(len(P))   # avoid draw-order colour bias
        P, C = P[order], C[order]
    else:
        P = np.zeros((0, 3), np.float32)
        C = np.zeros((0, 3), np.uint8)

    image_px = {}
    for plane, (h, v) in PLANES.items():
        img, wh = _rasterize(P, C, bounds, h, v, target_px, supersample)
        Image.fromarray(img, "RGB").save(out_dir / f"{plane}.png")
        image_px[plane] = wh

    meta = {
        "planes": {p: f"{p}.png" for p in PLANES},
        "bounds": [round(b, 3) for b in bounds],
        "image_px": image_px,
    }
    (out_dir / "overview.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Render overview projection posters for a 3D pyramid")
    ap.add_argument("meshes_dir", help="directory of source *.obj meshes")
    ap.add_argument("pyramid_dir", help="pyramid dir (tileset.json + features.json)")
    ap.add_argument("--max-neurons", type=int, default=8000, help="sample at most this many neurons")
    ap.add_argument("--cap-verts", type=int, default=1500, help="subsample at most this many verts/neuron")
    ap.add_argument("--target-px", type=int, default=1024, help="longest poster edge in pixels")
    ap.add_argument("--supersample", type=int, default=2, help="render scale for anti-aliasing (1=off)")
    ap.add_argument("--workers", type=int, default=None, help="OBJ-read processes (<=1 = serial)")
    args = ap.parse_args()

    meta = render_overview(
        args.meshes_dir, args.pyramid_dir,
        max_neurons=args.max_neurons, cap_verts=args.cap_verts,
        target_px=args.target_px, supersample=args.supersample, workers=args.workers,
    )
    out = Path(args.pyramid_dir) / "overview"
    print(f"Wrote {out}/overview.json + {', '.join(meta['planes'].values())} "
          f"(image_px={meta['image_px']}, bounds={meta['bounds']})")


if __name__ == "__main__":
    main()
