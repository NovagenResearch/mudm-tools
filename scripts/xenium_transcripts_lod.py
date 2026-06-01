#!/usr/bin/env python3
"""muDM point-LOD tiling for transcript point clouds (e.g. Xenium).

Builds a multi-resolution LOD pyramid for a transcript point cloud: at each
zoom level the points are spatially-stratified onto a quadtree grid and capped
at ``cap`` points per cell, so coarse zoom = spatially-uniform sparse sampling
and the finest zoom = (near) full density. This is the point-geometry analog of
QEM mesh LOD, and the substrate for density/resolution-robust ML on points.

Two modes: ``merge`` (default) emits one weighted point per (grid-cell, gene)
carrying a ``count`` -- the semantic point analog of mesh QEM LOD; ``random``
does uniform decimation (the ablation baseline). Output: one Parquet with
columns [zoom, tile_x, tile_y, x, y, gene_id, count, cell_id], queryable
per-zoom via predicate pushdown (a gene-id vocabulary is written alongside as
<output>.genes.json).

Usage::
    uv run python scripts/xenium_transcripts_lod.py \
        --transcripts data/xenium/transcripts.parquet \
        --output data/xenium/transcripts_lod.parquet \
        --base-cells 32 --max-zoom 4 --cap 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]


def _pick(cols, *cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c in low:
            return low[c]
    raise KeyError(f"none of {cands} in {cols}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--base-cells", type=int, default=32,
                    help="grid cells per axis at zoom 0")
    ap.add_argument("--max-zoom", type=int, default=4)
    ap.add_argument("--cap", type=int, default=4,
                    help="[random mode] max points kept per grid cell per zoom level")
    ap.add_argument("--mode", choices=["merge", "random"], default="merge",
                    help="merge = one weighted point per (grid-cell, gene) with a count "
                         "(semantic LOD); random = uniform decimation (ablation baseline)")
    ap.add_argument("--top-k", type=int, default=0,
                    help="[merge mode] keep only the top-k genes by count per grid cell "
                         "(0 = keep all)")
    ap.add_argument("--qv-min", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    t0 = time.perf_counter()
    pf = pq.ParquetFile(args.transcripts)
    names = pf.schema_arrow.names
    cx = _pick(names, "x_location", "x"); cy = _pick(names, "y_location", "y")
    cg = _pick(names, "feature_name", "gene"); cc = _pick(names, "cell_id", "cell")
    cqv = next((c for c in names if c.lower() in ("qv", "quality_value")), None)
    cols = [cx, cy, cg, cc] + ([cqv] if cqv else [])
    t = pf.read(columns=cols)
    x = np.asarray(t.column(cx), dtype=np.float64)
    y = np.asarray(t.column(cy), dtype=np.float64)
    gene = np.asarray(t.column(cg).cast("string").to_pylist(), dtype=object)
    cell = np.asarray(t.column(cc).cast("string").to_pylist(), dtype=object)
    keep = np.ones(len(x), dtype=bool)
    if cqv:
        keep &= np.asarray(t.column(cqv)) >= args.qv_min
    keep &= ~np.array([g.lower().startswith(("negcontrol", "blank", "antisense",
                                              "deprecated", "unassigned")) for g in gene])
    x, y, gene, cell = x[keep], y[keep], gene[keep], cell[keep]
    print(f"Loaded {len(x):,} transcripts (qv>={args.qv_min}, controls dropped) "
          f"in {time.perf_counter()-t0:.1f}s", flush=True)

    genes = sorted(set(gene.tolist()))
    gid_map = {g: i for i, g in enumerate(genes)}
    gid = np.array([gid_map[g] for g in gene], dtype=np.int32)
    n_genes = len(genes)

    xmin, xmax = x.min(), x.max(); ymin, ymax = y.min(), y.max()
    ext = max(xmax - xmin, ymax - ymin) or 1.0

    out = {k: [] for k in ("zoom", "tile_x", "tile_y", "x", "y",
                           "gene_id", "count", "cell_id")}
    for z in range(args.max_zoom + 1):
        ncells = args.base_cells * (2 ** z)
        cell_sz = ext / ncells
        tx = np.clip(((x - xmin) / cell_sz).astype(np.int64), 0, ncells - 1)
        ty = np.clip(((y - ymin) / cell_sz).astype(np.int64), 0, ncells - 1)
        if args.mode == "random":
            # uniform decimation: keep up to `cap` random points per grid cell
            key = tx * ncells + ty
            order = rng.permutation(len(key))
            srt = np.argsort(key[order], kind="stable")
            kk = key[order][srt]
            grp = np.r_[0, np.where(kk[1:] != kk[:-1])[0] + 1]
            rank = np.arange(len(kk)) - np.repeat(grp, np.diff(np.r_[grp, len(kk)]))
            sel = order[srt[rank < args.cap]]
            zx, zy = x[sel].astype(np.float32), y[sel].astype(np.float32)
            ztx, zty = tx[sel].astype(np.int32), ty[sel].astype(np.int32)
            zg, zcnt, zc = gid[sel], np.ones(len(sel), np.int32), cell[sel]
        else:
            # semantic merge: one weighted point per (grid-cell, gene), with count
            key = (tx * ncells + ty) * n_genes + gid
            order = np.argsort(key, kind="stable")
            ks = key[order]
            bnd = np.r_[0, np.where(ks[1:] != ks[:-1])[0] + 1]
            cnt = np.diff(np.r_[bnd, len(ks)]).astype(np.int32)
            zx = (np.add.reduceat(x[order], bnd) / cnt).astype(np.float32)
            zy = (np.add.reduceat(y[order], bnd) / cnt).astype(np.float32)
            zg = gid[order][bnd]
            ztx = tx[order][bnd].astype(np.int32)
            zty = ty[order][bnd].astype(np.int32)
            zcnt = cnt
            if args.top_k > 0:
                # keep only the top-k genes by count within each grid cell
                ck = ztx.astype(np.int64) * ncells + zty
                o2 = np.lexsort((-zcnt, ck))
                cks = ck[o2]
                cb = np.r_[0, np.where(cks[1:] != cks[:-1])[0] + 1]
                rnk = np.arange(len(cks)) - np.repeat(cb, np.diff(np.r_[cb, len(cks)]))
                keep = o2[rnk < args.top_k]
                zx, zy, zg = zx[keep], zy[keep], zg[keep]
                ztx, zty, zcnt = ztx[keep], zty[keep], zcnt[keep]
            zc = np.full(len(zx), "", dtype=object)
        n = len(zx)
        out["zoom"].append(np.full(n, z, dtype=np.int16))
        out["tile_x"].append(ztx); out["tile_y"].append(zty)
        out["x"].append(zx); out["y"].append(zy)
        out["gene_id"].append(zg); out["count"].append(zcnt.astype(np.int32))
        out["cell_id"].append(zc)
        print(f"  zoom {z}: grid {ncells}x{ncells}, {n:,} points from {len(x):,} "
              f"raw (mean count {float(zcnt.mean()):.1f})", flush=True)

    tbl = pa.table({
        "zoom": pa.array(np.concatenate(out["zoom"])),
        "tile_x": pa.array(np.concatenate(out["tile_x"])),
        "tile_y": pa.array(np.concatenate(out["tile_y"])),
        "x": pa.array(np.concatenate(out["x"])),
        "y": pa.array(np.concatenate(out["y"])),
        "gene_id": pa.array(np.concatenate(out["gene_id"])),
        "count": pa.array(np.concatenate(out["count"])),
        "cell_id": pa.array(np.concatenate([np.asarray(c, dtype=object)
                                            for c in out["cell_id"]])),
    })
    outp = Path(args.output); outp.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, outp, compression="zstd")
    Path(str(outp) + ".genes.json").write_text(json.dumps(gid_map))
    print(f"\nWrote {tbl.num_rows:,} rows -> {outp} "
          f"({outp.stat().st_size/1e6:.0f} MB) in {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
