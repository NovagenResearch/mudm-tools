#!/usr/bin/env python3
"""PROBE (go/no-go): does MULTI-SCALE (fine-center + coarse-surround) beat
single-scale input under a fixed budget, for Xenium cell typing?

Budget B = total points the model may ingest to classify one cell. Three input
policies, all spending the SAME B:
  * fine   : B raw transcripts from the cell itself (full detail, no context)
  * coarse : B merged+counted points from a wide window (context, identity blurred)
  * multi  : B/2 fine (the cell) + B/2 coarse (the surround)   <- muDM multi-LOD

Hypothesis (GO): `multi` dominates BOTH single-scale policies at low budget --
when the cell's own transcripts are too few to be decisive, the coarse
microenvironment disambiguates -- and the curves converge as B grows.

Labels: graphclust (intrinsic cell type). Circularity caveat noted; the relative
policy comparison is valid regardless. Coarse source = merge pyramid; fine source
= raw transcripts; gene ids are shared via <pyramid>.genes.json so the gene
embedding is common to both scales.

Usage::
    uv run python scripts/probe_xenium_multiscale.py \
        --transcripts ../data/xenium/transcripts.parquet \
        --pyramid ../data/xenium/transcripts_lod_merge.parquet \
        --clusters ../data/xenium/clusters.csv \
        --coarse-zoom 1 --radius 150 --budgets 16,64,256 \
        --max-cells 25000 --epochs 10 \
        --output ../results/xenium_multiscale_probe.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree
from torch.utils.data import DataLoader, Dataset

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]


def _pick(cols, *cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c in low:
            return low[c]
    raise KeyError(f"none of {cands} in {cols}")


def load(transcripts, pyramid, clusters, genes_json, coarse_zoom, qv_min,
         max_cells, min_tx, seed):
    gid_map = json.loads(Path(genes_json).read_text())  # gene_name -> id
    n_genes = len(gid_map)

    cl = {}
    with open(clusters) as f:
        r = csv.DictReader(f)
        bc, cc = r.fieldnames[0], r.fieldnames[1]
        for row in r:
            try:
                cl[row[bc]] = int(float(row[cc]))
            except (ValueError, TypeError):
                pass

    # --- raw transcripts -> per-cell own point clouds (the FINE source) ---
    pf = pq.ParquetFile(transcripts)
    names = pf.schema_arrow.names
    cx = _pick(names, "x_location", "x"); cy = _pick(names, "y_location", "y")
    cg = _pick(names, "feature_name", "gene"); cc_ = _pick(names, "cell_id", "cell")
    cqv = next((c for c in names if c.lower() in ("qv", "quality_value")), None)
    t = pf.read(columns=[cx, cy, cg, cc_] + ([cqv] if cqv else []))
    x = np.asarray(t.column(cx), float); y = np.asarray(t.column(cy), float)
    gene = np.asarray(t.column(cg).cast("string").to_pylist(), object)
    cell = np.asarray(t.column(cc_).cast("string").to_pylist(), object)
    keep = np.ones(len(x), bool)
    if cqv:
        keep &= np.asarray(t.column(cqv)) >= qv_min
    keep &= np.array([g in gid_map for g in gene])
    keep &= np.array([str(c) not in ("UNASSIGNED", "-1", "0", "None", "") for c in cell])
    x, y, gene, cell = x[keep], y[keep], gene[keep], cell[keep]
    gid = np.array([gid_map[g] for g in gene], np.int32)

    order = np.argsort(cell, kind="stable")
    cell, x, y, gid = cell[order], x[order], y[order], gid[order]
    bnds = np.where(cell[1:] != cell[:-1])[0] + 1
    fine = {}; cent = {}; lab = {}
    for s, e in zip(np.r_[0, bnds], np.r_[bnds, len(cell)]):
        cid = str(cell[s])
        if cid not in cl or (e - s) < min_tx:
            continue
        px, py = x[s:e].astype(np.float32), y[s:e].astype(np.float32)
        fine[cid] = (px, py, gid[s:e].copy())
        cent[cid] = (float(px.mean()), float(py.mean())); lab[cid] = cl[cid]

    ids = list(fine.keys())
    rng = np.random.RandomState(seed)
    if len(ids) > max_cells:
        ids = list(rng.choice(np.array(ids, dtype=object), max_cells, replace=False))

    # --- merge pyramid at coarse zoom (the COARSE source) ---
    pp = pq.ParquetFile(pyramid)
    tabp = pp.read(columns=["zoom", "x", "y", "gene_id", "count"])
    pz = np.asarray(tabp.column("zoom"))
    m = pz == coarse_zoom
    co = dict(x=np.asarray(tabp.column("x"), float)[m],
              y=np.asarray(tabp.column("y"), float)[m],
              g=np.asarray(tabp.column("gene_id"))[m].astype(np.int32),
              cnt=np.asarray(tabp.column("count"))[m].astype(np.float32))
    print(f"Loaded {len(ids)} cells; coarse zoom {coarse_zoom} = {m.sum():,} points",
          flush=True)
    return fine, cent, lab, ids, n_genes, co


def precompute_coarse_idx(ids, cent, co, radius):
    tree = KDTree(np.c_[co["x"], co["y"]])
    pts = np.array([cent[c] for c in ids], float)
    neigh = tree.query_radius(pts, r=radius)
    return {cid: neigh[i].astype(np.int64) for i, cid in enumerate(ids)}


class CellDS(Dataset):
    def __init__(self, ids, fine, cent, lab, co, cidx, radius, budget, policy):
        self.ids = ids; self.fine = fine; self.cent = cent; self.lab = lab
        self.co = co; self.cidx = cidx; self.R = radius
        self.B = budget; self.policy = policy

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _sample(arr, k):
        n = arr.shape[0]
        if n == 0:
            return np.zeros((k, 5), np.float32)
        idx = np.random.choice(n, k, replace=n < k)
        return arr[idx]

    def _fine(self, cid, cx, cy):
        fx, fy, fg = self.fine[cid]
        return np.c_[(fx - cx) / self.R, (fy - cy) / self.R, fg.astype(np.float32),
                     np.zeros(len(fx), np.float32), np.zeros(len(fx), np.float32)
                     ].astype(np.float32)

    def _coarse(self, cid, cx, cy):
        ci = self.cidx[cid]
        if len(ci) == 0:
            return np.zeros((0, 5), np.float32)
        return np.c_[(self.co["x"][ci] - cx) / self.R, (self.co["y"][ci] - cy) / self.R,
                     self.co["g"][ci].astype(np.float32),
                     np.log1p(self.co["cnt"][ci]), np.ones(len(ci), np.float32)
                     ].astype(np.float32)

    def __getitem__(self, i):
        cid = self.ids[i]; cx, cy = self.cent[cid]
        if self.policy == "fine":
            pts = self._sample(self._fine(cid, cx, cy), self.B)
        elif self.policy == "coarse":
            pts = self._sample(self._coarse(cid, cx, cy), self.B)
        else:
            h = self.B // 2
            pts = np.concatenate([self._sample(self._fine(cid, cx, cy), h),
                                  self._sample(self._coarse(cid, cx, cy), self.B - h)])
        return (torch.from_numpy(pts[:, :2]),
                torch.from_numpy(pts[:, 2].astype(np.int64)),
                torch.from_numpy(pts[:, 3:5].copy()),
                self.lab[cid])


class Net(nn.Module):
    def __init__(self, n_genes, n_cls, emb=32):
        super().__init__()
        self.emb = nn.Embedding(n_genes, emb)
        self.mlp = nn.Sequential(nn.Conv1d(2 + emb + 2, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                                 nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
                                nn.Linear(64, n_cls))

    def forward(self, xy, g, feat):
        e = self.emb(g)
        x = torch.cat([xy, e, feat], dim=2).transpose(1, 2)
        x = self.mlp(x).max(dim=2)[0]
        return self.fc(x)


def run_epoch(model, loader, opt, dev, train):
    model.train() if train else model.eval()
    crit = nn.CrossEntropyLoss(); preds = []; labs = []
    torch.set_grad_enabled(train)
    for xy, g, feat, yb in loader:
        xy, g, feat, yb = xy.to(dev), g.to(dev), feat.to(dev), yb.to(dev)
        out = model(xy, g, feat)
        if train:
            opt.zero_grad(); crit(out, yb).backward(); opt.step()
        preds += out.argmax(1).cpu().tolist(); labs += yb.cpu().tolist()
    acc = float(np.mean(np.array(preds) == np.array(labs)))
    return acc, f1_score(labs, preds, average="macro", zero_division=0)


def train_eval(tr, va, te, fine, cent, lab, co, cidx, radius, budget, policy,
               n_genes, n_cls, epochs, dev, bs=128):
    def mk(ids, shuffle):
        return DataLoader(CellDS(ids, fine, cent, lab, co, cidx, radius, budget, policy),
                          bs, shuffle=shuffle, num_workers=6)
    model = Net(n_genes, n_cls).to(dev)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    tl, vl, tel = mk(tr, True), mk(va, False), mk(te, False)
    best = 0.0; best_te = (0.0, 0.0)
    for ep in range(epochs):
        run_epoch(model, tl, opt, dev, True)
        if (ep + 1) % 2 == 0 or ep == epochs - 1:
            va_acc, _ = run_epoch(model, vl, None, dev, False)
            if va_acc >= best:
                best = va_acc; best_te = run_epoch(model, tel, None, dev, False)
    return best_te  # (acc, f1) at best-val epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", required=True)
    ap.add_argument("--pyramid", required=True)
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--genes-json", default=None,
                    help="defaults to <pyramid>.genes.json")
    ap.add_argument("--coarse-zoom", type=int, default=1)
    ap.add_argument("--radius", type=float, default=150.0, help="coarse window radius (um)")
    ap.add_argument("--budgets", default="16,64,256")
    ap.add_argument("--max-cells", type=int, default=25000)
    ap.add_argument("--min-tx", type=int, default=5)
    ap.add_argument("--qv-min", type=float, default=20.0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="../results/xenium_multiscale_probe.json")
    args = ap.parse_args()
    genes_json = args.genes_json or (args.pyramid + ".genes.json")
    budgets = [int(b) for b in args.budgets.split(",")]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}", flush=True)

    t0 = time.perf_counter()
    fine, cent, lab, ids, n_genes, co = load(
        args.transcripts, args.pyramid, args.clusters, genes_json,
        args.coarse_zoom, args.qv_min, args.max_cells, args.min_tx, args.seed)
    cidx = precompute_coarse_idx(ids, cent, co, args.radius)
    avg_ctx = float(np.mean([len(cidx[c]) for c in ids]))
    print(f"Setup in {time.perf_counter()-t0:.1f}s; mean coarse neighbors/cell={avg_ctx:.0f}",
          flush=True)

    uniq = sorted({lab[i] for i in ids}); lm = {c: j for j, c in enumerate(uniq)}
    lab = {i: lm[lab[i]] for i in ids}; n_cls = len(uniq)
    tr, te = train_test_split(ids, test_size=0.3, random_state=args.seed,
                              stratify=[lab[i] for i in ids])
    tr, va = train_test_split(tr, test_size=0.15, random_state=args.seed,
                              stratify=[lab[i] for i in tr])
    print(f"{len(ids)} cells, {n_cls} clusters, {n_genes} genes; "
          f"train/val/test = {len(tr)}/{len(va)}/{len(te)}", flush=True)

    results = {"budgets": budgets, "n_cells": len(ids), "n_clusters": n_cls,
               "coarse_zoom": args.coarse_zoom, "radius": args.radius,
               "mean_ctx_neighbors": round(avg_ctx, 1), "matrix": {}}
    for policy in ("fine", "coarse", "multi"):
        results["matrix"][policy] = {}
        for B in budgets:
            ts = time.perf_counter()
            acc, f1 = train_eval(tr, va, te, fine, cent, lab, co, cidx,
                                  args.radius, B, policy, n_genes, n_cls,
                                  args.epochs, dev)
            results["matrix"][policy][str(B)] = {"acc": round(acc, 4), "f1": round(f1, 4)}
            print(f"  {policy:>6} B={B:<4} acc={acc:.4f} f1={f1:.4f} "
                  f"({time.perf_counter()-ts:.0f}s)", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.output}", flush=True)
    lo = str(budgets[0]); mat = results["matrix"]
    print(f"VERDICT @B={lo}: fine={mat['fine'][lo]['acc']} "
          f"coarse={mat['coarse'][lo]['acc']} multi={mat['multi'][lo]['acc']} "
          f"(multi-best_single="
          f"{mat['multi'][lo]['acc']-max(mat['fine'][lo]['acc'],mat['coarse'][lo]['acc']):+.4f})")


if __name__ == "__main__":
    main()
