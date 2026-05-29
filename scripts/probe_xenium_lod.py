#!/usr/bin/env python3
"""PROBE (go/no-go): does multi-density LOD training on Xenium transcript POINTS help?

Tests whether a model trained across transcript-density levels (multi-LOD) is
more robust to test-time density than one trained at full density — the
empirical question gating whether muDM should add point subsampling/LOD.

For each cell we take its transcript molecules (x, y, gene), subsample to a
fraction r (the "LOD"), sample K points, and classify the cell's graphclust
label with a PointNet. We compare:
  * M_full  : trained at r=1.0 only
  * M_multi : trained on a mix of r in --rates
evaluated across every test r → a train-density x test-density matrix.

GO if M_multi degrades gracefully / dominates M_full at low density.
NO-GO if M_full is already density-robust (fixed-point PointNet is scale-free).

Usage::
    uv run python scripts/probe_xenium_lod.py \
        --transcripts data/xenium/transcripts.parquet \
        --clusters data/xenium/clusters.csv \
        --max-cells 30000 --epochs 15 --output results/xenium_lod_probe.json
"""
from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader, Dataset

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]


def _pick(cols: list[str], *cands: str) -> str:
    low = {c.lower(): c for c in cols}
    for cand in cands:
        if cand in low:
            return low[cand]
    raise KeyError(f"none of {cands} in {cols}")


def load_cells(transcripts: Path, clusters: Path, min_tx: int, qv_min: float):
    """Return (cell_pts: dict[cell-> (N,2) xy], cell_gene: dict[cell-> (N,) gid],
    labels: dict[cell-> int], n_genes)."""
    # --- labels ---
    import csv
    cl: dict[str, int] = {}
    with open(clusters) as f:
        r = csv.DictReader(f)
        bc, cc = r.fieldnames[0], r.fieldnames[1]
        for row in r:
            try:
                cl[row[bc]] = int(float(row[cc]))
            except (ValueError, TypeError):
                pass
    # --- transcripts (columns vary across Xenium versions) ---
    pf = pq.ParquetFile(str(transcripts))
    names = pf.schema_arrow.names
    cx = _pick(names, "x_location", "x", "x_global_px")
    cy = _pick(names, "y_location", "y", "y_global_px")
    cg = _pick(names, "feature_name", "gene", "target")
    cc_ = _pick(names, "cell_id", "cell")
    cqv = next((c for c in names if c.lower() in ("qv", "quality_value")), None)
    cols = [cx, cy, cg, cc_] + ([cqv] if cqv else [])
    t = pf.read(columns=cols)
    x = np.asarray(t.column(cx)); y = np.asarray(t.column(cy))
    gene = np.asarray(t.column(cg).cast("string").to_pylist(), dtype=object)
    cell = np.asarray(t.column(cc_).cast("string").to_pylist(), dtype=object)
    keep = np.ones(len(x), dtype=bool)
    if cqv:
        keep &= np.asarray(t.column(cqv)) >= qv_min
    # drop control/blank probes and unassigned transcripts
    isctrl = np.array([g.lower().startswith(("negcontrol", "blank", "antisense", "deprecated"))
                       for g in gene])
    keep &= ~isctrl
    keep &= np.array([str(c) not in ("UNASSIGNED", "-1", "0", "None", "") for c in cell])
    x, y, gene, cell = x[keep], y[keep], gene[keep], cell[keep]

    genes = sorted(set(gene.tolist()))
    gid = {g: i for i, g in enumerate(genes)}
    order = np.argsort(cell, kind="stable")
    cell, x, y, gene = cell[order], x[order], y[order], gene[order]
    cell_pts: dict = {}; cell_gene: dict = {}; labels: dict = {}
    # group by contiguous cell id
    bounds = np.where(cell[1:] != cell[:-1])[0] + 1
    for s, e in zip(np.r_[0, bounds], np.r_[bounds, len(cell)]):
        cid = str(cell[s])
        if cid not in cl or (e - s) < min_tx:
            continue
        cell_pts[cid] = np.stack([x[s:e], y[s:e]], axis=1).astype(np.float32)
        cell_gene[cid] = np.array([gid[g] for g in gene[s:e]], dtype=np.int64)
        labels[cid] = cl[cid]
    return cell_pts, cell_gene, labels, len(genes)


class CellPointDS(Dataset):
    def __init__(self, ids, cell_pts, cell_gene, lab_map, k, rate):
        self.ids = ids; self.pts = cell_pts; self.gene = cell_gene
        self.lab = lab_map; self.k = k; self.rate = rate

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        cid = self.ids[i]
        pts = self.pts[cid]; g = self.gene[cid]
        n = len(pts)
        r = self.rate if self.rate is not None else float(np.random.choice(RATES))
        m = max(3, int(round(n * r)))
        if m < n:
            sel = np.random.choice(n, m, replace=False); pts = pts[sel]; g = g[sel]
        idx = np.random.choice(len(pts), self.k, replace=len(pts) < self.k)
        p = pts[idx].copy(); gg = g[idx]
        p -= p.mean(0)
        s = np.abs(p).max()
        if s > 0:
            p /= s
        return torch.from_numpy(p), torch.from_numpy(gg), self.lab[self.ids[i]]


class PointNetGene(nn.Module):
    def __init__(self, n_genes, n_cls, emb=32):
        super().__init__()
        self.emb = nn.Embedding(n_genes, emb)
        self.mlp = nn.Sequential(nn.Conv1d(2 + emb, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                                 nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n_cls))

    def forward(self, p, g):
        e = self.emb(g)                       # (B,K,emb)
        x = torch.cat([p, e], dim=2).transpose(1, 2)  # (B,2+emb,K)
        x = self.mlp(x).max(dim=2)[0]
        return self.fc(x)


def run_epoch(model, loader, opt, dev, train):
    model.train() if train else model.eval()
    crit = nn.CrossEntropyLoss(); preds = []; labs = []
    torch.set_grad_enabled(train)
    for p, g, y in loader:
        p, g, y = p.to(dev), g.to(dev), y.to(dev)
        out = model(p, g)
        if train:
            opt.zero_grad(); crit(out, y).backward(); opt.step()
        preds += out.argmax(1).cpu().tolist(); labs += y.cpu().tolist()
    acc = float(np.mean(np.array(preds) == np.array(labs)))
    return acc, f1_score(labs, preds, average="macro", zero_division=0)


def train_model(tr_ids, va_ids, cp, cg, lab, k, n_genes, n_cls, rate, epochs, dev, bs=128):
    model = PointNetGene(n_genes, n_cls).to(dev)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    tl = DataLoader(CellPointDS(tr_ids, cp, cg, lab, k, rate), bs, shuffle=True, num_workers=8)
    vl = DataLoader(CellPointDS(va_ids, cp, cg, lab, k, 1.0), bs, num_workers=8)
    for ep in range(epochs):
        a, _ = run_epoch(model, tl, opt, dev, True)
        if (ep + 1) % 5 == 0:
            va, _ = run_epoch(model, vl, opt, dev, False)
            print(f"    ep{ep+1} train_acc={a:.3f} val_acc={va:.3f}", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", required=True)
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--max-cells", type=int, default=30000)
    ap.add_argument("--k-points", type=int, default=128)
    ap.add_argument("--min-tx", type=int, default=20)
    ap.add_argument("--qv-min", type=float, default=20.0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--rates", default="1.0,0.5,0.25,0.1,0.05")
    ap.add_argument("--output", default="results/xenium_lod_probe.json")
    args = ap.parse_args()
    global RATES
    RATES = [float(x) for x in args.rates.split(",")]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")

    t0 = time.perf_counter()
    cp, cg, lab, n_genes = load_cells(Path(args.transcripts), Path(args.clusters),
                                      args.min_tx, args.qv_min)
    print(f"Loaded {len(cp)} cells, {n_genes} genes in {time.perf_counter()-t0:.1f}s")
    ids = list(cp.keys())
    if len(ids) > args.max_cells:
        ids = list(np.random.RandomState(42).choice(ids, args.max_cells, replace=False))
    # remap labels to dense ids
    uniq = sorted({lab[i] for i in ids}); lm = {c: j for j, c in enumerate(uniq)}
    lab = {i: lm[lab[i]] for i in ids}
    n_cls = len(uniq)
    print(f"Using {len(ids)} cells, {n_cls} clusters")
    tr, te = train_test_split(ids, test_size=0.3, random_state=42,
                              stratify=[lab[i] for i in ids])
    tr, va = train_test_split(tr, test_size=0.15, random_state=42,
                              stratify=[lab[i] for i in tr])

    results = {"n_cells": len(ids), "n_clusters": n_cls, "n_genes": n_genes,
               "rates": RATES, "k_points": args.k_points, "matrix": {}}
    for tag, rate in (("M_full", 1.0), ("M_multi", None)):
        print(f"\n=== Training {tag} (train rate={rate}) ===")
        m = train_model(tr, va, cp, cg, lab, args.k_points, n_genes, n_cls,
                        rate, args.epochs, dev)
        row = {}
        for tr_ in RATES:
            dl = DataLoader(CellPointDS(te, cp, cg, lab, args.k_points, tr_), 128, num_workers=8)
            acc, f1 = run_epoch(m, dl, None, dev, False)
            row[str(tr_)] = {"acc": round(acc, 4), "f1": round(f1, 4)}
            print(f"    test@r={tr_}: acc={acc:.4f} f1={f1:.4f}")
        results["matrix"][tag] = row

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.output}")
    # verdict hint
    mf = results["matrix"]["M_full"]; mm = results["matrix"]["M_multi"]
    lo = str(RATES[-1])
    print(f"\nVERDICT HINT: at lowest density r={lo} -> "
          f"M_full acc={mf[lo]['acc']}, M_multi acc={mm[lo]['acc']} "
          f"(gain={mm[lo]['acc']-mf[lo]['acc']:+.4f}); "
          f"peak r=1.0 -> M_full={mf['1.0']['acc']}, M_multi={mm['1.0']['acc']}")


if __name__ == "__main__":
    main()
