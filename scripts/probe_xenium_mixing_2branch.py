#!/usr/bin/env python3
"""PROBE (fair test): TWO-BRANCH multi-scale fusion for Xenium region mixing.

The unified-point-set probe (probe_xenium_mixing.py) structurally suppressed the
fine scale (one max-pool lets coarse win every channel). This is the fair test:
two SEPARATE PointNet encoders (one per scale), fused at the embedding level.

Identical data/labels to that probe (imports its `load_and_label`). All three
policies share the SAME two-branch architecture and the SAME total budget -- each
branch gets B/2 real points -- so only the SCALE MIX differs:
  * fine   : both branches fed fine points        (B/2 + B/2 fine)
  * coarse : both branches fed coarse points       (B/2 + B/2 coarse)
  * multi  : one branch fine, one branch coarse     (B/2 fine + B/2 coarse)
This isolates the multi-scale effect from model capacity. GO if multi > both.

Usage::
    uv run python scripts/probe_xenium_mixing_2branch.py \
        --transcripts ../data/xenium/transcripts.parquet \
        --pyramid ../data/xenium/transcripts_lod_merge.parquet \
        --region-um 120 --min-cells 35 --coarse-zoom 1 --split tertile \
        --budgets 64,256,1024 --epochs 40 \
        --output ../results/xenium_mixing_2branch.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from probe_xenium_mixing import load_and_label  # identical data + labels

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

POLICY_SRC = {"fine": ("fine", "fine"), "coarse": ("coarse", "coarse"),
              "multi": ("fine", "coarse")}


class TwoBranchDS(Dataset):
    def __init__(self, ids, fine, coarse, label, centers, half, budget, policy):
        self.ids = ids; self.fine = fine; self.coarse = coarse; self.label = label
        self.centers = centers; self.half = half
        self.h = budget // 2; self.src = POLICY_SRC[policy]

    def __len__(self):
        return len(self.ids)

    def _set(self, r, source, cx, cy):
        if source == "fine":
            fx, fy, fg = self.fine[r]
            arr = np.c_[(fx - cx) / self.half, (fy - cy) / self.half, fg.astype(np.float32),
                        np.zeros(len(fx), np.float32), np.zeros(len(fx), np.float32)]
        else:
            px, py, pg, pc = self.coarse[r]
            arr = np.c_[(px - cx) / self.half, (py - cy) / self.half, pg.astype(np.float32),
                        np.log1p(pc), np.ones(len(px), np.float32)]
        arr = arr.astype(np.float32)
        n = arr.shape[0]
        if n == 0:
            arr = np.zeros((self.h, 5), np.float32)
        else:
            arr = arr[np.random.choice(n, self.h, replace=n < self.h)]
        return (torch.from_numpy(arr[:, :2]), torch.from_numpy(arr[:, 2].astype(np.int64)),
                torch.from_numpy(arr[:, 3:5].copy()))

    def __getitem__(self, i):
        r = self.ids[i]; cx, cy = self.centers[r]
        a = self._set(r, self.src[0], cx, cy)
        b = self._set(r, self.src[1], cx, cy)
        return (*a, *b, self.label[r])


class Branch(nn.Module):
    def __init__(self, emb_layer, emb=32, out=128):
        super().__init__()
        self.emb = emb_layer
        self.mlp = nn.Sequential(nn.Conv1d(2 + emb + 2, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                                 nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                                 nn.Conv1d(128, out, 1), nn.BatchNorm1d(out), nn.ReLU())

    def forward(self, xy, g, feat):
        e = self.emb(g)
        x = torch.cat([xy, e, feat], dim=2).transpose(1, 2)
        return self.mlp(x).max(dim=2)[0]


class TwoBranchNet(nn.Module):
    def __init__(self, n_genes, n_cls, emb=32, out=128):
        super().__init__()
        self.emb = nn.Embedding(n_genes, emb)  # shared gene semantics
        self.ba = Branch(self.emb, emb, out)
        self.bb = Branch(self.emb, emb, out)
        self.fc = nn.Sequential(nn.Linear(2 * out, 128), nn.ReLU(), nn.Dropout(0.3),
                                nn.Linear(128, n_cls))

    def forward(self, axy, ag, af, bxy, bg, bf):
        va = self.ba(axy, ag, af); vb = self.bb(bxy, bg, bf)
        return self.fc(torch.cat([va, vb], dim=1))


def run_epoch(model, loader, opt, dev, train):
    model.train() if train else model.eval()
    crit = nn.CrossEntropyLoss(); preds = []; labs = []
    torch.set_grad_enabled(train)
    for axy, ag, af, bxy, bg, bf, yb in loader:
        axy, ag, af = axy.to(dev), ag.to(dev), af.to(dev)
        bxy, bg, bf = bxy.to(dev), bg.to(dev), bf.to(dev)
        yb = yb.to(dev)
        out = model(axy, ag, af, bxy, bg, bf)
        if train:
            opt.zero_grad(); crit(out, yb).backward(); opt.step()
        preds += out.argmax(1).cpu().tolist(); labs += yb.cpu().tolist()
    acc = float(np.mean(np.array(preds) == np.array(labs)))
    return acc, f1_score(labs, preds, average="macro", zero_division=0)


def train_eval(tr, va, te, fine, coarse, label, centers, half, budget, policy,
               n_genes, epochs, dev, bs=64):
    def mk(ids, sh):
        return DataLoader(TwoBranchDS(ids, fine, coarse, label, centers, half, budget, policy),
                          bs, shuffle=sh, num_workers=6)
    model = TwoBranchNet(n_genes, 2).to(dev)
    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-4)
    tl, vl, tel = mk(tr, True), mk(va, False), mk(te, False)
    best = -1.0; best_te = (0.0, 0.0)
    for ep in range(epochs):
        run_epoch(model, tl, opt, dev, True)
        if (ep + 1) % 3 == 0 or ep == epochs - 1:
            va_acc, _ = run_epoch(model, vl, None, dev, False)
            if va_acc >= best:
                best = va_acc; best_te = run_epoch(model, tel, None, dev, False)
    return best_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", required=True)
    ap.add_argument("--pyramid", required=True)
    ap.add_argument("--genes-json", default=None)
    ap.add_argument("--region-um", type=float, default=120.0)
    ap.add_argument("--coarse-zoom", type=int, default=1)
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--min-cells", type=int, default=35)
    ap.add_argument("--min-minor", type=float, default=0.2)
    ap.add_argument("--min-marker", type=int, default=2)
    ap.add_argument("--split", choices=["median", "tertile"], default="tertile")
    ap.add_argument("--qv-min", type=float, default=20.0)
    ap.add_argument("--budgets", default="64,256,1024")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="../results/xenium_mixing_2branch.json")
    args = ap.parse_args()
    genes_json = args.genes_json or (args.pyramid + ".genes.json")
    budgets = [int(b) for b in args.budgets.split(",")]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}", flush=True)

    t0 = time.perf_counter()
    fine, coarse, label, centers, ids, n_genes, half = load_and_label(
        args.transcripts, args.pyramid, genes_json, args.region_um, args.coarse_zoom,
        args.knn, args.min_cells, args.min_minor, args.min_marker, args.qv_min,
        args.split, args.seed)
    print(f"Setup in {time.perf_counter()-t0:.1f}s [TWO-BRANCH fusion]", flush=True)

    y = [label[r] for r in ids]
    tr, te = train_test_split(ids, test_size=0.3, random_state=args.seed, stratify=y)
    tr, va = train_test_split(tr, test_size=0.18, random_state=args.seed,
                              stratify=[label[r] for r in tr])
    print(f"train/val/test = {len(tr)}/{len(va)}/{len(te)}", flush=True)

    results = {"budgets": budgets, "n_regions": len(ids), "region_um": args.region_um,
               "coarse_zoom": args.coarse_zoom, "fusion": "two_branch", "matrix": {}}
    for policy in ("fine", "coarse", "multi"):
        results["matrix"][policy] = {}
        for B in budgets:
            ts = time.perf_counter()
            acc, f1 = train_eval(tr, va, te, fine, coarse, label, centers, half,
                                 B, policy, n_genes, args.epochs, dev)
            results["matrix"][policy][str(B)] = {"acc": round(acc, 4), "f1": round(f1, 4)}
            print(f"  {policy:>6} B={B:<5} acc={acc:.4f} f1={f1:.4f} "
                  f"({time.perf_counter()-ts:.0f}s)", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.output}", flush=True)
    mat = results["matrix"]
    for B in budgets:
        b = str(B)
        gain = mat["multi"][b]["acc"] - max(mat["fine"][b]["acc"], mat["coarse"][b]["acc"])
        print(f"VERDICT @B={B}: fine={mat['fine'][b]['acc']} coarse={mat['coarse'][b]['acc']} "
              f"multi={mat['multi'][b]['acc']} (multi-best_single={gain:+.4f})")


if __name__ == "__main__":
    main()
