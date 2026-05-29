#!/usr/bin/env python3
"""FEASIBILITY PROBE: can muDM vector polygon contours classify PanNuke nuclei,
and does vertex-LOD (contour budget) move accuracy?

This is a cheap (~minutes) go/no-go before committing to a full 2D experiment.
It is LEAKAGE-FREE by construction: every sample is a SINGLE nucleus described
ONLY by its own polygon contour geometry. No neighbor labels, no context branch.

Two diagnostics:
  (1) Hand-crafted shape features (area, perimeter, vertices, circularity,
      eccentricity, bbox) -> RandomForest, 3-fold CV macro-F1. This is the
      *shape ceiling*: the most class signal recoverable from geometry alone.
  (2) Single-branch PointNet on the contour points at vertex budgets
      {8,16,32,full}, fold1+2 train / fold3 test, class-weighted CE. The
      macro-F1-vs-budget curve answers the right-sized-LOD question.

Pre-registered decision rule (see paper revision plan):
  * DROP            if RF and PointNet best macro-F1 are both < ~0.30
                     (vector geometry carries too little class signal;
                      nucleus typing needs pixels -> PanNuke is the wrong
                      dataset for any vector-LOD ML demo).
  * REFRAME-ECONOMY if best macro-F1 >= ~0.40 AND the budget curve is roughly
                     FLAT (budget-8 within ~0.02 of full) -> "right-sized LOD"
                     holds in 2D; run the proper DP-re-tiled 3-fold/5-seed
                     economy experiment.
  * AMBIGUOUS       if signal exists but macro-F1 RISES with budget (fine detail
                     needed) -> no economy story; honest-scope only.
Reference: always-predict-majority (Neoplastic) baseline = acc 0.408, macro-F1 0.116.

Usage::
    uv run python scripts/probe_pannuke_feasibility.py \
        --geojson-dir /data/ai/mudm-paper/data/pannuke_mudm/geojson \
        --output /data/ai/mudm-paper/results/pannuke_feasibility.json \
        --device cuda:1
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

CLASS_NAMES = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]
N_CLS = 5


# ---------------------------------------------------------------- data loading
def _shoelace_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def load_nuclei(geojson_dir: str, cache: str, max_images: int = 0) -> dict:
    """Parse per-image GeoJSON into per-nucleus contours + labels + folds.

    Returns dict with: contours (list of centered float32 (M,2)), labels (int8),
    folds (int8), feats (float32 (N,F)), feat_names.
    """
    if cache and os.path.exists(cache) and not max_images:
        print(f"Loading cache {cache}", flush=True)
        with open(cache, "rb") as fh:
            return pickle.load(fh)

    files = sorted(glob.glob(os.path.join(geojson_dir, "*.geojson")))
    if max_images:
        files = files[:max_images]
    print(f"Parsing {len(files)} geojson files...", flush=True)

    contours: list[np.ndarray] = []
    labels: list[int] = []
    folds: list[int] = []
    feats: list[list[float]] = []
    t0 = time.perf_counter()
    for k, fp in enumerate(files):
        fold = int(os.path.basename(fp)[4])  # "foldN_xxxxx.geojson"
        fc = json.loads(Path(fp).read_text())
        for feat in fc["features"]:
            ring = feat["geometry"]["coordinates"][0]
            p = np.asarray(ring, dtype=np.float32)
            if len(p) >= 4 and np.allclose(p[0], p[-1]):
                p = p[:-1]  # drop closing duplicate
            if len(p) < 3:
                continue
            cid = int(feat["properties"]["cell_type_id"])
            cen = p.mean(axis=0)
            pc = p - cen
            # hand-crafted shape features
            area = _shoelace_area(p)
            per = float(np.sum(np.linalg.norm(np.diff(p, axis=0, append=p[:1]), axis=1)))
            nv = float(len(p))
            circ = float(4.0 * np.pi * area / (per * per)) if per > 0 else 0.0
            # eccentricity via PCA of centered points
            cov = np.cov(pc.T) if len(pc) > 1 else np.eye(2)
            ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
            ecc = float(np.sqrt(max(0.0, 1.0 - ev[1] / ev[0]))) if ev[0] > 1e-9 else 0.0
            bw = float(pc[:, 0].max() - pc[:, 0].min())
            bh = float(pc[:, 1].max() - pc[:, 1].min())
            contours.append(pc.astype(np.float32))
            labels.append(cid)
            folds.append(fold)
            feats.append([area, np.log1p(area), per, nv, circ, ecc, bw, bh])
        if (k + 1) % 2000 == 0:
            print(f"  {k+1}/{len(files)} files, {len(labels)} nuclei "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)

    out = {
        "contours": contours,
        "labels": np.asarray(labels, dtype=np.int8),
        "folds": np.asarray(folds, dtype=np.int8),
        "feats": np.asarray(feats, dtype=np.float32),
        "feat_names": ["area", "log_area", "perimeter", "n_verts", "circularity",
                       "eccentricity", "bbox_w", "bbox_h"],
    }
    print(f"Parsed {len(labels)} nuclei in {time.perf_counter()-t0:.0f}s", flush=True)
    if cache and not max_images:
        with open(cache, "wb") as fh:
            pickle.dump(out, fh)
        print(f"Cached to {cache}", flush=True)
    return out


# ------------------------------------------------------------- diagnostic (1)
def rf_shape_ceiling(data: dict) -> dict:
    X, y, folds = data["feats"], data["labels"].astype(int), data["folds"]
    print("\n=== Diagnostic 1: hand-crafted shape features -> RandomForest (3-fold) ===",
          flush=True)
    accs, f1s = [], []
    per_class = np.zeros(N_CLS)
    for test_fold in (1, 2, 3):
        tr = folds != test_fold
        te = folds == test_fold
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight="balanced",
                                     random_state=0, max_depth=None)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        acc = float(np.mean(pred == y[te]))
        f1 = f1_score(y[te], pred, average="macro", zero_division=0)
        pc = f1_score(y[te], pred, average=None, labels=list(range(N_CLS)), zero_division=0)
        accs.append(acc); f1s.append(f1); per_class += np.asarray(pc) / 3.0
        print(f"  fold{test_fold} test: acc={acc:.4f} macro-F1={f1:.4f}", flush=True)
    res = {"acc_mean": round(float(np.mean(accs)), 4),
           "macro_f1_mean": round(float(np.mean(f1s)), 4),
           "macro_f1_std": round(float(np.std(f1s)), 4),
           "per_class_f1": {CLASS_NAMES[i]: round(float(per_class[i]), 4) for i in range(N_CLS)}}
    print(f"  RF shape ceiling: acc={res['acc_mean']} macro-F1={res['macro_f1_mean']} "
          f"+/-{res['macro_f1_std']}", flush=True)
    print(f"  per-class F1: {res['per_class_f1']}", flush=True)
    return res


# ------------------------------------------------------------- diagnostic (2)
class NucDS(Dataset):
    def __init__(self, contours, labels, idx, budget, scale):
        self.contours = contours; self.labels = labels; self.idx = idx
        self.budget = budget; self.scale = scale

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        r = self.idx[i]; p = self.contours[r]; m = len(p)
        full = self.budget <= 0 or self.budget >= m
        b = m if full else self.budget
        sel = np.arange(m) if (full and m == b) else np.random.choice(m, b, replace=m < b)
        out = (p[sel] / self.scale).astype(np.float32)
        # pad/truncate to a fixed length so default collate can batch (use budget,
        # or for "full" policy a fixed cap)
        cap = self.budget if self.budget > 0 else 128
        if out.shape[0] >= cap:
            out = out[np.random.choice(out.shape[0], cap, replace=False)]
        else:
            pad = out[np.random.choice(out.shape[0], cap - out.shape[0], replace=True)]
            out = np.concatenate([out, pad], axis=0)
        return torch.from_numpy(out), int(self.labels[r])


class PointNet(nn.Module):
    def __init__(self, n_cls, width=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(2, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, width, 1), nn.BatchNorm1d(width), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(width, 128), nn.ReLU(), nn.Dropout(0.3),
                                nn.Linear(128, n_cls))

    def forward(self, x):  # x: (B, N, 2)
        h = self.mlp(x.transpose(1, 2)).max(dim=2)[0]
        return self.fc(h)


def run_epoch(model, loader, opt, dev, crit, train):
    model.train() if train else model.eval()
    preds, labs = [], []
    torch.set_grad_enabled(train)
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        out = model(xb)
        if train:
            opt.zero_grad(); crit(out, yb).backward(); opt.step()
        preds += out.argmax(1).cpu().tolist(); labs += yb.cpu().tolist()
    acc = float(np.mean(np.array(preds) == np.array(labs)))
    return acc, f1_score(labs, preds, average="macro", zero_division=0), preds, labs


def pointnet_budget_sweep(data: dict, budgets, epochs, dev, scale, bs=256) -> dict:
    contours, labels, folds = data["contours"], data["labels"], data["folds"]
    tr_idx = np.where(folds != 3)[0]
    te_idx = np.where(folds == 3)[0]
    counts = np.bincount(labels[tr_idx].astype(int), minlength=N_CLS).astype(np.float64)
    w = (counts.sum() / (N_CLS * np.clip(counts, 1, None))).astype(np.float32)
    crit = nn.CrossEntropyLoss(weight=torch.from_numpy(w).to(dev))
    print(f"\n=== Diagnostic 2: PointNet budget sweep "
          f"(train={len(tr_idx)} fold1+2, test={len(te_idx)} fold3) ===", flush=True)
    print(f"  class weights: {dict(zip(CLASS_NAMES, w.round(3).tolist()))}", flush=True)
    out = {}
    for B in budgets:
        ts = time.perf_counter()
        tl = DataLoader(NucDS(contours, labels, tr_idx, B, scale), bs, shuffle=True,
                        num_workers=8, drop_last=True)
        el = DataLoader(NucDS(contours, labels, te_idx, B, scale), bs, shuffle=False,
                        num_workers=8)
        model = PointNet(N_CLS).to(dev)
        opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-4)
        best = (0.0, 0.0)
        for ep in range(epochs):
            run_epoch(model, tl, opt, dev, crit, True)
            if (ep + 1) % 4 == 0 or ep == epochs - 1:
                acc, f1, pred, lab = run_epoch(model, el, None, dev, crit, False)
                if f1 >= best[1]:
                    best = (acc, f1)
                    pc = f1_score(lab, pred, average=None, labels=list(range(N_CLS)),
                                  zero_division=0)
                    best_pc = {CLASS_NAMES[i]: round(float(pc[i]), 4) for i in range(N_CLS)}
        tag = "full" if B <= 0 else str(B)
        out[tag] = {"acc": round(best[0], 4), "macro_f1": round(best[1], 4),
                    "per_class_f1": best_pc}
        print(f"  budget={tag:>5}: acc={best[0]:.4f} macro-F1={best[1]:.4f} "
              f"({time.perf_counter()-ts:.0f}s) per-class={best_pc}", flush=True)
    return out


def decide(rf: dict, pn: dict) -> dict:
    rf_f1 = rf["macro_f1_mean"]
    pn_best = max(v["macro_f1"] for v in pn.values())
    f1_8 = pn.get("8", {}).get("macro_f1", 0.0)
    f1_full = pn.get("full", {}).get("macro_f1", pn_best)
    flat = abs(f1_full - f1_8) <= 0.02
    if max(rf_f1, pn_best) < 0.30:
        verdict = "DROP"
        why = "vector geometry carries too little class signal (best macro-F1 < 0.30)"
    elif pn_best >= 0.40 and flat:
        verdict = "REFRAME-ECONOMY"
        why = f"signal present (best={pn_best}) and budget curve flat (full-8={f1_full-f1_8:+.3f})"
    else:
        verdict = "AMBIGUOUS"
        why = (f"signal={pn_best:.3f}, budget curve {'flat' if flat else 'rising'} "
               f"(full-8={f1_full-f1_8:+.3f}); not a clean economy story")
    return {"verdict": verdict, "rationale": why, "rf_macro_f1": rf_f1,
            "pointnet_best_macro_f1": round(pn_best, 4),
            "f1_budget8": round(f1_8, 4), "f1_full": round(f1_full, 4),
            "majority_baseline_macro_f1": 0.116}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geojson-dir", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--output", default="../results/pannuke_feasibility.json")
    ap.add_argument("--budgets", default="8,16,32,0")  # 0 = full
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--scale", type=float, default=32.0)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cache = args.cache or os.path.join(os.path.dirname(args.geojson_dir.rstrip("/")),
                                       "feasibility_cache.pkl")
    print(f"Device: {dev}", flush=True)

    data = load_nuclei(args.geojson_dir, cache, args.max_images)
    dist = {CLASS_NAMES[i]: int(c) for i, c in
            enumerate(np.bincount(data["labels"].astype(int), minlength=N_CLS))}
    print(f"Class distribution: {dist}", flush=True)

    rf = rf_shape_ceiling(data)
    budgets = [int(b) for b in args.budgets.split(",")]
    pn = pointnet_budget_sweep(data, budgets, args.epochs, dev, args.scale)
    verdict = decide(rf, pn)

    results = {"n_nuclei": len(data["labels"]), "class_distribution": dist,
               "rf_shape_ceiling": rf, "pointnet_budget_sweep": pn,
               "decision": verdict, "scale": args.scale, "epochs": args.epochs}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.output}", flush=True)
    print(f"\n*** VERDICT: {verdict['verdict']} — {verdict['rationale']} ***", flush=True)


if __name__ == "__main__":
    main()
