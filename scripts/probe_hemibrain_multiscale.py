#!/usr/bin/env python3
"""PROBE (headline test, 3D): two-branch multi-scale fusion for Hemibrain neuron
cell-type classification under a vertex budget.

A neuron has millions of vertices (too vast to ingest fully). muDM's QEM mesh-LOD
pyramid gives a coarse whole-neuron mesh (zoom 0 -- global arbor shape) and a fine
mesh (zoom 3 -- local branch/bouton detail). Same two-branch, matched-budget design
as probe_xenium_mixing_2branch.py so results are directly comparable:
  * coarse : both branches fed coarse-zoom vertices       (global shape only)
  * fine   : both branches fed fine-zoom vertices          (local detail only)
  * multi  : coarse branch coarse-zoom + fine branch fine-zoom
GO if multi > both (neuron type needs global shape AND local detail, non-redundant).

Loader streams parquet batches and caps vertices/neuron/zoom to bound memory.

Usage::
    uv run python scripts/probe_hemibrain_multiscale.py \
        --tiles ../data/hemibrain/tiles/hemibrain/tiles.parquet \
        --coarse-zoom 0 --fine-zoom 3 --top-types 15 \
        --budgets 256,1024,4096 --max-neurons 3000 --epochs 30 \
        --output ../results/hemibrain_multiscale_probe.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

POLICY_SRC = {"fine": ("fine", "fine"), "coarse": ("coarse", "coarse"),
              "multi": ("coarse", "fine")}

# Curated central-complex morphological cell-type families. Defined by distinct
# fine-scale arborization within SHARED neuropils -> hypothesized to require both
# global wiring (coarse) AND local detail (fine) -> the genuine multi-scale test.
CX_MORPHO_FAMILIES = {
    "EPG", "PEN", "PFN", "ER", "PFL", "hDelta", "Delta", "EL", "PFG",
}


def _family(t):
    """Strip subtype suffix to get the type-family prefix.
    PFL3 -> PFL, hDeltaB -> hDelta, ER4d -> ER, AVLP464 -> AVLP, EPG -> EPG."""
    if not t:
        return None
    f = re.split(r"[\d_(]", t, maxsplit=1)[0]
    return re.sub(r"(?<=[a-z])[A-Z]$", "", f)


def _label(t, scheme):
    """Map raw cellType to a label per the chosen scheme; None = drop the neuron.

    * ``neuropil_family``: top-N families by prefix (mostly neuropil-named SMP,
      AVLP, ...) -- the CONTROL, expected coarse-favorable since region is a
      global property.
    * ``cx_morpho``: only the curated CX morphological families -- the genuine
      scale-distributed TEST.
    * ``raw``: raw cellType (legacy / debug).
    """
    fam = _family(t)
    if scheme == "neuropil_family":
        return fam
    if scheme == "cx_morpho":
        return fam if fam in CX_MORPHO_FAMILIES else None
    if scheme == "cx_subtype":
        # Within-family subtypes: raw cellType restricted to CX morpho families.
        # Tests whether multi-LOD helps when global shape is SHARED (same family,
        # e.g. all PFL or all hDelta) but discriminative info is in the fine
        # arborization (PFL1 vs PFL2 vs PFL3, hDeltaA vs hDeltaB, ...).
        return t if fam in CX_MORPHO_FAMILIES else None
    return t


def _instance_base(inst):
    """Strip the bilateral _L / _R suffix so the two homologs share a key
    (used for train/test dedupe to avoid leakage from mirror-symmetric pairs)."""
    if not inst:
        return None
    return re.sub(r"_[LR]$", "", inst)


def load_neurons(tiles, coarse_zoom, fine_zoom, label_scheme, top_n, max_neurons,
                 cap, seed):
    dset = ds.dataset(str(tiles), format="parquet")
    print(f"schema: {dset.schema.names}", flush=True)
    coarse = {}; fine = {}; ctype = {}; inst_base = {}
    clen = {}; flen = {}
    scanner = dset.scanner(columns=["zoom", "positions", "tags"],
                           filter=(ds.field("zoom") == coarse_zoom)
                           | (ds.field("zoom") == fine_zoom))
    nrows = 0
    for batch in scanner.to_batches():
        zoom = batch.column("zoom").to_numpy(zero_copy_only=False)
        tags_list = batch.column("tags").to_pylist()
        pos_list = batch.column("positions").to_pylist()
        nrows += len(zoom)
        for z, tg, pb in zip(zoom, tags_list, pos_list):
            if tg is None or pb is None:
                continue
            td = dict(tg) if isinstance(tg, list) else tg
            bid = td.get("body_id")
            if bid is None:
                continue
            if bid not in ctype:
                ct = td.get("cell_type")
                if ct is None:
                    continue
                ctype[bid] = ct
                inst_base[bid] = _instance_base(td.get("instance"))
            if z == coarse_zoom:
                store, ln = coarse, clen
            else:
                store, ln = fine, flen
            if ln.get(bid, 0) >= cap:
                continue
            v = np.frombuffer(pb, dtype=np.float32).reshape(-1, 3)
            if len(v) == 0:
                continue
            store.setdefault(bid, []).append(v)
            ln[bid] = ln.get(bid, 0) + len(v)
    print(f"scanned {nrows:,} tiles (zoom {coarse_zoom}+{fine_zoom})", flush=True)

    # Apply label scheme (drops neurons whose label is None for the scheme).
    labeled = {b: _label(ctype[b], label_scheme) for b in ctype}
    labeled = {b: l for b, l in labeled.items() if l is not None}
    cnt = Counter(labeled[b] for b in labeled if b in coarse and b in fine)
    label_map = {t: i for i, (t, _) in enumerate(cnt.most_common(top_n))}
    cand = [b for b in coarse if b in fine and labeled.get(b) in label_map]
    # Bilateral dedupe: group by stripped instance, keep one per group.
    by_base = {}
    for b in cand:
        base = inst_base.get(b) or b
        by_base.setdefault(base, []).append(b)
    ids = [bs[0] for bs in by_base.values()]
    # Safety: when Hemibrain's `instance` doesn't distinguish individual neurons
    # (e.g. all 18 ELs share `EL(EQ5)_{L,R}`), dedup collapses a class to ~1.
    # If that happens, revert to no-dedup -- the relative policy comparison is
    # still valid (any bilateral leakage helps all policies equally).
    post = Counter(labeled[b] for b in ids)
    if post and min(post.values()) < 5:
        print(f"NOTE: bilateral dedup would leave a class with "
              f"{min(post.values())} members; reverting (no dedup)", flush=True)
        ids = list(cand)
        dropped_bilateral = 0
    else:
        dropped_bilateral = len(cand) - len(ids)
    rng = np.random.RandomState(seed)
    if len(ids) > max_neurons:
        ids = list(rng.choice(np.array(ids, dtype=object), max_neurons, replace=False))
    C = {}; F = {}; cen = {}; scl = {}; lab = {}
    for b in ids:
        c = np.concatenate(coarse[b]).astype(np.float32)
        f = np.concatenate(fine[b]).astype(np.float32)
        center = c.mean(0)
        scale = float(np.abs(c - center).max()) or 1.0
        C[b] = c; F[b] = f; cen[b] = center; scl[b] = scale
        lab[b] = label_map[labeled[b]]
    print(f"{len(ids)} neurons, {len(label_map)} classes (scheme={label_scheme}); "
          f"median coarse verts={int(np.median([len(C[b]) for b in ids]))}, "
          f"fine={int(np.median([len(F[b]) for b in ids]))}; "
          f"dropped {dropped_bilateral} bilateral homologs", flush=True)
    print(f"label counts: {dict(cnt.most_common(top_n))}", flush=True)
    return C, F, cen, scl, lab, ids, len(label_map)


class NeuronDS(Dataset):
    def __init__(self, ids, C, F, cen, scl, lab, budget, policy):
        self.ids = ids; self.C = C; self.F = F; self.cen = cen; self.scl = scl
        self.lab = lab; self.h = budget // 2; self.src = POLICY_SRC[policy]

    def __len__(self):
        return len(self.ids)

    def _set(self, b, source):
        v = (self.C if source == "coarse" else self.F)[b]
        v = (v - self.cen[b]) / self.scl[b]
        n = len(v)
        idx = np.random.choice(n, self.h, replace=n < self.h)
        return torch.from_numpy(v[idx].astype(np.float32))

    def __getitem__(self, i):
        b = self.ids[i]
        return self._set(b, self.src[0]), self._set(b, self.src[1]), self.lab[b]


class Branch(nn.Module):
    def __init__(self, out=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                                 nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                                 nn.Conv1d(128, out, 1), nn.BatchNorm1d(out), nn.ReLU())

    def forward(self, p):  # p: (B, K, 3)
        return self.mlp(p.transpose(1, 2)).max(dim=2)[0]


class TwoBranchNet(nn.Module):
    def __init__(self, n_cls, out=256):
        super().__init__()
        self.ba = Branch(out); self.bb = Branch(out)
        self.fc = nn.Sequential(nn.Linear(2 * out, 128), nn.ReLU(), nn.Dropout(0.3),
                                nn.Linear(128, n_cls))

    def forward(self, a, b):
        return self.fc(torch.cat([self.ba(a), self.bb(b)], dim=1))


def run_epoch(model, loader, opt, dev, train):
    model.train() if train else model.eval()
    crit = nn.CrossEntropyLoss(); preds = []; labs = []
    torch.set_grad_enabled(train)
    for a, b, yb in loader:
        a, b, yb = a.to(dev), b.to(dev), yb.to(dev)
        out = model(a, b)
        if train:
            opt.zero_grad(); crit(out, yb).backward(); opt.step()
        preds += out.argmax(1).cpu().tolist(); labs += yb.cpu().tolist()
    acc = float(np.mean(np.array(preds) == np.array(labs)))
    return acc, f1_score(labs, preds, average="macro", zero_division=0)


def train_eval(tr, va, te, C, F, cen, scl, lab, budget, policy, n_cls, epochs, dev, bs=64):
    def mk(ids, sh):
        return DataLoader(NeuronDS(ids, C, F, cen, scl, lab, budget, policy),
                          bs, shuffle=sh, num_workers=6)
    model = TwoBranchNet(n_cls).to(dev)
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
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--coarse-zoom", type=int, default=0)
    ap.add_argument("--fine-zoom", type=int, default=3)
    ap.add_argument("--label-scheme",
                    choices=["neuropil_family", "cx_morpho", "cx_subtype", "raw"],
                    default="neuropil_family",
                    help="neuropil_family = top-N family prefixes (control, "
                         "likely coarse-favorable); cx_morpho = curated CX "
                         "morphological families (test, scale-distributed)")
    ap.add_argument("--top-n", type=int, default=15)
    ap.add_argument("--cap", type=int, default=16384, help="max vertices/neuron/zoom held")
    ap.add_argument("--max-neurons", type=int, default=3000)
    ap.add_argument("--budgets", default="256,1024,4096")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="../results/hemibrain_multiscale_probe.json")
    args = ap.parse_args()
    budgets = [int(b) for b in args.budgets.split(",")]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}", flush=True)

    t0 = time.perf_counter()
    C, F, cen, scl, lab, ids, n_cls = load_neurons(
        args.tiles, args.coarse_zoom, args.fine_zoom, args.label_scheme,
        args.top_n, args.max_neurons, args.cap, args.seed)
    print(f"Setup in {time.perf_counter()-t0:.1f}s [TWO-BRANCH 3D]", flush=True)

    y = [lab[b] for b in ids]
    tr, te = train_test_split(ids, test_size=0.3, random_state=args.seed, stratify=y)
    tr, va = train_test_split(tr, test_size=0.18, random_state=args.seed,
                              stratify=[lab[b] for b in tr])
    print(f"train/val/test = {len(tr)}/{len(va)}/{len(te)}", flush=True)

    results = {"budgets": budgets, "n_neurons": len(ids), "n_types": n_cls,
               "coarse_zoom": args.coarse_zoom, "fine_zoom": args.fine_zoom,
               "fusion": "two_branch", "matrix": {}}
    for policy in ("coarse", "fine", "multi"):
        results["matrix"][policy] = {}
        for B in budgets:
            ts = time.perf_counter()
            acc, f1 = train_eval(tr, va, te, C, F, cen, scl, lab, B, policy,
                                 n_cls, args.epochs, dev)
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
        print(f"VERDICT @B={B}: coarse={mat['coarse'][b]['acc']} fine={mat['fine'][b]['acc']} "
              f"multi={mat['multi'][b]['acc']} (multi-best_single={gain:+.4f})")


if __name__ == "__main__":
    main()
