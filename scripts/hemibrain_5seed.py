#!/usr/bin/env python3
"""5-seed re-run of the Hemibrain coarse-LOD economy HEADLINE (paper-grade stats).

The single-seed probe (probe_hemibrain_multiscale.py, cx_morpho) showed the
right-sized-LOD result: a coarse-LOD pipeline (zoom 0, B=256) reaches ~0.93 acc
on 7-class central-complex neuron typing while the full-res pipeline (zoom 3,
B=4096) reaches only ~0.59 -- equal-or-better accuracy at ~16x FLOPs / ~13x peak
GPU mem / ~247x storage (results/hemibrain_economy.json). This wrapper repeats
the SAME experiment across N seeds and reports mean +/- SE with paired t-tests,
so the headline rests on multi-seed statistics rather than one run.

Design:
  * load_neurons() runs ONCE (the 160 GB parquet scan is the expensive step) and
    is cached to disk; the neuron SET is identical across seeds (proper paired
    comparison). Seeds vary only the train/val/test split + model init.
  * For each seed: torch + numpy seeded; full coarse/fine/multi x budget matrix.
  * Crash-safe: the per-seed matrix is flushed to --output after every seed.
  * Paired tests across seeds:
      - economy headline : coarse@(z0,Bmin)  vs fine@(z3,Bmax)
      - matched budget   : coarse vs fine at Bmax (and at Bmin) -> isolates LOD
      - multi NO-GO      : multi vs coarse at each budget (retired-fusion check)

Usage::
    .venv/bin/python scripts/hemibrain_5seed.py \
        --tiles data/hemibrain/tiles/hemibrain/tiles.parquet \
        --label-scheme cx_morpho --coarse-zoom 0 --fine-zoom 3 \
        --budgets 256,1024,4096 --epochs 30 --seeds 0,1,2,3,4 \
        --device cuda:1 \
        --cache data/hemibrain/neuron_cache_cx_morpho_z0z3.pkl \
        --output ../results/hemibrain_5seed.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_hemibrain_multiscale import load_neurons, train_eval  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

POLICIES = ("coarse", "fine", "multi")


def load_or_cache(args):
    if args.cache and Path(args.cache).exists():
        print(f"Loading neuron cache {args.cache}", flush=True)
        with open(args.cache, "rb") as fh:
            d = pickle.load(fh)
        print(f"  {len(d['ids'])} neurons, {d['n_cls']} classes (cached)", flush=True)
        return d["C"], d["F"], d["cen"], d["scl"], d["lab"], d["ids"], d["n_cls"]
    t0 = time.perf_counter()
    C, F, cen, scl, lab, ids, n_cls = load_neurons(
        args.tiles, args.coarse_zoom, args.fine_zoom, args.label_scheme,
        args.top_n, args.max_neurons, args.cap, args.load_seed)
    print(f"load_neurons in {time.perf_counter()-t0:.0f}s", flush=True)
    if args.cache:
        Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
        with open(args.cache, "wb") as fh:
            pickle.dump({"C": C, "F": F, "cen": cen, "scl": scl, "lab": lab,
                         "ids": ids, "n_cls": n_cls}, fh)
        print(f"Cached neurons to {args.cache}", flush=True)
    return C, F, cen, scl, lab, ids, n_cls


def aggregate(per_seed, seeds, budgets):
    n = len(seeds)
    agg = {}
    for pol in POLICIES:
        agg[pol] = {}
        for B in budgets:
            accs = np.array([per_seed[s][pol][str(B)]["acc"] for s in seeds])
            f1s = np.array([per_seed[s][pol][str(B)]["f1"] for s in seeds])
            se = (lambda x: float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0)
            agg[pol][str(B)] = {
                "acc_mean": round(float(accs.mean()), 4), "acc_se": round(se(accs), 4),
                "f1_mean": round(float(f1s.mean()), 4), "f1_se": round(se(f1s), 4),
                "acc_per_seed": [round(float(a), 4) for a in accs],
            }
    return agg


def paired(per_seed, seeds, polA, BA, polB, BB, metric="acc"):
    n = len(seeds)
    a = np.array([per_seed[s][polA][str(BA)][metric] for s in seeds])
    b = np.array([per_seed[s][polB][str(BB)][metric] for s in seeds])
    diff = a - b
    if n > 1 and np.any(diff != 0):
        t, p = stats.ttest_rel(a, b)
    else:
        t, p = float("nan"), float("nan")
    return {"A": f"{polA}@B{BA}", "B": f"{polB}@B{BB}", "metric": metric,
            "mean_A": round(float(a.mean()), 4), "mean_B": round(float(b.mean()), 4),
            "mean_diff": round(float(diff.mean()), 4),
            "se_diff": round(float(diff.std(ddof=1) / np.sqrt(n)), 4) if n > 1 else 0.0,
            "t": None if np.isnan(t) else round(float(t), 3),
            "p": None if np.isnan(p) else float(p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--coarse-zoom", type=int, default=0)
    ap.add_argument("--fine-zoom", type=int, default=3)
    ap.add_argument("--label-scheme", default="cx_morpho",
                    choices=["neuropil_family", "cx_morpho", "cx_subtype", "raw"])
    ap.add_argument("--top-n", type=int, default=15)
    ap.add_argument("--cap", type=int, default=16384)
    ap.add_argument("--max-neurons", type=int, default=3000)
    ap.add_argument("--budgets", default="256,1024,4096")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--load-seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--output", default="../results/hemibrain_5seed.json")
    args = ap.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev} | scheme={args.label_scheme} z{args.coarse_zoom}/z{args.fine_zoom} "
          f"| budgets={budgets} | seeds={seeds} | epochs={args.epochs}", flush=True)

    C, F, cen, scl, lab, ids, n_cls = load_or_cache(args)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_seed = {}
    grand = time.perf_counter()
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        y = [lab[b] for b in ids]
        tr, te = train_test_split(ids, test_size=0.3, random_state=s, stratify=y)
        tr, va = train_test_split(tr, test_size=0.18, random_state=s,
                                  stratify=[lab[b] for b in tr])
        per_seed[s] = {p: {} for p in POLICIES}
        for pol in POLICIES:
            for B in budgets:
                ts = time.perf_counter()
                acc, f1 = train_eval(tr, va, te, C, F, cen, scl, lab, B, pol,
                                     n_cls, args.epochs, dev)
                per_seed[s][pol][str(B)] = {"acc": round(acc, 4), "f1": round(f1, 4)}
                print(f"  seed={s} {pol:>6} B={B:<5} acc={acc:.4f} f1={f1:.4f} "
                      f"({time.perf_counter()-ts:.0f}s)", flush=True)
        # crash-safe incremental flush
        out_path.write_text(json.dumps(
            {"status": "in_progress", "seeds_done": list(per_seed.keys()),
             "n_neurons": len(ids), "n_classes": n_cls, "budgets": budgets,
             "per_seed": {str(k): v for k, v in per_seed.items()}}, indent=2))
        print(f"  [flushed seed {s}; elapsed {time.perf_counter()-grand:.0f}s]", flush=True)

    agg = aggregate(per_seed, seeds, budgets)
    Bs = sorted(budgets)
    tests = {
        f"economy_headline_coarseZ{args.coarse_zoom}B{Bs[0]}_vs_fineZ{args.fine_zoom}B{Bs[-1]}":
            paired(per_seed, seeds, "coarse", Bs[0], "fine", Bs[-1]),
        f"matched_budget_coarse_vs_fine_B{Bs[-1]}":
            paired(per_seed, seeds, "coarse", Bs[-1], "fine", Bs[-1]),
        f"matched_budget_coarse_vs_fine_B{Bs[0]}":
            paired(per_seed, seeds, "coarse", Bs[0], "fine", Bs[0]),
    }
    multi_tests = {str(B): paired(per_seed, seeds, "multi", B, "coarse", B) for B in budgets}

    out = {"label_scheme": args.label_scheme, "coarse_zoom": args.coarse_zoom,
           "fine_zoom": args.fine_zoom, "n_neurons": len(ids), "n_classes": n_cls,
           "budgets": budgets, "seeds": seeds, "epochs": args.epochs, "status": "complete",
           "per_seed": {str(k): v for k, v in per_seed.items()}, "aggregate": agg,
           "paired_tests": tests, "multi_vs_coarse": multi_tests}
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.output} (total {time.perf_counter()-grand:.0f}s)", flush=True)

    h = tests[f"economy_headline_coarseZ{args.coarse_zoom}B{Bs[0]}_vs_fineZ{args.fine_zoom}B{Bs[-1]}"]
    cm = agg["coarse"][str(Bs[0])]; fm = agg["fine"][str(Bs[-1])]
    pstr = "n/a" if h["p"] is None else f"{h['p']:.2e}"
    print(f"\n*** HEADLINE: coarse@z{args.coarse_zoom},B{Bs[0]} acc={cm['acc_mean']}+/-{cm['acc_se']} "
          f"(f1={cm['f1_mean']}+/-{cm['f1_se']}) vs fine@z{args.fine_zoom},B{Bs[-1]} "
          f"acc={fm['acc_mean']}+/-{fm['acc_se']} (f1={fm['f1_mean']}+/-{fm['f1_se']}) "
          f"| diff={h['mean_diff']}+/-{h['se_diff']} t={h['t']} p={pstr} ***", flush=True)
    for B in budgets:
        mt = multi_tests[str(B)]
        print(f"    multi-vs-coarse @B{B}: diff={mt['mean_diff']}+/-{mt['se_diff']} "
              f"(multi {'>' if mt['mean_diff'] > 0 else '<='} coarse)", flush=True)


if __name__ == "__main__":
    main()
