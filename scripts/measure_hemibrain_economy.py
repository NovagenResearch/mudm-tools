#!/usr/bin/env python3
"""Rigorous economy measurements for the muDM "right-sized LOD" claim.

Produces, for each input budget B, the three numbers we want behind the paper's
compute-economy claim:
  - analytical forward-pass FLOPs per sample (exact, from the architecture);
  - measured peak GPU memory for one training step (forward + backward);
  - on-disk storage per LOD, read from the parquet metadata (no decompression).

Output: results/hemibrain_economy.json — a single artifact that grounds the
"~16x fewer vertices / FLOPs and ~233x less storage at equal-or-better accuracy"
claim with measured numbers rather than handwaving.

Notes:
  * The two-branch model and total budget B are identical across the fine /
    coarse / multi policies, so per-step FLOPs and memory depend only on B, not
    on the scale composition. Hence we sweep B, not policies.
  * Runs on cuda:1 by default to avoid contention with a probe running on cuda:0.

Usage::
    uv run python scripts/measure_hemibrain_economy.py \
        --tiles data/hemibrain/tiles/hemibrain/tiles.parquet \
        --budgets 256,1024,4096 --device cuda:1 \
        --output ../results/hemibrain_economy.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_hemibrain_multiscale import TwoBranchNet  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]


def analytical_flops_forward_per_sample(B: int, n_cls: int) -> int:
    """Forward FLOPs per sample for TwoBranchNet at total budget B.

    Each branch: PointNet over K = B/2 points via Conv1d(3->64->128->256) +
    max-pool. FC head: Linear(512->128) -> Linear(128->n_cls). BN/ReLU/max-pool
    are O(K) with small constants, ignored.
    """
    K = B // 2
    # Per-point MACs across the three Conv1d(kernel=1) layers in one branch
    per_branch_macs = K * (3 * 64 + 64 * 128 + 128 * 256)  # = K * 41152
    fc_macs = (2 * 256) * 128 + 128 * n_cls
    total_macs = 2 * per_branch_macs + fc_macs
    return 2 * total_macs  # 2 FLOPs / MAC


def measure_peak_memory(B: int, n_cls: int, device: torch.device, bs: int) -> dict:
    """One training step (fwd + bwd + Adam) at total budget B; record peak mem."""
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)
        # Force CUDA context initialization on this device so reset_peak_memory_stats works
        _ = torch.empty(1, device=device); torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    model = TwoBranchNet(n_cls).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    K = B // 2
    a = torch.randn(bs, K, 3, device=device)
    b = torch.randn(bs, K, 3, device=device)
    y = torch.randint(0, n_cls, (bs,), device=device)
    opt.zero_grad()
    out = model(a, b)
    loss = nn.functional.cross_entropy(out, y)
    loss.backward()
    opt.step()
    peak = torch.cuda.max_memory_allocated(device)
    n_params = sum(p.numel() for p in model.parameters())
    del model, opt, a, b, y, out, loss
    torch.cuda.empty_cache()
    return {"peak_bytes": int(peak), "peak_MB": round(peak / 1048576, 1),
            "n_params": int(n_params)}


def lod_storage_per_zoom(tiles_path: str) -> dict:
    """Per-LOD on-disk storage from parquet metadata (no decompression)."""
    pf = pq.ParquetFile(tiles_path)
    z = pf.read(columns=["zoom"]).column("zoom").to_numpy()
    per = collections.defaultdict(lambda: {"compressed": 0, "uncompressed": 0, "rows": 0})
    offset = 0
    for rg in range(pf.metadata.num_row_groups):
        rgm = pf.metadata.row_group(rg)
        n = rgm.num_rows
        rg_zooms = z[offset:offset + n]; offset += n
        zc = collections.Counter(rg_zooms.tolist())
        pos_comp = pos_uncomp = 0
        for ci in range(rgm.num_columns):
            col = rgm.column(ci)
            if col.path_in_schema == "positions":
                pos_comp = col.total_compressed_size
                pos_uncomp = col.total_uncompressed_size
                break
        for zoom, cnt in zc.items():
            per[int(zoom)]["compressed"] += pos_comp * cnt // n
            per[int(zoom)]["uncompressed"] += pos_uncomp * cnt // n
            per[int(zoom)]["rows"] += cnt
    fine_unc = per.get(3, {}).get("uncompressed", 1) or 1
    out = {}
    for zoom, v in per.items():
        out[str(zoom)] = {
            "rows": v["rows"],
            "total_verts": v["uncompressed"] // 12,
            "compressed_GB": round(v["compressed"] / 1e9, 2),
            "uncompressed_GB": round(v["uncompressed"] / 1e9, 2),
            "ratio_to_zoom3": round(v["uncompressed"] / fine_unc, 5),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--budgets", default="256,1024,4096")
    ap.add_argument("--n-cls", type=int, default=10)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--output", default="../results/hemibrain_economy.json")
    args = ap.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # 1) FLOPs + peak memory per budget
    per_budget = {}
    print("\n=== per-budget compute ===")
    for B in budgets:
        flops = analytical_flops_forward_per_sample(B, args.n_cls)
        mem = measure_peak_memory(B, args.n_cls, device, args.bs)
        per_budget[str(B)] = {
            "forward_FLOPs_per_sample": flops,
            "forward_MFLOPs_per_sample": round(flops / 1e6, 1),
            "peak_step_MB": mem["peak_MB"],
            "n_params": mem["n_params"],
        }
        print(f"  B={B:<5} forward {flops/1e6:>6.1f} MFLOPs/sample  "
              f"peak step mem {mem['peak_MB']:>6.1f} MB  "
              f"params={mem['n_params']:,}", flush=True)

    # 2) Storage per LOD
    storage = lod_storage_per_zoom(args.tiles)
    print("\n=== per-zoom storage ===")
    for zoom in sorted(storage, key=int):
        s = storage[zoom]
        print(f"  zoom {zoom}: {s['compressed_GB']:>6.2f} GB compressed / "
              f"{s['uncompressed_GB']:>6.2f} GB uncompressed  "
              f"{s['rows']:>10,} rows / {s['total_verts']:>14,} verts  "
              f"ratio_to_z3={s['ratio_to_zoom3']*100:>6.2f}%", flush=True)

    # 3) Headline economy: coarse-LOD-pipeline (B=min budget, zoom=0) vs
    #    fine-LOD-pipeline (B=max budget, zoom=3).
    Bs = sorted(budgets)
    headline = {
        "coarse_pipeline": {
            "budget_B": Bs[0],
            "lod": 0,
            "forward_MFLOPs": per_budget[str(Bs[0])]["forward_MFLOPs_per_sample"],
            "peak_step_MB": per_budget[str(Bs[0])]["peak_step_MB"],
            "lod_storage_GB": storage.get("0", {}).get("uncompressed_GB"),
        },
        "fine_pipeline": {
            "budget_B": Bs[-1],
            "lod": 3,
            "forward_MFLOPs": per_budget[str(Bs[-1])]["forward_MFLOPs_per_sample"],
            "peak_step_MB": per_budget[str(Bs[-1])]["peak_step_MB"],
            "lod_storage_GB": storage.get("3", {}).get("uncompressed_GB"),
        },
    }
    # ratios fine/coarse (how much MORE the fine-LOD pipeline costs)
    c = headline["coarse_pipeline"]; f = headline["fine_pipeline"]
    ratios = {
        "input_vertices_ratio": f["budget_B"] / c["budget_B"],
        "forward_FLOPs_ratio": f["forward_MFLOPs"] / c["forward_MFLOPs"],
        "peak_memory_ratio": (f["peak_step_MB"] / c["peak_step_MB"]) if c["peak_step_MB"] else None,
        "storage_ratio": ((f["lod_storage_GB"] / c["lod_storage_GB"])
                          if c["lod_storage_GB"] else None),
    }
    headline["fine_over_coarse_ratios"] = {k: round(v, 1) if v else v for k, v in ratios.items()}
    print("\n=== headline economy: coarse-LOD pipeline vs fine-LOD pipeline ===")
    print(f"  Input vertices ratio (fine/coarse):  {ratios['input_vertices_ratio']:>6.1f}x")
    print(f"  Forward FLOPs ratio:                  {ratios['forward_FLOPs_ratio']:>6.1f}x")
    if ratios["peak_memory_ratio"]:
        print(f"  Peak step memory ratio:               {ratios['peak_memory_ratio']:>6.1f}x")
    if ratios["storage_ratio"]:
        print(f"  On-disk storage ratio:                {ratios['storage_ratio']:>6.1f}x")

    out = {
        "device": str(device), "n_cls": args.n_cls, "batch_size": args.bs,
        "budgets": budgets,
        "per_budget": per_budget,
        "per_zoom_storage": storage,
        "headline": headline,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
