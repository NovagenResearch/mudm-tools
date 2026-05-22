"""B3-Xenium: Active-learning experiment on 10x Xenium FFPE breast cancer Rep1.

Drives the dataset-agnostic AL harness with:
  classifier_factory : XeniumMLPClassifier
  featurizer         : XeniumParquetFeaturizer (muDM Parquet read; fast)
  oracle             : SyntheticOracle over Xenium graph-cluster labels
  uncertainty_fn     : entropy (default) or margin (--uncertainty-fn margin)

Also runs the random-sampling baseline in the same harness, then writes a
single JSON conforming to paper/benchmark_results/_schema_validator.py.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from mudm_tools.active_learning.baselines import random_uncertainty
from mudm_tools.active_learning.classifiers.xenium_mlp import XeniumMLPClassifier
from mudm_tools.active_learning.featurizers.xenium_parquet import XeniumParquetFeaturizer
from mudm_tools.active_learning.harness import run_active_learning
from mudm_tools.active_learning.oracle import SyntheticOracle
from mudm_tools.active_learning.uncertainty import entropy, margin


def _load_cluster_labels(clusters_csv: Path) -> dict[str, int]:
    """Load 10x graph-cluster assignments. Convention: 'Barcode' column,
    'Cluster' column (1-indexed). Convert to 0-indexed int labels."""
    df = pd.read_csv(clusters_csv)
    return {str(b): int(c) - 1 for b, c in zip(df["Barcode"], df["Cluster"])}


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", required=True,
                    help="muDM-tiled Xenium Parquet path (output of xenium_to_mudm + tiling)")
    ap.add_argument("--clusters-csv", required=True,
                    help="10x analysis/clustering/gene_expression_graphclust/clusters.csv")
    ap.add_argument("--n-classes", type=int, default=10,
                    help="Number of cluster classes (default 10 -- Xenium Rep1 graph-cluster default)")
    ap.add_argument("--initial-labels", type=int, default=50)
    ap.add_argument("--budget-per-round", type=int, default=50)
    ap.add_argument("--n-rounds", type=int, default=20)
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--seeds", default="0,1,2",
                    help="Comma-separated seed list (e.g. '0,1,2')")
    ap.add_argument("--uncertainty-fn", choices=("entropy", "margin"), default="entropy")
    ap.add_argument("--mlp-hidden", type=int, default=128)
    ap.add_argument("--mlp-epochs", type=int, default=100)
    ap.add_argument("--output", required=True,
                    help="Output JSON path (must end with 'B3_xenium_al.json' to satisfy validator)")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    uncertainty_fn = entropy if args.uncertainty_fn == "entropy" else margin

    featurizer = XeniumParquetFeaturizer(parquet_path=args.parquet)
    feature_table = featurizer.feature_table()
    cluster_labels = _load_cluster_labels(Path(args.clusters_csv))

    # Restrict to feature_ids that exist in both the Parquet and the cluster CSV.
    common_ids = sorted(set(feature_table.keys()) & set(cluster_labels.keys()))
    if not common_ids:
        raise SystemExit(
            "No overlap between Parquet feature_ids and clusters.csv Barcode column. "
            "Check that the Xenium ingest produced feature_ids that match the 10x barcodes."
        )

    per_seed_results: dict[str, dict] = {}
    last_pool_ids: list[str] = []
    last_test_ids: list[str] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        shuffled = list(common_ids)
        rng.shuffle(shuffled)
        n_test = int(len(shuffled) * args.test_fraction)
        test_ids = shuffled[:n_test]
        pool_ids = shuffled[n_test:]
        initial = pool_ids[: args.initial_labels]
        last_pool_ids = pool_ids
        last_test_ids = test_ids

        ground_truth = {fid: cluster_labels[fid] for fid in shuffled}
        oracle_uncert = SyntheticOracle(ground_truth=ground_truth, test_set=set(test_ids))
        oracle_random = SyntheticOracle(ground_truth=ground_truth, test_set=set(test_ids))

        def factory_uncert():
            return XeniumMLPClassifier(
                feature_table=feature_table, n_classes=args.n_classes,
                hidden=args.mlp_hidden, epochs=args.mlp_epochs, seed=seed,
            )

        def factory_random():
            return XeniumMLPClassifier(
                feature_table=feature_table, n_classes=args.n_classes,
                hidden=args.mlp_hidden, epochs=args.mlp_epochs, seed=seed,
            )

        uncert_history = run_active_learning(
            classifier_factory=factory_uncert,
            pool_ids=pool_ids,
            initial_labeled_ids=initial,
            test_ids=test_ids,
            oracle=oracle_uncert,
            uncertainty_fn=uncertainty_fn,
            budget_per_round=args.budget_per_round,
            n_rounds=args.n_rounds,
        )
        random_history = run_active_learning(
            classifier_factory=factory_random,
            pool_ids=pool_ids,
            initial_labeled_ids=initial,
            test_ids=test_ids,
            oracle=oracle_random,
            uncertainty_fn=random_uncertainty,
            budget_per_round=args.budget_per_round,
            n_rounds=args.n_rounds,
        )
        per_seed_results[str(seed)] = {
            "uncertainty": uncert_history.to_json_dict(),
            "random": random_history.to_json_dict(),
        }

    out_path = Path(args.output)
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark_id": out_path.stem,  # must match filename per schema validator
        "git_sha_mudm_tools": _git_sha(Path(__file__).resolve().parents[1]),
        "hardware": {
            "cpu_cores": os.cpu_count(),
            "platform": platform.platform(),
        },
        "config": {
            "parquet": args.parquet,
            "clusters_csv": args.clusters_csv,
            "n_classes": args.n_classes,
            "initial_labels": args.initial_labels,
            "budget_per_round": args.budget_per_round,
            "n_rounds": args.n_rounds,
            "test_fraction": args.test_fraction,
            "seeds": seeds,
            "uncertainty_fn": args.uncertainty_fn,
            "mlp_hidden": args.mlp_hidden,
            "mlp_epochs": args.mlp_epochs,
        },
        "dataset": "Xenium_FFPE_Human_Breast_Cancer_Rep1",
        "n_pool": len(last_pool_ids),
        "n_test": len(last_test_ids),
        "per_seed": per_seed_results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
