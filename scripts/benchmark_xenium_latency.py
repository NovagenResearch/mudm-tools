"""B3-latency: Per-round latency comparison muDM Parquet vs re-extract from raw.

Two AL runs in the same harness with identical seeds, the only difference being
the featurizer:
  - muDM       : XeniumParquetFeaturizer (load once, dict lookup per round)
  - re-extract : XeniumReextractFeaturizer (re-parse cell_feature_matrix every round)

RoundTripLatencyRecorder times each round between the 'uncertainty computed'
marker and the 'oracle returned' marker.

This script re-implements a minimal AL loop locally (rather than calling
run_active_learning) so that the recorder marks can be inserted around the
featurizer call. The canonical harness is intentionally unchanged; this is
the documented Wave 1 trade-off (see plan section "Task 10").
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from mudm_tools.active_learning.baselines import random_uncertainty
from mudm_tools.active_learning.classifiers.xenium_mlp import XeniumMLPClassifier
from mudm_tools.active_learning.featurizers.xenium_parquet import XeniumParquetFeaturizer
from mudm_tools.active_learning.featurizers.xenium_reextract import XeniumReextractFeaturizer
from mudm_tools.active_learning.latency import RoundTripLatencyRecorder
from mudm_tools.active_learning.oracle import SyntheticOracle
from mudm_tools.active_learning.uncertainty import entropy


def _load_cluster_labels(clusters_csv: Path) -> dict[str, int]:
    df = pd.read_csv(clusters_csv)
    return {str(b): int(c) - 1 for b, c in zip(df["Barcode"], df["Cluster"])}


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _run_one(
    *,
    classifier_factory: Callable[[], XeniumMLPClassifier],
    featurizer,
    pool_ids: list[str],
    initial: list[str],
    test_ids: list[str],
    oracle: SyntheticOracle,
    n_rounds: int,
    budget: int,
) -> dict:
    """Run a single AL trajectory with the given featurizer, recording per-round
    latency. The round_scope captures the interval from uncertainty-computed
    to oracle-returned. The featurizer.load_features() call happens INSIDE
    the round scope (after the recorder enters scope and before
    mark_uncertainty_computed) so the muDM-vs-re-extract gap shows up in the
    sample interval.
    """
    recorder = RoundTripLatencyRecorder()
    pool: set[str] = set(pool_ids)
    labeled: list[str] = list(initial)
    labels: list[int] = [oracle.query(fid) for fid in labeled]
    pool -= set(labeled)
    test_labels = oracle.reveal_test_labels(list(test_ids))

    histories: list[dict] = []
    for round_idx in range(n_rounds):
        with recorder.round_scope():
            # Mark the start of the round BEFORE featurizer.load_features so the
            # featurizer cost is included in the recorded interval. The method is
            # named mark_uncertainty_computed for API compatibility, but in this
            # benchmark it semantically marks "start of per-round work". The
            # sampled interval is then: featurizer load + classifier fit + predict
            # + uncertainty + selection + oracle query -- i.e. the full per-round
            # cost the paper compares between muDM and re-extract paths.
            recorder.mark_uncertainty_computed()

            # Re-load features each round. This is the cost being measured:
            # for muDM, it is a dict lookup; for re-extract, it re-parses the
            # gzipped MatrixMarket.
            all_relevant = list(set(labeled) | pool | set(test_ids))
            _ = featurizer.load_features(all_relevant)

            clf = classifier_factory()
            clf.fit(labeled, labels)
            pool_list = list(pool)
            if not pool_list:
                break
            probs = clf.predict_proba(pool_list)
            scores = entropy(probs)

            k = min(budget, len(pool_list))
            top_idx = np.argsort(scores)[-k:][::-1]
            selected = [pool_list[int(i)] for i in top_idx]
            for fid in selected:
                labels.append(oracle.query(fid))
                labeled.append(fid)
                pool.discard(fid)
            recorder.mark_oracle_returned()

            test_probs = clf.predict_proba(list(test_ids))
            test_preds = np.argmax(test_probs, axis=1).tolist()
            acc = float(np.mean([p == t for p, t in zip(test_preds, test_labels)]))
            histories.append({
                "round": round_idx,
                "queries_this_round": len(selected),
                "total_queries": len(labeled),
                "test_accuracy": acc,
            })

    return {
        "rounds": histories,
        "latency_summary": recorder.summary(),
        "latency_samples_s": recorder.samples(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", required=True,
                    help="muDM-tiled Xenium Parquet (input to XeniumParquetFeaturizer)")
    ap.add_argument("--cell-feature-matrix", required=True,
                    help="Raw 10x cell_feature_matrix/ directory (input to XeniumReextractFeaturizer)")
    ap.add_argument("--clusters-csv", required=True,
                    help="10x analysis/clustering/gene_expression_graphclust/clusters.csv")
    ap.add_argument("--n-classes", type=int, default=10)
    ap.add_argument("--initial-labels", type=int, default=50)
    ap.add_argument("--budget-per-round", type=int, default=50)
    ap.add_argument("--n-rounds", type=int, default=10,
                    help="Latency benchmark uses fewer rounds because re-extract is slow")
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--mlp-hidden", type=int, default=128)
    ap.add_argument("--mlp-epochs", type=int, default=50)
    ap.add_argument("--output", required=True,
                    help="Output JSON path (must end with 'B3_latency.json')")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    parquet_feat = XeniumParquetFeaturizer(parquet_path=args.parquet)
    cluster_labels = _load_cluster_labels(Path(args.clusters_csv))
    parquet_table = parquet_feat.feature_table()
    common_ids = sorted(set(parquet_table.keys()) & set(cluster_labels.keys()))
    if not common_ids:
        raise SystemExit(
            "No overlap between Parquet feature_ids and clusters.csv Barcode column."
        )

    reextract_feat = XeniumReextractFeaturizer(cell_feature_matrix_dir=args.cell_feature_matrix)

    per_seed: dict[str, dict] = {}
    last_pool: list[str] = []
    last_test: list[str] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        ids = list(common_ids)
        rng.shuffle(ids)
        n_test = int(len(ids) * args.test_fraction)
        test_ids = ids[:n_test]
        pool_ids = ids[n_test:]
        initial = pool_ids[: args.initial_labels]
        last_pool, last_test = pool_ids, test_ids
        ground_truth = {fid: cluster_labels[fid] for fid in ids}

        def factory():
            return XeniumMLPClassifier(
                feature_table=parquet_table, n_classes=args.n_classes,
                hidden=args.mlp_hidden, epochs=args.mlp_epochs, seed=seed,
            )

        # muDM path
        mudm_run = _run_one(
            classifier_factory=factory,
            featurizer=parquet_feat,
            pool_ids=pool_ids, initial=initial, test_ids=test_ids,
            oracle=SyntheticOracle(ground_truth=ground_truth, test_set=set(test_ids)),
            n_rounds=args.n_rounds, budget=args.budget_per_round,
        )
        # Re-extract path
        reextract_run = _run_one(
            classifier_factory=factory,
            featurizer=reextract_feat,
            pool_ids=pool_ids, initial=initial, test_ids=test_ids,
            oracle=SyntheticOracle(ground_truth=ground_truth, test_set=set(test_ids)),
            n_rounds=args.n_rounds, budget=args.budget_per_round,
        )
        per_seed[str(seed)] = {"mudm": mudm_run, "reextract": reextract_run}

    out_path = Path(args.output)
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark_id": out_path.stem,
        "git_sha_mudm_tools": _git_sha(Path(__file__).resolve().parents[1]),
        "hardware": {
            "cpu_cores": os.cpu_count(),
            "platform": platform.platform(),
        },
        "config": {
            "parquet": args.parquet,
            "cell_feature_matrix": args.cell_feature_matrix,
            "clusters_csv": args.clusters_csv,
            "n_classes": args.n_classes,
            "initial_labels": args.initial_labels,
            "budget_per_round": args.budget_per_round,
            "n_rounds": args.n_rounds,
            "test_fraction": args.test_fraction,
            "seeds": seeds,
            "mlp_hidden": args.mlp_hidden,
            "mlp_epochs": args.mlp_epochs,
        },
        "dataset": "Xenium_FFPE_Human_Breast_Cancer_Rep1",
        "n_pool": len(last_pool),
        "n_test": len(last_test),
        "per_seed": per_seed,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
