"""Dataset-agnostic active-learning loop.

The harness operates on feature IDs and delegates all dataset-specific work
(feature extraction, classifier training) to pluggable components. This is the
core invariant of the paper: the harness is identical across datasets;
only classifier_factory varies."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .classifiers.base import Classifier
from .oracle import SyntheticOracle


@dataclass
class RoundResult:
    round_index: int
    queries_this_round: int
    total_queries: int
    test_accuracy: float
    test_macro_f1: float
    round_wall_time_s: float


@dataclass
class ALHistory:
    rounds: list[RoundResult] = field(default_factory=list)
    total_queries: int = 0

    def to_json_dict(self) -> dict:
        return {
            "total_queries": int(self.total_queries),
            "rounds": [
                {
                    "round": int(r.round_index),
                    "queries_this_round": int(r.queries_this_round),
                    "total_queries": int(r.total_queries),
                    "test_accuracy": float(r.test_accuracy),
                    "test_macro_f1": float(r.test_macro_f1),
                    "round_wall_time_s": float(r.round_wall_time_s),
                }
                for r in self.rounds
            ],
        }


def run_active_learning(
    *,
    classifier_factory: Callable[[], Classifier],
    pool_ids: Sequence[str],
    initial_labeled_ids: Sequence[str],
    test_ids: Sequence[str],
    oracle: SyntheticOracle,
    uncertainty_fn: Callable[[np.ndarray], np.ndarray],
    budget_per_round: int,
    n_rounds: int,
) -> ALHistory:
    """Run an active-learning experiment.

    The harness is dataset-agnostic. classifier_factory must produce a fresh
    Classifier each round; pool_ids and initial_labeled_ids are feature-id strings
    that the classifier knows how to materialize from disk. test_ids are held-out
    feature_ids that the oracle will refuse to label (prevents leakage); their
    labels are revealed only via oracle.reveal_test_labels() for evaluation."""
    pool: set[str] = set(pool_ids)
    labeled: list[str] = list(initial_labeled_ids)
    labels: list[int] = [oracle.query(fid) for fid in labeled]
    pool -= set(labeled)

    test_labels = oracle.reveal_test_labels(list(test_ids))

    history = ALHistory()
    history.total_queries = len(labeled)

    for round_idx in range(n_rounds):
        round_t0 = perf_counter()

        clf = classifier_factory()
        clf.fit(labeled, labels)

        pool_list = list(pool)
        if not pool_list:
            break

        probs = clf.predict_proba(pool_list)
        scores = uncertainty_fn(probs)
        # Top-budget most uncertain
        select_n = min(budget_per_round, len(pool_list))
        selected_idx = np.argsort(scores)[-select_n:][::-1]
        selected = [pool_list[int(i)] for i in selected_idx]

        for fid in selected:
            labels.append(oracle.query(fid))
            labeled.append(fid)
            pool.discard(fid)

        # Evaluate on test set
        test_probs = clf.predict_proba(list(test_ids))
        test_preds = np.argmax(test_probs, axis=1).tolist()
        acc = float(accuracy_score(test_labels, test_preds))
        mf1 = float(f1_score(test_labels, test_preds, average="macro", zero_division=0.0))

        round_t1 = perf_counter()
        history.rounds.append(
            RoundResult(
                round_index=round_idx,
                queries_this_round=len(selected),
                total_queries=len(labeled),
                test_accuracy=acc,
                test_macro_f1=mf1,
                round_wall_time_s=round_t1 - round_t0,
            )
        )
        history.total_queries = len(labeled)

    return history
