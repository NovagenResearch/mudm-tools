"""AL harness tests: dataset-agnostic loop, pluggable classifier, budget bookkeeping."""
import numpy as np
import pytest


def test_classifier_protocol_accepts_fit_and_predict_proba():
    """A Classifier must expose fit(feature_ids, labels) and predict_proba(feature_ids)."""
    from mudm_tools.active_learning.classifiers.base import Classifier
    assert hasattr(Classifier, "fit")
    assert hasattr(Classifier, "predict_proba")


def test_al_harness_runs_full_budget():
    """End-to-end: synthetic 100-sample binary problem; harness trains, queries
    uncertain samples, returns history with the requested round count and budget bookkeeping."""
    from collections import Counter

    from mudm_tools.active_learning.harness import run_active_learning
    from mudm_tools.active_learning.oracle import SyntheticOracle
    from mudm_tools.active_learning.uncertainty import entropy

    rng = np.random.default_rng(0)
    feature_ids = [f"c{i}" for i in range(100)]
    true_labels = dict(zip(feature_ids, rng.integers(0, 2, size=100).tolist()))
    test_ids = feature_ids[:20]
    pool_ids = feature_ids[20:]

    oracle = SyntheticOracle(ground_truth=true_labels, test_set=set(test_ids))

    class ToyClassifier:
        n_classes = 2

        def __init__(self):
            self._modal = 0

        def fit(self, ids, labels):
            if len(labels) > 0:
                self._modal = Counter(labels).most_common(1)[0][0]

        def predict_proba(self, ids):
            p = np.zeros((len(ids), 2))
            for i in range(len(ids)):
                p[i, self._modal] = 0.55 + 0.05 * rng.normal()
                p[i, 1 - self._modal] = 1 - p[i, 0]
            return np.clip(p, 1e-6, 1 - 1e-6)

    history = run_active_learning(
        classifier_factory=ToyClassifier,
        pool_ids=pool_ids,
        initial_labeled_ids=pool_ids[:5],
        test_ids=test_ids,
        oracle=oracle,
        uncertainty_fn=entropy,
        budget_per_round=5,
        n_rounds=10,
    )
    assert len(history.rounds) == 10
    # Initial 5 + 10 rounds × 5 budget = 55 total labeled
    assert history.total_queries == 55
    assert all(0.0 <= r.test_accuracy <= 1.0 for r in history.rounds)
    assert all(r.queries_this_round == 5 for r in history.rounds)


def test_al_harness_initial_label_count_correct():
    """If initial_labeled_ids has 7 items and budget is 3, after 0 rounds the
    history should be empty but oracle.queries_made should reflect 7."""
    from collections import Counter

    from mudm_tools.active_learning.harness import run_active_learning
    from mudm_tools.active_learning.oracle import SyntheticOracle
    from mudm_tools.active_learning.uncertainty import entropy

    rng = np.random.default_rng(0)
    fids = [f"f{i}" for i in range(50)]
    truth = dict(zip(fids, rng.integers(0, 2, size=50).tolist()))
    test_ids = fids[:10]
    pool = fids[10:]
    oracle = SyntheticOracle(ground_truth=truth, test_set=set(test_ids))

    class C:
        n_classes = 2
        def fit(self, ids, labels):
            self._m = Counter(labels).most_common(1)[0][0] if labels else 0
        def predict_proba(self, ids):
            p = np.full((len(ids), 2), 0.5)
            return p

    history = run_active_learning(
        classifier_factory=C,
        pool_ids=pool,
        initial_labeled_ids=pool[:7],
        test_ids=test_ids,
        oracle=oracle,
        uncertainty_fn=entropy,
        budget_per_round=3,
        n_rounds=0,
    )
    assert len(history.rounds) == 0
    assert oracle.queries_made == 7  # initial labels were drawn from oracle


def test_random_baseline_runs():
    """Random baseline is the same harness with a random uncertainty function."""
    from mudm_tools.active_learning.baselines import random_uncertainty

    probs = np.random.rand(50, 3)
    u = random_uncertainty(probs)
    assert u.shape == (50,)
    assert u.min() >= 0.0 and u.max() <= 1.0


def test_latency_recorder_captures_roundtrip():
    """RoundTripLatencyRecorder times the interval from uncertainty computation
    to oracle return across a round."""
    import time

    from mudm_tools.active_learning.latency import RoundTripLatencyRecorder

    rec = RoundTripLatencyRecorder()
    with rec.round_scope():
        rec.mark_uncertainty_computed()
        time.sleep(0.01)
        rec.mark_oracle_returned()
    samples = rec.samples()
    assert len(samples) == 1
    assert samples[0] >= 0.01


def test_latency_recorder_summary_stats():
    """summary() returns mean/median/stdev across samples."""
    from mudm_tools.active_learning.latency import RoundTripLatencyRecorder

    rec = RoundTripLatencyRecorder()
    rec._samples = [0.01, 0.02, 0.03]  # direct injection for test stability
    s = rec.summary()
    assert s["n"] == 3
    assert s["mean_s"] == pytest.approx(0.02)
    assert s["median_s"] == pytest.approx(0.02)


def test_latency_recorder_summary_empty():
    from mudm_tools.active_learning.latency import RoundTripLatencyRecorder
    rec = RoundTripLatencyRecorder()
    s = rec.summary()
    assert s == {"n": 0}


def test_history_to_json_dict_is_json_serializable():
    """ALHistory.to_json_dict() must produce JSON-serializable types only
    (no numpy scalars slipping through)."""
    import json

    from mudm_tools.active_learning.harness import ALHistory, RoundResult

    h = ALHistory(
        rounds=[
            RoundResult(
                round_index=np.int64(0),
                queries_this_round=np.int64(5),
                total_queries=np.int64(5),
                test_accuracy=np.float64(0.8),
                test_macro_f1=np.float64(0.75),
                round_wall_time_s=0.123,
            )
        ],
        total_queries=np.int64(5),
    )
    blob = json.dumps(h.to_json_dict())
    assert "0.8" in blob
    assert "5" in blob


def test_package_top_level_exports():
    """The top-level package re-exports the public API."""
    import mudm_tools.active_learning as al

    expected = {
        "run_active_learning",
        "ALHistory",
        "RoundResult",
        "SyntheticOracle",
        "Classifier",
        "margin",
        "entropy",
        "random_uncertainty",
        "RoundTripLatencyRecorder",
    }
    assert expected.issubset(set(al.__all__))
    for name in expected:
        assert hasattr(al, name), f"missing top-level export {name!r}"
