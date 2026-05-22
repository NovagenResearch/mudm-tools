"""XeniumMLPClassifier conforms to the AL Classifier protocol and learns from
gene-expression vectors fed via feature IDs."""
from __future__ import annotations

import numpy as np
import pytest


def _toy_feature_table(n: int = 30, n_genes: int = 16) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {f"c{i}": rng.standard_normal(n_genes).astype(np.float32) for i in range(n)}


def test_xenium_mlp_conforms_to_classifier_protocol():
    from mudm_tools.active_learning.classifiers.base import Classifier
    from mudm_tools.active_learning.classifiers.xenium_mlp import XeniumMLPClassifier

    assert hasattr(XeniumMLPClassifier, "fit")
    assert hasattr(XeniumMLPClassifier, "predict_proba")


def test_xenium_mlp_fit_predict_shapes():
    from mudm_tools.active_learning.classifiers.xenium_mlp import XeniumMLPClassifier

    features = _toy_feature_table(n=30, n_genes=16)
    labels_map = {fid: int(i % 3) for i, fid in enumerate(features)}

    clf = XeniumMLPClassifier(
        feature_table=features, n_classes=3, hidden=32, epochs=20, lr=1e-2, seed=0,
    )
    train_ids = list(features.keys())[:25]
    train_labels = [labels_map[fid] for fid in train_ids]
    clf.fit(train_ids, train_labels)

    pred_ids = list(features.keys())[25:]
    probs = clf.predict_proba(pred_ids)
    assert probs.shape == (5, 3)
    # Probability rows sum to ~1
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_xenium_mlp_learns_separable_problem():
    """On a clearly separable 2-class problem, accuracy should exceed random."""
    from mudm_tools.active_learning.classifiers.xenium_mlp import XeniumMLPClassifier

    rng = np.random.default_rng(0)
    n_per_class = 50
    n_genes = 8
    feats: dict[str, np.ndarray] = {}
    labels: dict[str, int] = {}
    for i in range(n_per_class):
        fid = f"a{i}"
        feats[fid] = (rng.standard_normal(n_genes) + 3.0).astype(np.float32)
        labels[fid] = 0
    for i in range(n_per_class):
        fid = f"b{i}"
        feats[fid] = (rng.standard_normal(n_genes) - 3.0).astype(np.float32)
        labels[fid] = 1

    clf = XeniumMLPClassifier(feature_table=feats, n_classes=2, hidden=16, epochs=50, lr=1e-2, seed=0)
    all_ids = list(feats.keys())
    train_ids = all_ids[:80]
    train_labels = [labels[fid] for fid in train_ids]
    clf.fit(train_ids, train_labels)

    test_ids = all_ids[80:]
    probs = clf.predict_proba(test_ids)
    preds = probs.argmax(axis=1)
    truth = np.array([labels[fid] for fid in test_ids])
    accuracy = (preds == truth).mean()
    assert accuracy > 0.8
