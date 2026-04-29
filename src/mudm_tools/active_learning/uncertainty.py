"""Uncertainty scorers for the AL harness.

All functions take class probabilities of shape (N, n_classes) and return (N,)
uncertainty values where higher = more uncertain."""
from __future__ import annotations

import numpy as np


def margin(probs: np.ndarray) -> np.ndarray:
    """Margin uncertainty: 1 - (top-1 prob - top-2 prob). Higher = more uncertain."""
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    return 1.0 - (sorted_probs[:, 0] - sorted_probs[:, 1])


def entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy per row. Higher = more uncertain."""
    safe = np.clip(probs, 1e-12, 1.0)
    return -np.sum(safe * np.log(safe), axis=1)
