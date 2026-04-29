"""Baselines for the AL comparison: random sampling.

Pool-uncertainty baseline differs from the muDM-synced arm only in *how* features
are accessed between rounds (raw-data re-extract vs. direct Parquet read), not in
the uncertainty scorer. That distinction is implemented in the latency benchmark
(scripts/benchmark_al_latency.py) rather than as a dedicated function here."""
from __future__ import annotations

import numpy as np


_rng = np.random.default_rng()


def random_uncertainty(probs: np.ndarray) -> np.ndarray:
    """Random uncertainty: ignore probs, return uniform random scores."""
    return _rng.random(size=probs.shape[0])
