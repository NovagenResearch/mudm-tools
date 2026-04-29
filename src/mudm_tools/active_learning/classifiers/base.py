"""Pluggable classifier protocol: the AL harness operates on feature IDs and does not
see classifier internals. fit() trains on labeled IDs; predict_proba() scores unlabeled."""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class Classifier(Protocol):
    """Minimal interface the AL harness requires from every dataset-specific classifier."""

    def fit(self, feature_ids: Sequence[str], labels: Sequence[int]) -> None:
        """Train on the given labeled feature IDs."""
        ...

    def predict_proba(self, feature_ids: Sequence[str]) -> np.ndarray:
        """Return class probabilities, shape (N, n_classes), for the given feature IDs."""
        ...

    @property
    def n_classes(self) -> int:
        ...
