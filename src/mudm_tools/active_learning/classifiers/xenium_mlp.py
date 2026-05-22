"""Xenium MLP classifier: simple feed-forward net over fixed-length gene-expression
vectors keyed by feature_id. Conforms to the AL Classifier protocol so it plugs
into the dataset-agnostic harness without changing the harness."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class XeniumMLPClassifier:
    """MLP over per-feature_id gene-expression vectors.

    Parameters
    ----------
    feature_table : mapping feature_id -> 1D numpy array (gene_panel_dimension,)
        The MLP reads features from this dict; the harness only ever sees the IDs.
    n_classes : int
        Number of output classes (e.g. number of cell-type clusters in Xenium Rep1).
    hidden : int
        Hidden layer size.
    epochs : int
        Training epochs per ``fit`` call.
    lr : float
        Adam learning rate.
    seed : int
        Reproducibility seed for torch + numpy.
    """

    def __init__(
        self,
        feature_table: Mapping[str, np.ndarray],
        n_classes: int,
        hidden: int = 128,
        epochs: int = 100,
        lr: float = 1e-3,
        seed: int = 0,
    ) -> None:
        self._features = feature_table
        self._n_classes = int(n_classes)
        self._hidden = int(hidden)
        self._epochs = int(epochs)
        self._lr = float(lr)
        self._seed = int(seed)

        # Infer feature dimensionality from the first entry
        first = next(iter(feature_table.values()))
        self._dim = int(first.shape[0])

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: nn.Module | None = None

    @property
    def n_classes(self) -> int:
        return self._n_classes

    def _build_model(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self._dim, self._hidden),
            nn.ReLU(),
            nn.Linear(self._hidden, self._hidden),
            nn.ReLU(),
            nn.Linear(self._hidden, self._n_classes),
        )

    def fit(self, feature_ids: Sequence[str], labels: Sequence[int]) -> None:
        torch.manual_seed(self._seed)
        np.random.seed(self._seed)

        X = np.stack([self._features[fid] for fid in feature_ids]).astype(np.float32)
        y = np.asarray(labels, dtype=np.int64)
        X_t = torch.from_numpy(X).to(self._device)
        y_t = torch.from_numpy(y).to(self._device)

        self._model = self._build_model().to(self._device)
        opt = torch.optim.Adam(self._model.parameters(), lr=self._lr)
        loss_fn = nn.CrossEntropyLoss()
        self._model.train()
        for _ in range(self._epochs):
            opt.zero_grad()
            logits = self._model(X_t)
            loss = loss_fn(logits, y_t)
            loss.backward()
            opt.step()

    def predict_proba(self, feature_ids: Sequence[str]) -> np.ndarray:
        assert self._model is not None, "predict_proba called before fit"
        X = np.stack([self._features[fid] for fid in feature_ids]).astype(np.float32)
        X_t = torch.from_numpy(X).to(self._device)
        self._model.eval()
        with torch.no_grad():
            probs = F.softmax(self._model(X_t), dim=1).cpu().numpy()
        return probs
