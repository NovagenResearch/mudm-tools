"""Read pre-tiled muDM Parquet for Xenium AL training.

The B2 ingest validation confirmed the schema: tags column is a JSON-encoded
mapping with at least 'cell_id' and 'expression' keys; expression is itself
a JSON-encoded list of floats of length equal to gene_panel_dimension.
"""
from __future__ import annotations

import json
from typing import Sequence

import numpy as np
import pyarrow.parquet as pq


class XeniumParquetFeaturizer:
    def __init__(self, parquet_path: str) -> None:
        self._path = parquet_path
        # Eager load on construction; the Parquet is small enough for Xenium Rep1
        # at scale (gene_panel_dimension=541, ~167k cells).
        table = pq.read_table(parquet_path)
        feature_ids = table.column("feature_id").to_pylist()
        tag_strs = table.column("tags").to_pylist()
        self._features: dict[str, np.ndarray] = {}
        for fid, tag_str in zip(feature_ids, tag_strs):
            tags = json.loads(tag_str) if isinstance(tag_str, str) else tag_str
            expr_str = tags.get("expression")
            if expr_str is None:
                continue
            expr = json.loads(expr_str) if isinstance(expr_str, str) else expr_str
            self._features[fid] = np.asarray(expr, dtype=np.float32)

    def load_features(self, feature_ids: Sequence[str]) -> dict[str, np.ndarray]:
        return {fid: self._features[fid] for fid in feature_ids}

    def feature_table(self) -> dict[str, np.ndarray]:
        """Return the full feature table for direct injection into a classifier."""
        return dict(self._features)
