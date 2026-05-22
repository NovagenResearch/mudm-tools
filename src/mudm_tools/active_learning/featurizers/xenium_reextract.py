"""Re-extract Xenium gene-expression vectors from raw cell_feature_matrix every
time ``load_features`` is called. This is the baseline against which the
muDM Parquet path is compared in the round-trip latency benchmark.

The slowness is intentional: each AL round re-pays the cost of parsing the
gzipped MatrixMarket file, mapping barcodes to cell_ids, and assembling a
dense expression vector. The point of the benchmark is to show that the
muDM Parquet path eliminates this cost."""
from __future__ import annotations

import gzip
import io
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.io import mmread


def _parse_cell_feature_matrix(cfm_dir: Path, cell_ids: Sequence[str]) -> dict[str, np.ndarray]:
    """Parse a 10x cell_feature_matrix directory and return the gene-expression
    vector for the given cell_ids. Implementation calls mmread on the gzipped
    MatrixMarket file each time -- no caching.

    Note: tests monkeypatch this function, so its real-data behaviour does not
    need to be exercised by unit tests. The integration test (run as part of B3-
    latency benchmark) validates against the actual 10x Rep1 cell_feature_matrix.
    """
    matrix_path = cfm_dir / "matrix.mtx.gz"
    barcodes_path = cfm_dir / "barcodes.tsv.gz"

    with gzip.open(matrix_path, "rb") as fh:
        matrix = mmread(io.BytesIO(fh.read()))
    matrix = matrix.tocsc()

    with gzip.open(barcodes_path, "rt") as fh:
        barcodes = [line.strip() for line in fh]

    barcode_to_idx = {bc: i for i, bc in enumerate(barcodes)}

    out: dict[str, np.ndarray] = {}
    for cid in cell_ids:
        # 10x barcode == cell_id in our convention; if not, the caller has
        # already mapped them.
        idx = barcode_to_idx.get(cid)
        if idx is None:
            raise KeyError(f"cell_id {cid!r} not found in barcodes.tsv.gz")
        out[cid] = np.asarray(matrix[:, idx].toarray()).flatten().astype(np.float32)
    return out


class XeniumReextractFeaturizer:
    def __init__(self, cell_feature_matrix_dir: str) -> None:
        self._cfm = Path(cell_feature_matrix_dir)

    def load_features(self, feature_ids: Sequence[str]) -> dict[str, np.ndarray]:
        return _parse_cell_feature_matrix(self._cfm, feature_ids)
