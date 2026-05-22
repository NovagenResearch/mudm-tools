"""Xenium featurizers: muDM Parquet path vs re-extract-from-raw baseline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _write_toy_parquet(tmp_path: Path) -> Path:
    """Synthesize a small muDM-style Parquet with cell_id + JSON-encoded
    expression vector under the tags column."""
    out = tmp_path / "toy_xenium.parquet"
    n = 10
    gene_dim = 5
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        expr = rng.standard_normal(gene_dim).round(4).tolist()
        rows.append({
            "feature_id": f"cell_{i}",
            "tags": json.dumps({"cell_id": str(i), "expression": json.dumps(expr)}),
        })
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out)
    return out


def _write_toy_cell_feature_matrix(tmp_path: Path) -> Path:
    """Synthesize a tiny cell_feature_matrix directory with the three 10x files.

    The actual 10x format uses matrix.mtx.gz + barcodes + features; for the
    unit test we monkeypatch the parser to return synthetic vectors. The test
    below validates the call path, not the 10x parser."""
    cfm = tmp_path / "cell_feature_matrix"
    cfm.mkdir()
    (cfm / "barcodes.tsv.gz").write_bytes(b"stub")
    (cfm / "features.tsv.gz").write_bytes(b"stub")
    (cfm / "matrix.mtx.gz").write_bytes(b"stub")
    return cfm


def test_parquet_featurizer_returns_expression_vectors(tmp_path):
    from mudm_tools.active_learning.featurizers.xenium_parquet import (
        XeniumParquetFeaturizer,
    )

    pq_path = _write_toy_parquet(tmp_path)
    feat = XeniumParquetFeaturizer(parquet_path=str(pq_path))
    out = feat.load_features([f"cell_{i}" for i in range(3)])
    assert set(out.keys()) == {"cell_0", "cell_1", "cell_2"}
    for arr in out.values():
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32
        assert arr.shape == (5,)


def test_reextract_featurizer_calls_raw_parser(tmp_path, monkeypatch):
    """The re-extract featurizer must re-parse raw cell_feature_matrix every
    call (no caching). Verify by counting parser invocations."""
    from mudm_tools.active_learning.featurizers import xenium_reextract as mod

    calls = {"n": 0}

    def fake_parse(cfm_dir: Path, cell_ids):
        calls["n"] += 1
        return {cid: np.zeros(5, dtype=np.float32) for cid in cell_ids}

    monkeypatch.setattr(mod, "_parse_cell_feature_matrix", fake_parse)

    cfm = _write_toy_cell_feature_matrix(tmp_path)
    feat = mod.XeniumReextractFeaturizer(cell_feature_matrix_dir=str(cfm))
    _ = feat.load_features(["cell_0", "cell_1"])
    _ = feat.load_features(["cell_2"])
    assert calls["n"] == 2  # No caching


def test_featurizers_package_exports_both():
    """The featurizers package re-exports both classes."""
    from mudm_tools.active_learning.featurizers import (
        XeniumParquetFeaturizer,
        XeniumReextractFeaturizer,
    )
    assert XeniumParquetFeaturizer is not None
    assert XeniumReextractFeaturizer is not None
