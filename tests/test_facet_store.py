"""Task 1 + 1b — Xenium converter emits a joined facet store; nothing high-card in tiles.

The converter's NEW ``facets`` path moves high-cardinality / multifaceted per-object
metadata (the per-cell gene-expression vector + a derived categorical) OUT of the
vector tiles and into a sibling, object-ID-keyed parquet store joined at view time.
Tiles keep only ``cell_id`` + the cheap ``total_counts`` scalar — NO ``expression``
key anywhere. Contract: ``facet_contract.md``.

Run with the xenium extra:
    uv run --extra xenium pytest tests/test_facet_store.py -x -q
"""

from __future__ import annotations

import glob
import gzip
import json
from pathlib import Path

import pytest

# Exercises the optional [xenium] extra + scipy/pyarrow for the facet store.
pytest.importorskip("polars")
pytest.importorskip("scipy")
pytest.importorskip("pyarrow")

import polars as pl
import pyarrow.parquet as pq

from mudm_tools.converters.xenium import XeniumConverter

# ---------------------------------------------------------------------------
# Tiny bundle: 3 closed-square cells + a 4-gene MEX matrix (nnz == 4)
# ---------------------------------------------------------------------------

# Panel / matrix-row order is DELIBERATELY NOT alphabetical, so a converter that sorted the long-form by
# matrix-row index would leave the `gene` string column unsorted — distinguishing index-sort (wrong) from
# name-sort (right). fieldenums.gene must preserve THIS order (the CSC var axis); expression.parquet must be
# sorted by gene NAME. Row indices (1-based): GENE_C=1, GENE_A=2, GENE_D=3, GENE_B=4.
GENES = ["GENE_C", "GENE_A", "GENE_D", "GENE_B"]
BARCODES = ["1", "2", "3"]

# Non-zeros (1-based MatrixMarket coords, features × cells):
#   cell 1: GENE_A=5, GENE_B=3   (total 8)
#   cell 2: GENE_C=1             (total 1)
#   cell 3: GENE_A=9             (total 9)
# => nnz == 4
NONZEROS = [
    (2, 1, 5),  # GENE_A (row 2), cell 1
    (4, 1, 3),  # GENE_B (row 4), cell 1
    (1, 2, 1),  # GENE_C (row 1), cell 2
    (2, 3, 9),  # GENE_A (row 2), cell 3
]


def _square(cell_id: str, x0: float, y0: float, s: float) -> dict:
    """A closed-square polygon as cell_boundaries vertex rows."""
    ring = [(x0, y0), (x0 + s, y0), (x0 + s, y0 + s), (x0, y0 + s), (x0, y0)]
    return {
        "cell_id": [cell_id] * len(ring),
        "vertex_x": [p[0] for p in ring],
        "vertex_y": [p[1] for p in ring],
    }


def _build_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    # cell_boundaries.parquet — 3 closed squares, ids "1"/"2"/"3"
    rows = {"cell_id": [], "vertex_x": [], "vertex_y": []}
    for cid, (x0, y0) in zip(BARCODES, [(10.0, 10.0), (40.0, 10.0), (10.0, 40.0)]):
        sq = _square(cid, x0, y0, 20.0)
        rows["cell_id"].extend(sq["cell_id"])
        rows["vertex_x"].extend(sq["vertex_x"])
        rows["vertex_y"].extend(sq["vertex_y"])
    pl.DataFrame(rows).write_parquet(bundle / "cell_boundaries.parquet")

    # experiment.xenium — pixel size only
    (bundle / "experiment.xenium").write_text(json.dumps({"pixel_size": 0.2125}))

    # cell_feature_matrix/ MEX triplet (4 genes × 3 cells)
    mtx_dir = bundle / "cell_feature_matrix"
    mtx_dir.mkdir()
    header = (
        "%%MatrixMarket matrix coordinate integer general\n"
        "%\n"
        f"{len(GENES)} {len(BARCODES)} {len(NONZEROS)}\n"
    )
    body = "".join(f"{r} {c} {v}\n" for (r, c, v) in NONZEROS)
    with gzip.open(mtx_dir / "matrix.mtx.gz", "wt") as f:
        f.write(header + body)
    with gzip.open(mtx_dir / "barcodes.tsv.gz", "wt") as f:
        f.write("\n".join(BARCODES) + "\n")
    with gzip.open(mtx_dir / "features.tsv.gz", "wt") as f:
        f.write("".join(f"ENSG_{g[-1]}\t{g}\tGene Expression\n" for g in GENES))

    # NO morphology image -> raster auto-skips.
    return bundle


def _all_cells_tags(out: Path) -> list[dict]:
    """Collect every cells_*.parquet tags map across all zoom partitions."""
    tags: list[dict] = []
    leaves = glob.glob(str(out / "features.parquet" / "zoom=*" / "cells_*.parquet"))
    assert leaves, "no cells_*.parquet partitions found"
    for leaf in leaves:
        t = pq.ParquetFile(leaf).read()  # per-leaf avoids hive `zoom` merge
        for row in t.column("tags").to_pylist():
            tags.append(dict(row))
    return tags


# ---------------------------------------------------------------------------
# Task 1 — facets enabled
# ---------------------------------------------------------------------------


@pytest.fixture
def converted(tmp_path):
    bundle = _build_bundle(tmp_path)
    out = tmp_path / "out"
    XeniumConverter().convert(
        str(bundle),
        str(out),
        {"facets": {"enabled": True}, "max_zoom": 1, "temp_dir": str(tmp_path / "t")},
    )
    return out


def test_no_expression_key_total_counts_present_in_every_cells_tag(converted):
    """(a) cells tile tags carry cell_id + total_counts ONLY — never expression."""
    tags = _all_cells_tags(converted)
    assert tags, "expected cells rows"
    for t in tags:
        assert "expression" not in t, f"expression leaked into a cells tag: {t}"
        assert "total_counts" in t, f"total_counts missing from a cells tag: {t}"
        assert "cell_id" in t, f"cell_id missing from a cells tag: {t}"


def test_expression_parquet_sorted_by_gene_nnz_rows(converted):
    """(b) facets/expression.parquet exists, gene non-decreasing, rows == nnz, has cell_id."""
    path = converted / "facets" / "expression.parquet"
    assert path.exists(), "facets/expression.parquet not written"
    table = pq.read_table(path)
    assert table.num_rows == len(NONZEROS), f"rows {table.num_rows} != nnz {len(NONZEROS)}"
    cols = set(table.column_names)
    assert {"cell_id", "gene", "count"} <= cols, f"columns {cols}"
    genes = table.column("gene").to_pylist()
    assert genes == sorted(genes), f"gene column not sorted ascending: {genes}"
    # The four nonzeros: GENE_A appears twice (cells 1 & 3), GENE_B once, GENE_C once.
    assert sorted(genes) == ["GENE_A", "GENE_A", "GENE_B", "GENE_C"]


def test_metadata_facets_block(converted):
    """(c) metadata.json facets block: ordered gene panel, storage=asset, key=cell_id."""
    meta = json.loads((converted / "metadata.json").read_text())
    assert "facets" in meta, "metadata.json missing facets block"
    facets = meta["facets"]
    assert facets["storage"] == "asset"
    assert facets["key"] == "cell_id"
    assert facets["layer"] == "cells"
    assert facets["fieldenums"]["gene"] == GENES
    assert "expr_tier" not in facets["fieldenums"]  # synthetic categorical dropped
    assert facets["fields"] == {"gene": "vector<int>"}
    # Assets: only the numeric-vector (gene) parquet facet, href relative to dataset dir.
    by_href = {a["href"]: a for a in facets["assets"]}
    assert "facets/expression.parquet" in by_href
    assert "facets/categorical.parquet" not in by_href
    expr_asset = by_href["facets/expression.parquet"]
    assert expr_asset["media_type"] == "application/vnd.apache.parquet"
    assert expr_asset["facet"] == "gene"
    assert expr_asset["layout"] == "long"
    assert expr_asset["sorted_by"] == "gene"
    assert expr_asset["rows"] == len(NONZEROS)


def test_no_categorical_parquet(converted):
    """(d) the synthetic expr_tier categorical was dropped — only the numeric-vector (gene) facet is
    emitted. A categorical facet returns when a dataset carries a REAL one (cell type / cluster / etc.).
    """
    assert not (converted / "facets" / "categorical.parquet").exists()


# ---------------------------------------------------------------------------
# Back-compat — no facets key => no facets/ dir, no metadata facets block
# ---------------------------------------------------------------------------


def test_no_facets_key_emits_no_facet_store(tmp_path):
    bundle = _build_bundle(tmp_path)
    out = tmp_path / "out_nofacets"
    XeniumConverter().convert(
        str(bundle),
        str(out),
        {"max_zoom": 1, "temp_dir": str(tmp_path / "t2")},
    )
    assert not (out / "facets").exists(), "facets/ dir written without facets config"
    meta = json.loads((out / "metadata.json").read_text())
    assert "facets" not in meta, "metadata.json gained a facets block without config"


# ---------------------------------------------------------------------------
# Task 1b — Zarr CSC companion (skipped when zarr absent)
# ---------------------------------------------------------------------------


def test_emit_zarr_appends_zarr_asset(tmp_path):
    pytest.importorskip("zarr")
    bundle = _build_bundle(tmp_path)
    out = tmp_path / "out_zarr"
    XeniumConverter().convert(
        str(bundle),
        str(out),
        {
            "facets": {"enabled": True, "emit_zarr": True},
            "max_zoom": 1,
            "temp_dir": str(tmp_path / "t3"),
        },
    )
    assert (out / "facets" / "expression.zarr").exists(), "expression.zarr not written"
    meta = json.loads((out / "metadata.json").read_text())
    media = [a["media_type"] for a in meta["facets"]["assets"]]
    assert "application/zarr" in media, f"no zarr asset declared: {media}"
    zarr_asset = next(a for a in meta["facets"]["assets"] if a["media_type"] == "application/zarr")
    assert zarr_asset["href"] == "facets/expression.zarr"
    assert zarr_asset["layout"] == "csc"
