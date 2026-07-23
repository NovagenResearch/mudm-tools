"""Path B — zoom-banded per-cell tags in BOTH outputs (PBF + Parquet).

Exercises the tiler capability that the Xenium converter's ``expression_lod``
uses: ``attach_tags_by_id`` (post-attach per-cell tags by matching an existing
tag) + ``set_tag_min_zoom`` (emit a tag only at zoom >= K). The per-cell
``expression`` tag must appear ONLY on fine tiles (z>=3 here) while the cheap
``total_counts`` scalar rides all zooms — in the PBF *and* the Parquet, with no
sidecar. This is the muDM-native representation (like CODEX ``m_<marker>``).
"""

from __future__ import annotations

import glob
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from mudm_tools._rs import StreamingTileGenerator2D
from mudm_tools.tiling2d.pbf_writer import generate_pbf
from mudm_tools.tiling2d.pbf_reader import read_pbf

BOUNDS = (0.0, 0.0, 1.0, 1.0)
EXPR_MIN_ZOOM = 3


def _square(x: float, y: float, s: float, cell_id: str) -> dict:
    """A small square polygon in [0,1]² tagged with a cell_id (mimics what
    ``add_parquet_polygons`` writes: layer_type + cell_id)."""
    ring = [(x, y), (x + s, y), (x + s, y + s), (x, y + s)]
    xy: list[float] = []
    for cx, cy in ring:
        xy.extend([cx, cy])
    return {
        "xy": xy,
        "geom_type": 3,
        "ring_lengths": [len(ring)],
        "min_x": x,
        "min_y": y,
        "max_x": x + s,
        "max_y": y + s,
        "tags": {"layer_type": "cells", "cell_id": cell_id},
    }


# Per-cell sparse expression payload, as the converter builds it.
ATTRS = {
    "1": [("total_counts", "42"), ("expression", '{"GENE_A":5,"GENE_B":3}')],
    "2": [("total_counts", "7"), ("expression", '{"GENE_C":1}')],
    "3": [("total_counts", "100"), ("expression", '{"GENE_A":9}')],
}


@pytest.fixture
def tiled():
    """Tile 3 cells (each present at every zoom 0-4), attach expression, gate it
    to z>=3, and emit BOTH PBF and Parquet. Yields (pbf_dir, parquet_dir)."""
    gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=4)
    gen.add_feature(_square(0.10, 0.10, 0.05, "1"))
    gen.add_feature(_square(0.70, 0.10, 0.05, "2"))
    gen.add_feature(_square(0.10, 0.70, 0.05, "3"))

    n = gen.attach_tags_by_id("cell_id", ATTRS)
    assert n == 3, f"attach_tags_by_id matched {n}/3 cells by cell_id"
    gen.set_tag_min_zoom("expression", EXPR_MIN_ZOOM)  # total_counts ungated

    with tempfile.TemporaryDirectory() as d:
        pbf_dir = Path(d) / "pbf"
        pq_dir = Path(d) / "parquet"
        generate_pbf(gen, pbf_dir, BOUNDS, simplify=False, layer_name="cells")
        gen.generate_parquet_native(str(pq_dir), BOUNDS, simplify=False)
        yield pbf_dir, pq_dir


def _pbf_tags_by_zoom(pbf_dir: Path, zoom: int) -> list[dict]:
    return [r["tags"] for r in read_pbf(pbf_dir, BOUNDS, zoom=zoom)]


def _parquet_tags_by_zoom(pq_dir: Path, zoom: int) -> list[dict]:
    tags: list[dict] = []
    for leaf in glob.glob(str(pq_dir / f"zoom={zoom}" / "*.parquet")):
        t = pq.ParquetFile(leaf).read()  # per-leaf: avoids hive `zoom` merge error
        for row in t.column("tags").to_pylist():
            tags.append(dict(row))
    return tags


# ---------------------------------------------------------------------------
# PBF (the viewer's source)
# ---------------------------------------------------------------------------


class TestPbfGate:
    def test_coarse_tiles_have_total_counts_but_not_expression(self, tiled):
        pbf_dir, _ = tiled
        for z in range(0, EXPR_MIN_ZOOM):  # 0,1,2
            tags = _pbf_tags_by_zoom(pbf_dir, z)
            assert tags, f"expected features at z{z}"
            for t in tags:
                assert "total_counts" in t, f"total_counts missing at z{z}"
                assert "cell_id" in t
                assert "expression" not in t, f"expression leaked onto coarse z{z}"

    def test_fine_tiles_carry_expression(self, tiled):
        pbf_dir, _ = tiled
        for z in range(EXPR_MIN_ZOOM, 5):  # 3,4
            tags = _pbf_tags_by_zoom(pbf_dir, z)
            assert tags, f"expected features at z{z}"
            for t in tags:
                assert "total_counts" in t
                assert "expression" in t, f"expression missing at fine z{z}"
            # value matches the per-cell payload joined by cell_id
            by_id = {t["cell_id"]: t for t in tags}
            assert by_id["1"]["expression"] == '{"GENE_A":5,"GENE_B":3}'
            assert by_id["1"]["total_counts"] == "42"


# ---------------------------------------------------------------------------
# Parquet (the ML / agent source) — same gate, same data
# ---------------------------------------------------------------------------


class TestParquetGate:
    def test_coarse_partitions_have_total_counts_but_not_expression(self, tiled):
        _, pq_dir = tiled
        for z in range(0, EXPR_MIN_ZOOM):
            tags = _parquet_tags_by_zoom(pq_dir, z)
            assert tags, f"expected rows at zoom={z}"
            for t in tags:
                assert "total_counts" in t
                assert "expression" not in t, f"expression leaked into parquet zoom={z}"

    def test_fine_partitions_carry_expression(self, tiled):
        _, pq_dir = tiled
        for z in range(EXPR_MIN_ZOOM, 5):
            tags = _parquet_tags_by_zoom(pq_dir, z)
            assert tags, f"expected rows at zoom={z}"
            for t in tags:
                assert "expression" in t, f"expression missing in parquet zoom={z}"
            by_id = {t["cell_id"]: t for t in tags}
            assert by_id["3"]["expression"] == '{"GENE_A":9}'
            assert by_id["3"]["total_counts"] == "100"


# ---------------------------------------------------------------------------
# Regression: no gate set ⇒ tags appear at all zooms (back-compat)
# ---------------------------------------------------------------------------


def test_no_gate_emits_tags_at_all_zooms():
    gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=4)
    gen.add_feature(_square(0.10, 0.10, 0.05, "1"))
    gen.attach_tags_by_id("cell_id", ATTRS)
    # NO set_tag_min_zoom → expression rides every zoom (unchanged behavior).
    with tempfile.TemporaryDirectory() as d:
        pbf_dir = Path(d) / "pbf"
        generate_pbf(gen, pbf_dir, BOUNDS, simplify=False, layer_name="cells")
        for z in range(0, 5):
            tags = [r["tags"] for r in read_pbf(pbf_dir, BOUNDS, zoom=z)]
            assert all("expression" in t for t in tags), f"z{z} should carry expression"
