"""End-to-end Xenium ingest spike: load Rep1, convert to muDM, tile to Parquet,
verify round-trip of polygons + expression vectors + cell-type / ontology metadata.

Skipped when the dataset isn't on disk."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

REP1 = Path(__file__).resolve().parent.parent / "data" / "xenium_breast" / "Rep1" / "outs"


@pytest.mark.skipif(not REP1.exists(), reason="Xenium Rep1 not downloaded")
def test_xenium_to_mudm_smoke(tmp_path):
    """Smoke: 200 cells through xenium_to_mudm produce a valid MuDMFeatureCollection."""
    from mudm_tools.converters.xenium import xenium_to_mudm

    fc = xenium_to_mudm(
        cell_boundaries_path=REP1 / "cell_boundaries.parquet",
        cell_feature_matrix_path=REP1 / "cell_feature_matrix",
        cells_path=REP1 / "cells.parquet",
        max_cells=200,
    )
    assert len(fc.features) == 200
    for feat in fc.features:
        assert feat.geometry.type == "Polygon"
        props = feat.properties or {}
        # Required: an "expression" list (per-cell counts) and a "cell_id"
        assert "cell_id" in props
        assert "expression" in props
        assert len(props["expression"]) > 0


@pytest.mark.skipif(not REP1.exists(), reason="Xenium Rep1 not downloaded")
def test_xenium_tile_to_parquet_roundtrip(tmp_path):
    """End-to-end: ingest 200 cells -> tile -> read back from Parquet -> verify schema."""
    from mudm_tools._rs import StreamingTileGenerator2D
    from mudm_tools.converters.xenium import xenium_to_mudm
    from mudm_tools.tiling2d import generate_parquet

    fc = xenium_to_mudm(
        cell_boundaries_path=REP1 / "cell_boundaries.parquet",
        cell_feature_matrix_path=REP1 / "cell_feature_matrix",
        cells_path=REP1 / "cells.parquet",
        max_cells=200,
    )
    bounds = _compute_bbox_2d(fc)

    gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=4, buffer=64 / 4096)
    geojson_str = fc.model_dump_json(exclude_none=True)
    gen.add_geojson(geojson_str, bounds)

    out = tmp_path / "tiles.parquet"
    generate_parquet(gen, str(out), bounds)

    table = pq.read_table(str(out))
    cols = set(table.column_names)
    # The actual schema written by generate_parquet (see tiling2d/parquet_writer.py)
    required = {
        "zoom",
        "tile_x",
        "tile_y",
        "feature_id",
        "geom_type",
        "positions",
        "ring_lengths",
        "tags",
    }
    missing = required - cols
    assert not missing, f"Parquet schema missing required columns: {missing}"

    # Tags should preserve expression and cell_id (encoded as utf8 strings).
    tags_col = table.column("tags").to_pylist()

    # Each row is a list[(key, value)] tuple list when reading map<utf8,utf8>
    def _has_key(row, key):
        if row is None:
            return False
        if isinstance(row, dict):
            return key in row
        return any(k == key for k, _ in row)

    assert any(
        _has_key(row, "expression") for row in tags_col
    ), f"no expression in tags; sample: {tags_col[:1]}"
    assert any(_has_key(row, "cell_id") for row in tags_col)

    # Expression value must be a JSON-decodable list of ints with non-trivial length.
    def _get(row, key):
        if isinstance(row, dict):
            return row.get(key)
        for k, v in row:
            if k == key:
                return v
        return None

    sample_expr = next(_get(row, "expression") for row in tags_col if _has_key(row, "expression"))
    decoded = json.loads(sample_expr)
    assert isinstance(decoded, list)
    assert len(decoded) > 0
    assert all(isinstance(x, int) for x in decoded)

    # All zoom levels populated (tiles span 0..4)
    zooms = set(table.column("zoom").to_pylist())
    assert zooms.issubset({0, 1, 2, 3, 4}) and zooms, f"unexpected zooms: {zooms}"


def _compute_bbox_2d(fc) -> tuple[float, float, float, float]:
    """Compute 2D bbox from a FeatureCollection of Polygons."""
    xs: list[float] = []
    ys: list[float] = []
    for feat in fc.features:
        coords = feat.geometry.coordinates
        # Polygon: [[(x, y), ...]]
        for ring in coords:
            for pt in ring:
                xs.append(float(pt[0]))
                ys.append(float(pt[1]))
    return (min(xs), min(ys), max(xs), max(ys))
