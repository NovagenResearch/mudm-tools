"""Fidelity-gated test for the migration transcoder (Task 4).

The fidelity gate IS the spec for the encoder: transcoded tile geometry must be
byte-faithful to the original PBF geometry, while the high-card marker (``m_*``)
keys are removed from the tiles and re-homed into ``facets/markers.parquet``.

The ``codex_fixture`` copies the smallest real single-cell marker dataset
(``mibi-uterus``: 2,235 cells; MIBI; ``vectors/`` + ``features.parquet`` +
``metadata.json``; markers are ``m_*`` keys) into ``tmp_path`` and yields it.
"""

import json
import shutil
from pathlib import Path

import mapbox_vector_tile as mvt
import pyarrow as pa
import pyarrow.parquet as pq  # noqa: F401  (asserts the parquet file is readable downstream)
import pytest

from mudm_tools.facets import FacetPolicy
from mudm_tools.migrate_facets import tile_from_features_parquet

# Smallest real dataset, in the sibling mudm-data repo.
_SRC = Path(__file__).resolve().parents[2] / "mudm-data" / "tiles2d" / "mibi-uterus"


@pytest.fixture
def codex_fixture(tmp_path):
    """Copy the smallest real single-cell marker dataset into tmp_path and yield it."""
    if not (_SRC / "metadata.json").is_file():
        pytest.skip(f"source dataset not present: {_SRC}")
    ds = tmp_path / "mibi-uterus"
    shutil.copytree(_SRC, ds)
    yield ds


def _geom_signature(pbf_bytes):
    t = mvt.decode(pbf_bytes)
    return sorted(
        json.dumps(f["geometry"], sort_keys=True)
        for L in t.values()
        for f in L["features"]
    )


def test_transcode_preserves_geometry_strips_markers(codex_fixture):  # fixture: a small copied dataset dir
    ds = codex_fixture
    before = {p.relative_to(ds): _geom_signature(p.read_bytes()) for p in (ds / "vectors").rglob("*.pbf")}
    pol = FacetPolicy.from_config({"select": {"include": ["m_*"]}, "keep_inline": ["cell_type"]})
    report = tile_from_features_parquet(str(ds), pol, markers=["m_CD8", "m_CD4"])
    assert report["geometry_ok"] is True
    after = {p.relative_to(ds): _geom_signature(p.read_bytes()) for p in (ds / "vectors").rglob("*.pbf")}
    assert before == after                                  # geometry byte-faithful
    raw = next((ds / "vectors").rglob("*.pbf")).read_bytes()
    assert b"m_CD8" not in raw                               # markers stripped
    assert (ds / "facets" / "markers.parquet").is_file()
    meta = json.loads((ds / "metadata.json").read_text())
    assert meta["facets"]["layout"] == "wide"


# --------------------------------------------------------------------------- #
# Synthetic regression: the non-CODEX shape that broke the rollout.
#   - markers are ``g_*`` (NOT ``m_*``)
#   - the join key in tags/tiles is ``cell`` (NOT ``cell_id``)
#   - a few stray ``m_*`` keys exist that must NOT be faceted/stripped
# This pins: (1) facet build + tile strip use the SAME resolved set (markers),
# (2) the ``cell`` join key is normalized to ``cell_id`` in BOTH facet & tiles,
# (3) geometry is byte-faithful, (4) conservation (faceted set == stripped set).
# --------------------------------------------------------------------------- #

# Three cells; each is a distinct square polygon. ``cell`` is the join key
# (value c_*), ``cell_type`` stays inline, g_* are markers (faceted), and the
# stray m_* keys must survive in the tiles (only the dataset's own markers go).
_G_KEYS = ["g_VIM", "g_B2M"]
_CELLS = [
    {"cell": "c_1", "cell_type": "a", "g_VIM": "3", "g_B2M": "0", "m_Stray": "x",
     "ring": [[10, 10], [10, 20], [20, 20], [20, 10], [10, 10]]},
    {"cell": "c_2", "cell_type": "b", "g_VIM": "7", "g_B2M": "5", "m_Stray": "y",
     "ring": [[30, 30], [30, 45], [45, 45], [45, 30], [30, 30]]},
    {"cell": "c_3", "cell_type": "a", "g_VIM": "0", "g_B2M": "9", "m_Stray": "z",
     "ring": [[50, 5], [50, 15], [60, 15], [60, 5], [50, 5]]},
]
_TAG_KEYS = ["cell", "cell_type", *_G_KEYS, "m_Stray"]


def _build_gstar_cell_dataset(root: Path) -> Path:
    """Hand-build a tiny g_*/`cell`-key dataset: a features.parquet/zoom=0 with a
    ``tags`` map column and a matching vectors/0/0/0.pbf, plus a metadata.json."""
    ds = root / "cosmx-synth"
    (ds / "features.parquet" / "zoom=0").mkdir(parents=True)
    (ds / "vectors" / "0" / "0").mkdir(parents=True)

    # features.parquet/zoom=0: one row per cell, tags as map<string,string>.
    tags = pa.array(
        [[(k, c[k]) for k in _TAG_KEYS] for c in _CELLS],
        type=pa.map_(pa.string(), pa.string()),
    )
    pq.write_table(
        pa.table({"tags": tags}),
        ds / "features.parquet" / "zoom=0" / "part.parquet",
    )

    # vectors/0/0/0.pbf: one polygon feature per cell, same props as the tags.
    feats = [
        {
            "geometry": {"type": "Polygon", "coordinates": [c["ring"]]},
            "properties": {k: c[k] for k in _TAG_KEYS},
        }
        for c in _CELLS
    ]
    (ds / "vectors" / "0" / "0" / "0.pbf").write_bytes(
        mvt.encode([{"name": "cells", "features": feats}])
    )

    # metadata.json with a single cells layer; markers = the g_* keys.
    (ds / "metadata.json").write_text(json.dumps({
        "name": "cosmx-synth",
        "vectors": {"path": "vectors", "markers": _G_KEYS,
                    "layers": [{"id": "cells", "min_zoom": 0}]},
    }))
    return ds


def test_transcode_gstar_cell_key(tmp_path):
    """g_* markers + a ``cell`` join key: facet build & tile strip must agree,
    ``cell`` normalizes to ``cell_id`` everywhere, geometry stays byte-faithful."""
    ds = _build_gstar_cell_dataset(tmp_path)
    tile = ds / "vectors" / "0" / "0" / "0.pbf"
    before = _geom_signature(tile.read_bytes())

    # Policy mirrors the fixed driver: include = the dataset's markers, key=cell.
    pol = FacetPolicy.from_config(
        {"select": {"include": _G_KEYS}, "keep_inline": ["cell_type"], "key": "cell"}
    )
    report = tile_from_features_parquet(str(ds), pol, markers=_G_KEYS)

    # (3) geometry byte-faithful.
    assert report["geometry_ok"] is True
    assert _geom_signature(tile.read_bytes()) == before

    # Facet store: rows == unique cells, has the g_* columns + a cell_id column.
    facet = pq.read_table(ds / "facets" / "markers.parquet")
    assert facet.num_rows == len(_CELLS)
    assert "cell_id" in facet.column_names
    for g in _G_KEYS:
        assert g in facet.column_names
    # cell_id is the NORMALIZED key value (the original `cell` values).
    assert set(facet.column("cell_id").to_pylist()) == {c["cell"] for c in _CELLS}
    # g_* values round-tripped from the tags map.
    vim = dict(zip(facet.column("cell_id").to_pylist(), facet.column("g_VIM").to_pylist()))
    assert vim["c_1"] == 3.0 and vim["c_2"] == 7.0 and vim["c_3"] == 0.0

    # Tile: ALL g_* stripped, `cell` renamed to cell_id, stray m_* + cell_type kept.
    decoded = mvt.decode(tile.read_bytes())
    props = decoded["cells"]["features"][0]["properties"]
    for g in _G_KEYS:
        assert g not in props
    assert "cell" not in props and "cell_id" in props
    assert props["cell_type"] in ("a", "b")
    assert "m_Stray" in props  # not a dataset marker -> must survive

    # (4) conservation: the set of faceted keys == the set of stripped keys.
    tile_keys_after = set(decoded["cells"]["features"][0]["properties"].keys())
    stripped = set(_TAG_KEYS) - tile_keys_after - {"cell"}  # `cell` was renamed, not dropped
    assert stripped == set(_G_KEYS)

    # facets-block key is normalized to cell_id.
    meta = json.loads((ds / "metadata.json").read_text())
    assert meta["facets"]["key"] == "cell_id"
    assert meta["facets"]["layout"] == "wide"
