import json, gzip
from pathlib import Path
import pyarrow.parquet as pq
from mudm_tools.converters import convert

def _toy_geojson(p: Path, n=200):
    feats = [{"type": "Feature",
              "geometry": {"type": "Polygon", "coordinates": [[[i,0],[i+1,0],[i+1,1],[i,1],[i,0]]]},
              "properties": {"cell_id": f"c{i}", "cell_type": "T", "m_CD8": (i % 50) * 0.1}}
             for i in range(n)]
    p.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))

def test_markers_faceted_not_inline(tmp_path):
    src = tmp_path / "cells.geojson"; _toy_geojson(src)
    out = tmp_path / "out"
    convert("geojson", str(src), str(out),
            {"facets": {"select": {"include": ["m_*"]}, "keep_inline": ["cell_type"]}})
    assert (out / "facets" / "markers.parquet").is_file()
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["facets"]["layout"] == "wide" and "m_CD8" in meta["facets"]["fields"]
    # a sample tile must NOT contain the m_CD8 key
    tile = next((out / "vectors").rglob("*.pbf"))
    raw = tile.read_bytes()
    raw = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
    assert b"m_CD8" not in raw and b"cell_type" in raw  # faceted stripped, categorical kept
