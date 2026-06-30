import numpy as np, pyarrow.parquet as pq
from mudm_tools.facets import FacetPolicy, emit_facet_store

def test_wide_store_bss_applied(tmp_path):
    n = 1000
    cids = [f"c{i}" for i in range(n)]
    attrs = {"m_A": np.random.rand(n).astype("float32"),
             "m_B": np.random.rand(n).astype("float32")}
    pol = FacetPolicy.from_config({"layout": "wide", "encoding": {"byte_stream_split": True}})
    block = emit_facet_store(tmp_path, cids, attrs, pol)
    p = tmp_path / "facets" / "markers.parquet"
    assert p.is_file()
    md = pq.ParquetFile(p).metadata
    # BSS must actually be applied (the use_dictionary=False gotcha)
    rg = md.row_group(0)
    encs = {rg.column(i).path_in_schema: rg.column(i).encodings for i in range(rg.num_columns)}
    assert "BYTE_STREAM_SPLIT" in encs["m_A"]
    assert block["layout"] == "wide" and block["key"] == "cell_id"
    assert block["assets"][0]["href"] == "facets/markers.parquet"
    assert set(block["fields"]) == {"m_A", "m_B"}
