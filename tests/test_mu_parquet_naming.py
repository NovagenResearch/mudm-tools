"""Guard: muDM tiling writes Parquet leaves as *.mu.parquet (custom schema, NOT GeoParquet)."""

from __future__ import annotations
import pyarrow.parquet as pq


def test_reader_roundtrips_mu_parquet(tmp_path):
    import pyarrow as pa

    d = tmp_path / "zoom=0"
    d.mkdir()
    t = pa.table({"a": [1, 2, 3]})
    pq.write_table(t, d.parent / "zoom=0" / "part_000.mu.parquet")
    # A directory read must still pick up the .mu.parquet file (ends in .parquet).
    got = pq.read_table(str(tmp_path))
    assert got.num_rows == 3
