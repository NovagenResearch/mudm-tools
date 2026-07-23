"""General, format-agnostic facet store: per-cell attributes -> facets/*.parquet + metadata block."""

from __future__ import annotations
import fnmatch
from pathlib import Path
from typing import Any
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_RG = 65536


class FacetPolicy:
    def __init__(self, **kw):
        self.key = kw.get("key", "cell_id")
        self.keep_inline = list(kw.get("keep_inline", ["cell_type"]))
        sel = kw.get("select", {}) or {}
        self.include = list(sel.get("include", []))
        self.exclude = list(sel.get("exclude", []))
        self.numeric_high_card = bool(sel.get("numeric_high_card", True))
        self.card_threshold = int(sel.get("card_threshold", 64))
        self.layout = kw.get("layout", "wide")
        enc = kw.get("encoding", {}) or {}
        self.codec = enc.get("codec", "snappy")
        self.byte_stream_split = bool(enc.get("byte_stream_split", True))
        self.quantize = enc.get("quantize")  # None | "uint16"
        self.prefetch_max_mb = int(kw.get("prefetch_max_mb", 30))

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "FacetPolicy":
        return cls(**(cfg or {}))


def _matches(name: str, globs: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, g) for g in globs)


def select_facet_keys(
    policy: FacetPolicy, columns: dict[str, Any], cardinality: dict[str, int]
) -> tuple[list[str], list[str]]:
    """Return (facet_keys, inline_keys). Heuristic: high-card numeric -> facet; plus include/exclude globs."""
    facet, inline = [], []
    for name, dtype in columns.items():
        if name == policy.key or name in policy.keep_inline:
            inline.append(name)
            continue
        if _matches(name, policy.exclude):
            inline.append(name)
            continue
        forced = _matches(name, policy.include)
        is_num = np.issubdtype(np.dtype(dtype), np.number)
        high = cardinality.get(name, 0) > policy.card_threshold
        (facet if forced or (policy.numeric_high_card and is_num and high) else inline).append(name)
    return facet, inline


def emit_facet_store(
    out_dir, cell_ids, attributes, policy: FacetPolicy, *, long_table=None
) -> dict:
    facets_dir = Path(out_dir) / "facets"
    facets_dir.mkdir(parents=True, exist_ok=True)
    if long_table is not None:  # sparse long form (Xenium genes)
        href = "facets/expression.parquet"
        pq.write_table(
            long_table,
            facets_dir / "expression.parquet",
            compression=policy.codec,
            row_group_size=_RG,
        )
        rows = long_table.num_rows
        asset = {
            "role": "facets",
            "href": href,
            "media_type": "application/vnd.apache.parquet",
            "facet": "gene",
            "layout": "long",
            "key": policy.key,
            "columns": ["cell_id", "gene", "count"],
            "sorted_by": "gene",
            "rows": int(rows),
        }
        block = {
            "layer": "cells",
            "key": policy.key,
            "storage": "asset",
            "layout": "long",
            "fields": {"gene": "vector<int>"},
            "assets": [asset],
        }
        return block
    # wide form (markers): one float column per attribute
    names = list(attributes.keys())
    cols = {"cell_id": pa.array([str(c) for c in cell_ids], pa.string())}
    for k in names:
        a = np.asarray(attributes[k])
        if policy.quantize == "uint16" and np.issubdtype(a.dtype, np.floating):
            scale = (np.quantile(a, 0.999) or 1.0) / 65535.0
            cols[k] = pa.array(np.clip(np.round(a / scale), 0, 65535).astype(np.uint16))
        else:
            cols[k] = pa.array(a.astype("float32"))
    table = pa.table(cols)
    href = "facets/markers.parquet"
    bss = (
        [k for k in names if pa.types.is_floating(table.schema.field(k).type)]
        if policy.byte_stream_split
        else False
    )
    pq.write_table(
        table,
        facets_dir / "markers.parquet",
        compression=policy.codec,
        row_group_size=_RG,
        use_dictionary=False,
        use_byte_stream_split=bss,
    )
    size_mb = (facets_dir / "markers.parquet").stat().st_size / 1e6
    asset = {
        "role": "facets",
        "href": href,
        "media_type": "application/vnd.apache.parquet",
        "layout": "wide",
        "key": policy.key,
        "columns": ["cell_id"] + names,
    }
    return {
        "layer": "cells",
        "key": policy.key,
        "storage": "asset",
        "layout": "wide",
        "fields": {k: "scalar<float>" for k in names},
        "assets": [asset],
        "prefetch": bool(size_mb <= policy.prefetch_max_mb),
    }
