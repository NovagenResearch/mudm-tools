"""Public API for Arrow/GeoParquet export."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pyarrow as pa
import pyarrow.parquet as pq

from mudm.model import MuDMFeature, MuDMFeatureCollection
from ._table_builder import build_table
from .models import ArrowConfig


def to_arrow_table(
    data: Union[MuDMFeature, MuDMFeatureCollection],
    config: ArrowConfig | None = None,
) -> pa.Table:
    """Convert MuDM data to a pyarrow.Table.

    The table contains WKB-encoded geometry, feature metadata, and
    property columns with types inferred from the data.  GeoParquet 1.1
    metadata is attached to the schema.

    Args:
        data: A MuDMFeature or MuDMFeatureCollection.
        config: Export configuration.

    Returns:
        A pyarrow.Table.
    """
    return build_table(data, config)


def to_geoparquet(
    data: Union[MuDMFeature, MuDMFeatureCollection],
    output_path: str | Path,
    config: ArrowConfig | None = None,
) -> pa.Table:
    """Convert MuDM data and write to a GeoParquet file.

    Args:
        data: A MuDMFeature or MuDMFeatureCollection.
        output_path: Path for the output ``.parquet`` file.
        config: Export configuration.

    Returns:
        The pyarrow.Table that was written.
    """
    table = build_table(data, config)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))

    return table
