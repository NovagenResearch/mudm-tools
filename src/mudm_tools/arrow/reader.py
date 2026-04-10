"""Read GeoParquet / Arrow tables back into MuDM models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import pyarrow as pa
import pyarrow.parquet as pq
import shapely

from mudm.model import MuDMFeature, MuDMFeatureCollection, OntologyTerm, Vocabulary
from ._from_geometry import shapely_to_microjson


# Columns with special meaning — not copied to feature.properties
_RESERVED_COLUMNS = {"id", "featureClass", "geometry"}


def _row_to_dict(table: pa.Table, idx: int) -> dict[str, Any]:
    """Extract a single row from an Arrow table as a plain dict."""
    return {col: table[col][idx].as_py() for col in table.column_names}


def _geometry_from_row(row: dict[str, Any], geom_col: str = "geometry") -> Any:
    """Decode WKB geometry from a row dict, returning a Shapely object or None."""
    wkb = row.get(geom_col)
    if wkb is None:
        return None
    return shapely.from_wkb(wkb)


def _feature_from_row(row: dict[str, Any], geom_col: str = "geometry") -> MuDMFeature:
    """Build a MuDMFeature from a plain row."""
    shapely_geom = _geometry_from_row(row, geom_col)
    geometry = shapely_to_microjson(shapely_geom)

    # Collect user properties (everything not reserved)
    props: dict[str, Any] = {}
    for k, v in row.items():
        if k not in _RESERVED_COLUMNS and k != geom_col:
            props[k] = v

    return MuDMFeature(
        type="Feature",
        id=row.get("id"),
        geometry=geometry,
        properties=props if props else {},
        featureClass=row.get("featureClass"),
    )


def _parse_vocabularies(
    raw: str | bytes,
) -> dict[str, Vocabulary] | str | None:
    """Parse vocabularies from schema metadata value."""
    import json as _json

    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    parsed = _json.loads(text)
    if isinstance(parsed, str):
        return parsed
    if isinstance(parsed, dict):
        return {
            k: Vocabulary(**v) if isinstance(v, dict) else v
            for k, v in parsed.items()
        }
    return None


def from_arrow_table(
    table: pa.Table,
    geometry_column: str = "geometry",
) -> MuDMFeatureCollection:
    """Convert a pyarrow.Table (with WKB geometry) to a MuDMFeatureCollection.

    Args:
        table: A pyarrow.Table, typically read from a GeoParquet file.
        geometry_column: Name of the WKB geometry column.

    Returns:
        A MuDMFeatureCollection.
    """
    features: list[MuDMFeature] = []

    for i in range(len(table)):
        row = _row_to_dict(table, i)
        features.append(_feature_from_row(row, geometry_column))

    # Read vocabularies from schema metadata
    vocabularies = None
    meta = table.schema.metadata or {}
    vocab_raw = meta.get(b"mudm:vocabularies")
    if vocab_raw is not None:
        vocabularies = _parse_vocabularies(vocab_raw)

    return MuDMFeatureCollection(
        type="FeatureCollection",
        features=features,
        vocabularies=vocabularies,
    )


def from_geoparquet(
    path: str | Path,
    geometry_column: str | None = None,
) -> MuDMFeatureCollection:
    """Read a GeoParquet file into a MuDMFeatureCollection.

    Args:
        path: Path to the ``.parquet`` file.
        geometry_column: Override the geometry column name.  If None, it
            is read from the GeoParquet ``geo`` metadata.

    Returns:
        A MuDMFeatureCollection.
    """
    import json

    table = pq.read_table(str(path))

    # Auto-detect geometry column from GeoParquet metadata
    if geometry_column is None:
        geo_meta = table.schema.metadata.get(b"geo")
        if geo_meta:
            meta = json.loads(geo_meta)
            geometry_column = meta.get("primary_column", "geometry")
        else:
            geometry_column = "geometry"

    return from_arrow_table(table, geometry_column)
