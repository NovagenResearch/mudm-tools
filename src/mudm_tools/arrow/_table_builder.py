"""Core logic: MuDM features → pyarrow.Table with WKB geometry."""

from __future__ import annotations

import json
from typing import Any

import pyarrow as pa
import shapely

from mudm.model import (
    MuDMFeature,
    MuDMFeatureCollection,
)
from ._geometry import (
    geometry_type_name,
    to_shapely,
    to_wkb,
)
from .models import ArrowConfig


# ---------------------------------------------------------------------------
# Property type inference
# ---------------------------------------------------------------------------

def _infer_pa_type(values: list[Any]) -> pa.DataType:
    """Infer a pyarrow type from a list of non-None values."""
    if not values:
        return pa.string()

    all_bool = all(isinstance(v, bool) for v in values)
    if all_bool:
        return pa.bool_()

    # Check int/float (bool is subclass of int, so check bool first)
    all_int = all(isinstance(v, int) and not isinstance(v, bool) for v in values)
    if all_int:
        return pa.int64()

    all_numeric = all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in values
    )
    if all_numeric:
        return pa.float64()

    all_str = all(isinstance(v, str) for v in values)
    if all_str:
        return pa.string()

    # Mixed / complex → JSON string
    return pa.string()


def _needs_json(value: Any) -> bool:
    """Check if a value needs JSON serialization (dict/list/complex)."""
    return isinstance(value, (dict, list))


def _coerce_value(value: Any, pa_type: pa.DataType) -> Any:
    """Coerce a property value to match the target Arrow type."""
    if value is None:
        return None
    if pa_type == pa.string() and not isinstance(value, str):
        return json.dumps(value, separators=(",", ":"))
    if pa_type == pa.float64() and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


# ---------------------------------------------------------------------------
# Row extraction
# ---------------------------------------------------------------------------

def _extract_rows(
    features: list[MuDMFeature],
    config: ArrowConfig,
) -> list[dict[str, Any]]:
    """Convert features to flat row dicts with WKB geometry."""
    rows: list[dict[str, Any]] = []
    gcol = config.primary_geometry_column

    for feat in features:
        geom = feat.geometry
        props = dict(feat.properties) if feat.properties else {}

        shapely_geom = to_shapely(geom)
        row: dict[str, Any] = {
            "id": str(feat.id) if feat.id is not None else None,
            "featureClass": feat.featureClass,
            gcol: to_wkb(shapely_geom),
        }
        row.update(props)
        rows.append(row)
        row["_shapely_geom"] = shapely_geom

    return rows


# ---------------------------------------------------------------------------
# Schema construction
# ---------------------------------------------------------------------------

_RESERVED_KEYS = {"id", "featureClass", "_shapely_geom"}


def _build_schema(
    rows: list[dict[str, Any]],
    config: ArrowConfig,
) -> pa.Schema:
    """Build Arrow schema from row dicts."""
    gcol = config.primary_geometry_column
    fields: list[pa.Field] = [
        pa.field("id", pa.string()),
        pa.field("featureClass", pa.string()),
        pa.field(gcol, pa.binary()),
    ]

    reserved = _RESERVED_KEYS | {gcol}

    # Collect property keys (preserve insertion order)
    prop_keys: list[str] = []
    prop_values: dict[str, list[Any]] = {}
    for r in rows:
        for k, v in r.items():
            if k in reserved:
                continue
            if k not in prop_values:
                prop_keys.append(k)
                prop_values[k] = []
            if v is not None:
                prop_values[k].append(v)

    # Add property fields with inferred types
    for key in prop_keys:
        pa_type = _infer_pa_type(prop_values[key])
        fields.append(pa.field(key, pa_type))

    return pa.schema(fields)


# ---------------------------------------------------------------------------
# GeoParquet metadata
# ---------------------------------------------------------------------------

def _geoparquet_metadata(
    rows: list[dict[str, Any]],
    config: ArrowConfig,
) -> dict[str, Any]:
    """Build GeoParquet 1.1 metadata dict."""
    geom_types: set[str] = set()
    xs: list[float] = []
    ys: list[float] = []

    for r in rows:
        shapely_geom = r.get("_shapely_geom")
        gtype = geometry_type_name(shapely_geom)
        if gtype is not None:
            geom_types.add(gtype)
        if shapely_geom is not None and not shapely_geom.is_empty:
            bounds = shapely_geom.bounds  # (minx, miny, maxx, maxy)
            xs.extend([bounds[0], bounds[2]])
            ys.extend([bounds[1], bounds[3]])

    bbox = [min(xs), min(ys), max(xs), max(ys)] if xs else []

    col_meta: dict[str, Any] = {
        "encoding": "WKB",
        "geometry_types": sorted(geom_types),
    }
    if bbox:
        col_meta["bbox"] = bbox

    return {
        "version": "1.1.0",
        "primary_column": config.primary_geometry_column,
        "columns": {
            config.primary_geometry_column: col_meta,
        },
    }


# ---------------------------------------------------------------------------
# Public: build table
# ---------------------------------------------------------------------------

def build_table(
    data: MuDMFeature | MuDMFeatureCollection,
    config: ArrowConfig | None = None,
) -> pa.Table:
    """Convert MuDM data to a pyarrow.Table.

    Args:
        data: A MuDMFeature or MuDMFeatureCollection.
        config: Export configuration. Uses defaults if None.

    Returns:
        A pyarrow.Table with WKB geometry and GeoParquet metadata.
    """
    if config is None:
        config = ArrowConfig()

    # Normalize to list of features
    if isinstance(data, MuDMFeatureCollection):
        features = list(data.features)
    else:
        features = [data]

    rows = _extract_rows(features, config)
    schema = _build_schema(rows, config)
    geo_meta = _geoparquet_metadata(rows, config)

    # Build column arrays
    columns: dict[str, list[Any]] = {f.name: [] for f in schema}

    for r in rows:
        for field in schema:
            name = field.name
            val = r.get(name)
            columns[name].append(_coerce_value(val, field.type))

    arrays = [pa.array(columns[f.name], type=f.type) for f in schema]

    # Attach GeoParquet metadata
    existing = schema.metadata or {}
    new_meta = {
        **existing,
        b"geo": json.dumps(geo_meta).encode("utf-8"),
    }

    # Store collection-level vocabularies in schema metadata
    if isinstance(data, MuDMFeatureCollection) and data.vocabularies is not None:
        if isinstance(data.vocabularies, str):
            vocab_json = json.dumps(data.vocabularies)
        else:
            vocab_json = json.dumps(
                {k: v.model_dump(exclude_none=True) for k, v in data.vocabularies.items()}
            )
        new_meta[b"mudm:vocabularies"] = vocab_json.encode("utf-8")

    return pa.table(
        {f.name: arr for f, arr in zip(schema, arrays)},
        schema=schema.with_metadata(new_meta),
    )
