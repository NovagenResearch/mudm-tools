"""Neuroglancer segment_properties writer.

Converts MuDM feature properties into the Neuroglancer
``neuroglancer_segment_properties`` inline JSON format.

Reference: https://github.com/google/neuroglancer/blob/master/
    src/neuroglancer/datasource/precomputed/segment_properties.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from mudm.model import MuDMFeature
from .models import (
    SegmentPropertiesInfo,
    SegmentPropertiesInline,
    SegmentPropertyField,
)


def features_to_segment_properties(
    features: Sequence[MuDMFeature],
    segment_ids: Sequence[int],
) -> dict[str, Any]:
    """Build segment_properties inline dict from MuDM features.

    Each feature's ``properties`` dict is read. All unique property keys
    across features become columns. Values are ordered to match
    ``segment_ids``.

    Args:
        features: MuDM features with properties dicts.
        segment_ids: Numeric segment IDs matching each feature.

    Returns:
        A dict matching the ``neuroglancer_segment_properties`` info schema.
    """
    ids = [str(sid) for sid in segment_ids]

    # Collect all property keys and determine types
    all_keys: list[str] = []
    seen: set[str] = set()
    for feat in features:
        if feat.properties:
            for k in feat.properties:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

    fields: list[SegmentPropertyField] = []
    for key in all_keys:
        values: list[Any] = []
        for feat in features:
            val = feat.properties.get(key) if feat.properties else None
            values.append(val if val is not None else "")

        # Determine field type
        if all(isinstance(v, (int, float)) for v in values if v != ""):
            field_type = "number"
            # Neuroglancer requires data_type for number properties
            if all(isinstance(v, int) for v in values if v != ""):
                data_type = "uint32"
            else:
                data_type = "float32"
            fields.append(
                SegmentPropertyField(
                    id=key, type=field_type, data_type=data_type, values=values
                )
            )
        else:
            field_type = "label"
            fields.append(
                SegmentPropertyField(id=key, type=field_type, values=values)
            )

    inline = SegmentPropertiesInline(ids=ids, properties=fields)
    info = SegmentPropertiesInfo(inline=inline)
    return info.to_info_dict()


def write_segment_properties(
    output_dir: str | Path,
    features: Sequence[MuDMFeature],
    segment_ids: Sequence[int],
) -> Path:
    """Write segment_properties info JSON to disk.

    Creates ``{output_dir}/info`` with the segment properties.

    Args:
        output_dir: Directory to write to (created if needed).
        features: MuDM features.
        segment_ids: Segment IDs corresponding to each feature.

    Returns:
        Path to the output directory.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    info_dict = features_to_segment_properties(features, segment_ids)
    (out / "info").write_text(json.dumps(info_dict, indent=2))

    return out
