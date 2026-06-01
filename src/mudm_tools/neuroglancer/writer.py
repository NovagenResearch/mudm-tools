"""Unified Neuroglancer export orchestrator.

Auto-dispatches MuDM data to the appropriate Neuroglancer writer:
- Point features   -> point annotations
- LineString features -> line annotations
- Properties -> segment_properties

For skeleton export, use ``write_skeleton()`` directly with a
NeuronMorphology from ``mudm.swc``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from geojson_pydantic import LineString, Point

from mudm.model import (
    MuDMFeature,
    MuDMFeatureCollection,
)
from .annotation_writer import write_annotations
from .state import (
    build_annotation_layer,
    build_viewer_state,
    viewer_state_to_url,
)


def to_neuroglancer(
    data: Union[MuDMFeature, MuDMFeatureCollection],
    output_dir: str | Path,
    *,
    base_url: Optional[str] = None,
) -> dict[str, Any]:
    """Export MuDM data to Neuroglancer precomputed format.

    Auto-dispatches based on geometry type:
    - Point -> point annotations
    - LineString -> line annotations

    For skeleton export, call ``write_skeleton()`` directly.

    Args:
        data: A MuDMFeature or MuDMFeatureCollection.
        output_dir: Root output directory.
        base_url: Optional Neuroglancer instance URL for viewer state.

    Returns:
        Dict with keys: ``paths`` (written dirs), ``viewer_state`` (if base_url).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if isinstance(data, MuDMFeatureCollection):
        features = list(data.features)
    else:
        features = [data]

    point_features: list[MuDMFeature] = []
    line_features: list[MuDMFeature] = []

    for feat in features:
        geom = feat.geometry
        if isinstance(geom, Point):
            point_features.append(feat)
        elif isinstance(geom, LineString):
            line_features.append(feat)

    paths: dict[str, str] = {}
    layers: list[dict[str, Any]] = []

    if point_features:
        ann_dir = out / "point_annotations"
        write_annotations(ann_dir, point_features, "point")
        paths["point_annotations"] = str(ann_dir)
        if base_url:
            layers.append(
                build_annotation_layer(
                    "point_annotations",
                    f"precomputed://{base_url}/point_annotations",
                )
            )

    if line_features:
        ann_dir = out / "line_annotations"
        write_annotations(ann_dir, line_features, "line")
        paths["line_annotations"] = str(ann_dir)
        if base_url:
            layers.append(
                build_annotation_layer(
                    "line_annotations",
                    f"precomputed://{base_url}/line_annotations",
                )
            )

    result: dict[str, Any] = {"paths": paths}

    if base_url and layers:
        state = build_viewer_state(layers)
        result["viewer_state"] = state
        result["viewer_url"] = viewer_state_to_url(state, base_url)

    return result
