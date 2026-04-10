"""Neuroglancer viewer state URL generation.

Builds Neuroglancer JSON viewer state dicts and encodes them into
shareable URLs.
"""

from __future__ import annotations

import json
import urllib.parse
from typing import Any, Dict, List, Optional


_RADIUS_SHADER = """\
#uicontrol float radiusScale slider(min=0.1, max=100.0, default=10.0, step=0.1)
#uicontrol float minWidth slider(min=0.5, max=10.0, default=2.0, step=0.5)
void main() {
  setColor(defaultSegmentColor());
  setLineWidth(max(getDataValue(0) * radiusScale, minWidth));
  setEndpointMarkerSize(0.0, 0.0);
}
"""


def build_skeleton_layer(
    name: str,
    source_url: str,
    *,
    use_radius: bool = True,
) -> dict[str, Any]:
    """Build a Neuroglancer layer dict for a skeleton source.

    Args:
        name: Layer display name.
        source_url: ``precomputed://`` URL to the skeleton data.
        use_radius: If True, include a skeleton shader that reads the
            radius vertex attribute to set line width. Adds a
            ``radiusScale`` slider control in the Neuroglancer UI.

    Returns:
        Layer dict suitable for the viewer state ``layers`` list.
    """
    layer: dict[str, Any] = {
        "type": "segmentation",
        "source": source_url,
        "name": name,
        "selectedAlpha": 0,
        "notSelectedAlpha": 0,
    }
    if use_radius:
        layer["skeletonRendering"] = {
            "mode2d": "lines",
            "mode3d": "lines",
            "shader": _RADIUS_SHADER,
        }
    return layer


def build_annotation_layer(
    name: str,
    source_url: str,
) -> dict[str, Any]:
    """Build a Neuroglancer layer dict for an annotation source.

    Args:
        name: Layer display name.
        source_url: ``precomputed://`` URL to the annotation data.

    Returns:
        Layer dict suitable for the viewer state ``layers`` list.
    """
    return {
        "type": "annotation",
        "source": source_url,
        "name": name,
    }


def build_viewer_state(
    layers: List[dict[str, Any]],
    position: Optional[List[float]] = None,
    projection_scale: Optional[float] = None,
    layout: str = "3d",
) -> dict[str, Any]:
    """Build a complete Neuroglancer viewer state dict.

    Args:
        layers: List of layer dicts (from build_*_layer functions).
        position: Optional 3D camera position [x, y, z] in nm.
        projection_scale: Camera zoom (field of view in nm). Larger = more zoomed out.
        layout: Viewer layout. ``"3d"`` for 3D only, ``"4panel"`` for all views.

    Returns:
        Viewer state dict that can be serialized to JSON.
    """
    state: dict[str, Any] = {"layers": layers, "layout": layout}
    if position is not None:
        nav: dict[str, Any] = {
            "pose": {"position": {"voxelCoordinates": position}},
        }
        if projection_scale is not None:
            nav["zoomFactor"] = projection_scale
        state["navigation"] = nav
    elif projection_scale is not None:
        state["navigation"] = {"zoomFactor": projection_scale}
    return state


def viewer_state_to_url(
    state: dict[str, Any],
    base_url: str = "https://neuroglancer-demo.appspot.com",
) -> str:
    """Encode a viewer state dict into a Neuroglancer URL.

    Args:
        state: Viewer state dict.
        base_url: Base Neuroglancer instance URL.

    Returns:
        Full URL with JSON-encoded fragment.
    """
    fragment = json.dumps(state, separators=(",", ":"))
    # URL-encode the JSON so browsers don't mangle special characters
    # ({, }, ", [, ] etc.). Neuroglancer decodes before JSON.parse.
    encoded = urllib.parse.quote(fragment, safe=",:/@")
    return f"{base_url}/#!{encoded}"
