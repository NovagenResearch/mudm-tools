"""Build a ``features.json`` feature→tile index from an existing tileset.json.

The Three.js 3D Tiles viewer (``viewers/viewer3d/js/TileManager.js``) is
feature-driven: per frame it iterates ``selectedFeatures``, looks each
one up in ``featureIndex``, and requests tiles from
``feat.tiles[zoom]``. Our Rust ``generate_3dtiles`` emits the octree +
GLBs but not this per-feature index, so we derive it here.

The index is built by bbox intersection: walk the tileset octree; for
each tile, check whether its bounding volume overlaps the feature's
world-space bbox; if yes, record the tile URI under that feature at the
zoom parsed from the URI prefix.

This is **conservative**: every tile that actually contains geometry
for a feature will be listed (no false negatives). A few tiles that the
bbox merely touches but which don't contain the feature's geometry may
also be listed (benign over-fetch).

Output shape matches the legacy ``features.json`` the viewer reads::

    {
      "features": {
        "<feature_name>": {
          "tiles": {"0": ["0/0/0/0.glb"], "1": [...], ...},
          ...feature_properties...
        },
        ...
      }
    }
"""

from __future__ import annotations

import colorsys
import hashlib
import json
from pathlib import Path
from typing import Any

Bbox = tuple[float, float, float, float, float, float]


def _hsl_to_hex(h: float, s: float, l: float) -> str:
    """h,s,l in [0,1] -> '#rrggbb'."""
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return "#{:02x}{:02x}{:02x}".format(
        int(round(r * 255)), int(round(g * 255)), int(round(b * 255))
    )


def _color_for_name(name: str) -> str:
    """Deterministic unique-ish color per feature name.

    Uses SHA-1 of the name to pick a hue; saturation and lightness are
    fixed for consistent visual weight. 357 features → 357 well-spread
    colors across the hue wheel.
    """
    h = int(hashlib.sha1(name.encode("utf-8")).hexdigest()[:8], 16)
    hue = (h % 3600) / 3600.0  # 0.0..1.0
    return _hsl_to_hex(hue, 0.65, 0.55)


def _tile_bbox_from_node(node: dict) -> Bbox | None:
    """Derive an axis-aligned (minx,miny,minz, maxx,maxy,maxz) from an OGC
    3D Tiles ``boundingVolume.box``.

    3D Tiles box: 12 floats = [cx,cy,cz, hxx,hxy,hxz, hyx,hyy,hyz, hzx,hzy,hzz]
    where the three half-axis vectors describe an oriented box. Our Rust
    generator writes axis-aligned boxes (only diagonals are non-zero), so the
    half-extents live at indices 3 (hx on x), 7 (hy on y), 11 (hz on z).
    """
    bv = node.get("boundingVolume") or {}
    box = bv.get("box")
    if not box or len(box) < 12:
        return None
    cx, cy, cz = box[0], box[1], box[2]
    hx = abs(box[3])
    hy = abs(box[7])
    hz = abs(box[11])
    return (cx - hx, cy - hy, cz - hz, cx + hx, cy + hy, cz + hz)


def _bbox_intersects(a: Bbox, b: Bbox) -> bool:
    return not (
        a[0] > b[3] or a[3] < b[0]
        or a[1] > b[4] or a[4] < b[1]
        or a[2] > b[5] or a[5] < b[2]
    )


def _walk(
    node: dict,
    feat_bbox: Bbox,
    acc: dict[str, list[str]],
) -> None:
    tb = _tile_bbox_from_node(node)
    # If we can't read a bbox, conservatively descend (don't prune).
    intersects = True
    if tb is not None:
        intersects = _bbox_intersects(tb, feat_bbox)
    if intersects:
        content = node.get("content")
        if content and "uri" in content:
            uri = content["uri"]
            zoom = uri.split("/", 1)[0]
            acc.setdefault(zoom, []).append(uri)
    for child in node.get("children", []) or []:
        _walk(child, feat_bbox, acc)


def build_features_index(
    tileset_path: str | Path,
    features: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build a features.json dict from an existing tileset + per-feature data.

    Args:
        tileset_path: Path to ``tileset.json``.
        features: Mapping ``{feature_name: {"bbox": Bbox, "properties": dict}}``.
            ``bbox`` is the feature's world-space AABB (min/max x/y/z). The
            properties dict is copied into the output alongside ``tiles``.

    Returns:
        A dict matching the legacy features.json shape (see module docstring).
        Tile lists are deduplicated and sorted for stable output.
    """
    tileset = json.loads(Path(tileset_path).read_text())
    root = tileset.get("root", {})

    out: dict[str, Any] = {}
    for name, entry in features.items():
        bbox = entry["bbox"]
        by_zoom: dict[str, list[str]] = {}
        _walk(root, bbox, by_zoom)
        for z, uris in by_zoom.items():
            by_zoom[z] = sorted(set(uris))
        feat_out: dict[str, Any] = {"tiles": by_zoom}
        feat_out.update(entry.get("properties", {}))
        # Default unique-ish color keyed on the feature name. The viewer's
        # "Original" color-by path picks this up (see TileManager._getFeatureColor).
        # It's overridden when the user picks an attribute from the Color By UI.
        feat_out.setdefault("color", _color_for_name(name))
        out[name] = feat_out
    return {"features": out}


def write_features_index(
    tileset_path: str | Path,
    features: dict[str, dict[str, Any]],
    out_path: str | Path,
) -> Path:
    """Write features.json next to tileset.json."""
    index = build_features_index(tileset_path, features)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index, indent=2))
    return out
