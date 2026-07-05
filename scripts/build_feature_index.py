#!/usr/bin/env python3
"""Scan .glb tiles and build a MuDM FeatureCollection index for the viewer.

Reads the JSON chunk from each GLB file to extract feature names and
properties from node extras, then writes a features.json manifest as a
valid MuDM/muDM FeatureCollection.

Output format (MuDM FeatureCollection):
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "id": "Caudoputamen",
      "geometry": {
        "type": "TIN",
        "coordinates": [],
        "tiles": ["0/0/0/0", "2/0/0/0", "2/0/0/1", "3/0/0/0", ...]
      },
      "properties": {
        "color": "#98D9AA",
        "acronym": "CP",
        "ccf_id": 672
      }
    },
    ...
  ]
}

A separate tilejson3d.json file contains encoding and zoom metadata.
"""

import argparse
import json
import struct
from collections import defaultdict
from pathlib import Path


def read_glb_json(path: Path) -> dict:
    """Read only the JSON chunk from a GLB file (fast, skips binary data)."""
    with open(path, "rb") as f:
        header = f.read(12)
        if len(header) < 12:
            return {}
        magic, version, total_len = struct.unpack("<III", header)
        if magic != 0x46546C67:  # 'glTF'
            return {}

        chunk_header = f.read(8)
        if len(chunk_header) < 8:
            return {}
        chunk_len, chunk_type = struct.unpack("<II", chunk_header)
        if chunk_type != 0x4E4F534A:  # 'JSON'
            return {}

        json_bytes = f.read(chunk_len)
        return json.loads(json_bytes)


def extract_features(glb_json: dict) -> list[dict]:
    """Extract feature info from glTF node extras."""
    features = []
    for node in glb_json.get("nodes", []):
        extras = node.get("extras")
        if extras and ("name" in extras or "acronym" in extras or "body_id" in extras):
            features.append(extras)
    return features


# Structural extras handled elsewhere (color -> its own entry; name -> the feature id; tile_ids ->
# accumulated separately), so they are not copied into the property bag.
_NON_PROPERTY_EXTRAS = ("color", "name", "tile_ids")


def feature_properties(extras: dict) -> dict:
    """Scalar feature properties kept from glTF node extras for viewer color-by / filter / hover.

    Keeps EVERY scalar field (str/int/float/bool) rather than a hardcoded allowlist — otherwise any
    custom per-feature attribute (e.g. a segmentation's volume/sphericity, an EMT score) is silently
    dropped. Subsumes the old connectome allowlist (acronym/ccf_id/body_id/… are all scalars, still
    captured). Nested/list values are skipped (the viewer's color-by only handles scalars)."""
    return {k: v for k, v in extras.items()
            if k not in _NON_PROPERTY_EXTRAS and isinstance(v, (str, int, float, bool))}


def build_index(tiles_dir: Path, id_fields: list[str] | None = None,
                unique_by: str | None = None,
                ) -> tuple[dict, dict[int, int], int]:
    """Build feature index from all .glb files.

    Args:
        tiles_dir: Path to the 3D tiles directory.
        id_fields: List of metadata keys that are identifiers (excluded from
            filter/color-by in the viewer). Passed through to build_tilejson().
        unique_by: If set (e.g. "body_id"), group features by a UNIQUE key
            synthesized as "{base_name} ({feat[unique_by]})" instead of by bare
            name. Needed when the baked display name is NOT unique per feature
            (e.g. hemibrain's neuPrint `instance` is shared across bodies, which
            silently collapsed 25000 neurons → 12552). Mirrors flywire's baked
            "{label} ({root_id})" convention, applied at index time (no re-tile).

    Returns:
        A tuple of (collection_dict, zoom_counts, max_zoom) where:
        - collection_dict is the MuDM FeatureCollection with TIN geometry
        - zoom_counts maps zoom level to tile count
        - max_zoom is the highest zoom level found
    """
    # feature_name → {color, acronym, ccf_id, tile_ids: set()}
    feature_map: dict[str, dict] = {}
    zoom_counts: dict[int, int] = defaultdict(int)
    max_zoom = 0

    glb_files = sorted(tiles_dir.rglob("*.glb"))
    print(f"Scanning {len(glb_files)} .glb files...")

    for glb_path in glb_files:
        rel = glb_path.relative_to(tiles_dir)
        parts = rel.parts  # e.g., ('2', '0', '1', '0.glb')
        if len(parts) != 4:
            continue

        z = int(parts[0])
        zoom_counts[z] += 1
        max_zoom = max(max_zoom, z)

        # Convert URI "2/0/0/0.glb" to bare tile ID "2/0/0/0"
        tile_id = str(rel.with_suffix(""))

        glb_json = read_glb_json(glb_path)
        features = extract_features(glb_json)

        for feat in features:
            name = (feat.get("name")
                    or feat.get("acronym")
                    or feat.get("instance")
                    or str(feat.get("body_id", "")))
            if not name:
                continue
            # Force per-feature uniqueness when the baked name is shared across
            # features (e.g. hemibrain's `instance`). Synthesize "{base} ({uid})".
            if unique_by is not None and feat.get(unique_by) is not None:
                uid = feat[unique_by]
                name = name if str(uid) in name else f"{name} ({uid})"

            if name not in feature_map:
                entry: dict = {
                    "color": feat.get("color", "#888888"),
                    "tile_ids": set(),
                }
                # Keep every scalar extra as a property (no hardcoded allowlist) so custom color-by
                # fields survive; see feature_properties().
                entry.update(feature_properties(feat))
                feature_map[name] = entry

            feature_map[name]["tile_ids"].add(tile_id)

    print(f"Found {len(feature_map)} unique features")
    for z in sorted(zoom_counts):
        print(f"  Zoom {z}: {zoom_counts[z]} tiles")

    # Build MuDM FeatureCollection
    features_list = []
    for name in sorted(feature_map):
        entry = feature_map[name]
        tile_ids = entry.pop("tile_ids")
        props = {}
        for key, val in entry.items():
            props[key] = val
        features_list.append({
            "type": "Feature",
            "id": name,
            "geometry": {
                "type": "TIN",
                "coordinates": [],
                "tiles": sorted(tile_ids),
            },
            "properties": props,
        })

    result = {
        "type": "FeatureCollection",
        "features": features_list,
    }
    return result, dict(sorted(zoom_counts.items())), max_zoom


def build_tilejson(zoom_counts: dict[int, int], max_zoom: int,
                   id_fields: list[str] | None = None,
                   bounds3d: list[float] | None = None) -> dict:
    """Build a tilejson3d.json dict with encodings and zoom metadata."""
    tj: dict = {
        "tilejson": "3.0.0",
        "tiles": ["{z}/{x}/{y}/{d}"],
        "minzoom": 0,
        "maxzoom": max_zoom,
        "vector_layers": [{"id": "default", "fields": {}}],
        "zoom_counts": {str(k): v for k, v in sorted(zoom_counts.items())},
        "encodings": [
            {"format": "glb", "compression": "meshopt", "path": "3dtiles", "extension": ".glb"},
        ],
    }
    if id_fields:
        tj["id_fields"] = id_fields
    if bounds3d:
        tj["bounds3d"] = bounds3d
    return tj


def main():
    parser = argparse.ArgumentParser(description="Build feature index for 3D viewer")
    parser.add_argument(
        "--tiles-dir",
        default="data/mouselight/tiles/3dtiles",
        help="3D Tiles directory",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path for features.json (default: <tiles-dir>/features.json)",
    )
    parser.add_argument(
        "--tilejson-output",
        default=None,
        help="Output path for tilejson3d.json (default: sibling of --output)",
    )
    parser.add_argument(
        "--id-fields",
        nargs="*",
        default=None,
        help="Metadata keys that are identifiers (excluded from filter/color-by)",
    )
    args = parser.parse_args()

    tiles_dir = Path(args.tiles_dir)
    output = Path(args.output) if args.output else tiles_dir / "features.json"

    index, zoom_counts, max_zoom = build_index(tiles_dir, id_fields=args.id_fields)

    # Write features.json
    output.write_text(json.dumps(index, indent=2))
    print(f"Wrote {output} ({output.stat().st_size / 1024:.1f} KB)")

    # Write tilejson3d.json
    tilejson_path = (
        Path(args.tilejson_output) if args.tilejson_output
        else output.parent / "tilejson3d.json"
    )
    tj = build_tilejson(zoom_counts, max_zoom, id_fields=args.id_fields)
    tilejson_path.write_text(json.dumps(tj, indent=2))
    print(f"Wrote {tilejson_path} ({tilejson_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
