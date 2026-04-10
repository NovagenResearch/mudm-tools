#!/usr/bin/env python3
"""Generate Draco-compressed 3D Tiles for a single MouseLight brain.

Deletes existing 3dtiles/ directory and rebuilds with use_draco=True.

Usage:
    uv run python scripts/mouselight_3dtiles_draco.py --brain 2016-10-31
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

_DATA_DIR = _ROOT / "data" / "mouselight"
_TILES_DIR = _DATA_DIR / "tiles"

MAX_ZOOM = 3


def _fmt_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{s:.0f}s"


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.2f} MB"
    return f"{n / (1024 * 1024 * 1024):.2f} GB"


def _dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _build_mouselight_tags(obj_path: Path, ontology: dict | None) -> dict:
    from obj_to_microjson import match_region

    stem = obj_path.stem
    tags: dict = {"mesh_name": stem, "source": obj_path.name}

    if ontology is not None:
        region = match_region(stem, ontology)
        if region:
            tags["ccf_id"] = region["id"]
            tags["name"] = region["name"]
            tags["acronym"] = region["acronym"]
            if region["parent_structure_id"] is not None:
                tags["parent_id"] = region["parent_structure_id"]
            if region["color_hex_triplet"]:
                tags["color"] = f"#{region['color_hex_triplet']}"
            if region["structure_id_path"]:
                tags["hierarchy_path"] = region["structure_id_path"]

    return tags


def main():
    parser = argparse.ArgumentParser(
        description="Generate Draco-compressed 3D Tiles for a MouseLight brain",
    )
    parser.add_argument(
        "--brain", type=str, required=True,
        help="Brain date folder name (e.g. 2016-10-31)",
    )
    parser.add_argument(
        "--no-ontology", action="store_true",
        help="Skip Allen CCF ontology lookup",
    )
    args = parser.parse_args()

    brain_dir = _DATA_DIR / args.brain
    if not brain_dir.is_dir():
        print(f"ERROR: {brain_dir} not found", file=sys.stderr)
        sys.exit(1)

    obj_paths = sorted(brain_dir.glob("*.obj"))
    if not obj_paths:
        print(f"ERROR: No OBJ files in {brain_dir}", file=sys.stderr)
        sys.exit(1)

    # Ontology
    ontology = None
    if not args.no_ontology:
        from obj_to_microjson import fetch_allen_ontology
        print("Fetching Allen CCF ontology...")
        t0 = time.perf_counter()
        ontology = fetch_allen_ontology()
        print(f"  Loaded {len(ontology)} entries ({_fmt_time(time.perf_counter() - t0)})")

    path_strs = [str(p) for p in obj_paths]
    tags_list = [_build_mouselight_tags(p, ontology) for p in obj_paths]
    total_mb = sum(p.stat().st_size for p in obj_paths) / (1024 * 1024)

    print(f"\nBrain: {args.brain}")
    print(f"  {len(obj_paths)} OBJ files ({total_mb:.0f} MB)")

    from mudm_tools._rs import StreamingTileGenerator, scan_obj_bounds

    # Scan bounds
    t0 = time.perf_counter()
    bounds = scan_obj_bounds(path_strs)
    print(f"  Bounds: x=[{bounds[0]:.0f}, {bounds[3]:.0f}] "
          f"y=[{bounds[1]:.0f}, {bounds[4]:.0f}] "
          f"z=[{bounds[2]:.0f}, {bounds[5]:.0f}]  ({_fmt_time(time.perf_counter() - t0)})")

    # Delete existing 3dtiles
    tiles3d_dir = _TILES_DIR / args.brain / "3dtiles"
    if tiles3d_dir.exists():
        old_size = _dir_size(tiles3d_dir)
        print(f"\n  Deleting existing 3dtiles/ ({_fmt_bytes(old_size)})...")
        shutil.rmtree(tiles3d_dir)
    tiles3d_dir.mkdir(parents=True, exist_ok=True)

    # Ingest
    gen = StreamingTileGenerator(min_zoom=0, max_zoom=MAX_ZOOM)
    t0 = time.perf_counter()
    gen.add_obj_files(path_strs, bounds, tags_list)
    t_ingest = time.perf_counter() - t0
    print(f"  Ingest: {_fmt_time(t_ingest)} ({len(obj_paths) / max(t_ingest, 0.001):.0f} files/s)")

    # Generate Draco-compressed 3D Tiles
    print("  Encoding 3D Tiles with Draco...")
    t0 = time.perf_counter()
    n_tiles = gen.generate_3dtiles(
        str(tiles3d_dir), bounds,
        use_draco=True,
    )
    t_gen = time.perf_counter() - t0
    del gen

    new_size = _dir_size(tiles3d_dir)
    print(f"  {n_tiles} tiles in {_fmt_time(t_gen)} ({_fmt_bytes(new_size)})")

    # Build features.json
    from build_feature_index import build_index

    print("  Building features.json...")
    index = build_index(tiles3d_dir)
    features_path = tiles3d_dir / "features.json"
    features_path.write_text(json.dumps(index, indent=2))
    n_features = len(index.get("features", {}))
    print(f"  {n_features} features indexed")

    print(f"\nDone. Output: {tiles3d_dir}")


if __name__ == "__main__":
    main()
