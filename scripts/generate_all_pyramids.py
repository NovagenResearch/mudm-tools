#!/usr/bin/env python3
"""Generate 3D tile pyramids for all MouseLight brains.

Loops over date directories containing OBJ files, converts each to
MuDM, generates 3D Tiles via TileGenerator3D, builds per-pyramid
feature indexes, and writes a pyramids.json manifest.

Usage::

    .venv/bin/python scripts/generate_all_pyramids.py \
      --obj-base data/mouselight/ \
      --output-base data/mouselight/tiles/ \
      --max-zoom 3

    # Single brain only:
    .venv/bin/python scripts/generate_all_pyramids.py \
      --obj-base data/mouselight/ \
      --output-base data/mouselight/tiles/ \
      --only 2021-09-16
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from mudm_tools.tiling3d.generator3d import TileGenerator3D
from mudm_tools.tiling3d.octree import OctreeConfig

from build_feature_index import build_index, build_tilejson
from obj_to_microjson import (
    build_collection,
    fetch_allen_ontology,
    obj_to_feature,
)


def _fmt_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{s:.0f}s"


def _fmt_bytes(n: int) -> str:
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def discover_brains(obj_base: Path) -> list[str]:
    """Find date-named directories with OBJ files."""
    date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    brains = []
    for d in sorted(obj_base.iterdir()):
        if d.is_dir() and date_re.match(d.name):
            if list(d.glob("*.obj")):
                brains.append(d.name)
    return brains


def generate_pyramid(
    brain_id: str,
    obj_dir: Path,
    output_dir: Path,
    *,
    ontology: dict | None,
    max_zoom: int = 3,
    workers: int | None = None,
) -> dict:
    """Generate 3D Tiles pyramid for a single brain.

    Returns metadata dict for pyramids.json.
    """
    tiles_dir = output_dir / brain_id / "3dtiles"

    # Convert OBJ → MuDM
    obj_paths = sorted(obj_dir.glob("*.obj"))
    print(f"  Converting {len(obj_paths)} OBJ files...")
    t0 = time.perf_counter()
    features = []
    for i, obj_path in enumerate(obj_paths, 1):
        if i % 100 == 0 or i == len(obj_paths):
            print(f"    [{i}/{len(obj_paths)}]", end="\r", file=sys.stderr)
        features.append(obj_to_feature(str(obj_path), ontology))
    print(file=sys.stderr)
    collection = build_collection(features, ontology)
    convert_time = time.perf_counter() - t0
    print(f"  Conversion: {_fmt_time(convert_time)}")

    # Generate tiles
    if tiles_dir.exists():
        shutil.rmtree(tiles_dir)
    tiles_dir.mkdir(parents=True, exist_ok=True)

    gen = TileGenerator3D(
        OctreeConfig(max_zoom=max_zoom),
        output_format="3dtiles",
        workers=workers,
    )
    print(f"  Generating 3D Tiles (zoom 0-{max_zoom})...")
    t0 = time.perf_counter()
    gen.add_features(collection)
    n_tiles = gen.generate(tiles_dir)
    gen.write_metadata(tiles_dir)
    gen_time = time.perf_counter() - t0
    print(f"  Generated {n_tiles} tiles in {_fmt_time(gen_time)}")

    # Build feature index and tilejson3d
    print("  Building feature index...")
    pyramid_root = output_dir / brain_id
    index, zoom_counts, max_zoom_found = build_index(tiles_dir)
    n_features = len(index.get("features", []))

    # Write features.json to pyramid root
    (pyramid_root / "features.json").write_text(json.dumps(index, indent=2))

    # Write tilejson3d.json to pyramid root
    tj = build_tilejson(zoom_counts, max_zoom_found)
    (pyramid_root / "tilejson3d.json").write_text(json.dumps(tj, indent=2))

    total_size = _dir_size(tiles_dir)
    print(f"  Size: {_fmt_bytes(total_size)}")

    return {
        "id": brain_id,
        "label": f"MouseLight {brain_id}",
        "tilejson": "tilejson3d.json",
        "features": "features.json",
        "tiles": n_tiles,
        "feature_count": n_features,
        "size_bytes": total_size,
    }


def write_manifest(output_base: Path, pyramids: list[dict]) -> None:
    """Write pyramids.json manifest in PyramidJSON format."""
    manifest = {"version": "1.0", "pyramids": pyramids}
    path = output_base / "pyramids.json"
    path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 3D tile pyramids for all MouseLight brains",
    )
    parser.add_argument(
        "--obj-base", type=Path, required=True,
        help="Base directory with date-named brain dirs (e.g. data/mouselight/)",
    )
    parser.add_argument(
        "--output-base", type=Path, required=True,
        help="Output base for tiles (e.g. data/mouselight/tiles/)",
    )
    parser.add_argument(
        "--max-zoom", type=int, default=3,
        help="Max zoom level (default: 3)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Worker processes: None/0=auto, 1=serial",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate even if tiles already exist",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="Only process this brain date (e.g. 2020-11-26)",
    )
    parser.add_argument(
        "--no-ontology", action="store_true",
        help="Skip Allen CCF ontology lookup",
    )
    args = parser.parse_args()

    brains = discover_brains(args.obj_base)
    if args.only:
        if args.only not in brains:
            print(f"ERROR: {args.only} not found. Available: {brains}", file=sys.stderr)
            sys.exit(1)
        brains = [args.only]

    print(f"Found {len(brains)} brains: {', '.join(brains)}")

    # Fetch ontology once (shared across all brains)
    ontology = None
    if not args.no_ontology:
        print("Fetching Allen CCF ontology...")
        ontology = fetch_allen_ontology()
        print(f"  Loaded {len(ontology)} entries")

    effective_workers = args.workers if args.workers and args.workers >= 1 else (os.cpu_count() or 1)
    print(f"Workers: {effective_workers}, Max zoom: {args.max_zoom}\n")

    pyramids: list[dict] = []
    total_t0 = time.perf_counter()

    for i, brain_id in enumerate(brains, 1):
        tiles_dir = args.output_base / brain_id / "3dtiles"
        exists = tiles_dir.exists() and (tiles_dir / "tileset.json").exists()

        if exists and not args.force:
            print(f"[{i}/{len(brains)}] {brain_id} — already exists, skipping (use --force)")
            pyramid_root = args.output_base / brain_id
            # Rebuild feature index if needed
            features_path = pyramid_root / "features.json"
            if not features_path.exists():
                # Try old location inside 3dtiles/
                old_path = tiles_dir / "features.json"
                if old_path.exists():
                    features_path = old_path
            if features_path.exists():
                index = json.loads(features_path.read_text())
            else:
                index, _, _ = build_index(tiles_dir)
                (pyramid_root / "features.json").write_text(json.dumps(index, indent=2))

            features = index.get("features", [])
            n_features = len(features) if isinstance(features, list) else len(features)

            # Read tilejson3d.json for zoom_counts if available
            tj_path = pyramid_root / "tilejson3d.json"
            if tj_path.exists():
                tj = json.loads(tj_path.read_text())
                n_tiles = sum(int(v) for v in tj.get("zoom_counts", {}).values())
            else:
                # Fallback: count .glb files
                n_tiles = len(list(tiles_dir.rglob("*.glb")))

            pyramids.append({
                "id": brain_id,
                "label": f"MouseLight {brain_id}",
                "tilejson": "tilejson3d.json",
                "features": "features.json",
                "tiles": n_tiles,
                "feature_count": n_features,
                "size_bytes": _dir_size(tiles_dir),
            })
            continue

        print(f"[{i}/{len(brains)}] {brain_id}")
        obj_dir = args.obj_base / brain_id
        meta = generate_pyramid(
            brain_id,
            obj_dir,
            args.output_base,
            ontology=ontology,
            max_zoom=args.max_zoom,
            workers=args.workers,
        )
        pyramids.append(meta)
        print()

    write_manifest(args.output_base, pyramids)

    total_time = time.perf_counter() - total_t0
    print(f"Total time: {_fmt_time(total_time)}")
    print(f"Pyramids: {len(pyramids)}")
    for p in pyramids:
        print(f"  {p['id']}: {p['tiles']} tiles, {p.get('feature_count', '?')} features, {_fmt_bytes(p['size_bytes'])}")


if __name__ == "__main__":
    main()
