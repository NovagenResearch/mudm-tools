#!/usr/bin/env python3
"""Generate partitioned Parquet pyramids for all MouseLight brains.

One pyramid per brain (date folder). Uses the streaming batch API with
size-based file rotation for O(batch_size) memory.

Usage:
    uv run python scripts/mouselight_parquet.py
    uv run python scripts/mouselight_parquet.py --brain 2021-09-16  # single brain
    uv run python scripts/mouselight_parquet.py --prime             # also create Arrow IPC
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
BATCH_SIZE = 10_000


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
    """Build tags dict for a MouseLight OBJ file from Allen CCF ontology."""
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


def process_brain(
    brain_dir: Path,
    output_dir: Path,
    ontology: dict | None,
    *,
    do_prime: bool = False,
) -> dict:
    """Generate partitioned Parquet pyramid for one brain.

    Returns summary dict with brain_id, rows, size, timing.
    """
    from mudm_tools._rs import StreamingTileGenerator, scan_obj_bounds
    from mudm_tools.tiling3d.parquet_writer import generate_parquet
    from mudm_tools.tiling3d.parquet_prime import prime_parquet

    brain_id = brain_dir.name
    obj_paths = sorted(brain_dir.glob("*.obj"))
    if not obj_paths:
        return {"brain_id": brain_id, "skipped": True}

    path_strs = [str(p) for p in obj_paths]
    tags_list = [_build_mouselight_tags(p, ontology) for p in obj_paths]
    total_mb = sum(p.stat().st_size for p in obj_paths) / (1024 * 1024)

    # Scan bounds
    t0 = time.perf_counter()
    bounds = scan_obj_bounds(path_strs)
    t_bounds = time.perf_counter() - t0

    print(f"  {len(obj_paths)} OBJ files ({total_mb:.0f} MB)")
    print(f"  Bounds: x=[{bounds[0]:.0f}, {bounds[3]:.0f}] "
          f"y=[{bounds[1]:.0f}, {bounds[4]:.0f}] "
          f"z=[{bounds[2]:.0f}, {bounds[5]:.0f}]  ({_fmt_time(t_bounds)})")

    # Output directory
    pq_dir = output_dir / brain_id / "parquet_partitioned"
    if pq_dir.exists():
        shutil.rmtree(pq_dir)

    # Ingest + generate
    gen = StreamingTileGenerator(min_zoom=0, max_zoom=MAX_ZOOM)

    t0 = time.perf_counter()
    gen.add_obj_files(path_strs, bounds, tags_list)
    t_ingest = time.perf_counter() - t0
    print(f"  Ingest: {_fmt_time(t_ingest)} ({len(obj_paths) / max(t_ingest, 0.001):.0f} files/s)")

    t0 = time.perf_counter()
    n_rows = generate_parquet(
        gen, pq_dir, bounds,
        batch_size=BATCH_SIZE, partitioned=True,
    )
    t_write = time.perf_counter() - t0
    del gen

    pq_size = _dir_size(pq_dir)

    # Partition layout
    for z_dir in sorted(pq_dir.iterdir()):
        if z_dir.is_dir() and z_dir.name.startswith("zoom="):
            parts = sorted(z_dir.glob("part_*.parquet"))
            z_size = sum(p.stat().st_size for p in parts)
            print(f"    {z_dir.name}: {len(parts)} part(s), {_fmt_bytes(z_size)}")

    print(f"  Parquet: {n_rows:,} rows in {_fmt_time(t_write)} ({_fmt_bytes(pq_size)})")

    # Prime (optional)
    n_arrow = 0
    if do_prime:
        t0 = time.perf_counter()
        n_arrow = prime_parquet(pq_dir)
        t_prime = time.perf_counter() - t0
        print(f"  Primed: {n_arrow} Arrow IPC files ({_fmt_time(t_prime)})")

    return {
        "brain_id": brain_id,
        "obj_files": len(obj_paths),
        "obj_mb": round(total_mb, 1),
        "rows": n_rows,
        "pq_bytes": pq_size,
        "ingest_s": round(t_ingest, 1),
        "write_s": round(t_write, 1),
        "arrow_files": n_arrow,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate partitioned Parquet pyramids for MouseLight brains",
    )
    parser.add_argument(
        "--brain", type=str, default=None,
        help="Process a single brain (date folder name, e.g. 2021-09-16)",
    )
    parser.add_argument(
        "--prime", action="store_true",
        help="Also create Arrow IPC siblings (prime_parquet)",
    )
    parser.add_argument(
        "--no-ontology", action="store_true",
        help="Skip Allen CCF ontology lookup (faster, fewer tags)",
    )
    args = parser.parse_args()

    # Fetch ontology once
    ontology = None
    if not args.no_ontology:
        from obj_to_microjson import fetch_allen_ontology
        print("Fetching Allen CCF ontology...")
        t0 = time.perf_counter()
        ontology = fetch_allen_ontology()
        print(f"  Loaded {len(ontology)} ontology entries ({_fmt_time(time.perf_counter() - t0)})")

    # Find brain directories
    if args.brain:
        brain_dirs = [_DATA_DIR / args.brain]
        if not brain_dirs[0].is_dir():
            print(f"ERROR: {brain_dirs[0]} not found", file=sys.stderr)
            sys.exit(1)
    else:
        brain_dirs = sorted(
            d for d in _DATA_DIR.iterdir()
            if d.is_dir() and list(d.glob("*.obj"))
        )

    if not brain_dirs:
        print(f"ERROR: No brain directories with OBJ files in {_DATA_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{len(brain_dirs)} brain(s) to process, zoom 0–{MAX_ZOOM}")
    print(f"Output: {_TILES_DIR}/{{brain}}/parquet_partitioned/")
    print()

    t_total_start = time.perf_counter()
    results = []

    for i, brain_dir in enumerate(brain_dirs):
        print(f"{'='*60}")
        print(f"[{i+1}/{len(brain_dirs)}] {brain_dir.name}")
        print(f"{'='*60}")

        result = process_brain(brain_dir, _TILES_DIR, ontology, do_prime=args.prime)
        results.append(result)
        print()

    t_total = time.perf_counter() - t_total_start

    # Summary
    processed = [r for r in results if not r.get("skipped")]
    total_rows = sum(r["rows"] for r in processed)
    total_pq = sum(r["pq_bytes"] for r in processed)
    total_ingest = sum(r["ingest_s"] for r in processed)
    total_write = sum(r["write_s"] for r in processed)

    print(f"{'='*60}")
    print(f"ALL DONE — {len(processed)} brains")
    print(f"{'='*60}")
    print(f"  Total rows:   {total_rows:,}")
    print(f"  Total size:   {_fmt_bytes(total_pq)}")
    print(f"  Ingest time:  {_fmt_time(total_ingest)}")
    print(f"  Write time:   {_fmt_time(total_write)}")
    print(f"  Wall time:    {_fmt_time(t_total)}")

    # Write manifest
    manifest_path = _TILES_DIR / "parquet_manifest.json"
    manifest_path.write_text(json.dumps({"brains": processed}, indent=2))
    print(f"\n  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
