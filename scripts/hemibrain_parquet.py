#!/usr/bin/env python3
"""Generate partitioned Parquet pyramid for the Hemibrain dataset.

Uses the streaming batch API for O(batch_size) memory.

Usage:
    uv run python scripts/hemibrain_parquet.py
"""

from __future__ import annotations

import json
import shutil
import sys
import time
import tracemalloc
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

_DATA_DIR = _ROOT / "data" / "hemibrain"
_MESH_DIR = _DATA_DIR / "meshes"
_META_PATH = _DATA_DIR / "metadata.json"
_TILES_DIR = _DATA_DIR / "tiles" / "hemibrain"

MAX_ZOOM = 3
BASE_CELLS = 100
BATCH_SIZE = 500


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


def _build_tags(obj_path: Path, meta_lookup: dict[str, dict]) -> dict:
    import colorsys
    body_id = obj_path.stem
    tags: dict = {
        "body_id": int(body_id) if body_id.isdigit() else body_id,
        "source": obj_path.name,
    }
    meta = meta_lookup.get(body_id, {})
    if meta:
        if meta.get("cellType"):
            tags["cell_type"] = meta["cellType"]
        if meta.get("instance"):
            tags["instance"] = meta["instance"]
        if meta.get("status"):
            tags["status"] = meta["status"]
        if meta.get("statusLabel"):
            tags["status_label"] = meta["statusLabel"]
        if meta.get("pre") is not None:
            tags["pre"] = str(meta["pre"])
        if meta.get("post") is not None:
            tags["post"] = str(meta["post"])
        if meta.get("somaRadius") is not None:
            tags["soma_radius"] = str(meta["somaRadius"])
        if meta.get("cropped") is not None:
            tags["cropped"] = str(meta["cropped"]).lower()
        # Derive dominant neuropil region from roiInfo
        roi_info = meta.get("roiInfo")
        if roi_info and isinstance(roi_info, dict):
            best_roi, best_count = None, 0
            for roi, counts in roi_info.items():
                if isinstance(counts, dict):
                    total = (counts.get("pre", 0) or 0) + (counts.get("post", 0) or 0)
                    if total > best_count:
                        best_roi, best_count = roi, total
            if best_roi:
                tags["primary_roi"] = best_roi
    tags["name"] = tags.get("instance") or str(body_id)
    h = hash(str(body_id)) & 0x7FFFFFFF
    hue = (h % 360) / 360.0
    r, g, b = colorsys.hls_to_rgb(hue, 0.5, 0.75)
    tags["color"] = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    return tags


def _ingest(gen, obj_paths, bounds, meta_lookup):
    """Ingest all OBJ files in parallel via rayon."""
    path_strs = [str(p) for p in obj_paths]
    tags_list = [_build_tags(p, meta_lookup) for p in obj_paths]
    total_mb = sum(p.stat().st_size for p in obj_paths) / (1024 * 1024)
    print(f"  {len(obj_paths)} files ({total_mb:.0f} MB total), parallel rayon...",
          flush=True)
    t0 = time.perf_counter()
    gen.add_obj_files(path_strs, bounds, tags_list)
    dt = time.perf_counter() - t0
    print(f"  Done: {_fmt_time(dt)} ({len(obj_paths) / dt:.1f} files/s)",
          flush=True)


def main():
    from mudm_tools._rs import StreamingTileGenerator, scan_obj_bounds
    from mudm_tools.tiling3d.parquet_writer import generate_parquet
    from mudm_tools.tiling3d.parquet_reader import read_parquet

    obj_paths = sorted(_MESH_DIR.glob("*.obj"))
    if not obj_paths:
        print(f"ERROR: No .obj files in {_MESH_DIR}", file=sys.stderr)
        sys.exit(1)

    meta_lookup: dict[str, dict] = {}
    if _META_PATH.exists():
        raw = json.loads(_META_PATH.read_text())
        for neuron in raw.get("neurons", []):
            meta_lookup[str(neuron["bodyId"])] = neuron

    # Load cached bounds
    bounds_cache = _MESH_DIR / "bounds.json"
    if bounds_cache.exists():
        bounds = tuple(json.loads(bounds_cache.read_text()))
        print(f"Bounds (cached): x=[{bounds[0]:.0f}, {bounds[3]:.0f}] "
              f"y=[{bounds[1]:.0f}, {bounds[4]:.0f}] "
              f"z=[{bounds[2]:.0f}, {bounds[5]:.0f}]")
    else:
        print("Scanning bounds...")
        bounds = scan_obj_bounds([str(p) for p in obj_paths])
        bounds_cache.write_text(json.dumps(list(bounds)))

    total_mb = sum(p.stat().st_size for p in obj_paths) / (1024 * 1024)

    print(f"\n{'=' * 70}")
    print(f"Hemibrain Partitioned Parquet Pyramid")
    print(f"  {len(obj_paths)} OBJ files ({total_mb:.0f} MB total)")
    print(f"  Zoom 0–{MAX_ZOOM}, base_cells={BASE_CELLS}")
    print(f"  Batch size: {BATCH_SIZE:,} (O(batch) memory)")
    print(f"{'=' * 70}")

    # --- Partitioned streaming ---
    part_dir = _TILES_DIR / "parquet_partitioned"
    if part_dir.exists():
        shutil.rmtree(part_dir)

    gen = StreamingTileGenerator(
        min_zoom=0, max_zoom=MAX_ZOOM, base_cells=BASE_CELLS,
    )

    print(f"\nPhase 1: Ingest OBJ → fragment file...")
    t0 = time.perf_counter()
    _ingest(gen, obj_paths, bounds, meta_lookup)
    t_ingest = time.perf_counter() - t0
    print(f"  Ingest total: {_fmt_time(t_ingest)}")

    print(f"\nPhase 2: Write partitioned Parquet (batch={BATCH_SIZE:,})...")
    tracemalloc.start()
    t0 = time.perf_counter()
    n_rows = generate_parquet(
        gen, part_dir, bounds,
        batch_size=BATCH_SIZE, partitioned=True,
    )
    t_write = time.perf_counter() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del gen

    part_size = _dir_size(part_dir)
    throughput = n_rows / t_write if t_write > 0 else 0

    print(f"  {n_rows:,} rows in {_fmt_time(t_write)}")
    print(f"  Size: {_fmt_bytes(part_size)}")
    print(f"  Peak memory (write phase): {peak_mem / (1024*1024):.1f} MB")
    print(f"  Throughput: {throughput:,.0f} rows/s")

    # Partition layout
    print(f"\n  Partition layout:")
    for z_dir in sorted(part_dir.iterdir()):
        if z_dir.is_dir() and z_dir.name.startswith("zoom="):
            pq_files = sorted(z_dir.glob("*.parquet"))
            z_total = sum(f.stat().st_size for f in pq_files)
            print(f"    {z_dir.name}: {len(pq_files)} file(s), "
                  f"{_fmt_bytes(z_total)}")

    # --- Quick read-back verification ---
    print(f"\nPhase 3: Verify read-back...")
    t0 = time.perf_counter()
    rows_z0 = read_parquet(part_dir, zoom=0)
    t_read_z0 = time.perf_counter() - t0
    print(f"  zoom=0: {len(rows_z0):,} rows ({_fmt_time(t_read_z0)})")

    t0 = time.perf_counter()
    rows_z3 = read_parquet(part_dir, zoom=MAX_ZOOM)
    t_read_z3 = time.perf_counter() - t0
    print(f"  zoom={MAX_ZOOM}: {len(rows_z3):,} rows ({_fmt_time(t_read_z3)})")

    all_fids = set()
    for z in range(MAX_ZOOM + 1):
        rows = read_parquet(part_dir, zoom=z)
        all_fids.update(r["feature_id"] for r in rows)
    print(f"  Unique features across all zooms: {len(all_fids)}")
    print(f"  Zoom levels present: {list(range(MAX_ZOOM + 1))}")

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print(f"DONE")
    print(f"{'=' * 70}")
    print(f"  Rows:        {n_rows:,}")
    print(f"  Ingest:      {_fmt_time(t_ingest)}")
    print(f"  Write:       {_fmt_time(t_write)}")
    print(f"  Total:       {_fmt_time(t_ingest + t_write)}")
    print(f"  Peak memory: {peak_mem / (1024*1024):.1f} MB")
    print(f"  Output size: {_fmt_bytes(part_size)}")
    print(f"  Output:      {part_dir}/")


if __name__ == "__main__":
    main()
