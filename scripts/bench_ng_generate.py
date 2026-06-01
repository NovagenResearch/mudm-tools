#!/usr/bin/env python3
"""One-off bench: ingest N largest hemibrain neurons + generate Neuroglancer multilod.

Measures NG generate wall-time after the SOTA-NG round (in-memory Draco, no temp-OBJ
round-trip) vs the old ~40min/40n baseline. Default ceiling -> k=1 (apples-to-apples
with the old whole-corpus run). Usage:
    TMPDIR=/data/tmp uv run python scripts/bench_ng_generate.py [N] [mesh_dir] [out]
"""
import glob
import os
import sys
import time

from mudm_tools._rs import StreamingTileGenerator, scan_obj_bounds

n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
mesh_dir = sys.argv[2] if len(sys.argv) > 2 else "data/hemibrain/meshes"
out = sys.argv[3] if len(sys.argv) > 3 else "/data/tmp/ng_bench_out"

# Match the driver EXACTLY: sorted(mesh_dir.glob("*.obj"))[:n] = first N by
# filename/body-id (lexical), NOT largest-by-size. This is the SAME 40 the old
# ~40min NG run used (download_hemibrain.py:600/604) -> apples-to-apples.
paths = sorted(glob.glob(os.path.join(mesh_dir, "*.obj")))[:n]
print(f"{len(paths)} neurons (lexical first-{n}, matches driver); total OBJ MB="
      f"{sum(os.path.getsize(p) for p in paths)/1e6:.0f}")

bounds = tuple(scan_obj_bounds(paths))
tags = [{} for _ in paths]

gen = StreamingTileGenerator(min_zoom=0, max_zoom=3, base_cells=100)
gen._reset_ng_peak_resident_bytes()

t0 = time.perf_counter()
gen.add_obj_files(paths, bounds, tags)
t_ingest = time.perf_counter() - t0
print(f"ingest: {t_ingest:.1f}s")

t0 = time.perf_counter()
count = gen.generate_neuroglancer_multilod(out, bounds, 10)  # qbits=10, ceiling=0 -> resolver (k=1 on big-RAM host)
t_gen = time.perf_counter() - t0

print(f"NG generate: {t_gen:.1f}s  segments={count}  "
      f"k_buckets={gen._get_ng_bucket_count()}  "
      f"peak_resident_bytes={gen._get_ng_peak_resident_bytes()}")
print(f"=== NG generate {t_gen:.1f}s vs old ~40min(2400s) baseline ===")
