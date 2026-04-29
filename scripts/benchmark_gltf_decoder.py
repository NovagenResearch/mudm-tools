"""Head-to-head decode benchmark: Rust gltf crate vs. pygltflib on identical
GLB tiles. Reports decode latency. Closes the methodological asymmetry between
the prior pbf3-vs-pygltflib comparison."""
import argparse
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import pygltflib

from mudm_tools._rs import decode_glb_buffer


def bench_rust(tile_bytes: list[bytes], iters: int = 20) -> dict:
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        for b in tile_bytes:
            decode_glb_buffer(b)
        times.append(time.perf_counter() - t0)
    n_tiles = len(tile_bytes)
    mean_total_s = statistics.mean(times)
    min_total_s = min(times)
    return {
        "mean_total_s": mean_total_s,
        "stdev_total_s": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_total_s": min_total_s,
        "mean_per_tile_s": mean_total_s / n_tiles,
        "min_per_tile_s": min_total_s / n_tiles,
        "n_tiles": n_tiles,
        "iters": iters,
    }


def bench_pygltflib(tile_paths: list[Path], iters: int = 20) -> dict:
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        for p in tile_paths:
            pygltflib.GLTF2.load(str(p))
        times.append(time.perf_counter() - t0)
    n_tiles = len(tile_paths)
    mean_total_s = statistics.mean(times)
    min_total_s = min(times)
    return {
        "mean_total_s": mean_total_s,
        "stdev_total_s": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_total_s": min_total_s,
        "mean_per_tile_s": mean_total_s / n_tiles,
        "min_per_tile_s": min_total_s / n_tiles,
        "n_tiles": n_tiles,
        "iters": iters,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-dir", required=True, help="Directory containing .glb tiles")
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    tile_paths = sorted(Path(args.tiles_dir).rglob("*.glb"))[: args.sample]
    if not tile_paths:
        raise SystemExit(f"no GLB tiles found under {args.tiles_dir}")
    tile_bytes = [p.read_bytes() for p in tile_paths]

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark_id": "B6_gltf_decoder_comparison",
        "n_tiles_sampled": len(tile_paths),
        "iters": args.iters,
        "hardware": {"cpu_cores": os.cpu_count()},
        "rust_gltf": bench_rust(tile_bytes, args.iters),
        "pygltflib": bench_pygltflib(tile_paths, args.iters),
    }
    Path(args.output).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
