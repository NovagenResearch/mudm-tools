#!/usr/bin/env python3
"""Benchmark PBF3 (gzipped) vs Draco-compressed meshes vs PBF3-with-Draco hybrid.

Usage::

    uv run python scripts/benchmark_draco.py
    uv run python scripts/benchmark_draco.py --data-dir data/mouselight/2021-09-16/HortaObj/
    uv run python scripts/benchmark_draco.py --csv draco_results.csv

Metrics per quantization level (8, 10, 14, 16, 20 bits):
    - Compressed size (raw Draco bytes, gzipped PBF3, hybrid PBF3-with-Draco)
    - Encode time
    - Decode latency (median, P95)
    - L2 vertex error (mean, max, P95, P99, relative to bbox diagonal)
"""

from __future__ import annotations

import argparse
import gzip
import statistics
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

# Force unbuffered stdout for progress output
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_bytes(n: int) -> str:
    """Human-readable byte count."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.2f} MB"


def _require_dracopy():
    """Lazily import DracoPy."""
    try:
        import DracoPy
    except ImportError:
        print("ERROR: DracoPy is required. Install with: uv add DracoPy")
        sys.exit(1)
    return DracoPy


# ---------------------------------------------------------------------------
# L2 Fidelity
# ---------------------------------------------------------------------------


def compute_l2_error(
    original: np.ndarray,
    decoded: np.ndarray,
    bbox_diagonal: float | None = None,
) -> dict[str, float]:
    """Compute per-vertex L2 error between original and Draco-decoded vertices.

    DracoPy preserves vertex ordering, so we can directly compare.

    Args:
        original: Nx3 float64 original vertex positions.
        decoded: Nx3 float32/64 decoded vertex positions.
        bbox_diagonal: Bounding box diagonal for relative error. Computed if None.

    Returns:
        Dict with mean, max, p95, p99, and relative-to-diagonal metrics.
    """
    orig = np.asarray(original, dtype=np.float64)
    dec = np.asarray(decoded, dtype=np.float64)

    if orig.shape != dec.shape:
        raise ValueError(
            f"Shape mismatch: original {orig.shape} vs decoded {dec.shape}"
        )

    diffs = np.linalg.norm(orig - dec, axis=1)

    if bbox_diagonal is None:
        bbox_min = orig.min(axis=0)
        bbox_max = orig.max(axis=0)
        bbox_diagonal = float(np.linalg.norm(bbox_max - bbox_min))

    diag = max(bbox_diagonal, 1e-10)

    sorted_diffs = np.sort(diffs)
    n = len(sorted_diffs)

    return {
        "mean": float(np.mean(diffs)),
        "max": float(np.max(diffs)),
        "p95": float(sorted_diffs[int(n * 0.95)]) if n > 0 else 0.0,
        "p99": float(sorted_diffs[int(n * 0.99)]) if n > 0 else 0.0,
        "rel_mean": float(np.mean(diffs)) / diag,
        "rel_max": float(np.max(diffs)) / diag,
    }


# ---------------------------------------------------------------------------
# Mesh loading
# ---------------------------------------------------------------------------


def load_obj_meshes(data_dir: Path, max_files: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load OBJ meshes from a directory using the Rust parser.

    Returns list of (vertices_Nx3_f64, faces_Mx3_u32).
    """
    from mudm_tools._rs import parse_obj

    obj_files = sorted(data_dir.glob("*.obj"))
    if max_files > 0:
        obj_files = obj_files[:max_files]

    meshes = []
    for f in obj_files:
        verts, faces = parse_obj(str(f))
        verts = np.array(verts, dtype=np.float64).reshape(-1, 3)
        faces = np.array(faces, dtype=np.uint32).reshape(-1, 3)
        if len(verts) > 0 and len(faces) > 0:
            meshes.append((verts, faces))

    return meshes


def generate_synthetic_mesh(n_faces: int = 1000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic mesh for testing."""
    rng = np.random.default_rng(seed)
    n_verts = n_faces + 2
    verts = rng.random((n_verts, 3), dtype=np.float64) * 1000
    faces = []
    for i in range(n_faces):
        faces.append([i % n_verts, (i + 1) % n_verts, (i + 2) % n_verts])
    return verts, np.array(faces, dtype=np.uint32)


# ---------------------------------------------------------------------------
# PBF3 baseline size
# ---------------------------------------------------------------------------


def pbf3_gzipped_size(vertices: np.ndarray, faces: np.ndarray) -> int:
    """Compute gzipped size of PBF3-style raw vertex/index data.

    PBF3 stores positions as float32 LE + indices as uint32 LE.
    """
    pos_bytes = np.ascontiguousarray(vertices, dtype=np.float32).tobytes()
    idx_bytes = np.ascontiguousarray(faces, dtype=np.uint32).tobytes()
    raw = pos_bytes + idx_bytes
    return len(gzip.compress(raw, compresslevel=6))


def pbf3_raw_size(vertices: np.ndarray, faces: np.ndarray) -> int:
    """Raw PBF3 mesh data size (float32 positions + uint32 indices)."""
    return vertices.shape[0] * 3 * 4 + faces.shape[0] * 3 * 4


# ---------------------------------------------------------------------------
# Draco encode/decode
# ---------------------------------------------------------------------------


def draco_encode(
    vertices: np.ndarray,
    faces: np.ndarray,
    quantization_bits: int = 14,
    compression_level: int = 7,
) -> bytes:
    """Encode mesh with Draco at given quantization level."""
    DracoPy = _require_dracopy()
    points = np.ascontiguousarray(vertices, dtype=np.float32)
    face_idx = np.ascontiguousarray(faces, dtype=np.uint32)
    return DracoPy.encode(
        points, faces=face_idx,
        quantization_bits=quantization_bits,
        compression_level=compression_level,
    )


def draco_decode(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Decode Draco bytes back to vertices and faces."""
    DracoPy = _require_dracopy()
    mesh = DracoPy.decode(data)
    verts = np.array(mesh.points, dtype=np.float32).reshape(-1, 3)
    faces = np.array(mesh.faces, dtype=np.uint32).reshape(-1, 3)
    return verts, faces


# ---------------------------------------------------------------------------
# Hybrid PBF3-with-Draco
# ---------------------------------------------------------------------------


def hybrid_pbf3_draco_size(
    vertices: np.ndarray,
    faces: np.ndarray,
    quantization_bits: int = 14,
) -> tuple[int, int]:
    """Compute gzipped size of PBF3 tile with Draco-encoded mesh bytes.

    Replaces the raw float32 positions with Draco-encoded bytes,
    keeping the PBF3 protobuf framing and gzip.

    Returns:
        (gzipped_size, draco_raw_size)
    """
    draco_bytes = draco_encode(vertices, faces, quantization_bits)
    # Simulate PBF3 framing: length-prefixed Draco blob + indices overhead
    # In practice, the Draco blob replaces mesh_positions in the protobuf
    raw = draco_bytes
    gz_size = len(gzip.compress(raw, compresslevel=6))
    return gz_size, len(draco_bytes)


# ---------------------------------------------------------------------------
# Benchmark per quantization level
# ---------------------------------------------------------------------------


def benchmark_single(
    meshes: list[tuple[np.ndarray, np.ndarray]],
    quant_bits: int,
    decode_iterations: int = 20,
) -> dict[str, Any]:
    """Run full benchmark for one quantization level across all meshes."""
    total_draco_size = 0
    total_pbf3_gz_size = 0
    total_pbf3_raw_size = 0
    total_hybrid_gz_size = 0
    total_hybrid_draco_size = 0
    total_verts = 0
    total_faces = 0
    encode_times: list[float] = []
    decode_times: list[float] = []
    all_l2: list[dict[str, float]] = []

    for verts, faces in meshes:
        total_verts += len(verts)
        total_faces += len(faces)

        # PBF3 baseline
        total_pbf3_raw_size += pbf3_raw_size(verts, faces)
        total_pbf3_gz_size += pbf3_gzipped_size(verts, faces)

        # Draco encode
        t0 = time.perf_counter()
        draco_bytes = draco_encode(verts, faces, quant_bits)
        encode_times.append(time.perf_counter() - t0)
        total_draco_size += len(draco_bytes)

        # Hybrid
        hz, hd = hybrid_pbf3_draco_size(verts, faces, quant_bits)
        total_hybrid_gz_size += hz
        total_hybrid_draco_size += hd

        # Decode latency
        for _ in range(decode_iterations):
            t0 = time.perf_counter()
            decoded_verts, _ = draco_decode(draco_bytes)
            decode_times.append(time.perf_counter() - t0)

        # L2 error (once per mesh)
        decoded_verts, _ = draco_decode(draco_bytes)
        if len(decoded_verts) == len(verts):
            l2 = compute_l2_error(verts, decoded_verts)
            all_l2.append(l2)

    # Aggregate L2
    l2_agg: dict[str, float] = {}
    if all_l2:
        for key in ["mean", "max", "p95", "p99", "rel_mean", "rel_max"]:
            vals = [d[key] for d in all_l2]
            l2_agg[f"l2_{key}"] = float(np.mean(vals))

    sorted_decode = sorted(decode_times)
    n_d = len(sorted_decode)

    return {
        "quant_bits": quant_bits,
        "n_meshes": len(meshes),
        "total_verts": total_verts,
        "total_faces": total_faces,
        "pbf3_raw": total_pbf3_raw_size,
        "pbf3_gzip": total_pbf3_gz_size,
        "draco_size": total_draco_size,
        "hybrid_draco_size": total_hybrid_draco_size,
        "hybrid_gz": total_hybrid_gz_size,
        "encode_median_ms": statistics.median(encode_times) * 1000 if encode_times else 0,
        "decode_median_ms": statistics.median(sorted_decode) * 1000 if sorted_decode else 0,
        "decode_p95_ms": sorted_decode[int(n_d * 0.95)] * 1000 if n_d > 0 else 0,
        **l2_agg,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_report(label: str, results: list[dict[str, Any]]) -> None:
    """Print formatted comparison table."""
    print(f"\n{'=' * 90}")
    print(f"  {label}")
    print(f"{'=' * 90}")

    if not results:
        print("  No results.")
        return

    r0 = results[0]
    print(f"  Meshes: {r0['n_meshes']}, Vertices: {r0['total_verts']:,}, Faces: {r0['total_faces']:,}")
    print(f"  PBF3 raw: {_fmt_bytes(r0['pbf3_raw'])}, PBF3 gzip: {_fmt_bytes(r0['pbf3_gzip'])}")

    print(f"\n  {'Quant':>6s}  {'Draco':>10s}  {'Hybrid gz':>10s}  {'Enc ms':>8s}  {'Dec ms':>8s}  {'Dec P95':>8s}  {'L2 mean':>10s}  {'L2 max':>10s}  {'L2 rel%':>8s}")
    print(f"  {'─' * 86}")

    for r in results:
        l2_mean = r.get("l2_mean", 0)
        l2_max = r.get("l2_max", 0)
        l2_rel = r.get("l2_rel_mean", 0) * 100
        print(
            f"  {r['quant_bits']:>5d}b  "
            f"{_fmt_bytes(r['draco_size']):>10s}  "
            f"{_fmt_bytes(r['hybrid_gz']):>10s}  "
            f"{r['encode_median_ms']:>7.2f}  "
            f"{r['decode_median_ms']:>7.3f}  "
            f"{r['decode_p95_ms']:>7.3f}  "
            f"{l2_mean:>10.4f}  "
            f"{l2_max:>10.4f}  "
            f"{l2_rel:>7.4f}"
        )

    # Compression ratio summary
    print(f"\n  Compression ratio vs PBF3 gzip:")
    for r in results:
        ratio = r['pbf3_gzip'] / r['draco_size'] if r['draco_size'] > 0 else 0
        hybrid_ratio = r['pbf3_gzip'] / r['hybrid_gz'] if r['hybrid_gz'] > 0 else 0
        print(f"    {r['quant_bits']:>2d}-bit: Draco {ratio:.2f}x smaller, Hybrid {hybrid_ratio:.2f}x smaller")

    print()


def results_to_csv_rows(label: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten results for CSV export."""
    rows = []
    for r in results:
        row = {"dataset": label}
        row.update(r)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark PBF3 vs Draco compression",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing .obj files (MouseLight HortaObj)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Max OBJ files to load (0=all)",
    )
    parser.add_argument(
        "--quant-levels",
        nargs="+",
        type=int,
        default=[8, 10, 14, 16, 20],
        help="Draco quantization bit levels (default: 8 10 14 16 20)",
    )
    parser.add_argument(
        "--decode-iters",
        type=int,
        default=20,
        help="Iterations for decode latency benchmark (default: 20)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Write results to CSV file",
    )
    args = parser.parse_args()

    csv_rows: list[dict[str, Any]] = []

    # --- Real dataset ---
    if args.data_dir and args.data_dir.exists():
        print(f"\n>>> Loading OBJ meshes from {args.data_dir}...")
        meshes = load_obj_meshes(args.data_dir, args.max_files)
        print(f"    Loaded {len(meshes)} meshes")

        if meshes:
            label = f"obj_{args.data_dir.name}"
            results = []
            for qb in args.quant_levels:
                print(f"    Benchmarking {qb}-bit quantization...")
                r = benchmark_single(meshes, qb, args.decode_iters)
                results.append(r)

            print_report(label, results)
            csv_rows.extend(results_to_csv_rows(label, results))

    # --- Synthetic fallback ---
    if not args.data_dir or not args.data_dir.exists():
        for n_faces in [1000, 10000]:
            label = f"synthetic_{n_faces}"
            print(f"\n>>> Generating synthetic mesh ({n_faces} faces)...")
            verts, faces = generate_synthetic_mesh(n_faces)
            meshes = [(verts, faces)]

            results = []
            for qb in args.quant_levels:
                print(f"    Benchmarking {qb}-bit quantization...")
                r = benchmark_single(meshes, qb, args.decode_iters)
                results.append(r)

            print_report(label, results)
            csv_rows.extend(results_to_csv_rows(label, results))

    # --- CSV export ---
    if args.csv and csv_rows:
        import csv

        all_keys = list(dict.fromkeys(k for row in csv_rows for k in row.keys()))
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Results written to {args.csv}")

    print("Done.")


if __name__ == "__main__":
    main()
