#!/usr/bin/env python3
"""Decisive consolidation I/O-bottleneck probe (Task 1 of the parallel-rewrite plan).

PURPOSE
    Decide ONE of three I/O regimes for the Hemibrain post-ingest consolidation
    read path, against a *real* fragment directory of ZSTD-compressed ``.mjf``
    shards on the DEPLOYMENT box. The consolidation is I/O-bound (ZSTD decode is
    ~0.06% utilized per the analysis), so the decisive signal is RAW BYTE READ
    throughput: open each shard and read it to EOF, counting bytes. This needs
    NO knowledge of the MJF2 container and runs on any box with plain Python.

        (A) BANDWIDTH / QUEUE-DEPTH-BOUND -- aggregate MB/s rises with thread
            count k to a knee well above the ~14 MB/s serial baseline (e.g.
            >=150 MB/s). The parallel reader is the whole fix; set
            io_threads ~= the measured knee. (Plan: do Tasks 2-7, SKIP Task 8.)
        (B) IOPS / open()-SYSCALL-BOUND -- files/sec plateaus at a few hundred
            regardless of k and MB/s barely rises. The parallel reader still
            helps ~7-15x, but the rank-1 fix is write-side shard coalescing.
            Pick a conservative io_threads. (Plan: Tasks 2-7 + schedule Task 8.)
        (C) HDD SEEK-THRASH REGRESSION -- MB/s at k>1 DROPS below the serial
            baseline. Default io_threads=1 (serial), keep inode ordering,
            prioritize coalescing. (Plan: io_threads=1 + prioritize Task 8.)

READ-ONLY
    This tool NEVER modifies, renames, or deletes anything. It only ever opens
    shard files for READING. The optional ``posix_fadvise(POSIX_FADV_DONTNEED)``
    call is a cache hint, not a write -- it cannot alter or remove file content.

PAGE-CACHE CAVEAT (the single most important correctness property)
    Re-reading a shard hits the OS page cache (RAM), not the disk, which would
    make every run after the first meaningless. This probe therefore partitions
    the sampled shards into DISJOINT subsets, one per (order, mode, k)
    measurement, so every measured run reads shards that NO earlier run in this
    process touched -- no file is ever measured twice. The partition is carved
    in CANONICAL (lexical) index space ONCE, then each run's fixed file SET is
    reordered into the requested order; this is what guarantees disjointness
    holds ACROSS orders, where lexical and inode sort the same files into
    different positions. On Linux it additionally evicts each file from cache
    after reading (best-effort ``posix_fadvise(POSIX_FADV_DONTNEED)``); disjoint
    sampling is the primary defense, fadvise is belt-and-suspenders.
    RESULTS ARE INVALID if the directory was just written or read (warm cache).
    For a clean read, drop caches first (root):
        sync; echo 3 > /proc/sys/vm/drop_caches

DEV-BOX != DEPLOY-BOX CAVEAT
    Numbers from the macOS dev box (APFS SSD, few cores) DO NOT transfer to the
    Linux deployment box (many-core, RAID, UNKNOWN storage media). The verdict
    is only meaningful when run on the deployment box against the real frag dir.

RUN
    uv run scripts/measure_consolidation.py /path/to/real/frag_dir

INTERPRETING THE VERDICT
    Read the VERDICT block at the end. It states the regime (A/B/C), a
    recommended ``io_threads`` value, whether to set the ``MUDM_IO_THREADS``
    environment variable, and the exact plan-task branch to follow. The
    aligned table above it shows, per (order, mode, k): aggregate MB/s,
    files/sec, p95 per-file latency, CPU%, and peak-RSS delta. Compare the best
    MB/s and files/sec against the k=1 serial baseline:
      * MB/s climbs to a high knee   -> regime A (bandwidth/queue-depth-bound)
      * files/sec flat, MB/s flat     -> regime B (IOPS / open()-bound)
      * MB/s at k>1 below serial       -> regime C (HDD seek-thrash)
    The regime is decided from the LEXICAL curve -- the order that
    ``Fragment3DReader::open_dir`` actually ships by default (it does
    ``paths.sort()`` on full paths) -- so a thrashing default is never masked
    by a faster inode order. inode order is reported only as a possible
    MITIGATION (plan branch C keeps inode ordering).
    NOTE: only AGGREGATE MB/s is the load-bearing metric. Per-file p95 latency
    at k>1 includes ThreadPoolExecutor queue wait, so it conflates device
    latency with queueing delay -- read it as a trend, not an absolute. The
    lexical-vs-inode delta is DIRECTIONAL only: each run's inode order is the
    st_ino sort of a scattered disjoint subset, not a contiguous physical sweep.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001 -- not all stdout objects support this
    pass

# --- Optional dependencies: all degrade gracefully. -------------------------
try:
    import psutil  # type: ignore
    _HAVE_PSUTIL = True
except Exception:  # noqa: BLE001
    _HAVE_PSUTIL = False

try:
    import resource  # POSIX only (Linux + macOS); absent on Windows.
    _HAVE_RESOURCE = True
except Exception:  # noqa: BLE001
    _HAVE_RESOURCE = False

_IS_LINUX = sys.platform.startswith("linux")
_IS_MACOS = sys.platform == "darwin"

# os.posix_fadvise + POSIX_FADV_DONTNEED exist only where the C library has
# them (Linux Python builds). Guard so macOS / Windows are unaffected.
_HAVE_FADVISE = hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED")

_READ_CHUNK = 1 << 20  # 1 MiB read chunk
_KNEE_GAIN = 0.15      # >=15% MB/s improvement over previous k counts as a rise


# ---------------------------------------------------------------------------
# Formatting helpers (match scripts/ house style)
# ---------------------------------------------------------------------------

def _fmt_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{s:.0f}s"


def _fmt_bytes(n: float) -> str:
    if n < 1024:
        return f"{n:.0f} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.2f} MB"
    return f"{n / (1024 * 1024 * 1024):.2f} GB"


def _maxrss_bytes() -> int:
    """Peak RSS so far, normalized to BYTES.

    ``ru_maxrss`` is KiB on Linux but BYTES on macOS/BSD -- normalize and label.
    Returns 0 if the ``resource`` module is unavailable (e.g. Windows).
    """
    if not _HAVE_RESOURCE:
        return 0
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if _IS_MACOS:
        return int(raw)          # already bytes on macOS/BSD
    return int(raw) * 1024       # Linux + most other POSIX report KiB


# ---------------------------------------------------------------------------
# Per-file read worker (the only thing that touches the disk)
# ---------------------------------------------------------------------------

def _read_one_raw(path: str, fadvise: bool) -> tuple[float, int, bool]:
    """Open + read one shard to EOF in 1 MiB chunks; count bytes. READ-ONLY.

    Returns ``(per_file_seconds, n_bytes, ok)``. On any error returns ok=False
    so the caller can skip and continue. Best-effort POSIX_FADV_DONTNEED after
    reading evicts the file from page cache (Linux only; never modifies data).
    """
    t0 = time.perf_counter()
    n = 0
    fd = -1
    try:
        # Open read-only by fd so we can fadvise(DONTNEED) on the same fd.
        fd = os.open(path, os.O_RDONLY)
        while True:
            chunk = os.read(fd, _READ_CHUNK)
            if not chunk:
                break
            n += len(chunk)
        if fadvise and _HAVE_FADVISE:
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass  # best-effort cache eviction; ignore failures
        return (time.perf_counter() - t0, n, True)
    except (OSError, IOError):
        return (time.perf_counter() - t0, n, False)
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass


def _decode_one_zstd(path: str, fadvise: bool) -> tuple[float, int, bool]:
    """Open + ZSTD-decode one shard to EOF; count UNCOMPRESSED bytes. READ-ONLY.

    Adds decompression on top of the raw read so the measurement reflects
    decode+I/O. Only used with --decode when ``zstandard`` is importable. Each
    ``.mjf`` is a single ZSTD frame (see fragment.rs ShardReader3D::open).
    """
    import zstandard  # imported lazily; presence checked by caller

    t0 = time.perf_counter()
    n = 0
    fd = -1
    try:
        fd = os.open(path, os.O_RDONLY)
        f = os.fdopen(fd, "rb", closefd=False)
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            while True:
                chunk = reader.read(_READ_CHUNK)
                if not chunk:
                    break
                n += len(chunk)
        if fadvise and _HAVE_FADVISE:
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
        return (time.perf_counter() - t0, n, True)
    except Exception:  # noqa: BLE001 -- decode errors must not abort the sweep
        return (time.perf_counter() - t0, n, False)
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# One measured run over a DISJOINT subset of shards
# ---------------------------------------------------------------------------

def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = pct / 100.0 * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _measure_run(
    paths: list[str],
    order: str,
    mode: str,
    k: int,
    fadvise: bool,
) -> dict:
    """Time reading a DISJOINT set of shards with k threads. Returns metrics.

    Wall-clock times the whole batch. ThreadPoolExecutor with k workers gives
    real I/O concurrency because CPython releases the GIL during the blocking
    os.read()/decode. Per-file errors are counted and skipped.
    """
    worker = _decode_one_zstd if mode == "decode" else _read_one_raw

    rss_before = _maxrss_bytes()
    proc = None
    if _HAVE_PSUTIL:
        proc = psutil.Process()
        proc.cpu_percent(None)  # prime the interval baseline
    cpu_t0 = time.process_time()

    per_file: list[float] = []
    total_bytes = 0
    skipped = 0

    wall_t0 = time.perf_counter()
    if k <= 1:
        for p in paths:
            dt, nbytes, ok = worker(p, fadvise)
            if ok:
                per_file.append(dt)
                total_bytes += nbytes
            else:
                skipped += 1
    else:
        with ThreadPoolExecutor(max_workers=k) as ex:
            for dt, nbytes, ok in ex.map(lambda p: worker(p, fadvise), paths):
                if ok:
                    per_file.append(dt)
                    total_bytes += nbytes
                else:
                    skipped += 1
    wall = time.perf_counter() - wall_t0

    cpu_seconds = time.process_time() - cpu_t0
    if _HAVE_PSUTIL and proc is not None:
        cpu_pct = proc.cpu_percent(None)  # avg over the interval, may exceed 100
    else:
        cpu_pct = (cpu_seconds / wall * 100.0) if wall > 0 else 0.0
    rss_after = _maxrss_bytes()

    n_ok = len(per_file)
    mb = total_bytes / (1024 * 1024)
    per_file.sort()
    mean_ms = (sum(per_file) / n_ok * 1000.0) if n_ok else 0.0

    return {
        "order": order,
        "mode": mode,
        "k": k,
        "n_files": n_ok,
        "skipped": skipped,
        "wall_s": wall,
        "total_bytes": total_bytes,
        "mb_s": (mb / wall) if wall > 0 else 0.0,
        "files_s": (n_ok / wall) if wall > 0 else 0.0,
        "mean_ms": mean_ms,
        "p50_ms": _percentile(per_file, 50) * 1000.0,
        "p95_ms": _percentile(per_file, 95) * 1000.0,
        "cpu_pct": cpu_pct,
        "rss_delta_bytes": max(0, rss_after - rss_before),
    }


# ---------------------------------------------------------------------------
# Knee detection + verdict classification
# ---------------------------------------------------------------------------

def _detect_knee(runs_by_k: dict[int, float]) -> tuple[int, float]:
    """Largest k whose MB/s improves >= _KNEE_GAIN over the previous k.

    Returns ``(knee_k, best_mb_s)``. The knee is the recommended concurrency:
    beyond it, throughput stops rising (or regresses).
    """
    ks = sorted(runs_by_k)
    if not ks:
        return (1, 0.0)
    best_mb = max(runs_by_k.values())
    knee = ks[0]
    for i in range(1, len(ks)):
        prev = runs_by_k[ks[i - 1]]
        cur = runs_by_k[ks[i]]
        if prev > 0 and (cur - prev) / prev >= _KNEE_GAIN:
            knee = ks[i]
        # else: improvement stalled -- keep the last good knee.
    return (knee, best_mb)


def _classify(
    serial_mb: float,
    best_mb: float,
    knee_k: int,
    best_mb_at_k_gt1: float,
) -> str:
    """Return regime 'A', 'B', or 'C' from the scaling signal.

    C (regression) takes precedence: if even the BEST multithreaded throughput
    cannot hold ~90% of the serial baseline, adding reader threads actively
    hurts -- that is HDD seek-thrash, and io_threads MUST stay at 1. We key on
    the BEST of k>1 (not the min) so a single jittery slow run cannot spuriously
    force C; conversely we do NOT gate the C check on knee_k, because a spurious
    internal >=15% local rise (e.g. 8->11 MB/s) can flag a knee>=2 on a curve
    that is still entirely below serial -- the unconditional best-vs-serial test
    catches that and returns C (the catastrophic-misconfig case the plan's Risk
    #1 warns about). A 10% tolerance absorbs measurement noise.

    files/sec is intentionally NOT used as a separate signal: shard sizes are
    near-uniform (~580 KB), so files/sec is collinear with MB/s. It is reported
    in the table/JSON for transparency only.
    """
    # C: seek-thrash. The decisive signal is that the BEST multithreaded
    # throughput fails to beat ~90% of serial (parallelism actively hurts),
    # regardless of any spurious internal knee.
    if serial_mb > 0 and best_mb_at_k_gt1 < serial_mb * 0.90:
        return "C"

    speedup = (best_mb / serial_mb) if serial_mb > 0 else 0.0
    # A: bandwidth/queue-depth-bound -- scaled well past serial to a real knee.
    if knee_k >= 2 and speedup >= 3.0 and best_mb >= 150.0:
        return "A"
    # A (weaker box / lower ceiling): still clearly scales with k.
    if knee_k >= 2 and speedup >= 2.0:
        return "A"
    # B: IOPS/open()-bound -- more threads barely helped MB/s.
    return "B"


# ---------------------------------------------------------------------------
# Output: aligned table + verdict
# ---------------------------------------------------------------------------

def _print_table(runs: list[dict]) -> None:
    hdr = (
        f"{'order':<8} {'mode':<7} {'k':>3} "
        f"{'MB/s':>9} {'files/s':>9} {'p95 ms':>8} "
        f"{'CPU%':>7} {'RSS_d':>9} {'files':>6} {'skip':>5}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in runs:
        print(
            f"{r['order']:<8} {r['mode']:<7} {r['k']:>3} "
            f"{r['mb_s']:>9.1f} {r['files_s']:>9.1f} {r['p95_ms']:>8.2f} "
            f"{r['cpu_pct']:>7.1f} {_fmt_bytes(r['rss_delta_bytes']):>9} "
            f"{r['n_files']:>6} {r['skipped']:>5}"
        )
    print("(per-file p95 at k>1 includes pool queue wait; only aggregate MB/s "
          "is load-bearing.)")


_VERDICT_TEXT = {
    "A": (
        "REGIME A: BANDWIDTH / QUEUE-DEPTH-BOUND",
        "The parallel reader is the WHOLE fix. Aggregate MB/s rose with thread "
        "count to a knee well above the ~14 MB/s serial baseline.",
        "Plan branch: implement Tasks 2-7 only. SKIP Task 8 (write-side "
        "coalescing) -- the file count is not the bottleneck.",
    ),
    "B": (
        "REGIME B: IOPS / open()-SYSCALL-BOUND",
        "The parallel reader helps (~7-15x) but is NOT the rank-1 fix. files/sec "
        "plateaued and MB/s barely rose with more threads -- per-file open() "
        "overhead dominates at this shard count.",
        "Plan branch: implement Tasks 2-7 (conservative io_threads = measured "
        "knee), AND schedule Task 8 (write-side shard coalescing) as the rank-1 "
        "fix.",
    ),
    "C": (
        "REGIME C: HDD SEEK-THRASH REGRESSION",
        "Adding reader threads made throughput WORSE than the serial baseline -- "
        "concurrent reads on this media flip sequential I/O into seek-thrash.",
        "Plan branch: default io_threads=1 (serial fallback), keep inode "
        "ordering, and PRIORITIZE Task 8 (write-side coalescing).",
    ),
}


def _print_verdict(summary: dict) -> None:
    regime = summary["regime"]
    title, why, branch = _VERDICT_TEXT[regime]
    rec = summary["recommended_io_threads"]
    bar = "=" * 78
    print()
    print(bar)
    print(f"  VERDICT: {title}")
    print(bar)
    print(f"  Baseline = lexical k=1 (the order open_dir actually ships).")
    print(f"  Serial baseline (k=1):   {summary['serial_mb_s']:.1f} MB/s")
    if summary["serial_mb_s"] > 0:
        speedup = summary["best_mb_s"] / summary["serial_mb_s"]
        print(f"  Best aggregate:          {summary['best_mb_s']:.1f} MB/s "
              f"at the k={summary['knee_k']} knee ({speedup:.1f}x serial)")
    else:
        print(f"  Best aggregate:          {summary['best_mb_s']:.1f} MB/s "
              f"at the k={summary['knee_k']} knee")
    if summary.get("verdict_order") and summary["verdict_order"] != "lexical":
        print(f"  (lexical not measured; verdict derived from "
              f"'{summary['verdict_order']}' order instead.)")
    if not summary.get("has_serial_baseline", True):
        print("  WARNING: no k=1 run in the sweep; serial baseline is the "
              "SLOWEST measured k. Verdict is PROVISIONAL -- re-run including "
              "k=1 for a defensible baseline.")
    print()
    print(f"  Why: {why}")
    print()
    print(f"  {branch}")
    print()
    print(f"  RECOMMENDED io_threads = {rec}")
    if regime == "C":
        print(f"  Set: export MUDM_IO_THREADS=1   (force serial; do NOT raise it)")
    else:
        print(f"  Set: export MUDM_IO_THREADS={rec}")
    print(bar)


# ---------------------------------------------------------------------------
# Sampling + page-cache-disjoint partitioning
# ---------------------------------------------------------------------------

def _sorted_paths(all_paths: list[Path], order: str) -> list[Path]:
    """Return paths in the requested order.

    'lexical' mirrors Fragment3DReader::open_dir's sorted glob order.
    'inode'   sorts by st_ino (Linux physical-order proxy; harmless elsewhere).
    """
    if order == "inode":
        def _ino(p: Path) -> int:
            try:
                return p.stat().st_ino
            except OSError:
                return 0
        return sorted(all_paths, key=_ino)
    return sorted(all_paths)  # lexical


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Decisive consolidation I/O-bottleneck probe (READ-ONLY). Reads real "
            ".mjf shards serial vs k-threaded, lexical vs inode order, and prints "
            "an A/B/C regime verdict. Run on the DEPLOYMENT box."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("frag_dir", type=Path, help="Directory of *.mjf shards")
    parser.add_argument(
        "--sample", type=int, default=2000,
        help="Number of DISTINCT shards to measure across ALL runs combined "
             "(default: 2000; total reads NEVER exceed this). Partitioned "
             "disjointly so no file is read twice. For a steady deploy-box "
             "verdict use >=4000 so each run gets >=~200 files.",
    )
    parser.add_argument(
        "--threads", default="1,2,4,8,16,32",
        help="Comma-separated k values to sweep (default: 1,2,4,8,16,32). "
             "Include 1 for a true serial baseline.",
    )
    parser.add_argument(
        "--orders", default="lexical,inode",
        help="Comma-separated orderings: lexical (open_dir order) and/or inode "
             "(Linux physical-order proxy). Default: lexical,inode.",
    )
    parser.add_argument(
        "--decode", action="store_true",
        help="Also measure decode+I/O (ZSTD). Tries mudm_tools._rs, then "
             "'zstandard'; skips with a note if neither is available. "
             "DEFAULT OFF -- raw byte read is the decisive signal.",
    )
    parser.add_argument(
        "--json", type=Path, default=None,
        help="Also write the machine-readable summary dict to this path.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for deterministic shard sampling (default: 0).",
    )
    args = parser.parse_args()

    frag_dir: Path = args.frag_dir
    if not frag_dir.is_dir():
        print(f"ERROR: not a directory: {frag_dir}", file=sys.stderr)
        sys.exit(1)

    # --- Enumerate shards ---
    print(f"Scanning {frag_dir} for *.mjf shards ...", flush=True)
    all_paths = sorted(frag_dir.glob("*.mjf"))
    shard_count = len(all_paths)
    if shard_count == 0:
        print(f"ERROR: no *.mjf files in {frag_dir}", file=sys.stderr)
        sys.exit(1)

    # Mean/median size from a bounded stat sample (516k files is too many to
    # stat fully just for a summary; this is a size estimate, not a measurement).
    rng = random.Random(args.seed)
    stat_idx = list(range(shard_count))
    if shard_count > 4000:
        stat_idx = rng.sample(stat_idx, 4000)
    sizes: list[int] = []
    for i in stat_idx:
        try:
            sizes.append(all_paths[i].stat().st_size)
        except OSError:
            pass
    sizes.sort()
    mean_kb = (sum(sizes) / len(sizes) / 1024.0) if sizes else 0.0
    median_kb = (sizes[len(sizes) // 2] / 1024.0) if sizes else 0.0

    print(f"files={shard_count} mean_kb={mean_kb:.1f} median_kb={median_kb:.1f} "
          f"(size from {len(sizes)} sampled shards)", flush=True)

    # --- Resolve modes ---
    modes = ["raw"]
    decode_note = ""
    if args.decode:
        # Fragment3DReader is NOT exposed to Python (confirmed: rust/src/lib.rs
        # registers only the obj/clip/tile/encoder/projector/streaming/decoder
        # symbols in the _rs pymodule; Fragment3DReader is internal Rust). Fall
        # back to whole-frame zstandard decode (each .mjf is a single ZSTD frame
        # per fragment.rs ShardReader3D::open).
        rs_present = False
        try:
            import mudm_tools._rs as _rs  # type: ignore  # noqa: F401
            rs_present = True
        except Exception:  # noqa: BLE001
            rs_present = False
        if rs_present:
            decode_note = "mudm_tools._rs present (no per-shard Python reader exported)"
        try:
            import zstandard  # noqa: F401
            modes.append("decode")
            decode_note = (decode_note + "; " if decode_note else "") + \
                "decode via zstandard whole-frame"
        except Exception:  # noqa: BLE001
            decode_note = (decode_note + "; " if decode_note else "") + \
                "zstandard NOT importable -- decode mode SKIPPED"
            print(f"NOTE: --decode requested but {decode_note}", flush=True)

    # --- Parse sweep params ---
    try:
        ks = sorted({int(x) for x in args.threads.split(",") if x.strip()})
        ks = [k for k in ks if k >= 1]
    except ValueError:
        print(f"ERROR: bad --threads list: {args.threads}", file=sys.stderr)
        sys.exit(1)
    if not ks:
        ks = [1]
    valid_orders = {"lexical", "inode"}
    orders = [o.strip() for o in args.orders.split(",") if o.strip() in valid_orders]
    if not orders:
        print(f"ERROR: bad --orders list: {args.orders} "
              f"(valid: lexical, inode)", file=sys.stderr)
        sys.exit(1)

    has_serial_baseline = 1 in ks
    if not has_serial_baseline:
        print(
            "WARNING: --threads has no k=1 entry. There is no true SERIAL "
            "baseline, so the A/B/C speedup and seek-thrash comparison are made "
            "against the SLOWEST measured k. Add 1 to --threads for a defensible "
            "verdict.",
            file=sys.stderr, flush=True,
        )

    # --- Plan the DISJOINT partition (page-cache correctness, HARD REQ #2) ---
    # One run per (order, mode, k). To keep ratios comparable across k, every
    # run reads the SAME number of files (files_per_run), drawn from disjoint
    # slices of the shuffled sample so no file is ever touched twice. The total
    # number of distinct shards read (files_per_run * n_runs) must NEVER exceed
    # min(--sample, shard_count): --sample is the ceiling on DISTINCT shards.
    # When the budget is too small for one file per run, SCALE RUNS DOWN (drop
    # the largest k values first, then the second order) rather than inflating
    # files-per-run past the requested ceiling.
    requested_sample = min(args.sample, shard_count)
    if requested_sample < 1:
        print(f"ERROR: --sample resolves to {requested_sample} shards "
              f"(have {shard_count}); nothing to measure.", file=sys.stderr)
        sys.exit(1)

    def _plan_runs(orders_in: list[str], ks_in: list[int]) -> int:
        return len(orders_in) * len(modes) * len(ks_in)

    sweep_orders = list(orders)
    sweep_ks = list(ks)
    trimmed = False
    # Trim the sweep until at least 1 file per run fits inside requested_sample.
    while _plan_runs(sweep_orders, sweep_ks) > requested_sample:
        trimmed = True
        if len(sweep_ks) > 1:
            sweep_ks = sweep_ks[:-1]            # drop the largest k
        elif len(sweep_orders) > 1:
            sweep_orders = sweep_orders[:-1]    # drop the second order
        else:
            break  # 1 order x 1 mode-set x 1 k; cannot trim further

    orders = sweep_orders
    ks = sweep_ks
    n_runs = _plan_runs(orders, ks)
    files_per_run = max(1, requested_sample // n_runs)
    needed = files_per_run * n_runs
    # Final guards: never exceed the distinct-shard ceiling.
    if needed > requested_sample:
        files_per_run = max(1, requested_sample // n_runs)
        needed = files_per_run * n_runs
    if needed > shard_count:
        files_per_run = max(1, shard_count // n_runs)
        needed = files_per_run * n_runs

    if trimmed:
        print(
            f"WARNING: --sample={args.sample} too small for the full sweep; "
            f"TRIMMED to orders={orders} threads={ks} so total distinct shards "
            f"({needed}) stays within the requested ceiling "
            f"({requested_sample}). Disjoint partitioning preserved; no file is "
            f"read twice. Increase --sample to restore the full sweep.",
            file=sys.stderr, flush=True,
        )
    if files_per_run < 30:
        print(
            f"WARNING: only {files_per_run} files per measured run "
            f"({n_runs} runs x disjoint slices = {needed} of {shard_count} "
            f"shards). Per-run timing will be NOISY. Increase --sample (>=4000 "
            f"recommended) or reduce --threads / --orders / --decode for a "
            f"stable verdict.",
            file=sys.stderr, flush=True,
        )

    # Build one shuffled pool of CANONICAL (all_paths / lexical) indices, then
    # carve disjoint contiguous slices. Disjointness is enforced in canonical
    # index space so it holds ACROSS orders: lexical and inode sort the same
    # files into different positions, so sharing pool indices across per-order
    # lists would alias the SAME physical file and break the cache guarantee.
    pool = list(range(shard_count))
    rng.shuffle(pool)
    pool = pool[:needed]

    # --- Caveats (always print up front) ---
    print()
    print("CAVEATS:")
    print("  * READ-ONLY: this tool never writes/renames/deletes shards.")
    print("  * Page cache: results are INVALID if this dir was just written or "
          "read (warm cache).")
    if _IS_LINUX:
        print("    To drop caches (root): sync; echo 3 > /proc/sys/vm/drop_caches")
        print(f"    posix_fadvise(DONTNEED) cache eviction: "
              f"{'ON' if _HAVE_FADVISE else 'unavailable'}.")
    else:
        print(f"    Running on {sys.platform}: this is the DEV box, NOT the "
              f"deploy box. posix_fadvise(DONTNEED) unavailable here; rely on "
              f"disjoint sampling only. Numbers DO NOT transfer to deploy.")
    print(f"  * Disjoint partition: {files_per_run} files/run x {n_runs} runs = "
          f"{needed} distinct shards; no shard is measured twice (carved in "
          f"canonical index space, valid across orders).")
    psutil_state = "available" if _HAVE_PSUTIL else "absent (CPU% from process_time)"
    print(f"  * psutil: {psutil_state}.")
    if args.decode:
        print(f"  * decode mode: {decode_note or 'skipped'}.")
    print()

    # --- Run the sweep (order outer, then mode, then k) ---
    #
    # PAGE-CACHE CORRECTNESS: every run reads PHYSICAL files that no other run
    # touched, REGARDLESS of order. We carve disjoint slices of CANONICAL
    # (all_paths / lexical) indices from the shuffled pool, one slice per run,
    # then reorder ONLY that slice's Path objects into the requested order.
    # Indexing a per-order-sorted list with shared pool indices would alias
    # different files across orders and break disjointness -- so we never do
    # that. The file SET per run is fixed by canonical index; the ORDER is
    # applied afterwards to that set alone.
    runs: list[dict] = []
    slice_i = 0
    interrupted = False
    measured_canonical: set[int] = set()  # defensive disjointness invariant
    try:
        for order in orders:
            for mode in modes:
                fadvise = True  # always attempt DONTNEED; no-op off Linux
                for k in ks:
                    lo = slice_i * files_per_run
                    hi = lo + files_per_run
                    canon_idxs = pool[lo:hi]
                    slice_i += 1
                    # Disjointness invariant: these canonical indices must be
                    # brand new. pool[:needed] is a permutation of distinct
                    # indices carved into non-overlapping slices, so this holds.
                    # Explicit raise (not assert) so it survives `python -O`.
                    overlap = measured_canonical.intersection(canon_idxs)
                    if overlap:
                        raise RuntimeError(
                            "PAGE-CACHE BUG: run would reuse already-measured "
                            f"shards {sorted(overlap)[:5]} -- disjoint partition "
                            "violated."
                        )
                    measured_canonical.update(canon_idxs)
                    # Fix the file SET by canonical index, then sort that set
                    # into the requested order (re-stat for inode is fine; it is
                    # OUTSIDE the timed read loop so it cannot corrupt MB/s).
                    sub_paths = _sorted_paths(
                        [all_paths[i] for i in canon_idxs], order
                    )
                    sub = [str(p) for p in sub_paths]
                    print(
                        f"[run {slice_i}/{n_runs}] order={order} mode={mode} "
                        f"k={k} files={len(sub)} ...", flush=True,
                    )
                    r = _measure_run(sub, order, mode, k, fadvise)
                    runs.append(r)
                    print(
                        f"    -> {r['mb_s']:.1f} MB/s  {r['files_s']:.1f} files/s  "
                        f"p95={r['p95_ms']:.2f}ms  cpu={r['cpu_pct']:.0f}%  "
                        f"wall={_fmt_time(r['wall_s'])}  skipped={r['skipped']}",
                        flush=True,
                    )
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted -- printing partial summary.", file=sys.stderr)

    # --- Shared summary skeleton ---
    base_summary = {
        "frag_dir": str(frag_dir),
        "shard_count": shard_count,
        "mean_kb": round(mean_kb, 2),
        "median_kb": round(median_kb, 2),
        "sample": needed,
        "files_per_run": files_per_run,
        "modes": modes,
        "orders": orders,
        "threads": ks,
        "decode_note": decode_note,
        "interrupted": interrupted,
        "has_serial_baseline": has_serial_baseline,
        "platform": sys.platform,
    }

    if not runs:
        # No run completed (e.g. Ctrl-C before the first). Emit an always-valid
        # JSON stub so --json output is machine-parseable, then exit.
        stub = dict(base_summary)
        stub.update({
            "verdict_order": None,
            "runs": [],
            "knee_k": None,
            "best_mb_s": None,
            "serial_mb_s": None,
            "best_mb_s_at_k_gt1": None,
            "regime": None,
            "regime_name": None,
            "recommended_io_threads": None,
            "set_mudm_io_threads": None,
        })
        print("ERROR: no runs completed -- emitting minimal JSON stub.",
              file=sys.stderr)
        print()
        print("JSON SUMMARY:")
        print(json.dumps(stub, indent=2, default=str))
        if args.json is not None:
            try:
                args.json.parent.mkdir(parents=True, exist_ok=True)
                args.json.write_text(json.dumps(stub, indent=2, default=str))
                print(f"\nWrote summary to {args.json}")
            except OSError as e:
                print(f"WARNING: could not write --json {args.json}: {e}",
                      file=sys.stderr)
        sys.exit(130 if interrupted else 1)

    # --- Build the verdict from the RAW mode (the decisive signal) ---
    raw_runs = [r for r in runs if r["mode"] == "raw"]
    decisive = raw_runs if raw_runs else runs

    # Build per-order k->MB/s curves. The REGIME is decided from the LEXICAL
    # curve (= Fragment3DReader::open_dir's shipped order: fragment.rs does
    # paths.sort() on full paths), because that is what production uses by
    # default. inode is reported only as a possible MITIGATION (plan branch C
    # keeps inode ordering). Fall back to the best-peak order only if lexical
    # was not measured (e.g. --orders inode, or trimmed away).
    by_order: dict[str, dict[int, float]] = {}
    for r in decisive:
        by_order.setdefault(r["order"], {})[r["k"]] = r["mb_s"]
    if "lexical" in by_order:
        verdict_order = "lexical"
    else:
        verdict_order = max(by_order, key=lambda o: max(by_order[o].values()))
    curve = by_order[verdict_order]

    knee_k, best_mb = _detect_knee(curve)
    # True serial baseline = k=1 if measured; else fall back to the SLOWEST
    # measured k (and we already WARNed loudly that the verdict is provisional).
    if 1 in curve:
        serial_mb = curve[1]
    else:
        serial_mb = min(curve.values()) if curve else 0.0
    ks_gt1 = [mb for kk, mb in curve.items() if kk > 1]
    best_mb_at_k_gt1 = max(ks_gt1) if ks_gt1 else serial_mb

    regime = _classify(serial_mb, best_mb, knee_k, best_mb_at_k_gt1)

    # Recommended io_threads per regime. If the sweep had no k>1 data point we
    # never measured concurrency, so do not extrapolate beyond what we observed.
    measured_concurrency = any(kk > 1 for kk in curve)
    if regime == "A":
        recommended = max(2, knee_k) if measured_concurrency else 1
    elif regime == "B":
        # conservative; write-side coalescing (Task 8) is the rank-1 fix.
        recommended = max(2, min(knee_k, 8)) if measured_concurrency else 1
    else:  # C
        recommended = 1

    summary = dict(base_summary)
    summary.update({
        "verdict_order": verdict_order,
        "runs": runs,
        "knee_k": knee_k,
        "best_mb_s": round(best_mb, 2),
        "serial_mb_s": round(serial_mb, 2),
        "best_mb_s_at_k_gt1": round(best_mb_at_k_gt1, 2),
        "regime": regime,
        "regime_name": {
            "A": "A_BANDWIDTH_QUEUE_DEPTH_BOUND",
            "B": "B_IOPS_OPEN_SYSCALL_BOUND",
            "C": "C_HDD_SEEK_THRASH_REGRESSION",
        }[regime],
        "recommended_io_threads": recommended,
        "set_mudm_io_threads": recommended,
    })

    # --- Output ---
    print()
    _print_table(runs)
    _print_verdict(summary)

    print()
    print("JSON SUMMARY:")
    print(json.dumps(summary, indent=2, default=str))

    if args.json is not None:
        try:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(summary, indent=2, default=str))
            print(f"\nWrote summary to {args.json}")
        except OSError as e:
            print(f"WARNING: could not write --json {args.json}: {e}",
                  file=sys.stderr)

    if interrupted:
        sys.exit(130)


if __name__ == "__main__":
    main()