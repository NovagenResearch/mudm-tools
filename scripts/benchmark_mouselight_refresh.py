"""B8: MouseLight 38-brain pipeline refresh on current hardware.

For each brain directory under `data/mouselight/<brain>/`, runs the muDM 3D
pipeline (octree -> simplify (QEM) -> encode GLB + Parquet) by invoking the
existing per-brain scripts, then captures per-brain wall time and output sizes
for three encodings of the same pyramid:
  - Parquet (partitioned)
  - GLB inside 3D Tiles on disk
  - GLB inside 3D Tiles, summed over per-file Brotli (HTTP transport size)

Wraps the existing per-brain scripts:
  - scripts/mouselight_parquet.py        (Parquet pyramid)
  - scripts/mouselight_3dtiles_draco.py  (GLB inside 3D Tiles, Draco-compressed)

NOTE ON COMPRESSION: the current per-brain 3D-tiles script encodes GLB with
Draco geometry compression (use_draco=True hard-coded). The plan referenced a
"meshopt" variant for MouseLight, but only `mouselight_3dtiles_draco.py`
exists in this repo for the MouseLight dataset. The wrapper therefore reports
`compression: "draco"` in the output JSON. Switching to meshopt would require
a one-line change in `mouselight_3dtiles_draco.py` (or a sibling
`mouselight_3dtiles_meshopt.py`) to pass `compression="meshopt"` to
`generate_3dtiles` (cf. `scripts/hemibrain_3dtiles_meshopt.py`).

CLI CONSTRAINTS: the per-brain scripts hard-code their input/output roots to
`<mudm-tools>/data/mouselight/` and `.../tiles/`, and accept only `--brain
<date-folder>`. This wrapper therefore requires `--brains-dir` to be
`<mudm-tools>/data/mouselight` (verified at startup) and reads output sizes
from the per-brain scripts' default output locations.

Usage (workstation, full 38-brain run):
    uv run python scripts/benchmark_mouselight_refresh.py \\
        --brains-dir data/mouselight \\
        --output ../mudm-paper/paper/benchmark_results/B8_mouselight_refresh.json

Local smoke (single brain):
    uv run python scripts/benchmark_mouselight_refresh.py \\
        --brains-dir data/mouselight \\
        --output /tmp/B8_mouselight_refresh.json \\
        --limit 1
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


# Path of the mudm-tools repo (parent of scripts/) — used as cwd for subprocess
# calls and as the anchor for the data/mouselight convention.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "mouselight"


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _du_bytes(path: Path) -> int:
    """Recursively sum file sizes under path; 0 if absent."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def _brotli_size(directory: Path) -> int:
    """Sum Brotli-compressed sizes of every file under the directory.

    Approximates what a static HTTP server with Content-Encoding: br would
    transmit. Brotli quality 11 is the default for static assets.

    Imports brotli lazily so that `--help` and the parquet-only branch work
    without the brotli package installed.
    """
    try:
        import brotli  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "brotli is required for B8 transport-size measurement. "
            "Install via: uv add brotli"
        ) from exc

    if not directory.exists():
        return 0
    total = 0
    for f in directory.rglob("*"):
        if f.is_file():
            data = f.read_bytes()
            total += len(brotli.compress(data, quality=11))
    return total


def _run_parquet(brain_name: str, repo: Path, extra_args: list[str]) -> float:
    """Run the per-brain Parquet pipeline for one brain.

    Per-brain CLI: `mouselight_parquet.py --brain <date>` (optionally
    --no-ontology, --prime). Output goes to
    `<repo>/data/mouselight/tiles/<brain>/parquet_partitioned/`.
    """
    t0 = time.perf_counter()
    subprocess.check_call(
        ["uv", "run", "python", "scripts/mouselight_parquet.py",
         "--brain", brain_name, *extra_args],
        cwd=str(repo),
    )
    return time.perf_counter() - t0


def _run_3dtiles(brain_name: str, repo: Path, extra_args: list[str]) -> float:
    """Run the per-brain 3D-tiles pipeline for one brain.

    Per-brain CLI: `mouselight_3dtiles_draco.py --brain <date>` (optionally
    --no-ontology). Output goes to
    `<repo>/data/mouselight/tiles/<brain>/3dtiles/`. Encoding is Draco
    (use_draco=True hard-coded in the per-brain script).
    """
    t0 = time.perf_counter()
    subprocess.check_call(
        ["uv", "run", "python", "scripts/mouselight_3dtiles_draco.py",
         "--brain", brain_name, *extra_args],
        cwd=str(repo),
    )
    return time.perf_counter() - t0


def _run_one_brain(
    brain_dir: Path,
    tiles_dir: Path,
    repo: Path,
    extra_args: list[str],
) -> dict:
    """Run both pipelines for one brain. Returns per-brain timings + sizes."""
    brain = brain_dir.name
    parquet_t = _run_parquet(brain, repo, extra_args)
    glb_t = _run_3dtiles(brain, repo, extra_args)

    parquet_out = tiles_dir / brain / "parquet_partitioned"
    glb_out = tiles_dir / brain / "3dtiles"

    return {
        "brain": brain,
        "parquet_wall_time_s": parquet_t,
        "glb_wall_time_s": glb_t,
        "parquet_size_bytes": _du_bytes(parquet_out),
        "glb_size_bytes": _du_bytes(glb_out),
        "glb_brotli_size_bytes": _brotli_size(glb_out),
        "parquet_out": str(parquet_out.relative_to(repo)) if parquet_out.exists() else None,
        "glb_out": str(glb_out.relative_to(repo)) if glb_out.exists() else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--brains-dir", default=str(_DEFAULT_DATA_DIR),
        help=(
            "Directory containing per-brain subdirectories. Must resolve to "
            "<mudm-tools>/data/mouselight because the per-brain scripts "
            "hard-code that path. Default: %(default)s"
        ),
    )
    ap.add_argument(
        "--output", required=True,
        help="Output JSON path (must end with 'B8_mouselight_refresh.json')",
    )
    ap.add_argument(
        "--limit", type=int, default=0,
        help="Process only the first N brains (0 = all). Useful for smoke tests.",
    )
    ap.add_argument(
        "--no-ontology", action="store_true",
        help="Pass --no-ontology to both per-brain scripts (skips Allen CCF fetch).",
    )
    ap.add_argument(
        "--skip-brotli", action="store_true",
        help=(
            "Skip Brotli transport-size measurement. Useful when the brotli "
            "package is not installed."
        ),
    )
    ap.add_argument(
        "--mudm-tools-repo", default=None,
        help="Path to mudm-tools repo (default: inferred from this script).",
    )
    args = ap.parse_args()

    repo = (Path(args.mudm_tools_repo).resolve() if args.mudm_tools_repo
            else _REPO_ROOT)
    brains_dir = Path(args.brains_dir).resolve()

    # Per-brain scripts hard-code their input root to <repo>/data/mouselight.
    # Refuse to run if the user pointed elsewhere, to avoid a silent mismatch
    # between this wrapper's measurement paths and what the scripts actually
    # produce.
    expected = (repo / "data" / "mouselight").resolve()
    if brains_dir != expected:
        raise SystemExit(
            f"--brains-dir must resolve to {expected} (the per-brain scripts "
            f"hard-code this path); got {brains_dir}"
        )
    if not brains_dir.is_dir():
        raise SystemExit(f"brains-dir does not exist: {brains_dir}")

    # A brain dir is a subdirectory containing at least one .obj file.
    brain_dirs = sorted(
        p for p in brains_dir.iterdir()
        if p.is_dir() and any(p.glob("*.obj"))
    )
    if args.limit:
        brain_dirs = brain_dirs[: args.limit]
    if not brain_dirs:
        raise SystemExit(
            f"No brain subdirectories with .obj files found under {brains_dir}"
        )

    tiles_dir = brains_dir / "tiles"

    extra_args: list[str] = []
    if args.no_ontology:
        extra_args.append("--no-ontology")

    # Patch _brotli_size into a no-op if the user passed --skip-brotli.
    if args.skip_brotli:
        global _brotli_size  # noqa: PLW0603 - explicit override is the point
        _brotli_size = lambda _d: 0  # type: ignore[assignment]  # noqa: E731

    per_brain: list[dict] = []
    overall_t0 = time.perf_counter()
    for i, bd in enumerate(brain_dirs, start=1):
        print(f"\n=== {bd.name} ({i}/{len(brain_dirs)}) ===", flush=True)
        per_brain.append(_run_one_brain(bd, tiles_dir, repo, extra_args))
    total_wall_s = time.perf_counter() - overall_t0

    out_path = Path(args.output)
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark_id": out_path.stem,
        "git_sha_mudm_tools": _git_sha(repo),
        "hardware": {
            "cpu_cores": os.cpu_count(),
            "platform": platform.platform(),
        },
        "config": {
            "brains_dir": str(brains_dir),
            "tiles_dir": str(tiles_dir),
            "n_brains": len(brain_dirs),
            "limit": args.limit,
            "no_ontology": args.no_ontology,
            "skip_brotli": args.skip_brotli,
            "compression": "draco",
            "per_brain_scripts": [
                "scripts/mouselight_parquet.py",
                "scripts/mouselight_3dtiles_draco.py",
            ],
        },
        "total_wall_time_s": total_wall_s,
        "per_brain": per_brain,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
