#!/usr/bin/env python3
"""End-to-end cnn-nmo builder (CLI wrapper around cnn_nmo.run_build)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mudm_tools.cnn_nmo import run_build, run_aux_only


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=ROOT / "data" / "cnn-nmo")
    parser.add_argument("--tiles-dir", type=Path,
                        default=ROOT / "data" / "cnn-nmo" / "tiles")
    parser.add_argument("--cortex-obj", type=Path,
                        default=ROOT / "data" / "ccf" / "isocortex.obj")
    parser.add_argument("--scene", type=str, default="hust-ccf")
    parser.add_argument(
        "--aux-only", action="store_true",
        help="Re-emit only tilejson3d.json + features.json + pyramids.json "
             "against an existing tileset. Skips SWC parsing, FC building, "
             "Rust tiling, and skeleton writing (minutes instead of hours).",
    )
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

    if args.aux_only:
        run_aux_only(
            data_dir=args.data_dir,
            tiles_dir=args.tiles_dir,
            cortex_obj=args.cortex_obj if args.cortex_obj.exists() else None,
            scene_id=args.scene,
        )
        print("Aux files regenerated.")
        return 0

    run_build(
        data_dir=args.data_dir,
        tiles_dir=args.tiles_dir,
        cortex_obj=args.cortex_obj if args.cortex_obj.exists() else None,
        scene_id=args.scene,
    )
    print("Build complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
