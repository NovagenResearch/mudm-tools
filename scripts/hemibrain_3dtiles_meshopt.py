#!/usr/bin/env python3
"""Generate meshopt-compressed 3D Tiles for random Hemibrain neurons.

Usage:
    uv run python scripts/hemibrain_3dtiles_meshopt.py --count 10
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

_DATA_DIR = _ROOT / "data" / "hemibrain"
_MESH_DIR = _DATA_DIR / "meshes"
_META_PATH = _DATA_DIR / "metadata.json"
_TILES_DIR = _DATA_DIR / "tiles" / "hemibrain_sample"

MAX_ZOOM = 4


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
        if meta.get("pre") is not None:
            tags["pre"] = str(meta["pre"])
        if meta.get("post") is not None:
            tags["post"] = str(meta["post"])
        if meta.get("status"):
            tags["status"] = meta["status"]
        if meta.get("statusLabel"):
            tags["status_label"] = meta["statusLabel"]
        if meta.get("size") is not None:
            tags["size_voxels"] = str(meta["size"])
        if meta.get("somaRadius") is not None:
            tags["soma_radius"] = str(meta["somaRadius"])
        if meta.get("cropped") is not None:
            tags["cropped"] = meta["cropped"]
        # Extract primary brain regions from roiInfo
        if meta.get("roiInfo"):
            import json as _json
            try:
                roi = _json.loads(meta["roiInfo"]) if isinstance(meta["roiInfo"], str) else meta["roiInfo"]
                # Sort regions by total synapses (pre+post), take top 3
                ranked = sorted(roi.items(), key=lambda kv: kv[1].get("pre", 0) + kv[1].get("post", 0), reverse=True)
                regions = [r[0] for r in ranked[:3]]
                if regions:
                    tags["brain_regions"] = ", ".join(regions)
            except Exception:
                pass
    instance = tags.get("instance")
    tags["name"] = f"{instance} ({body_id})" if instance else str(body_id)
    h = hash(str(body_id)) & 0x7FFFFFFF
    hue = (h % 360) / 360.0
    r, g, b = colorsys.hls_to_rgb(hue, 0.5, 0.75)
    tags["color"] = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    return tags


def main():
    parser = argparse.ArgumentParser(
        description="Generate meshopt 3D Tiles for random Hemibrain neurons",
    )
    parser.add_argument("--count", type=int, default=10, help="Number of random neurons (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--max-zoom", type=int, default=MAX_ZOOM, help=f"Max zoom (default: {MAX_ZOOM})")
    parser.add_argument("--output", type=Path, default=_TILES_DIR, help=f"Output directory (default: {_TILES_DIR})")
    parser.add_argument("--ingest-threads", type=int, default=0, help="Ingest thread limit (0=all cores)")
    args = parser.parse_args()

    # Pick random OBJ files
    all_objs = sorted(_MESH_DIR.glob("*.obj"))
    if not all_objs:
        print(f"ERROR: No OBJ files in {_MESH_DIR}", file=sys.stderr)
        sys.exit(1)

    random.seed(args.seed)
    selected = random.sample(all_objs, min(args.count, len(all_objs)))
    print(f"Selected {len(selected)} random neurons (seed={args.seed}):")
    for p in selected:
        print(f"  {p.name} ({_fmt_bytes(p.stat().st_size)})")

    # Load metadata
    meta_lookup: dict[str, dict] = {}
    if _META_PATH.exists():
        raw = json.loads(_META_PATH.read_text())
        for neuron in raw.get("neurons", []):
            meta_lookup[str(neuron["bodyId"])] = neuron

    from mudm_tools._rs import StreamingTileGenerator, scan_obj_bounds

    path_strs = [str(p) for p in selected]
    tags_list = [_build_tags(p, meta_lookup) for p in selected]

    # Scan bounds
    t0 = time.perf_counter()
    bounds = scan_obj_bounds(path_strs)
    print(f"\nBounds: x=[{bounds[0]:.0f}, {bounds[3]:.0f}] "
          f"y=[{bounds[1]:.0f}, {bounds[4]:.0f}] "
          f"z=[{bounds[2]:.0f}, {bounds[5]:.0f}]  ({_fmt_time(time.perf_counter() - t0)})")

    # Clean output
    tiles3d_dir = args.output / "3dtiles"
    if tiles3d_dir.exists():
        shutil.rmtree(tiles3d_dir)
    tiles3d_dir.mkdir(parents=True, exist_ok=True)

    # Ingest — use TMPDIR for fragment storage (large datasets need fast disk)
    import tempfile
    tmp_dir = tempfile.gettempdir()
    gen = StreamingTileGenerator(min_zoom=0, max_zoom=args.max_zoom, base_cells=100, temp_dir=tmp_dir)
    t0 = time.perf_counter()
    gen.add_obj_files(path_strs, bounds, tags_list, ingest_threads=args.ingest_threads)
    t_ingest = time.perf_counter() - t0
    print(f"Ingest: {_fmt_time(t_ingest)}")

    # Generate meshopt-q14 3D Tiles
    print("Encoding 3D Tiles with meshopt-q14...")
    t0 = time.perf_counter()
    n_tiles = gen.generate_3dtiles(str(tiles3d_dir), bounds, compression="meshopt-q14")
    t_gen = time.perf_counter() - t0
    del gen

    output_size = _dir_size(tiles3d_dir)
    print(f"\nDone: {n_tiles} tiles in {_fmt_time(t_gen)} ({_fmt_bytes(output_size)})")
    print(f"Output: {tiles3d_dir}")

    # Build features.json and tilejson3d.json
    from build_feature_index import build_index, build_tilejson

    id_fields = ["body_id", "instance"]
    index, zoom_counts, max_zoom_found = build_index(tiles3d_dir, id_fields=id_fields)
    n_features = len(index.get("features", []))

    # Write features.json to pyramid root (sibling of 3dtiles/)
    features_path = args.output / "features.json"
    features_path.write_text(json.dumps(index, indent=2))
    print(f"Features indexed: {n_features}")

    # Write tilejson3d.json to pyramid root
    tj = build_tilejson(zoom_counts, max_zoom_found, id_fields=id_fields,
                        bounds3d=list(bounds))
    tilejson_path = args.output / "tilejson3d.json"
    tilejson_path.write_text(json.dumps(tj, indent=2))
    print(f"Wrote {tilejson_path}")

    # Update pyramids.json manifest (PyramidJSON format)
    pyramids_path = args.output.parent / "pyramids.json"
    pyramid_id = args.output.name
    entry = {
        "id": pyramid_id,
        "label": f"Hemibrain v1.2.1 ({len(selected)} neurons, streaming meshopt)",
        "tilejson": "tilejson3d.json",
        "features": "features.json",
        "tiles": n_tiles,
        "feature_count": n_features,
        "size_bytes": output_size,
    }
    if pyramids_path.exists():
        manifest = json.loads(pyramids_path.read_text())
    else:
        manifest = {"version": "1.0", "pyramids": []}
    if "version" not in manifest:
        manifest["version"] = "1.0"
    # Replace existing entry with same id, or append
    manifest["pyramids"] = [
        p for p in manifest["pyramids"] if p["id"] != pyramid_id
    ] + [entry]
    pyramids_path.write_text(json.dumps(manifest, indent=2))
    print(f"Updated {pyramids_path}")


if __name__ == "__main__":
    main()
