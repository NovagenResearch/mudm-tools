#!/usr/bin/env python3
"""Download BANC (Brain And Nerve Cord — complete adult FEMALE Drosophila CNS,
brain + VNC) neuron meshes and tile them as meshopt 3D Tiles.

The FEMALE whole-CNS counterpart to MaleCNS (male). Fully ANONYMOUS — no CAVE
token: the BANC team publishes a public GCS bucket.
  - meshes:      precomputed://gs://lee-lab_brain-and-nerve-cord-fly-connectome/neuron_meshes
                 (LEGACY single-resolution precomputed mesh — NO LOD ladder, full-res;
                 so we DECIMATE in-worker to a web budget, unlike the multilod datasets);
  - annotations: gs://.../compiled_data/banc_888/banc_888_meta.feather (token-free, root_id-keyed).

Default curation `dn-an` = the largest descending + ascending neurons (the brain<->VNC
bridge), the female mirror of the MaleCNS dataset. License: CC BY-NC 4.0 (Bates,
Phelps, Kim, Yang et al. 2026; bioRxiv 2025.07.31.667571).

Usage (on oden):
    .venv/bin/python scripts/download_banc.py --download --tile \
        --select dn-an --max-neurons 1000 --decimate-faces 300000 --max-zoom 4

    # smoke test a single root_id (anonymous, no token; reports verts + OBJ size):
    .venv/bin/python scripts/download_banc.py --smoke 720575941472733451 --decimate-faces 0
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

_DATA_DIR = _ROOT / "data" / "banc"
_MESH_DIR = _DATA_DIR / "meshes"
_META_PATH = _DATA_DIR / "banc_888_meta.feather"
_TILES_DIR = _DATA_DIR / "tiles" / "banc"

_BANC_SEG = "precomputed://gs://lee-lab_brain-and-nerve-cord-fly-connectome/neuron_meshes"
_META_URL = ("https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/"
             "compiled_data/banc_888/banc_888_meta.feather")

MAX_ZOOM = 4
DECIMATE_FACES = 300_000  # legacy meshes are full-res; decimate to this many faces for the web

_SELECT = {
    "dn-an": ["descending", "ascending"],
    "dn": ["descending"],
    "ascending": ["ascending"],
    # sensorimotor periphery: the body's input (sensory, esp. sensory_ascending
    # that reach the brain) + output (motor to muscle). Complements dn-an.
    "sm": ["sensory", "sensory_ascending", "sensory_descending", "motor"],
    # the visual system / optic lobe (columnar retinotopy + projection neurons).
    "optic": ["optic_lobe_intrinsic", "visual_projection", "visual_centrifugal"],
}


def _hash_color(body_id) -> str:
    # Spread hue + saturation + lightness (from separate md5 bytes) so the palette
    # is ~3D, not 360-hue-capped — adjacent neurons get vivid, distinct colors.
    d = hashlib.md5(str(body_id).encode()).digest()
    hue = d[0] / 255.0
    sat = 0.65 + (d[1] / 255.0) * 0.32
    lig = 0.45 + (d[2] / 255.0) * 0.22
    r, g, b = colorsys.hls_to_rgb(hue, lig, sat)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def _fmt_time(s: float) -> str:
    if s < 1:
        return f"{s*1000:.1f}ms"
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(s, 60)
    return f"{int(m)}m{sec:.0f}s"


def _fmt_bytes(n: int) -> str:
    for unit, div in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if n >= div:
            return f"{n/div:.2f} {unit}"
    return f"{n} B"


def _dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


# ---------------------------------------------------------------------------
# Step 1: annotations (token-free feather) + curation
# ---------------------------------------------------------------------------

def load_meta(path: Path) -> list[dict]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading BANC metadata: {_META_URL}")
        urllib.request.urlretrieve(_META_URL, path)
        print(f"  saved {path} ({path.stat().st_size // (1024*1024)} MB)")
    import pyarrow.feather as feather
    rows = feather.read_table(path).to_pylist()
    print(f"  {len(rows)} annotated neurons in {path.name}")
    return rows


def select_neurons(rows: list[dict], select: str, max_neurons: int | None,
                   rank: str = "volume") -> list[dict]:
    keep = set(_SELECT[select])
    sel = [r for r in rows
           if r.get("super_class") in keep and str(r.get("root_id") or "").isdigit()]
    if rank == "random":
        # representative sample (e.g. optic columnar retinotopy) rather than the
        # biggest few neurons. Seeded for reproducibility.
        import random as _random
        _random.Random(0).shuffle(sel)
    else:
        sel.sort(key=lambda r: (r.get("volume_nm3") or 0), reverse=True)  # largest first
    if max_neurons and len(sel) > max_neurons:
        sel = sel[:max_neurons]
    return sel


# ---------------------------------------------------------------------------
# Step 2: Download meshes via CloudVolume (anonymous, legacy single-res) + decimate
# ---------------------------------------------------------------------------

_WORKER_CV = None
_WORKER_DECIMATE = 0


def _dl_init_worker(decimate_faces: int = 0) -> None:
    global _WORKER_CV, _WORKER_DECIMATE
    from cloudvolume import CloudVolume
    _WORKER_CV = CloudVolume(_BANC_SEG, use_https=True, fill_missing=True,
                             cache=False, progress=False)
    _WORKER_DECIMATE = decimate_faces


def _get_mesh(cv, root_id: int):
    """BANC meshes are LEGACY single-resolution precomputed — no LOD ladder."""
    res = cv.mesh.get(root_id)
    mesh = res[root_id] if isinstance(res, dict) else res
    if mesh is None or not len(getattr(mesh, "vertices", [])):
        raise RuntimeError("empty mesh")
    return mesh


def _decimate(verts: np.ndarray, faces: np.ndarray, target_faces: int):
    """Quadric-decimate to ~target_faces (legacy meshes are full-res ~M-vertex)."""
    if not target_faces or len(faces) <= target_faces:
        return verts, faces
    import fast_simplification
    reduction = max(0.0, min(0.99, 1.0 - target_faces / len(faces)))
    v2, f2 = fast_simplification.simplify(
        verts.astype(np.float32), faces.astype(np.int32), target_reduction=reduction)
    return np.asarray(v2, dtype=np.float64), np.asarray(f2, dtype=np.int64)


def _dl_one(task: tuple[int, str, bool]) -> tuple[int, str, int]:
    body_id, out_dir, skip_existing = task
    obj_path = Path(out_dir) / f"{body_id}.obj"
    if skip_existing and obj_path.exists():
        return (body_id, "skip", 0)
    try:
        mesh = _get_mesh(_WORKER_CV, body_id)
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)  # 0-indexed
        verts, faces = _decimate(verts, faces, _WORKER_DECIMATE)
        tmp = obj_path.with_suffix(".obj.tmp")
        with open(tmp, "w") as f:
            np.savetxt(f, verts, fmt="v %.7g %.7g %.7g")
            np.savetxt(f, faces + 1, fmt="f %d %d %d")  # OBJ is 1-indexed
        tmp.rename(obj_path)
        return (body_id, "ok", int(len(verts)))
    except Exception as e:  # noqa: BLE001
        return (body_id, f"err:{type(e).__name__}:{e}", 0)


def download_meshes(body_ids: list[int], output_dir: Path, *, decimate_faces: int = 0,
                    skip_existing: bool = True, workers: int | None = None) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    n_workers = workers or min(16, (os.cpu_count() or 8))
    tasks = [(int(b), str(output_dir), skip_existing) for b in body_ids]
    total = len(tasks)
    done = errors = 0
    print(f"  Downloading {total} meshes (decimate→{decimate_faces or 'none'} faces) "
          f"with {n_workers} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_dl_init_worker,
                             initargs=(decimate_faces,)) as ex:
        for i, (bid, status, _n) in enumerate(ex.map(_dl_one, tasks, chunksize=1), 1):
            if status in ("ok", "skip"):
                done += 1
            else:
                errors += 1
                if errors <= 8:
                    print(f"  {bid} — ERROR: {status}", flush=True)
            if i % 50 == 0 or i == total:
                print(f"  [{i}/{total}] ok={done} err={errors}", flush=True)
    print(f"  Downloaded {done}/{total} meshes ({errors} errors)")
    return done


# ---------------------------------------------------------------------------
# Step 3: per-mesh tags (per-id color + rich BANC annotations)
# ---------------------------------------------------------------------------

# BANC feather column -> muDM tag key (with remaps to the shared viewer keys).
_TAG_MAP = [
    ("cell_type", "cell_type"),
    ("super_class", "super_class"),
    ("cell_class", "cell_class"),
    ("cell_sub_class", "cell_sub_class"),
    ("flow", "flow"),
    ("side", "side"),
    ("region", "region"),                       # NEW: brain-vs-VNC discriminator
    ("hemilineage", "hemilineage"),
    ("nerve", "nerve"),
    ("sexually_dimorphic", "dimorphism"),        # remap -> shared 'dimorphism'
    ("neurotransmitter_predicted", "predicted_nt"),
    ("neurotransmitter_verified", "consensus_nt"),
    ("malecns_cell_type", "malecns_type"),       # NEW: cross-ref to the MALE counterpart
    ("manc_cell_type", "manc_type"),
    ("hemibrain_cell_type", "hemibrain_type"),
    ("fafb_cell_type", "flywire_type"),
    ("status", "status"),
]


def _build_tags(obj_path: Path, meta_lookup: dict[str, dict]) -> dict:
    root_id = obj_path.stem
    tags: dict = {
        "body_id": int(root_id) if root_id.isdigit() else root_id,
        "source": obj_path.name,
    }
    meta = meta_lookup.get(root_id, {})
    for col, key in _TAG_MAP:
        v = meta.get(col)
        if isinstance(v, str):
            v = v.strip()
        if v not in (None, "", "NA"):
            tags[key] = v
    _ct = tags.get("cell_type") or tags.get("hemibrain_type") or tags.get("malecns_type")
    tags["name"] = f"{_ct} ({root_id})" if _ct else str(root_id)
    tags["color"] = _hash_color(root_id)
    return tags


# ---------------------------------------------------------------------------
# Step 4: Tile meshopt 3D Tiles + features.json + tilejson3d + pyramids.json
# ---------------------------------------------------------------------------

def tile_meshopt(mesh_dir: Path, meta_lookup: dict[str, dict], output_dir: Path, *,
                 max_zoom: int = MAX_ZOOM, max_files: int | None = None,
                 ingest_threads: int = 0) -> None:
    import tempfile
    from collections import Counter

    from build_feature_index import build_index, build_tilejson
    from mudm_tools._rs import StreamingTileGenerator, scan_obj_bounds

    obj_paths = sorted(mesh_dir.glob("*.obj"))
    if not obj_paths:
        print(f"ERROR: No OBJ files in {mesh_dir}", file=sys.stderr)
        sys.exit(1)
    if max_files and max_files < len(obj_paths):
        obj_paths = obj_paths[:max_files]

    path_strs = [str(p) for p in obj_paths]
    tags_list = [_build_tags(p, meta_lookup) for p in obj_paths]

    print("super_class breakdown:")
    for sc, cnt in Counter(t.get("super_class", "(none)") for t in tags_list).most_common():
        print(f"  {cnt:5d}  {sc}")

    t0 = time.perf_counter()
    bounds = scan_obj_bounds(path_strs)
    print(f"\nBounds: x=[{bounds[0]:.0f}, {bounds[3]:.0f}] "
          f"y=[{bounds[1]:.0f}, {bounds[4]:.0f}] "
          f"z=[{bounds[2]:.0f}, {bounds[5]:.0f}]  ({_fmt_time(time.perf_counter() - t0)})")

    tiles3d_dir = output_dir / "3dtiles"
    if tiles3d_dir.exists():
        shutil.rmtree(tiles3d_dir)
    tiles3d_dir.mkdir(parents=True, exist_ok=True)

    gen = StreamingTileGenerator(min_zoom=0, max_zoom=max_zoom, base_cells=100,
                                 temp_dir=tempfile.gettempdir())
    t0 = time.perf_counter()
    gen.add_obj_files(path_strs, bounds, tags_list, ingest_threads=ingest_threads)
    print(f"Ingest: {_fmt_time(time.perf_counter() - t0)}")

    print("Encoding 3D Tiles with meshopt...")
    t0 = time.perf_counter()
    n_tiles = gen.generate_3dtiles(str(tiles3d_dir), bounds, compression="meshopt")
    print(f"  {n_tiles} tiles in {_fmt_time(time.perf_counter() - t0)}")
    del gen

    output_size = _dir_size(tiles3d_dir)
    print(f"Output: {tiles3d_dir} ({_fmt_bytes(output_size)})")

    id_fields = ["body_id"]
    index, zoom_counts, max_zoom_found = build_index(tiles3d_dir, id_fields=id_fields)
    n_features = len(index.get("features", []))
    (output_dir / "features.json").write_text(json.dumps(index, indent=2))
    print(f"Features indexed: {n_features}")

    tj = build_tilejson(zoom_counts, max_zoom_found, id_fields=id_fields, bounds3d=list(bounds))
    (output_dir / "tilejson3d.json").write_text(json.dumps(tj, indent=2))

    pyramids_path = output_dir.parent / "pyramids.json"
    pyramid_id = output_dir.name
    entry = {
        "id": pyramid_id,
        "label": "BANC (whole Drosophila FEMALE CNS — brain + VNC)",
        "tilejson": "tilejson3d.json", "features": "features.json",
        "tiles": n_tiles, "feature_count": n_features, "size_bytes": output_size,
    }
    manifest = json.loads(pyramids_path.read_text()) if pyramids_path.exists() else {"version": "1.0", "pyramids": []}
    manifest.setdefault("version", "1.0")
    manifest["pyramids"] = [p for p in manifest["pyramids"] if p.get("id") != pyramid_id] + [entry]
    pyramids_path.write_text(json.dumps(manifest, indent=2))
    print(f"Updated {pyramids_path} (entry: {pyramid_id}, {n_features} features, {n_tiles} tiles)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="BANC (female Drosophila CNS) download + meshopt 3D tiling")
    parser.add_argument("--download", action="store_true", help="Select neurons + download+decimate meshes")
    parser.add_argument("--tile", action="store_true", help="Tile downloaded meshes (meshopt 3D Tiles)")
    parser.add_argument("--smoke", type=int, default=None, metavar="ROOT_ID",
                        help="Download a single root_id mesh (anonymous), then exit")
    parser.add_argument("--select", default="dn-an", choices=list(_SELECT),
                        help="Curation (default: dn-an = descending + ascending = brain<->VNC bridge)")
    parser.add_argument("--rank", default="volume", choices=["volume", "random"],
                        help="Pick the largest-by-volume (default) or a random sample (e.g. for optic columns)")
    parser.add_argument("--max-neurons", type=int, default=1000, help="Cap on selected neurons (default: 1000)")
    parser.add_argument("--decimate-faces", type=int, default=DECIMATE_FACES,
                        help=f"Quadric-decimate each mesh to ~N faces (0=off; default {DECIMATE_FACES})")
    parser.add_argument("--max-zoom", type=int, default=MAX_ZOOM, help=f"Max zoom (default: {MAX_ZOOM})")
    parser.add_argument("--workers", type=int, default=None, help="Download worker processes")
    parser.add_argument("--ingest-threads", type=int, default=0, help="Tile ingest threads (0=all)")
    parser.add_argument("--data-dir", type=Path, default=_DATA_DIR, help="Data directory")
    parser.add_argument("--output", type=Path, default=_TILES_DIR, help="Output pyramid directory")
    args = parser.parse_args()

    data_dir = args.data_dir
    mesh_dir = data_dir / "meshes"
    meta_path = data_dir / "banc_888_meta.feather"

    if args.smoke is not None:
        mesh_dir.mkdir(parents=True, exist_ok=True)
        print(f"Smoke test: fetching root {args.smoke} (decimate→{args.decimate_faces or 'none'}) "
              f"from {_BANC_SEG} ...")
        _dl_init_worker(args.decimate_faces)
        rid, status, nverts = _dl_one((args.smoke, str(mesh_dir), False))
        sz = (mesh_dir / f"{args.smoke}.obj").stat().st_size if status == "ok" else 0
        print(f"  -> {status}  ({nverts} vertices, OBJ {_fmt_bytes(sz)})")
        sys.exit(0 if status == "ok" else 1)

    if not (args.download or args.tile):
        parser.print_help()
        sys.exit(1)

    rows = load_meta(meta_path)
    sel = select_neurons(rows, args.select, args.max_neurons, rank=args.rank)
    meta_lookup = {str(r["root_id"]): r for r in sel}
    print(f"Selected {len(sel)} neurons (--select {args.select}); "
          f"{len({r.get('cell_type') for r in sel})} cell types")

    if args.download:
        body_ids = [int(r["root_id"]) for r in sel]
        print(f"\nDownloading {len(body_ids)} BANC meshes (decimate→{args.decimate_faces or 'none'})...")
        t0 = time.perf_counter()
        download_meshes(body_ids, mesh_dir, decimate_faces=args.decimate_faces, workers=args.workers)
        print(f"  Download time: {_fmt_time(time.perf_counter() - t0)}")

    if args.tile:
        tile_meshopt(mesh_dir, meta_lookup, args.output,
                     max_zoom=args.max_zoom, max_files=args.max_neurons,
                     ingest_threads=args.ingest_threads)

    print("Done.")


if __name__ == "__main__":
    main()
