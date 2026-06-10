#!/usr/bin/env python3
"""Download FlyWire (adult Drosophila brain connectome, FAFB) neuron meshes and
tile them as meshopt-compressed 3D Tiles for the muDM 3D viewer.

Fully ANONYMOUS — no CAVE token. Mirrors download_manc.py, but:
  - meshes come from the public flat precomputed bucket gs://flywire_v141_m783
    (the 783 LTS release; cloud-volume use_https=True, no secret);
  - per-neuron annotations come from the token-free Schlegel et al. 2024 TSV in
    github.com/flyconnectome/flywire_annotations (keyed by the SAME 783 root_id).

Default curation = the ELLIPSOID-BODY / PROTOCEREBRAL-BRIDGE head-direction
"compass" (~506 neurons): ER ring neurons + EPG/PEN/PEG/EL/ExR columnar + Delta7.

License of the source data: CC BY-NC 4.0 (Dorkenwald et al. 2024; Schlegel et al.
2024; Zheng et al. 2018; flywire.ai).

Prerequisites: cloud-volume + DracoPy + requests in the venv (all already present
on oden / under mudm-data's [hemibrain]+[draco] extras). NO token.

Usage (on oden):
    .venv/bin/python scripts/download_flywire.py --download --tile \
        --select cx-compass --lod 1 --max-zoom 4

    # smoke test a single root id (confirm anonymous cv.mesh.get works):
    .venv/bin/python scripts/download_flywire.py --smoke 720575940614297640
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import hashlib
import json
import os
import re
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

_DATA_DIR = _ROOT / "data" / "flywire"
_MESH_DIR = _DATA_DIR / "meshes"
_ANNOT_PATH = _DATA_DIR / "Supplemental_file1_neuron_annotations.tsv"
_TILES_DIR = _DATA_DIR / "tiles" / "flywire"

# Public flat precomputed bucket for the 783 LTS release (neuroglancer_multilod_draco,
# LODs 0-3). Anonymous read via use_https=True — no CAVE token.
_FLYWIRE_SEG = "precomputed://gs://flywire_v141_m783"
_ANNOT_URL = ("https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
              "main/supplemental_files/Supplemental_file1_neuron_annotations.tsv")

MAX_ZOOM = 4

# Ellipsoid-body / protocerebral-bridge head-direction "compass" cell_types
# (CX cell_class). Ring neurons are matched via cell_sub_class == "ring neuron".
_COMPASS_RE = re.compile(r"^(EPG|EPGt|EL|PEN|PEG|ExR|Delta7|IbSpsP)")


def _hash_color(body_id) -> str:
    """Deterministic per-neuron color from the root id (the viewer's "Original"
    coloring — distinct per neuron). Categorical fields (super_class / cell_class /
    cell_type / flow / predicted_nt) drive the auto-built "Color By" dropdown, so
    the baked color is intentionally per-id, not per-class. md5 (not salted hash())
    for reproducibility."""
    h = int(hashlib.md5(str(body_id).encode()).hexdigest()[:8], 16)
    hue = (h % 360) / 360.0
    r, g, b = colorsys.hls_to_rgb(hue, 0.55, 0.6)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


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


# ---------------------------------------------------------------------------
# Step 1: Annotations (token-free flat TSV) + curation
# ---------------------------------------------------------------------------

def load_annotations(path: Path) -> list[dict]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading annotations: {_ANNOT_URL}")
        urllib.request.urlretrieve(_ANNOT_URL, path)
        print(f"  saved {path} ({path.stat().st_size // 1024} KB)")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    print(f"  {len(rows)} annotated neurons in {path.name}")
    return rows


def select_neurons(rows: list[dict], mode: str, max_neurons: int | None) -> list[dict]:
    """Curate the subset to tile. Returns rows with a numeric root_id."""
    rows = [r for r in rows if (r.get("root_id") or "").isdigit()]
    if mode == "cx-compass":
        sel = [r for r in rows if r.get("cell_class") == "CX" and (
            r.get("cell_sub_class") == "ring neuron"
            or _COMPASS_RE.match(r.get("cell_type", "") or ""))]
    elif mode == "cx-all":
        sel = [r for r in rows if r.get("cell_class") == "CX"]
    elif mode == "per-superclass":
        # stratified sample: up to (max_neurons / n_superclasses) per super_class
        from collections import defaultdict
        by_sc: dict[str, list] = defaultdict(list)
        for r in rows:
            if r.get("super_class"):
                by_sc[r["super_class"]].append(r)
        per = max(1, (max_neurons or 450) // max(1, len(by_sc)))
        sel = []
        for sc in sorted(by_sc):
            sel.extend(sorted(by_sc[sc], key=lambda r: r["root_id"])[:per])
    elif mode == "all":
        # the ENTIRE proofread connectome (every neuron with a numeric root_id)
        sel = list(rows)
    else:
        raise SystemExit(f"unknown --select mode: {mode}")

    # group by cell_type so the viewer list + symmetry stay coherent
    sel.sort(key=lambda r: (r.get("cell_type", ""), r.get("root_id", "")))
    if max_neurons and len(sel) > max_neurons:
        # cap by WHOLE cell_types (don't truncate mid-type -> preserve symmetry)
        kept, seen, types_order = [], set(), []
        for r in sel:
            ct = r.get("cell_type", "")
            if ct not in seen and len(kept) >= max_neurons:
                break
            seen.add(ct); types_order.append(ct)
        capped = []
        for r in sel:
            if r.get("cell_type", "") in seen:
                capped.append(r)
        sel = capped
    return sel


# ---------------------------------------------------------------------------
# Step 2: Download meshes via CloudVolume (anonymous, flat bucket)
# ---------------------------------------------------------------------------

_WORKER_CV = None


def _dl_init_worker() -> None:
    global _WORKER_CV
    from cloudvolume import CloudVolume
    _WORKER_CV = CloudVolume(_FLYWIRE_SEG, use_https=True, fill_missing=True,
                             cache=False, progress=False)


def _get_mesh(cv, root_id: int, lod: int):
    """Fetch one neuron mesh at the requested multires LOD (0=finest..3), falling
    back to finer levels if the requested LOD is unavailable for this neuron."""
    last = None
    for l in range(lod, -1, -1):
        try:
            res = cv.mesh.get(root_id, lod=l)
            mesh = res[root_id] if isinstance(res, dict) else res
            if mesh is not None and len(getattr(mesh, "vertices", [])):
                return mesh
        except Exception as e:  # noqa: BLE001
            last = e
    if last:
        raise last
    raise RuntimeError("empty mesh")


def _dl_one(task: tuple[int, str, bool, int]) -> tuple[int, str, int]:
    root_id, out_dir, skip_existing, lod = task
    obj_path = Path(out_dir) / f"{root_id}.obj"
    if skip_existing and obj_path.exists():
        return (root_id, "skip", 0)
    try:
        mesh = _get_mesh(_WORKER_CV, root_id, lod)
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3) + 1
        tmp = obj_path.with_suffix(".obj.tmp")
        with open(tmp, "w") as f:
            np.savetxt(f, verts, fmt="v %.7g %.7g %.7g")
            np.savetxt(f, faces, fmt="f %d %d %d")
        tmp.rename(obj_path)
        return (root_id, "ok", int(len(verts)))
    except Exception as e:  # noqa: BLE001
        return (root_id, f"err:{type(e).__name__}:{e}", 0)


def download_meshes(root_ids: list[int], output_dir: Path, *, lod: int = 1,
                    skip_existing: bool = True, workers: int | None = None) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    n_workers = workers or min(16, (os.cpu_count() or 8))
    tasks = [(int(b), str(output_dir), skip_existing, lod) for b in root_ids]
    total = len(tasks)
    done = errors = 0
    print(f"  Downloading {total} meshes (lod={lod}) with {n_workers} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_dl_init_worker) as ex:
        for i, (rid, status, _n) in enumerate(ex.map(_dl_one, tasks, chunksize=1), 1):
            if status in ("ok", "skip"):
                done += 1
            else:
                errors += 1
                if errors <= 8:
                    print(f"  {rid} — ERROR: {status}", flush=True)
            if i % 25 == 0 or i == total:
                print(f"  [{i}/{total}] ok={done} err={errors}", flush=True)
    print(f"  Downloaded {done}/{total} meshes ({errors} errors)")
    return done


# ---------------------------------------------------------------------------
# Step 3: per-mesh tags (per-id hash color + rich FlyWire annotations)
# ---------------------------------------------------------------------------

# FlyWire annotation column -> muDM tag key. Only emitted when the source value
# is non-empty (tolerant of absent columns / annotation-version drift).
_TAG_MAP = [
    ("cell_type", "cell_type"),
    ("super_class", "super_class"),
    ("cell_class", "cell_class"),
    ("cell_sub_class", "cell_sub_class"),
    ("hemibrain_type", "hemibrain_type"),
    ("flow", "flow"),
    ("side", "side"),
    ("nerve", "nerve"),
    ("top_nt", "predicted_nt"),
    ("top_nt_conf", "nt_conf"),
    ("ito_lee_hemilineage", "hemilineage"),
    ("supertype", "supertype"),
    ("status", "status"),
    ("fbbt_id", "fbbt_id"),
    ("vfb_id", "vfb_id"),
]


def _build_tags(obj_path: Path, meta_lookup: dict[str, dict]) -> dict:
    root_id = obj_path.stem
    tags: dict = {
        "body_id": int(root_id) if root_id.isdigit() else root_id,
        "source": obj_path.name,
    }
    meta = meta_lookup.get(root_id, {})
    for col, key in _TAG_MAP:
        val = (meta.get(col) or "").strip() if isinstance(meta.get(col), str) else meta.get(col)
        if val:
            tags[key] = val
    # name MUST be unique per neuron (build_feature_index groups by name) — mirror
    # MANC's "{instance} ({body_id})". cell_type alone is shared by many neurons.
    _ct = tags.get("cell_type") or tags.get("hemibrain_type")
    tags["name"] = f"{_ct} ({root_id})" if _ct else str(root_id)
    tags["color"] = _hash_color(root_id)
    return tags


# ---------------------------------------------------------------------------
# Step 4: Tile meshopt 3D Tiles + features.json + tilejson3d + pyramids.json
# ---------------------------------------------------------------------------

def tile_meshopt(mesh_dir: Path, meta_lookup: dict[str, dict], output_dir: Path, *,
                 max_zoom: int = MAX_ZOOM, max_files: int | None = None,
                 ingest_threads: int = 0,
                 emit_neuroglancer: bool = False, emit_parquet: bool = False,
                 label: str | None = None) -> None:
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

    print("cell_type breakdown:")
    for ct, cnt in Counter(t.get("cell_type", "(none)") for t in tags_list).most_common():
        print(f"  {cnt:4d}  {ct}")

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

    # Parquet first — it's streaming/bounded-memory, so it's safe even at huge scale.
    if emit_parquet:
        from mudm_tools.tiling3d import generate_parquet
        pq_path = output_dir / "features.parquet"
        if pq_path.is_dir():
            shutil.rmtree(pq_path)
        elif pq_path.exists():
            pq_path.unlink()
        print("Emitting Hive-partitioned Parquet...")
        t0 = time.perf_counter()
        n_rows = generate_parquet(gen, pq_path, bounds, partitioned=True)
        pq_size = _dir_size(pq_path) if pq_path.is_dir() else pq_path.stat().st_size
        print(f"  Parquet: {n_rows:,} rows in {_fmt_time(time.perf_counter() - t0)} "
              f"({_fmt_bytes(pq_size)})")

    # Neuroglancer last — the multilod resolver can be memory-heavy at large
    # feature counts, so run it after the durable outputs are already on disk.
    if emit_neuroglancer:
        ng_dir = output_dir / "neuroglancer"
        if ng_dir.exists():
            shutil.rmtree(ng_dir)
        print("Emitting Neuroglancer precomputed multilod...")
        t0 = time.perf_counter()
        gen.generate_neuroglancer_multilod(str(ng_dir), bounds)
        print(f"  Neuroglancer in {_fmt_time(time.perf_counter() - t0)} "
              f"({_fmt_bytes(_dir_size(ng_dir))})")

    del gen

    output_size = _dir_size(tiles3d_dir)
    print(f"Output: {tiles3d_dir} ({_fmt_bytes(output_size)})")

    # only body_id is a true identifier; keep cell_type/super_class/etc. available
    # as "Color By" options (they are the point of the dataset).
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
        "label": label or "FlyWire (Drosophila brain connectome — central complex)",
        "tilejson": "tilejson3d.json",
        "features": "features.json",
        "tiles": n_tiles,
        "feature_count": n_features,
        "size_bytes": output_size,
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
    parser = argparse.ArgumentParser(description="FlyWire (Drosophila brain) download + meshopt 3D tiling")
    parser.add_argument("--download", action="store_true", help="Select neurons + download meshes")
    parser.add_argument("--tile", action="store_true", help="Tile downloaded meshes (meshopt 3D Tiles)")
    parser.add_argument("--smoke", type=int, default=None, metavar="ROOT_ID",
                        help="Download a single root id to test anonymous CloudVolume mesh.get, then exit")
    parser.add_argument("--select", default="cx-compass",
                        choices=["cx-compass", "cx-all", "per-superclass", "all"],
                        help="Curation (default: cx-compass). 'all' = every proofread neuron (~139k).")
    parser.add_argument("--max-neurons", type=int, default=700,
                        help="Cap on selected neurons (default: 700; ignored for --select all)")
    parser.add_argument("--neuroglancer", action="store_true",
                        help="Also emit Neuroglancer precomputed multilod (neuroglancer/)")
    parser.add_argument("--parquet", action="store_true",
                        help="Also emit Hive-partitioned features.parquet")
    parser.add_argument("--lod", type=int, default=1,
                        help="Multires mesh LOD: 0=finest .. 3=coarsest (default: 1)")
    parser.add_argument("--max-zoom", type=int, default=MAX_ZOOM, help=f"Max zoom (default: {MAX_ZOOM})")
    parser.add_argument("--workers", type=int, default=None, help="Download worker processes")
    parser.add_argument("--ingest-threads", type=int, default=0, help="Tile ingest threads (0=all)")
    parser.add_argument("--data-dir", type=Path, default=_DATA_DIR, help="Data directory")
    parser.add_argument("--output", type=Path, default=_TILES_DIR, help="Output pyramid directory")
    args = parser.parse_args()

    data_dir = args.data_dir
    mesh_dir = data_dir / "meshes"
    annot_path = data_dir / "Supplemental_file1_neuron_annotations.tsv"

    if args.smoke is not None:
        mesh_dir.mkdir(parents=True, exist_ok=True)
        print(f"Smoke test: fetching root {args.smoke} (lod={args.lod}) from {_FLYWIRE_SEG} ...")
        _dl_init_worker()
        rid, status, nverts = _dl_one((args.smoke, str(mesh_dir), False, args.lod))
        print(f"  -> {status}  ({nverts} vertices)")
        sys.exit(0 if status == "ok" else 1)

    if not (args.download or args.tile):
        parser.print_help()
        sys.exit(1)

    # 'all' = the entire connectome: never cap (selection or tiling).
    cap = None if args.select == "all" else args.max_neurons
    rows = load_annotations(annot_path)
    sel = select_neurons(rows, args.select, cap)
    meta_lookup = {r["root_id"]: r for r in sel}
    print(f"Selected {len(sel)} neurons (--select {args.select}); "
          f"{len({r.get('cell_type') for r in sel})} cell types")

    if args.download:
        root_ids = [int(r["root_id"]) for r in sel]
        print(f"\nDownloading {len(root_ids)} FlyWire meshes (lod={args.lod})...")
        t0 = time.perf_counter()
        download_meshes(root_ids, mesh_dir, lod=args.lod, workers=args.workers)
        print(f"  Download time: {_fmt_time(time.perf_counter() - t0)}")

    if args.tile:
        label = ("FlyWire FAFB (Drosophila whole-brain connectome)"
                 if args.select == "all" else None)
        tile_meshopt(mesh_dir, meta_lookup, args.output,
                     max_zoom=args.max_zoom, max_files=cap,
                     ingest_threads=args.ingest_threads,
                     emit_neuroglancer=args.neuroglancer, emit_parquet=args.parquet,
                     label=label)

    print("Done.")


if __name__ == "__main__":
    main()
