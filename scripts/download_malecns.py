#!/usr/bin/env python3
"""Download MaleCNS (Janelia FlyEM adult MALE Drosophila central nervous system =
brain + optic lobes + VNC) neuron meshes and tile them as meshopt 3D Tiles.

Hybrid of download_manc.py (neuPrint custom-Cypher metadata + cloud-volume meshes)
and download_flywire.py (--select curation, unique per-neuron name, per-id color):
  - metadata: neuPrint REST `male-cns:v0.9` (NEUPRINT_TOKEN required — on oden .env);
  - meshes:   ANONYMOUS public bucket precomputed://gs://flyem-male-cns/v0.9/segmentation
              (multilod_draco, cv.mesh.get(lod=), no token).

Default curation `dn-an` = descending + ascending neurons — the brain<->VNC bridge
that physically spans the whole CNS (the MaleCNS-unique story vs brain-only
hemibrain/FlyWire and VNC-only MANC). License of source data: CC BY 4.0
(Berg et al. 2025, bioRxiv 10.1101/2025.10.09.680999).

Usage (on oden):
    set -a; . /data/ai/mudm-paper/.env; set +a
    .venv/bin/python scripts/download_malecns.py --download --tile \
        --select dn-an --max-neurons 1000 --lod 1 --max-zoom 4

    # smoke test a single bodyId (mesh only — no token needed):
    .venv/bin/python scripts/download_malecns.py --smoke <bodyId>
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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

_DATA_DIR = _ROOT / "data" / "malecns"
_MESH_DIR = _DATA_DIR / "meshes"
_META_PATH = _DATA_DIR / "metadata.json"
_TILES_DIR = _DATA_DIR / "tiles" / "malecns"

_MALECNS_SEG = "precomputed://gs://flyem-male-cns/v0.9/segmentation"
_NEUPRINT_URL = "https://neuprint.janelia.org"
_NEUPRINT_DATASET = "male-cns:v0.9"

MAX_ZOOM = 4

# --select mode -> (neuPrint Neuron property, allowed values). Verified live.
_SELECT = {
    "dn": ("superclass", ["descending_neuron"]),
    "dn-an": ("superclass", ["descending_neuron", "ascending_neuron"]),
    "ascending": ("superclass", ["ascending_neuron"]),
    # the sexually-dimorphic / male courtship-&-reproduction circuit (the MaleCNS
    # paper's headline). 1267 male-specific + 747 sexually dimorphic ≈ 2014.
    "courtship": ("dimorphism", ["male-specific", "sexually dimorphic"]),
}


def _hash_color(body_id) -> str:
    """Deterministic per-neuron color from bodyId (the viewer's "Original"). md5,
    not salted hash(), for reproducibility. Categorical fields (super_class /
    dimorphism / class / predicted_nt) drive the auto Color-By dropdown."""
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
# Step 1: neuPrint metadata (raw REST + token) + curation
# ---------------------------------------------------------------------------

def query_neuprint(token: str, select: str, max_neurons: int | None) -> list[dict]:
    import requests
    field, values = _SELECT[select]
    val_list = ", ".join(f'"{s}"' for s in values)
    cypher = f"""
    MATCH (n :Neuron)
    WHERE n.{field} IN [{val_list}] AND n.bodyId IS NOT NULL
    RETURN n.bodyId AS bodyId,
           n.type AS cellType,
           n.instance AS instance,
           n.systematicType AS systematicType,
           n.superclass AS superclass,
           n.class AS neuronClass,
           n.subclass AS subclass,
           n.somaSide AS somaSide,
           n.somaNeuromere AS somaNeuromere,
           n.hemilineage AS hemilineage,
           n.predictedNt AS predictedNt,
           n.consensusNt AS consensusNt,
           n.dimorphism AS dimorphism,
           n.flywireType AS flywireType,
           n.hemibrainType AS hemibrainType,
           n.mancType AS mancType,
           n.vfbId AS vfbId,
           n.entryNerve AS entryNerve,
           n.exitNerve AS exitNerve,
           n.status AS status,
           n.statusLabel AS statusLabel,
           n.size AS size,
           n.pre AS pre,
           n.post AS post,
           n.somaLocation AS somaLocation,
           n.roiInfo AS roiInfo
    ORDER BY n.size DESC
    """
    if max_neurons:
        cypher += f"\n    LIMIT {max_neurons}"
    print(f"Querying {_NEUPRINT_DATASET} for {field} IN [{val_list}] (--select {select})...")
    resp = requests.post(
        f"{_NEUPRINT_URL}/api/custom/custom",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"cypher": cypher, "dataset": _NEUPRINT_DATASET}, timeout=180)
    resp.raise_for_status()
    result = resp.json()
    neurons = [dict(zip(result["columns"], row)) for row in result["data"]]
    print(f"  Got {len(neurons)} neurons from neuPrint")
    return neurons


# ---------------------------------------------------------------------------
# Step 2: Download meshes via CloudVolume (anonymous, public bucket)
# ---------------------------------------------------------------------------

_WORKER_CV = None


def _dl_init_worker() -> None:
    global _WORKER_CV
    from cloudvolume import CloudVolume
    _WORKER_CV = CloudVolume(_MALECNS_SEG, use_https=True, fill_missing=True,
                             cache=False, progress=False)


def _get_mesh(cv, body_id: int, lod: int):
    last = None
    for l in range(lod, -1, -1):
        try:
            res = cv.mesh.get(body_id, lod=l)
            mesh = res[body_id] if isinstance(res, dict) else res
            if mesh is not None and len(getattr(mesh, "vertices", [])):
                return mesh
        except Exception as e:  # noqa: BLE001
            last = e
    if last:
        raise last
    raise RuntimeError("empty mesh")


def _dl_one(task: tuple[int, str, bool, int]) -> tuple[int, str, int]:
    body_id, out_dir, skip_existing, lod = task
    obj_path = Path(out_dir) / f"{body_id}.obj"
    if skip_existing and obj_path.exists():
        return (body_id, "skip", 0)
    try:
        mesh = _get_mesh(_WORKER_CV, body_id, lod)
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3) + 1
        tmp = obj_path.with_suffix(".obj.tmp")
        with open(tmp, "w") as f:
            np.savetxt(f, verts, fmt="v %.7g %.7g %.7g")
            np.savetxt(f, faces, fmt="f %d %d %d")
        tmp.rename(obj_path)
        return (body_id, "ok", int(len(verts)))
    except Exception as e:  # noqa: BLE001
        return (body_id, f"err:{type(e).__name__}:{e}", 0)


def download_meshes(body_ids: list[int], output_dir: Path, *, lod: int = 1,
                    skip_existing: bool = True, workers: int | None = None) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    n_workers = workers or min(16, (os.cpu_count() or 8))
    tasks = [(int(b), str(output_dir), skip_existing, lod) for b in body_ids]
    total = len(tasks)
    done = errors = 0
    print(f"  Downloading {total} meshes (lod={lod}) with {n_workers} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_dl_init_worker) as ex:
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
# Step 3: per-mesh tags (per-id color + rich MaleCNS annotations)
# ---------------------------------------------------------------------------

# neuPrint result key -> muDM tag key. Only emitted when non-empty.
_TAG_MAP = [
    ("cellType", "cell_type"),
    ("instance", "instance"),
    ("systematicType", "systematic_type"),
    ("superclass", "super_class"),
    ("neuronClass", "neuron_class"),
    ("subclass", "subclass"),
    ("somaSide", "soma_side"),
    ("somaNeuromere", "soma_neuromere"),
    ("hemilineage", "hemilineage"),
    ("predictedNt", "predicted_nt"),
    ("consensusNt", "consensus_nt"),
    ("dimorphism", "dimorphism"),
    ("flywireType", "flywire_type"),
    ("hemibrainType", "hemibrain_type"),
    ("mancType", "manc_type"),
    ("vfbId", "vfb_id"),
    ("entryNerve", "entry_nerve"),
    ("exitNerve", "exit_nerve"),
    ("status", "status"),
    ("statusLabel", "status_label"),
]


def _build_tags(obj_path: Path, meta_lookup: dict[str, dict]) -> dict:
    body_id = obj_path.stem
    tags: dict = {
        "body_id": int(body_id) if body_id.isdigit() else body_id,
        "source": obj_path.name,
    }
    meta = meta_lookup.get(body_id, {})
    for col, key in _TAG_MAP:
        v = meta.get(col)
        if isinstance(v, str):
            v = v.strip()
        if v:
            tags[key] = v
    if meta.get("pre") is not None:
        tags["pre"] = str(meta["pre"])
    if meta.get("post") is not None:
        tags["post"] = str(meta["post"])
    # dominant neuropils from roiInfo (top-3 by pre+post)
    if meta.get("roiInfo"):
        try:
            roi = json.loads(meta["roiInfo"]) if isinstance(meta["roiInfo"], str) else meta["roiInfo"]
            ranked = sorted(roi.items(),
                            key=lambda kv: (kv[1].get("pre", 0) or 0) + (kv[1].get("post", 0) or 0),
                            reverse=True)
            regions = [r[0] for r in ranked[:3]]
            if regions:
                tags["brain_regions"] = ", ".join(regions)
        except Exception:  # noqa: BLE001
            pass
    # name MUST be unique per neuron (build_feature_index groups by name)
    _ct = tags.get("cell_type") or tags.get("hemibrain_type") or tags.get("flywire_type")
    tags["name"] = f"{_ct} ({body_id})" if _ct else str(body_id)
    tags["color"] = _hash_color(body_id)
    return tags


# ---------------------------------------------------------------------------
# Step 4: Tile meshopt 3D Tiles + features.json + tilejson3d + pyramids.json
# ---------------------------------------------------------------------------

def tile_meshopt(mesh_dir: Path, meta_path: Path, output_dir: Path, *,
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

    meta_lookup: dict[str, dict] = {}
    if meta_path.exists():
        for n in json.loads(meta_path.read_text()).get("neurons", []):
            meta_lookup[str(n["bodyId"])] = n

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
        "label": "Janelia MaleCNS (whole Drosophila male CNS — brain + VNC)",
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
    parser = argparse.ArgumentParser(description="MaleCNS (Drosophila male CNS) download + meshopt 3D tiling")
    parser.add_argument("--download", action="store_true", help="Query neuPrint + download meshes")
    parser.add_argument("--tile", action="store_true", help="Tile downloaded meshes (meshopt 3D Tiles)")
    parser.add_argument("--smoke", type=int, default=None, metavar="BODY_ID",
                        help="Download a single bodyId mesh (anonymous, no token), then exit")
    parser.add_argument("--select", default="dn-an", choices=list(_SELECT),
                        help="Curation (default: dn-an = descending + ascending = brain<->VNC bridge)")
    parser.add_argument("--max-neurons", type=int, default=1000, help="Cap on selected neurons (default: 1000)")
    parser.add_argument("--lod", type=int, default=1, help="Multires mesh LOD: 0=finest..3=coarsest (default: 1)")
    parser.add_argument("--max-zoom", type=int, default=MAX_ZOOM, help=f"Max zoom (default: {MAX_ZOOM})")
    parser.add_argument("--workers", type=int, default=None, help="Download worker processes")
    parser.add_argument("--ingest-threads", type=int, default=0, help="Tile ingest threads (0=all)")
    parser.add_argument("--data-dir", type=Path, default=_DATA_DIR, help="Data directory")
    parser.add_argument("--output", type=Path, default=_TILES_DIR, help="Output pyramid directory")
    args = parser.parse_args()

    data_dir = args.data_dir
    mesh_dir = data_dir / "meshes"
    meta_path = data_dir / "metadata.json"

    if args.smoke is not None:
        mesh_dir.mkdir(parents=True, exist_ok=True)
        print(f"Smoke test: fetching body {args.smoke} (lod={args.lod}) from {_MALECNS_SEG} ...")
        _dl_init_worker()
        bid, status, nverts = _dl_one((args.smoke, str(mesh_dir), False, args.lod))
        print(f"  -> {status}  ({nverts} vertices)")
        sys.exit(0 if status == "ok" else 1)

    if not (args.download or args.tile):
        parser.print_help()
        sys.exit(1)

    if args.download:
        token = os.environ.get("NEUPRINT_TOKEN", "")
        if not token:
            print("ERROR: NEUPRINT_TOKEN not set (source /data/ai/mudm-paper/.env on oden).", file=sys.stderr)
            sys.exit(1)
        data_dir.mkdir(parents=True, exist_ok=True)
        neurons = query_neuprint(token, args.select, args.max_neurons)
        if not neurons:
            print("ERROR: neuPrint returned no neurons.", file=sys.stderr)
            sys.exit(1)
        meta_path.write_text(json.dumps({"neurons": neurons, "dataset": _NEUPRINT_DATASET,
                                         "select": args.select}, indent=2))
        print(f"  Saved metadata to {meta_path}")
        body_ids = [n["bodyId"] for n in neurons]
        print(f"\nDownloading {len(body_ids)} MaleCNS meshes (lod={args.lod})...")
        t0 = time.perf_counter()
        download_meshes(body_ids, mesh_dir, lod=args.lod, workers=args.workers)
        print(f"  Download time: {_fmt_time(time.perf_counter() - t0)}")

    if args.tile:
        tile_meshopt(mesh_dir, meta_path, args.output,
                     max_zoom=args.max_zoom, max_files=args.max_neurons,
                     ingest_threads=args.ingest_threads)

    print("Done.")


if __name__ == "__main__":
    main()
