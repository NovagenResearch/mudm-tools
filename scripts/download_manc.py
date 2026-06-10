#!/usr/bin/env python3
"""Download Janelia MANC (Male Adult Nerve Cord / Drosophila VNC) neuron meshes
and tile them as meshopt-compressed 3D Tiles for the muDM 3D viewer.

Mirrors the proven hemibrain pipeline (download_hemibrain.py +
hemibrain_3dtiles_meshopt.py) with MANC-specific sources, a richer neuPrint
query, and class-based coloring.

Sources (verified 2026-06-06):
    - meshes (CloudVolume):  precomputed://gs://manc-seg-v1p2/manc-seg-v1.2
    - metadata (neuPrint):   https://neuprint.janelia.org  dataset manc:v1.2.1
    - license:               CC BY 4.0

Prerequisites:
    cloud-volume + requests in the venv; NEUPRINT_TOKEN env var
    (get from https://neuprint.janelia.org -> Account).

Usage (on oden, where deps + token live):
    set -a; . /data/ai/mudm-paper/.env; set +a
    .venv/bin/python scripts/download_manc.py --download --tile \
        --max-neurons 250 --max-zoom 4 --output /data/ai/mudm-paper/mudm-data/tiles/manc

    # download only (no tiling):
    .venv/bin/python scripts/download_manc.py --download --max-neurons 250

    # smoke test a single body id (confirm CloudVolume mesh.get works):
    .venv/bin/python scripts/download_manc.py --smoke 10000
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

_DATA_DIR = _ROOT / "data" / "manc"
_MESH_DIR = _DATA_DIR / "meshes"
_META_PATH = _DATA_DIR / "metadata.json"
_TILES_DIR = _DATA_DIR / "tiles" / "manc"

_MANC_SEG = "precomputed://gs://manc-seg-v1p2/manc-seg-v1.2"
_NEUPRINT_URL = "https://neuprint.janelia.org"
_NEUPRINT_DATASET = "manc:v1.2.1"

MAX_ZOOM = 4

def _hash_color(body_id) -> str:
    """Deterministic per-neuron color from the body id (the viewer's default
    "Original" coloring — distinct per neuron, like hemibrain). Class/NT/etc.
    coloring is handled separately by the viewer's auto-built "Color By"
    dropdown from the neuron_class/predicted_nt/... fields, so the baked color
    is intentionally per-id, NOT per-class. Uses md5 (not Python's salted
    hash()) so colors are stable across runs and reproducible."""
    # Spread hue + saturation + lightness (from separate md5 bytes) so the palette
    # is ~3D, not 360-hue-capped — adjacent neurons get vivid, distinct colors.
    d = hashlib.md5(str(body_id).encode()).digest()
    hue = d[0] / 255.0
    sat = 0.65 + (d[1] / 255.0) * 0.32
    lig = 0.45 + (d[2] / 255.0) * 0.22
    r, g, b = colorsys.hls_to_rgb(hue, lig, sat)
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
# Step 1: Query neuPrint for MANC metadata (raw REST; no neuprint-python dep)
# ---------------------------------------------------------------------------

def query_neuprint(
    token: str,
    max_neurons: int | None = None,
    min_type_count: int = 0,
) -> list[dict]:
    """Query neuPrint (manc:v1.2.1) for richly-annotated traced neurons.

    Selects the MANC-specific annotation fields that make this dataset a far
    richer showcase than hemibrain. Absent properties return null (Cypher is
    tolerant), so over-selecting is safe across schema minor versions.

    When *min_type_count* > 0, restricts to neurons whose cell type has at least
    that many traced instances (well-represented types).
    """
    import requests

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    fields = """
               n.bodyId AS bodyId,
               n.type AS cellType,
               n.instance AS instance,
               n.systematicType AS systematicType,
               n.class AS neuronClass,
               n.subclass AS subclass,
               n.somaSide AS somaSide,
               n.somaNeuromere AS somaNeuromere,
               n.hemilineage AS hemilineage,
               n.predictedNt AS predictedNt,
               n.entryNerve AS entryNerve,
               n.exitNerve AS exitNerve,
               n.status AS status,
               n.statusLabel AS statusLabel,
               n.size AS size,
               n.pre AS pre,
               n.post AS post,
               n.somaLocation AS somaLocation,
               n.roiInfo AS roiInfo
    """

    if min_type_count > 0:
        cypher = f"""
        MATCH (n :Neuron)
        WHERE n.status = "Traced" AND n.type IS NOT NULL
        WITH n.type AS ct, count(n) AS cnt
        WHERE cnt >= {min_type_count}
        WITH collect(ct) AS validTypes
        MATCH (n :Neuron)
        WHERE n.type IN validTypes AND n.status = "Traced"
        RETURN {fields}
        ORDER BY n.size DESC
        """
        print(f"Querying MANC for neurons in types with >={min_type_count} instances...")
    else:
        cypher = f"""
        MATCH (n :Neuron)
        WHERE n.status = "Traced" AND n.type IS NOT NULL
        RETURN {fields}
        ORDER BY n.size DESC
        """
        print("Querying MANC for traced, typed neuron metadata...")

    if max_neurons:
        cypher += f"\n        LIMIT {max_neurons}"

    payload = {"cypher": cypher, "dataset": _NEUPRINT_DATASET}
    resp = requests.post(
        f"{_NEUPRINT_URL}/api/custom/custom",
        headers=headers,
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    result = resp.json()
    neurons = [dict(zip(result["columns"], row)) for row in result["data"]]
    print(f"  Got {len(neurons)} neurons from neuPrint")
    return neurons


# ---------------------------------------------------------------------------
# Step 2: Download meshes via CloudVolume
# ---------------------------------------------------------------------------

_WORKER_CV = None


def _dl_init_worker() -> None:
    global _WORKER_CV
    from cloudvolume import CloudVolume
    _WORKER_CV = CloudVolume(_MANC_SEG, use_https=True, progress=False)


def _get_mesh(cv, body_id: int, lod: int):
    """Fetch one neuron mesh at the requested multires LOD.

    MANC meshes are neuroglancer_multilod_draco with LODs 0 (finest, up to ~7M
    verts / ~450 MB on the giant fibers) through 3 (coarsest). lod=1 is the web
    sweet spot (~1M verts on giants). If the requested LOD is unavailable for a
    neuron, fall back to finer levels (which always exist).
    """
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


def download_meshes(
    body_ids: list[int],
    output_dir: Path,
    *,
    lod: int = 1,
    skip_existing: bool = True,
    workers: int | None = None,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    n_workers = workers or min(16, (os.cpu_count() or 8))
    tasks = [(int(b), str(output_dir), skip_existing, lod) for b in body_ids]
    total = len(tasks)
    done = errors = 0
    print(f"  Downloading {total} meshes with {n_workers} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_dl_init_worker) as ex:
        for i, (body_id, status, _n) in enumerate(ex.map(_dl_one, tasks, chunksize=1), 1):
            if status in ("ok", "skip"):
                done += 1
            else:
                errors += 1
                if errors <= 5:
                    print(f"  {body_id} — ERROR: {status}", flush=True)
            if i % 25 == 0 or i == total:
                print(f"  [{i}/{total}] ok={done} err={errors}", flush=True)
    print(f"  Downloaded {done}/{total} meshes ({errors} errors)")
    return done


# ---------------------------------------------------------------------------
# Step 3: Build per-mesh tags (color by neuron class + rich annotations)
# ---------------------------------------------------------------------------

def _build_tags(obj_path: Path, meta_lookup: dict[str, dict]) -> dict:
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
        if meta.get("systematicType"):
            tags["systematic_type"] = meta["systematicType"]
        if meta.get("neuronClass"):
            tags["neuron_class"] = meta["neuronClass"]
        if meta.get("subclass"):
            tags["subclass"] = meta["subclass"]
        if meta.get("somaSide"):
            tags["soma_side"] = meta["somaSide"]
        if meta.get("somaNeuromere"):
            tags["soma_neuromere"] = meta["somaNeuromere"]
        if meta.get("hemilineage"):
            tags["hemilineage"] = meta["hemilineage"]
        if meta.get("predictedNt"):
            tags["predicted_nt"] = meta["predictedNt"]
        if meta.get("entryNerve"):
            tags["entry_nerve"] = meta["entryNerve"]
        if meta.get("exitNerve"):
            tags["exit_nerve"] = meta["exitNerve"]
        if meta.get("status"):
            tags["status"] = meta["status"]
        if meta.get("statusLabel"):
            tags["status_label"] = meta["statusLabel"]
        if meta.get("pre") is not None:
            tags["pre"] = str(meta["pre"])
        if meta.get("post") is not None:
            tags["post"] = str(meta["post"])
        if meta.get("size") is not None:
            tags["size_voxels"] = str(meta["size"])
        # Dominant VNC neuropils from roiInfo (top-3 by pre+post synapses)
        if meta.get("roiInfo"):
            try:
                roi = (json.loads(meta["roiInfo"])
                       if isinstance(meta["roiInfo"], str) else meta["roiInfo"])
                ranked = sorted(
                    roi.items(),
                    key=lambda kv: (kv[1].get("pre", 0) or 0) + (kv[1].get("post", 0) or 0),
                    reverse=True,
                )
                regions = [r[0] for r in ranked[:3]]
                if regions:
                    tags["brain_regions"] = ", ".join(regions)
            except Exception:  # noqa: BLE001
                pass

    instance = tags.get("instance")
    tags["name"] = (
        f"{instance} ({body_id})" if instance
        else (tags.get("systematic_type") or tags.get("cell_type") or str(body_id))
    )
    tags["color"] = _hash_color(body_id)
    return tags


# ---------------------------------------------------------------------------
# Step 4: Tile meshopt 3D Tiles + features.json + tilejson3d.json + pyramids.json
# ---------------------------------------------------------------------------

def tile_meshopt(
    mesh_dir: Path,
    meta_path: Path,
    output_dir: Path,
    *,
    max_zoom: int = MAX_ZOOM,
    max_files: int | None = None,
    ingest_threads: int = 0,
) -> None:
    import tempfile

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
        raw = json.loads(meta_path.read_text())
        for n in raw.get("neurons", []):
            meta_lookup[str(n["bodyId"])] = n

    path_strs = [str(p) for p in obj_paths]
    tags_list = [_build_tags(p, meta_lookup) for p in obj_paths]

    # Class breakdown sanity report
    from collections import Counter
    classes = Counter(t.get("neuron_class", "(unknown)") for t in tags_list)
    print("Class breakdown:")
    for cls, cnt in classes.most_common():
        print(f"  {cnt:4d}  {cls}")

    t0 = time.perf_counter()
    bounds = scan_obj_bounds(path_strs)
    print(f"\nBounds: x=[{bounds[0]:.0f}, {bounds[3]:.0f}] "
          f"y=[{bounds[1]:.0f}, {bounds[4]:.0f}] "
          f"z=[{bounds[2]:.0f}, {bounds[5]:.0f}]  ({_fmt_time(time.perf_counter() - t0)})")

    tiles3d_dir = output_dir / "3dtiles"
    if tiles3d_dir.exists():
        shutil.rmtree(tiles3d_dir)
    tiles3d_dir.mkdir(parents=True, exist_ok=True)

    gen = StreamingTileGenerator(
        min_zoom=0, max_zoom=max_zoom, base_cells=100, temp_dir=tempfile.gettempdir())
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

    id_fields = ["body_id", "instance"]
    index, zoom_counts, max_zoom_found = build_index(tiles3d_dir, id_fields=id_fields)
    n_features = len(index.get("features", []))
    (output_dir / "features.json").write_text(json.dumps(index, indent=2))
    print(f"Features indexed: {n_features}")

    tj = build_tilejson(zoom_counts, max_zoom_found, id_fields=id_fields, bounds3d=list(bounds))
    (output_dir / "tilejson3d.json").write_text(json.dumps(tj, indent=2))

    # Upsert pyramids.json
    pyramids_path = output_dir.parent / "pyramids.json"
    pyramid_id = output_dir.name
    entry = {
        "id": pyramid_id,
        "label": "Janelia MANC (Drosophila VNC connectome)",
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
    manifest.setdefault("version", "1.0")
    manifest["pyramids"] = [
        p for p in manifest["pyramids"] if p.get("id") != pyramid_id
    ] + [entry]
    pyramids_path.write_text(json.dumps(manifest, indent=2))
    print(f"Updated {pyramids_path} (entry: {pyramid_id}, {n_features} features, {n_tiles} tiles)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="MANC (Drosophila VNC) download + meshopt 3D tiling")
    parser.add_argument("--download", action="store_true", help="Query neuPrint + download meshes")
    parser.add_argument("--tile", action="store_true", help="Tile downloaded meshes (meshopt 3D Tiles)")
    parser.add_argument("--smoke", type=int, default=None, metavar="BODY_ID",
                        help="Download a single body id to test CloudVolume mesh.get, then exit")
    parser.add_argument("--max-neurons", type=int, default=250, help="Max neurons (default: 250)")
    parser.add_argument("--lod", type=int, default=1,
                        help="Multires mesh LOD: 0=finest(~450MB giants) .. 3=coarsest (default: 1)")
    parser.add_argument("--min-type-count", type=int, default=0,
                        help="Only neurons whose type has >= N traced instances (default: 0)")
    parser.add_argument("--max-zoom", type=int, default=MAX_ZOOM, help=f"Max zoom (default: {MAX_ZOOM})")
    parser.add_argument("--workers", type=int, default=None, help="Download worker processes")
    parser.add_argument("--ingest-threads", type=int, default=0, help="Tile ingest threads (0=all)")
    parser.add_argument("--data-dir", type=Path, default=_DATA_DIR, help="Data directory")
    parser.add_argument("--output", type=Path, default=_TILES_DIR, help="Output pyramid directory")
    args = parser.parse_args()

    data_dir = args.data_dir
    mesh_dir = data_dir / "meshes"
    meta_path = data_dir / "metadata.json"

    # --- Smoke test: single mesh fetch ---
    if args.smoke is not None:
        mesh_dir.mkdir(parents=True, exist_ok=True)
        print(f"Smoke test: fetching body {args.smoke} (lod={args.lod}) from {_MANC_SEG} ...")
        _dl_init_worker()
        bid, status, nverts = _dl_one((args.smoke, str(mesh_dir), False, args.lod))
        print(f"  -> {status}  ({nverts} vertices)")
        sys.exit(0 if status == "ok" else 1)

    if not (args.download or args.tile):
        parser.print_help()
        sys.exit(1)

    # --- Download ---
    if args.download:
        token = os.environ.get("NEUPRINT_TOKEN", "")
        if not token:
            print("ERROR: NEUPRINT_TOKEN not set. Get it from https://neuprint.janelia.org -> Account",
                  file=sys.stderr)
            sys.exit(1)
        data_dir.mkdir(parents=True, exist_ok=True)
        neurons = query_neuprint(token, max_neurons=args.max_neurons, min_type_count=args.min_type_count)
        if not neurons:
            print("ERROR: neuPrint returned no neurons.", file=sys.stderr)
            sys.exit(1)
        meta_path.write_text(json.dumps({"neurons": neurons, "dataset": _NEUPRINT_DATASET}, indent=2))
        print(f"  Saved metadata to {meta_path}")

        body_ids = [n["bodyId"] for n in neurons]
        print(f"\nDownloading {len(body_ids)} neuron meshes (lod={args.lod})...")
        t0 = time.perf_counter()
        download_meshes(body_ids, mesh_dir, lod=args.lod, workers=args.workers)
        print(f"  Download time: {_fmt_time(time.perf_counter() - t0)}")

    # --- Tile ---
    if args.tile:
        tile_meshopt(
            mesh_dir, meta_path, args.output,
            max_zoom=args.max_zoom, max_files=args.max_neurons,
            ingest_threads=args.ingest_threads,
        )

    print("Done.")


if __name__ == "__main__":
    main()
