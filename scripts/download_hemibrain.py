#!/usr/bin/env python3
"""Download Hemibrain v1.2.1 neuron meshes and metadata for benchmarking.

Downloads neuron meshes from the Janelia Hemibrain dataset via CloudVolume,
queries neuPrint for cell type metadata, exports as OBJ files, converts to
MuDM, tiles with TileGenerator3D, and benchmarks.

Prerequisites:
    uv pip install --python .venv/bin/python cloud-volume requests

Usage::

    # Set neuPrint auth token (get from https://neuprint.janelia.org → Account):
    export NEUPRINT_TOKEN="eyJhbGciOi..."

    # Full pipeline:
    .venv/bin/python scripts/download_hemibrain.py --download --convert --tile --benchmark

    # Download only (top 1000 neurons):
    .venv/bin/python scripts/download_hemibrain.py --download --max-neurons 1000

    # From existing OBJ files:
    .venv/bin/python scripts/download_hemibrain.py --convert --tile --benchmark
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

_DATA_DIR = _ROOT / "data" / "hemibrain"
_MESH_DIR = _DATA_DIR / "meshes"
_META_PATH = _DATA_DIR / "metadata.json"
_TILES_DIR = _DATA_DIR / "tiles"

_HEMIBRAIN_SEG = (
    "precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation"
)
_NEUPRINT_URL = "https://neuprint.janelia.org"
_NEUPRINT_DATASET = "hemibrain:v1.2.1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    return f"{n / (1024 * 1024):.2f} MB"


def _surface_run_summary(pyramid_dir: Path, phase: str) -> dict | None:
    """Read ``<pyramid_dir>/run_summary.json`` (WS-D) and ``logging.info`` it.

    The Rust ErrorCollector streams ``errors.jsonl`` + ``run_summary.json`` under
    ``pyramid_dir`` (only when ``_set_run_dir`` is set and the phase wires the
    collector). The Rust side already raises on fatal; Python just surfaces the
    honest ok/fail counts and the errors.jsonl path. Returns the parsed summary,
    or ``None`` when no summary was written for this phase.
    """
    summary_path = pyramid_dir / "run_summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    errlog = pyramid_dir / "errors.jsonl"
    logging.info(
        "[%s] run_summary: ok=%s fail=%s fatal=%s elapsed_s=%.2f (errors: %s)",
        summary.get("phase", phase),
        summary.get("ok", "?"),
        summary.get("fail", "?"),
        summary.get("fatal", "?"),
        float(summary.get("elapsed_s", 0.0)),
        errlog if errlog.exists() else "none",
    )
    return summary


# ---------------------------------------------------------------------------
# Step 1: Query neuPrint for metadata
# ---------------------------------------------------------------------------

def query_neuprint(
    token: str,
    max_neurons: int | None = None,
    min_type_count: int = 0,
) -> list[dict]:
    """Query neuPrint for neuron metadata via REST API.

    When *min_type_count* > 0, a two-stage Cypher query first identifies cell
    types with at least that many traced instances, then returns neurons whose
    type is in that set (ordered by size DESC, up to *max_neurons*).

    Returns list of dicts with bodyId, type, instance, status, etc.
    """
    import requests

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    if min_type_count > 0:
        # Two-stage query: only neurons from well-represented cell types
        cypher = f"""
        MATCH (n :Neuron)
        WHERE n.status = "Traced" AND n.type IS NOT NULL
        WITH n.type AS ct, count(n) AS cnt
        WHERE cnt >= {min_type_count}
        WITH collect(ct) AS validTypes
        MATCH (n :Neuron)
        WHERE n.type IN validTypes
        RETURN n.bodyId AS bodyId,
               n.type AS cellType,
               n.instance AS instance,
               n.status AS status,
               n.statusLabel AS statusLabel,
               n.size AS size,
               n.pre AS pre,
               n.post AS post,
               n.somaLocation AS somaLocation,
               n.somaRadius AS somaRadius,
               n.cropped AS cropped,
               n.roiInfo AS roiInfo
        ORDER BY n.size DESC
        """
        if max_neurons:
            cypher += f"\nLIMIT {max_neurons}"
        print(f"Querying neuPrint for neurons in types with >={min_type_count} instances...")
    else:
        # Original query: all traced neurons
        cypher = """
        MATCH (n :Neuron)
        WHERE n.status = "Traced"
        RETURN n.bodyId AS bodyId,
               n.type AS cellType,
               n.instance AS instance,
               n.status AS status,
               n.statusLabel AS statusLabel,
               n.size AS size,
               n.pre AS pre,
               n.post AS post,
               n.somaLocation AS somaLocation,
               n.somaRadius AS somaRadius,
               n.cropped AS cropped,
               n.roiInfo AS roiInfo
        ORDER BY n.size DESC
        """
        if max_neurons:
            cypher += f"\nLIMIT {max_neurons}"
        print(f"Querying neuPrint for neuron metadata...")

    payload = {"cypher": cypher, "dataset": _NEUPRINT_DATASET}
    resp = requests.post(
        f"{_NEUPRINT_URL}/api/custom/custom",
        headers=headers,
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    result = resp.json()

    columns = result["columns"]
    rows = result["data"]
    neurons = [dict(zip(columns, row)) for row in rows]
    print(f"  Got {len(neurons)} neurons from neuPrint")
    return neurons


def query_neuprint_by_ids(token: str, body_ids: list[int]) -> list[dict]:
    """Query neuPrint for specific body IDs (for updating existing metadata).

    Queries in batches of 500 to avoid Cypher query size limits.
    """
    import requests

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    all_neurons: list[dict] = []
    batch_size = 500

    for start in range(0, len(body_ids), batch_size):
        chunk = body_ids[start : start + batch_size]
        id_list = ", ".join(str(bid) for bid in chunk)
        cypher = f"""
        MATCH (n :Neuron)
        WHERE n.bodyId IN [{id_list}]
        RETURN n.bodyId AS bodyId,
               n.type AS cellType,
               n.instance AS instance,
               n.status AS status,
               n.statusLabel AS statusLabel,
               n.size AS size,
               n.pre AS pre,
               n.post AS post,
               n.somaLocation AS somaLocation,
               n.somaRadius AS somaRadius,
               n.cropped AS cropped,
               n.roiInfo AS roiInfo
        """
        payload = {"cypher": cypher, "dataset": _NEUPRINT_DATASET}
        batch_num = start // batch_size + 1
        total_batches = (len(body_ids) + batch_size - 1) // batch_size
        print(f"  Querying batch {batch_num}/{total_batches} "
              f"({len(chunk)} body IDs)...", flush=True)
        resp = requests.post(
            f"{_NEUPRINT_URL}/api/custom/custom",
            headers=headers,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        result = resp.json()
        columns = result["columns"]
        rows = result["data"]
        all_neurons.extend(dict(zip(columns, row)) for row in rows)

    print(f"  Got metadata for {len(all_neurons)} neurons")
    return all_neurons


def update_metadata(mesh_dir: Path, meta_path: Path, token: str) -> None:
    """Re-query neuPrint for all body IDs found on disk, update metadata.json."""
    obj_paths = sorted(mesh_dir.glob("*.obj"))
    if not obj_paths:
        print(f"ERROR: No .obj files in {mesh_dir}", file=sys.stderr)
        sys.exit(1)

    body_ids = []
    for p in obj_paths:
        stem = p.stem
        if stem.isdigit():
            body_ids.append(int(stem))
    print(f"Found {len(body_ids)} OBJ files on disk, querying neuPrint...")

    neurons = query_neuprint_by_ids(token, body_ids)

    # Parse roiInfo JSON strings into dicts
    for n in neurons:
        if isinstance(n.get("roiInfo"), str):
            try:
                n["roiInfo"] = json.loads(n["roiInfo"])
            except (json.JSONDecodeError, TypeError):
                pass

    meta_path.write_text(json.dumps(
        {"neurons": neurons, "dataset": _NEUPRINT_DATASET},
        indent=2,
    ))

    # Print summary
    with_type = sum(1 for n in neurons if n.get("cellType"))
    with_pre = sum(1 for n in neurons if n.get("pre") is not None)
    with_roi = sum(1 for n in neurons if n.get("roiInfo"))
    print(f"  Updated {meta_path}")
    print(f"  {len(neurons)} neurons total")
    print(f"  {with_type} with cellType, {with_pre} with synapse counts, "
          f"{with_roi} with roiInfo")


# ---------------------------------------------------------------------------
# Step 2: Download meshes via CloudVolume
# ---------------------------------------------------------------------------

_WORKER_CV = None


def _dl_init_worker() -> None:
    """Process-pool initializer: one CloudVolume per worker process."""
    global _WORKER_CV
    from cloudvolume import CloudVolume
    _WORKER_CV = CloudVolume(_HEMIBRAIN_SEG, use_https=True, progress=False)


def _dl_one(task: tuple[int, str, bool]) -> tuple[int, str, int]:
    """Download one mesh to OBJ. Returns ``(body_id, status, n_verts)``.

    Writes to a temp file then atomically renames, so an interrupted run never
    leaves a partial ``.obj`` that ``skip_existing`` would later treat as done.
    """
    body_id, out_dir, skip_existing = task
    obj_path = Path(out_dir) / f"{body_id}.obj"
    if skip_existing and obj_path.exists():
        return (body_id, "skip", 0)
    try:
        mesh = _WORKER_CV.mesh.get(body_id, lod=0)[body_id]
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
    skip_existing: bool = True,
    workers: int | None = None,
) -> int:
    """Download neuron meshes as OBJ files via CloudVolume, in parallel.

    Uses a process pool (default ``min(16, cpu_count)``) so the network fetch
    and OBJ writing for different neurons run concurrently. Returns the number
    of successfully downloaded (or already-present) meshes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n_workers = workers or min(16, (os.cpu_count() or 8))
    tasks = [(int(b), str(output_dir), skip_existing) for b in body_ids]
    total = len(tasks)
    done = errors = 0
    print(f"  Downloading {total} meshes with {n_workers} workers...", flush=True)
    with ProcessPoolExecutor(
        max_workers=n_workers, initializer=_dl_init_worker
    ) as ex:
        for i, (body_id, status, _n) in enumerate(
            ex.map(_dl_one, tasks, chunksize=1), 1
        ):
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
# Step 3: Convert OBJ → MuDM
# ---------------------------------------------------------------------------

def convert_to_microjson(
    mesh_dir: Path,
    metadata_path: Path,
    *,
    max_files: int | None = None,
):
    """Convert Hemibrain OBJ meshes to MuDMFeatureCollection."""
    import numpy as np

    from mudm.model import (
        MuDMFeature,
        MuDMFeatureCollection,
        OntologyTerm,
        Vocabulary,
    )
    from mudm_tools.swc import _mesh_to_tin
    from obj_to_microjson import parse_obj

    obj_paths = sorted(mesh_dir.glob("*.obj"))
    if not obj_paths:
        print(f"ERROR: No .obj files in {mesh_dir}", file=sys.stderr)
        sys.exit(1)

    if max_files and max_files < len(obj_paths):
        obj_paths = obj_paths[:max_files]

    # Load metadata if available
    meta_lookup: dict[str, dict] = {}
    if metadata_path.exists():
        raw = json.loads(metadata_path.read_text())
        for neuron in raw.get("neurons", []):
            meta_lookup[str(neuron["bodyId"])] = neuron

    print(f"Converting {len(obj_paths)} OBJ files to MuDM...")
    t0 = time.perf_counter()
    features: list[MuDMFeature] = []
    total_verts = 0
    total_faces = 0

    for i, obj_path in enumerate(obj_paths, 1):
        if i % 50 == 0 or i == len(obj_paths):
            print(f"  [{i}/{len(obj_paths)}] {obj_path.name}", end="\r", file=sys.stderr)

        vertices, faces = parse_obj(str(obj_path))
        tin = _mesh_to_tin(vertices, faces)

        body_id = obj_path.stem
        props: dict = {
            "body_id": int(body_id) if body_id.isdigit() else body_id,
            "source": obj_path.name,
            "vertex_count": int(vertices.shape[0]),
            "face_count": int(faces.shape[0]),
        }

        # Add neuPrint metadata if available
        meta = meta_lookup.get(body_id, {})
        feature_class = body_id
        if meta:
            if meta.get("cellType"):
                props["cell_type"] = meta["cellType"]
                feature_class = meta["cellType"]
            if meta.get("instance"):
                props["instance"] = meta["instance"]
            if meta.get("status"):
                props["status"] = meta["status"]
            if meta.get("statusLabel"):
                props["status_label"] = meta["statusLabel"]
            if meta.get("somaLocation"):
                props["soma_location"] = meta["somaLocation"]
            if meta.get("pre") is not None:
                props["pre"] = meta["pre"]
            if meta.get("post") is not None:
                props["post"] = meta["post"]
            if meta.get("somaRadius") is not None:
                props["soma_radius"] = meta["somaRadius"]
            if meta.get("cropped") is not None:
                props["cropped"] = meta["cropped"]

        features.append(MuDMFeature(
            type="Feature",
            geometry=tin,
            properties=props,
            featureClass=feature_class,
        ))
        total_verts += vertices.shape[0]
        total_faces += faces.shape[0]

    print(file=sys.stderr)
    convert_time = time.perf_counter() - t0

    # Build vocabulary from cell types
    cell_types = set()
    for f in features:
        ct = (f.properties or {}).get("cell_type")
        if ct:
            cell_types.add(ct)

    vocabs = None
    if cell_types:
        terms = {
            ct: OntologyTerm(
                uri=f"https://neuprint.janelia.org/view/celltype/{ct}",
                label=ct,
            )
            for ct in sorted(cell_types)
        }
        vocabs = {
            "hemibrain_cell_types": Vocabulary(
                namespace="https://neuprint.janelia.org/",
                description="Hemibrain v1.2.1 cell type annotations",
                terms=terms,
            ),
        }

    collection = MuDMFeatureCollection(
        type="FeatureCollection",
        features=features,
        properties={
            "dataset": "hemibrain_v1.2.1",
            "mesh_count": len(features),
            "total_vertices": total_verts,
            "total_faces": total_faces,
            "cell_types": len(cell_types),
        },
        vocabularies=vocabs,
    )

    print(f"  {len(features)} features, {total_verts:,} vertices, {total_faces:,} faces")
    print(f"  {len(cell_types)} unique cell types")
    print(f"  Conversion time: {_fmt_time(convert_time)}")

    return collection, convert_time


# ---------------------------------------------------------------------------
# Step 3b: True streaming tiling (no MuDMFeatureCollection in memory)
# ---------------------------------------------------------------------------

def _build_tags(obj_path: Path, meta_lookup: dict[str, dict]) -> dict:
    """Build property tags for an OBJ file from metadata."""
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
        if meta.get("status"):
            tags["status"] = meta["status"]
        if meta.get("statusLabel"):
            tags["status_label"] = meta["statusLabel"]
        if meta.get("pre") is not None:
            tags["pre"] = str(meta["pre"])
        if meta.get("post") is not None:
            tags["post"] = str(meta["post"])
        if meta.get("somaRadius") is not None:
            tags["soma_radius"] = str(meta["somaRadius"])
        if meta.get("cropped") is not None:
            tags["cropped"] = str(meta["cropped"]).lower()
        # Derive dominant neuropil region from roiInfo
        roi_info = meta.get("roiInfo")
        if roi_info and isinstance(roi_info, dict):
            # Find the region with most total synapses (pre + post)
            best_roi, best_count = None, 0
            for roi, counts in roi_info.items():
                if isinstance(counts, dict):
                    total = (counts.get("pre", 0) or 0) + (counts.get("post", 0) or 0)
                    if total > best_count:
                        best_roi, best_count = roi, total
            if best_roi:
                tags["primary_roi"] = best_roi
    # Feature name for viewer: prefer instance, fall back to body_id
    tags["name"] = tags.get("instance") or str(body_id)
    # Deterministic color from body_id hash
    h = hash(str(body_id)) & 0x7FFFFFFF
    hue = (h % 360) / 360.0
    import colorsys
    r, g, b = colorsys.hls_to_rgb(hue, 0.5, 0.75)
    tags["color"] = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    return tags


def tile_streaming(
    mesh_dir: Path,
    metadata_path: Path,
    output_dir: Path,
    *,
    max_zoom: int = 3,
    max_files: int | None = None,
    skip_3dtiles: bool = False,
    skip_pbf3: bool = False,
    pyramid_name: str = "hemibrain",
) -> dict:
    """True streaming tiling — O(1 mesh) memory during ingest.

    All heavy work (OBJ parse, projection, clipping, fragment I/O) happens
    in Rust via ``add_obj_file()`` / ``add_obj_files()``.

    Giant meshes (>500 MB) are ingested one at a time to cap peak RAM.
    Smaller meshes are batched for parallel rayon ingest.

    EMIT-ALL (WS-A): one ``StreamingTileGenerator`` is ingested ONCE, then
    both outputs are emitted from it (``generate_3dtiles`` + ``generate_parquet``)
    before the generator is dropped — byte-identical to separate per-format
    ingests (see ``tests/test_emit_all.py``).

    ``skip_pbf3`` is retained for caller compatibility but is now a no-op: pbf3,
    feature-centric pbf3, and neuroglancer are no longer emitted here (their Rust
    ``generate_*`` implementations are kept untouched; NG is deferred).

    Output uses pyramid directory structure:
        {output_dir}/{pyramid_name}/3dtiles/      (tileset.json, features.json, *.glb)
        {output_dir}/{pyramid_name}/tiles.parquet/ (zoom=N/part_*.parquet)
    """
    import shutil
    import subprocess

    from mudm_tools._rs import StreamingTileGenerator, scan_obj_bounds

    # Size threshold for serial vs parallel ingest. Raised to 5 GB so all
    # meshes in this dataset (largest ~1.3 GB) go through the parallel rayon
    # batch instead of the serial giant path. Memory-safe: rayon bounds
    # concurrency and frags stream to TMPDIR on disk; the box has 772 GB RAM.
    _GIANT_THRESHOLD = 5 * 1024 * 1024 * 1024

    obj_paths = sorted(mesh_dir.glob("*.obj"))
    if not obj_paths:
        print(f"ERROR: No .obj files in {mesh_dir}", file=sys.stderr)
        sys.exit(1)
    if max_files and max_files < len(obj_paths):
        obj_paths = obj_paths[:max_files]

    meta_lookup: dict[str, dict] = {}
    if metadata_path.exists():
        raw = json.loads(metadata_path.read_text())
        for neuron in raw.get("neurons", []):
            meta_lookup[str(neuron["bodyId"])] = neuron

    results: dict = {}
    path_strs = [str(p) for p in obj_paths]

    # --- Pass 1: scan bounds (Rust per-file, vertex-only, cached) ---
    bounds_cache = mesh_dir / "bounds.json"
    if bounds_cache.exists():
        bounds = tuple(json.loads(bounds_cache.read_text()))
        print(f"Using cached bounds from {bounds_cache}", flush=True)
    else:
        print(f"Pass 1: Scanning {len(obj_paths)} OBJ vertex bounds "
              f"(parallel rayon)...", flush=True)
        t0 = time.perf_counter()
        # scan_obj_bounds() rayon-parallelizes across the whole list internally
        # (streaming.rs:scan_bounds_parallel, GIL released). One call uses all
        # cores; the old per-file loop pinned it to one core.
        bounds = tuple(scan_obj_bounds(path_strs))
        bounds_cache.write_text(json.dumps(list(bounds)))
        print(f"  Bounds: x=[{bounds[0]:.0f}, {bounds[3]:.0f}] "
              f"y=[{bounds[1]:.0f}, {bounds[4]:.0f}] "
              f"z=[{bounds[2]:.0f}, {bounds[5]:.0f}]  "
              f"({_fmt_time(time.perf_counter() - t0)})", flush=True)

    # Partition files into giants (serial) and small (parallel batches)
    giants = []
    smalls = []
    for p in obj_paths:
        if p.stat().st_size > _GIANT_THRESHOLD:
            giants.append(p)
        else:
            smalls.append(p)
    print(f"Files: {len(giants)} giant (>{_GIANT_THRESHOLD // (1024*1024)} MB), "
          f"{len(smalls)} small → chunked ingest", flush=True)

    def _ingest_chunked(gen: StreamingTileGenerator) -> float:
        """Chunked ingest: giants one-by-one, smalls in parallel batches."""
        t0 = time.perf_counter()

        # Giant files: serial, one at a time
        for i, obj_path in enumerate(giants, 1):
            t1 = time.perf_counter()
            tags = _build_tags(obj_path, meta_lookup)
            gen.add_obj_file(str(obj_path), bounds, tags)
            dt = time.perf_counter() - t1
            sz_mb = obj_path.stat().st_size / (1024 * 1024)
            print(f"  [giant {i}/{len(giants)}] {obj_path.name} "
                  f"({sz_mb:.0f} MB, {_fmt_time(dt)})", flush=True)

        # Small files: parallel batch via add_obj_files
        if smalls:
            small_strs = [str(p) for p in smalls]
            small_tags = [_build_tags(p, meta_lookup) for p in smalls]
            print(f"  Ingesting {len(smalls)} small files (parallel rayon)...",
                  flush=True)
            t1 = time.perf_counter()
            gen.add_obj_files(small_strs, bounds, small_tags)
            dt = time.perf_counter() - t1
            print(f"  Small batch: {_fmt_time(dt)} "
                  f"({len(smalls) / dt:.0f} files/s)", flush=True)

        return time.perf_counter() - t0

    # Pyramid directory structure
    pyramid_dir = output_dir / pyramid_name

    # --- EMIT-ALL: one generator, ONE ingest, multiple outputs ---
    # WS-A (2026-05-31 scope): emit 3dtiles + parquet from a SINGLE ingest.
    # Each generate_* opens frag_dir read-only; only Drop deletes it
    # (streaming.rs), so reusing one generator across generate_3dtiles +
    # generate_parquet is byte-identical to two separate generators that each
    # ingest the same inputs (proven by tests/test_emit_all.py, WS-A A.1).
    # NG is DEFERRED and pbf3/feature_pbf3 are dropped this round (their Rust
    # generate_* implementations are kept untouched, just no longer called
    # here). skip_pbf3 is retained as a no-op param for caller compatibility.
    from mudm_tools.tiling3d.parquet_writer import generate_parquet as _gen_pq

    # run_dir must exist before ingest: the Rust ErrorCollector opens
    # <run_dir>/errors.jsonl on the first generate/ingest phase and does not
    # mkdir it (error_log.rs). With one ingest up front (vs the old per-format
    # blocks that each mkdir'd their output dir first), create it here.
    pyramid_dir.mkdir(parents=True, exist_ok=True)

    gen = StreamingTileGenerator(min_zoom=0, max_zoom=max_zoom, base_cells=100)
    gen._set_run_dir(str(pyramid_dir))

    # Ingest ONCE (measured once now; per-format generate is measured below).
    print(f"\nStreaming ingest (zoom 0-{max_zoom}, base_cells=100)...")
    t_index = _ingest_chunked(gen)
    print(f"  Ingest: {_fmt_time(t_index)}")

    # pbf3/feature_pbf3/neuroglancer no longer emitted this round.
    results["pbf3_tiles"] = None
    results["pbf3_dir"] = None
    results["feature_pbf3_features"] = None
    results["feature_pbf3_dir"] = None
    results["neuroglancer_features"] = None
    results["neuroglancer_dir"] = None

    # --- 3dtiles (optional) ---
    if not skip_3dtiles:
        tiles3d_dir = pyramid_dir / "3dtiles"
        if tiles3d_dir.exists():
            shutil.rmtree(tiles3d_dir)
        tiles3d_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating 3D Tiles...")
        t0 = time.perf_counter()
        n_tiles_3d = gen.generate_3dtiles(str(tiles3d_dir), bounds)
        t_gen_3d = time.perf_counter() - t0
        _surface_run_summary(pyramid_dir, "3dtiles")

        tiles3d_size = sum(f.stat().st_size for f in tiles3d_dir.rglob("*") if f.is_file())
        results["3dtiles_tiles"] = n_tiles_3d
        results["3dtiles_index_time"] = t_index
        results["3dtiles_gen_time"] = t_gen_3d
        results["3dtiles_size_raw"] = tiles3d_size
        results["3dtiles_size_gzip"] = 0
        results["3dtiles_dir"] = tiles3d_dir

        print(f"  {n_tiles_3d} tiles in {_fmt_time(t_gen_3d)}")
        print(f"  Size: {_fmt_bytes(tiles3d_size)} raw")
        if t_gen_3d > 0:
            print(f"  Throughput: {n_tiles_3d / t_gen_3d:.0f} tiles/s")

        # Build features.json for the viewer
        feat_index = tiles3d_dir / "features.json"
        print(f"\nBuilding features.json...")
        subprocess.run(
            [
                str(_ROOT / ".venv" / "bin" / "python"),
                str(_ROOT / "scripts" / "build_feature_index.py"),
                "--tiles-dir", str(tiles3d_dir),
                "--output", str(feat_index),
            ],
            check=True,
        )

    # --- parquet (always; reuses the same ingested generator) ---
    pq_path = pyramid_dir / "tiles.parquet"
    print(f"\nGenerating Parquet...")
    t0 = time.perf_counter()
    # partitioned=True -> native Rust consolidation: bounded memory (O(batch),
    # not O(all rows)) + the read∥transform overlap path. Output is a
    # `tiles.parquet/zoom=N/part_*.parquet` DIRECTORY, not a single file.
    # parallel=True + io_threads=12 is the measured consolidation knee on oden
    # (80 was worse) — do NOT change these.
    n_rows_pq = _gen_pq(
        gen, pq_path, bounds,
        partitioned=True, parallel=True, io_threads=12,
    )
    t_gen_pq = time.perf_counter() - t0
    _surface_run_summary(pyramid_dir, "parquet")

    # frag_dir removed only here (Drop) — after the LAST generate_* call.
    del gen

    if pq_path.is_dir():
        pq_size = sum(f.stat().st_size for f in pq_path.rglob("*") if f.is_file())
    else:
        pq_size = pq_path.stat().st_size if pq_path.exists() else 0
    results["parquet_rows"] = n_rows_pq
    results["parquet_index_time"] = t_index
    results["parquet_gen_time"] = t_gen_pq
    results["parquet_size_raw"] = pq_size
    results["parquet_path"] = pq_path

    print(f"  {n_rows_pq} rows in {_fmt_time(t_gen_pq)}")
    print(f"  Size: {_fmt_bytes(pq_size)}")

    return results


# ---------------------------------------------------------------------------
# Step 4 & 5: Tile + Benchmark (reuse from benchmark_mouselight)
# ---------------------------------------------------------------------------

def tile_and_benchmark(
    collection,
    output_dir: Path,
    *,
    max_zoom: int = 3,
    workers: int | None = None,
    do_tile: bool = True,
    do_benchmark: bool = True,
    skip_3dtiles: bool = False,
    csv_path: Path | None = None,
) -> dict:
    """Tile the collection and run benchmarks."""
    from benchmark_mouselight import (
        bench_decode,
        bench_decode_3dtiles,
        bench_memory,
        export_csv,
        generate_tiles,
        print_report,
    )

    results: dict = {}

    if do_tile:
        tile_results = generate_tiles(
            collection,
            output_dir,
            max_zoom=max_zoom,
            workers=workers,
            skip_3dtiles=skip_3dtiles,
        )
        results["tile"] = tile_results

    if do_benchmark:
        pbf3_dir = output_dir / "pbf3"
        tiles3d_dir = output_dir / "3dtiles" if not skip_3dtiles else None

        if pbf3_dir.exists():
            print(f"\nBenchmarking pbf3 decode...")
            decode_pbf3 = bench_decode(pbf3_dir)
            results["decode_pbf3"] = decode_pbf3
        else:
            decode_pbf3 = {}

        decode_3dt: dict = {}
        if tiles3d_dir and tiles3d_dir.exists():
            print(f"Benchmarking 3D Tiles decode...")
            decode_3dt = bench_decode_3dtiles(tiles3d_dir)
            results["decode_3dt"] = decode_3dt

        print("Measuring peak memory...")
        memory = bench_memory(
            pbf3_dir if pbf3_dir.exists() else Path("/dev/null"),
            tiles3d_dir if tiles3d_dir and tiles3d_dir.exists() else None,
        )
        results["memory"] = memory

        if do_tile:
            print_report(
                results.get("convert_time", 0),
                tile_results,
                decode_pbf3,
                decode_3dt,
                memory,
                None,
            )

        if csv_path:
            export_csv(
                csv_path,
                results.get("convert_time", 0),
                tile_results if do_tile else {},
                decode_pbf3,
                decode_3dt,
                memory,
                None,
            )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hemibrain v1.2.1 download, conversion, tiling, and benchmark",
    )
    parser.add_argument("--download", action="store_true", help="Download meshes from CloudVolume + metadata from neuPrint")
    parser.add_argument("--update-metadata", action="store_true", help="Re-query neuPrint for richer metadata on already-downloaded neurons")
    parser.add_argument("--convert", action="store_true", help="Convert OBJ to MuDM")
    parser.add_argument("--tile", action="store_true", help="Generate tiles (pbf3 + 3dtiles)")
    parser.add_argument("--benchmark", action="store_true", help="Run decode/memory benchmarks")
    parser.add_argument("--max-neurons", type=int, default=1000, help="Max neurons to download (default: 1000)")
    parser.add_argument("--min-type-count", type=int, default=0, help="Only download neurons whose cell type has >= N instances (default: 0 = no filter)")
    parser.add_argument("--max-zoom", type=int, default=3, help="Max zoom level (default: 3)")
    parser.add_argument("--workers", type=int, default=None, help="Worker processes")
    parser.add_argument("--skip-3dtiles", action="store_true", help="Skip 3D Tiles generation")
    parser.add_argument("--skip-pbf3", action="store_true", help="Skip pbf3 generation (3D Tiles only)")
    parser.add_argument("--streaming", action="store_true", help="True streaming mode: O(1 mesh) memory, skips MuDMFeatureCollection")
    parser.add_argument("--data-dir", type=Path, default=_DATA_DIR, help="Data directory")
    parser.add_argument("--csv", type=Path, default=None, help="Export results to CSV")
    args = parser.parse_args()

    if not any([args.download, args.update_metadata, args.convert, args.tile, args.benchmark]):
        parser.print_help()
        sys.exit(1)

    data_dir = args.data_dir
    mesh_dir = data_dir / "meshes"
    meta_path = data_dir / "metadata.json"
    tiles_dir = data_dir / "tiles"

    # --- Update metadata only ---
    if args.update_metadata:
        token = os.environ.get("NEUPRINT_TOKEN", "")
        if not token:
            print("ERROR: NEUPRINT_TOKEN not set.", file=sys.stderr)
            print("  Set it via: export NEUPRINT_TOKEN='your-token-here'")
            print("  Get token from https://neuprint.janelia.org → Account")
            sys.exit(1)
        update_metadata(mesh_dir, meta_path, token)
        if not any([args.download, args.convert, args.tile, args.benchmark]):
            print("Done.")
            return

    # --- Download ---
    if args.download:
        data_dir.mkdir(parents=True, exist_ok=True)

        token = os.environ.get("NEUPRINT_TOKEN", "")
        neurons: list[dict] = []

        if token:
            neurons = query_neuprint(token, max_neurons=args.max_neurons, min_type_count=args.min_type_count)

            # Save metadata
            meta_path.write_text(json.dumps(
                {"neurons": neurons, "dataset": _NEUPRINT_DATASET},
                indent=2,
            ))
            print(f"  Saved metadata to {meta_path}")
        else:
            print("WARNING: NEUPRINT_TOKEN not set. Downloading without metadata.")
            print("  Set it via: export NEUPRINT_TOKEN='your-token-here'")
            print("  Get token from https://neuprint.janelia.org → Account")

            if meta_path.exists():
                raw = json.loads(meta_path.read_text())
                neurons = raw.get("neurons", [])
                print(f"  Using existing metadata ({len(neurons)} neurons)")

        if not neurons:
            print("ERROR: No neuron IDs to download. Set NEUPRINT_TOKEN.", file=sys.stderr)
            sys.exit(1)

        body_ids = [n["bodyId"] for n in neurons]
        print(f"\nDownloading {len(body_ids)} neuron meshes...")
        t0 = time.perf_counter()
        downloaded = download_meshes(body_ids, mesh_dir, workers=args.workers)
        dl_time = time.perf_counter() - t0
        print(f"  Download time: {_fmt_time(dl_time)}")

        # Invalidate bounds cache so it gets recomputed with new meshes
        bounds_cache = mesh_dir / "bounds.json"
        if bounds_cache.exists():
            bounds_cache.unlink()
            print(f"  Invalidated bounds cache ({bounds_cache})")

    # --- Streaming mode (bypass MuDMFeatureCollection entirely) ---
    if args.streaming and args.tile:
        print("Mode: True Streaming (O(1 mesh) memory)")
        tile_streaming(
            mesh_dir,
            meta_path,
            tiles_dir,
            max_zoom=args.max_zoom,
            max_files=args.max_neurons,
            skip_3dtiles=args.skip_3dtiles,
            skip_pbf3=getattr(args, "skip_pbf3", False),
        )
        print("Done.")
        return

    # --- Convert ---
    collection = None
    convert_time = 0.0
    if args.convert:
        collection, convert_time = convert_to_microjson(
            mesh_dir, meta_path, max_files=args.max_neurons,
        )

    # --- Tile + Benchmark ---
    if args.tile or args.benchmark:
        if collection is None:
            print("Loading MuDM collection (convert step)...")
            collection, convert_time = convert_to_microjson(
                mesh_dir, meta_path, max_files=args.max_neurons,
            )

        tile_and_benchmark(
            collection,
            tiles_dir,
            max_zoom=args.max_zoom,
            workers=args.workers,
            do_tile=args.tile,
            do_benchmark=args.benchmark,
            skip_3dtiles=args.skip_3dtiles,
            csv_path=args.csv,
        )

    print("Done.")


if __name__ == "__main__":
    main()
