#!/usr/bin/env python3
"""Download HuBMAP Human Reference Atlas (HRA) organ GLB files for benchmarking.

Downloads 77 organ GLB files from the HuBMAP HRA API, converts to MuDM
TIN features with ASCT+B ontology metadata, tiles with TileGenerator3D,
and benchmarks.

Usage::

    # Full pipeline:
    .venv/bin/python scripts/download_hubmap.py --download --convert --tile --benchmark

    # Download only:
    .venv/bin/python scripts/download_hubmap.py --download

    # From existing GLB files:
    .venv/bin/python scripts/download_hubmap.py --convert --tile --benchmark
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

_DATA_DIR = _ROOT / "data" / "hubmap"
_GLB_DIR = _DATA_DIR / "glb"
_META_PATH = _DATA_DIR / "metadata.json"
_TILES_DIR = _DATA_DIR / "tiles"

_HRA_API = "https://apps.humanatlas.io/api/v1/reference-organs"


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


# ---------------------------------------------------------------------------
# Step 1: Download GLB files
# ---------------------------------------------------------------------------

def download_glb_files(output_dir: Path, sex_filter: str | None = None) -> list[dict]:
    """Download organ GLB files from HuBMAP HRA API.

    Returns list of organ metadata dicts.
    """
    import requests

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching HRA organ manifest...")
    resp = requests.get(_HRA_API, timeout=30)
    resp.raise_for_status()
    organs = resp.json()
    print(f"  Found {len(organs)} organ references")

    if sex_filter:
        organs = [o for o in organs if o.get("sex", "").lower() == sex_filter.lower()]
        print(f"  Filtered to {len(organs)} ({sex_filter})")

    downloaded = 0
    errors = 0
    metadata: list[dict] = []

    for i, organ in enumerate(organs, 1):
        label = organ.get("label", "unknown")
        sex = organ.get("sex", "unknown")
        glb_url = organ["object"]["file"]
        safe_name = f"{label.replace(' ', '_')}_{sex.lower()}"
        glb_path = output_dir / f"{safe_name}.glb"

        meta = {
            "id": organ.get("@id", ""),
            "label": label,
            "sex": sex,
            "file": glb_path.name,
            "uberon": organ.get("representation_of", ""),
            "x_dim_mm": float(organ.get("x_dimension", 0)),
            "y_dim_mm": float(organ.get("y_dimension", 0)),
            "z_dim_mm": float(organ.get("z_dimension", 0)),
            "creation_date": organ.get("creation_date", ""),
            "file_subpath": organ["object"].get("file_subpath", ""),
        }
        metadata.append(meta)

        if glb_path.exists():
            downloaded += 1
            if i % 20 == 0:
                print(f"  [{i}/{len(organs)}] {safe_name} — exists, skipping")
            continue

        try:
            r = requests.get(glb_url, timeout=60)
            r.raise_for_status()
            glb_path.write_bytes(r.content)
            downloaded += 1
            print(f"  [{i}/{len(organs)}] {safe_name} — {_fmt_bytes(len(r.content))}")
        except Exception as e:
            errors += 1
            print(f"  [{i}/{len(organs)}] {safe_name} — ERROR: {e}")

    print(f"  Downloaded {downloaded}/{len(organs)} GLB files ({errors} errors)")
    return metadata


# ---------------------------------------------------------------------------
# Step 2: Convert GLB → MuDM
# ---------------------------------------------------------------------------

def glb_to_mesh(glb_path: Path) -> tuple | None:
    """Extract vertices and faces from a GLB file.

    Returns (vertices_ndarray, faces_ndarray, node_names_list) or None.
    """
    import numpy as np

    try:
        import pygltflib
    except ImportError:
        print("ERROR: pygltflib required for GLB parsing", file=sys.stderr)
        sys.exit(1)

    try:
        gltf = pygltflib.GLTF2.load_from_bytes(glb_path.read_bytes())
    except Exception as e:
        print(f"  WARNING: Could not load {glb_path.name}: {e}")
        return None

    if not gltf.meshes:
        return None

    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    node_names: list[str] = []
    vert_offset = 0

    binary_blob = gltf.binary_blob()
    if binary_blob is None:
        return None

    for mesh_idx, mesh in enumerate(gltf.meshes):
        mesh_name = mesh.name or f"mesh_{mesh_idx}"

        for prim in mesh.primitives:
            # Get position accessor
            pos_idx = prim.attributes.POSITION
            if pos_idx is None:
                continue

            pos_accessor = gltf.accessors[pos_idx]
            pos_bv = gltf.bufferViews[pos_accessor.bufferView]

            # Read vertex positions
            offset = (pos_bv.byteOffset or 0) + (pos_accessor.byteOffset or 0)
            n_verts = pos_accessor.count
            vert_data = binary_blob[offset : offset + n_verts * 12]
            verts = np.frombuffer(vert_data, dtype=np.float32).reshape(-1, 3).copy()
            all_verts.append(verts)

            # Get indices
            if prim.indices is not None:
                idx_accessor = gltf.accessors[prim.indices]
                idx_bv = gltf.bufferViews[idx_accessor.bufferView]
                idx_offset = (idx_bv.byteOffset or 0) + (idx_accessor.byteOffset or 0)

                # Determine index type
                if idx_accessor.componentType == 5123:  # UNSIGNED_SHORT
                    dtype = np.uint16
                    stride = 2
                elif idx_accessor.componentType == 5125:  # UNSIGNED_INT
                    dtype = np.uint32
                    stride = 4
                else:
                    dtype = np.uint16
                    stride = 2

                idx_data = binary_blob[idx_offset : idx_offset + idx_accessor.count * stride]
                indices = np.frombuffer(idx_data, dtype=dtype).copy()
                faces = indices.reshape(-1, 3).astype(np.uint32) + vert_offset
                all_faces.append(faces)

            node_names.append(mesh_name)
            vert_offset += n_verts

    if not all_verts:
        return None

    vertices = np.vstack(all_verts).astype(np.float64)
    faces = np.vstack(all_faces).astype(np.uint32) if all_faces else np.zeros((0, 3), dtype=np.uint32)

    return vertices, faces, node_names


def convert_to_microjson(
    glb_dir: Path,
    metadata_path: Path,
    *,
    max_files: int | None = None,
):
    """Convert HuBMAP GLB files to MuDMFeatureCollection."""
    import numpy as np

    from mudm.model import (
        MuDMFeature,
        MuDMFeatureCollection,
        OntologyTerm,
        Vocabulary,
    )
    from mudm_tools.swc import _mesh_to_tin

    glb_paths = sorted(glb_dir.glob("*.glb"))
    if not glb_paths:
        print(f"ERROR: No .glb files in {glb_dir}", file=sys.stderr)
        sys.exit(1)

    if max_files and max_files < len(glb_paths):
        glb_paths = glb_paths[:max_files]

    # Load metadata
    meta_lookup: dict[str, dict] = {}
    if metadata_path.exists():
        raw = json.loads(metadata_path.read_text())
        for m in raw:
            meta_lookup[m["file"]] = m

    print(f"Converting {len(glb_paths)} GLB files to MuDM...")
    t0 = time.perf_counter()
    features: list[MuDMFeature] = []
    total_verts = 0
    total_faces = 0
    uberon_terms: dict[str, str] = {}  # uberon_id → label

    for i, glb_path in enumerate(glb_paths, 1):
        print(f"  [{i}/{len(glb_paths)}] {glb_path.name}", end="", flush=True)

        result = glb_to_mesh(glb_path)
        if result is None:
            print(" — SKIP (no mesh data)")
            continue

        vertices, faces, node_names = result
        if len(faces) == 0:
            print(" — SKIP (no faces)")
            continue

        tin = _mesh_to_tin(vertices, faces)

        stem = glb_path.stem
        meta = meta_lookup.get(glb_path.name, {})
        props: dict = {
            "organ": meta.get("label", stem),
            "sex": meta.get("sex", ""),
            "source": glb_path.name,
            "vertex_count": int(vertices.shape[0]),
            "face_count": int(faces.shape[0]),
        }

        if meta.get("uberon"):
            props["uberon_id"] = meta["uberon"]
            uberon_terms[meta["uberon"]] = meta.get("label", stem)

        if meta.get("x_dim_mm"):
            props["dimensions_mm"] = [
                meta["x_dim_mm"], meta["y_dim_mm"], meta["z_dim_mm"],
            ]

        feature_class = meta.get("label", stem)

        features.append(MuDMFeature(
            type="Feature",
            geometry=tin,
            properties=props,
            featureClass=feature_class,
        ))
        total_verts += vertices.shape[0]
        total_faces += faces.shape[0]
        print(f" — {vertices.shape[0]:,} verts, {faces.shape[0]:,} faces")

    convert_time = time.perf_counter() - t0

    # Build vocabulary from Uberon terms
    vocabs = None
    if uberon_terms:
        terms = {
            label: OntologyTerm(uri=uri, label=label)
            for uri, label in sorted(uberon_terms.items(), key=lambda x: x[1])
        }
        vocabs = {
            "uberon": Vocabulary(
                namespace="http://purl.obolibrary.org/obo/",
                description="Uberon multi-species anatomy ontology",
                terms=terms,
            ),
        }

    collection = MuDMFeatureCollection(
        type="FeatureCollection",
        features=features,
        properties={
            "dataset": "HuBMAP_HRA",
            "organ_count": len(features),
            "total_vertices": total_verts,
            "total_faces": total_faces,
        },
        vocabularies=vocabs,
    )

    print(f"\n  {len(features)} organs, {total_verts:,} vertices, {total_faces:,} faces")
    print(f"  {len(uberon_terms)} Uberon terms")
    print(f"  Conversion time: {_fmt_time(convert_time)}")

    return collection, convert_time


# ---------------------------------------------------------------------------
# Step 3 & 4: Tile + Benchmark
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

        decode_pbf3: dict = {}
        if pbf3_dir.exists():
            print(f"\nBenchmarking pbf3 decode...")
            decode_pbf3 = bench_decode(pbf3_dir)

        decode_3dt: dict = {}
        if tiles3d_dir and tiles3d_dir.exists():
            print(f"Benchmarking 3D Tiles decode...")
            decode_3dt = bench_decode_3dtiles(tiles3d_dir)

        print("Measuring peak memory...")
        memory = bench_memory(
            pbf3_dir if pbf3_dir.exists() else Path("/dev/null"),
            tiles3d_dir if tiles3d_dir and tiles3d_dir.exists() else None,
        )

        results["decode_pbf3"] = decode_pbf3
        results["decode_3dt"] = decode_3dt
        results["memory"] = memory

        if do_tile:
            print_report(0, tile_results, decode_pbf3, decode_3dt, memory, None)

        if csv_path:
            export_csv(csv_path, 0, tile_results if do_tile else {},
                       decode_pbf3, decode_3dt, memory, None)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HuBMAP HRA download, conversion, tiling, and benchmark",
    )
    parser.add_argument("--download", action="store_true", help="Download GLB files from HRA API")
    parser.add_argument("--convert", action="store_true", help="Convert GLB to MuDM")
    parser.add_argument("--tile", action="store_true", help="Generate tiles")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument("--sex", type=str, default=None, help="Filter by sex (Female/Male)")
    parser.add_argument("--max-zoom", type=int, default=3, help="Max zoom level")
    parser.add_argument("--workers", type=int, default=None, help="Worker processes")
    parser.add_argument("--skip-3dtiles", action="store_true", help="Skip 3D Tiles")
    parser.add_argument("--data-dir", type=Path, default=_DATA_DIR, help="Data directory")
    parser.add_argument("--csv", type=Path, default=None, help="Export CSV")
    args = parser.parse_args()

    if not any([args.download, args.convert, args.tile, args.benchmark]):
        parser.print_help()
        sys.exit(1)

    data_dir = args.data_dir
    glb_dir = data_dir / "glb"
    meta_path = data_dir / "metadata.json"
    tiles_dir = data_dir / "tiles"

    # --- Download ---
    if args.download:
        metadata = download_glb_files(glb_dir, sex_filter=args.sex)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(metadata, indent=2))
        print(f"  Saved metadata to {meta_path}")

    # --- Convert ---
    collection = None
    if args.convert or args.tile or args.benchmark:
        collection, convert_time = convert_to_microjson(glb_dir, meta_path)

    # --- Tile + Benchmark ---
    if args.tile or args.benchmark:
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
