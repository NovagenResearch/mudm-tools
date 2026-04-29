"""Batch driver for the cnn-nmo dataset.

Wraps swc_to_feature_collection to attach parentId + identity tags per
compartment feature, and classifies each neuron by source institution.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from mudm.model import MuDMFeature, MuDMFeatureCollection

from mudm_tools.swc import _mesh_to_tin, swc_to_feature_collection


SOURCE_HUST = "HUST"
SOURCE_ALLEN = "Allen"
SOURCE_SEU_ALLEN = "SEU-Allen"
SOURCE_UNKNOWN = "unknown"


def classify_source(note: Optional[str]) -> str:
    """Classify a NeuroMorpho ``note`` string into a source institution."""
    if not note:
        return SOURCE_UNKNOWN
    if "HUST-Suzhou Institute for Brainsmatics" in note:
        return SOURCE_HUST
    if "SEU-ALLEN" in note or "Southeast University" in note:
        return SOURCE_SEU_ALLEN
    if "Allen Institute for Brain Science" in note:
        return SOURCE_ALLEN
    return SOURCE_UNKNOWN


def build_feature_collection(
    swc_path: str | Path,
    *,
    neuron_id: int,
    neuron_name: str,
    archive: str,
    source: str,
    segments: int = 8,
    min_radius: float = 0.1,
    smooth_subdivisions: int = 3,
    mesh_quality: float = 1.0,
) -> MuDMFeatureCollection:
    """Build a per-compartment MuDMFeatureCollection with muDM parentId + identity tags.

    Each feature gets:
      - feature.parentId = neuron_id (rides through Rust via ``_parent_id`` tag)
      - feature.properties = {neuron_id, neuron_name, source, archive,
                              compartment}
    """
    coll = swc_to_feature_collection(
        str(swc_path),
        name=neuron_name,
        segments=segments,
        min_radius=min_radius,
        smooth_subdivisions=smooth_subdivisions,
        mesh_quality=mesh_quality,
    )
    for feat in coll.features:
        compartment = (feat.properties or {}).get("compartment", "unknown")
        feat.parentId = neuron_id
        feat.properties = {
            "neuron_id": neuron_id,
            "neuron_name": neuron_name,
            "source": source,
            "archive": archive,
            "compartment": compartment,
        }
    return coll


# ---------------------------------------------------------------------------
# run_build: end-to-end HUST-CCF scene builder
# ---------------------------------------------------------------------------

CCF_CANONICAL_BOUNDS = (0.0, 0.0, 0.0, 13200.0, 8000.0, 11400.0)
BOUNDS_PADDING_UM = 500.0


@dataclass
class _NeuronCtx:
    path: Path
    record: "NeuronRecord"  # noqa: F821 — forward ref to neuron_meta.NeuronRecord
    coll: MuDMFeatureCollection
    bbox: tuple[float, float, float, float, float, float]


def _git_sha() -> str:
    """Best-effort git short SHA for provenance; 'unknown' when not available."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _parse_obj_to_feature_collection(obj_path: Path) -> MuDMFeatureCollection:
    """Parse a simple OBJ file into a single-feature MuDMFeatureCollection.

    Minimal standalone parser (duplicates ``scripts/obj_to_mudm.parse_obj``
    to avoid a cross-imports from ``scripts/``).
    """
    import numpy as np

    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    with open(obj_path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            tag = parts[0]
            if tag == "v" and len(parts) >= 4:
                vertices.append(
                    [float(parts[1]), float(parts[2]), float(parts[3])]
                )
            elif tag == "f" and len(parts) >= 4:
                face_verts: list[int] = []
                for tok in parts[1:]:
                    # Handle f v, f v/t, f v//n, f v/t/n
                    face_verts.append(int(tok.split("/")[0]) - 1)  # OBJ 1-based
                for i in range(1, len(face_verts) - 1):
                    faces.append([face_verts[0], face_verts[i], face_verts[i + 1]])

    if not vertices or not faces:
        raise ValueError(f"OBJ at {obj_path} has no usable geometry")

    verts = np.asarray(vertices, dtype=np.float64)
    idx = np.asarray(faces, dtype=np.uint32)
    tin = _mesh_to_tin(verts, idx)

    feat = MuDMFeature(
        type="Feature",
        geometry=tin,
        properties={"source": "obj", "mesh_name": Path(obj_path).stem},
        featureClass=Path(obj_path).stem,
    )
    return MuDMFeatureCollection(
        type="FeatureCollection",
        features=[feat],
        properties={"name": Path(obj_path).stem},
    )


def _compute_bounds(
    neurons: list[_NeuronCtx],
    cortex_bbox: tuple[float, float, float, float, float, float] | None,
) -> tuple[float, float, float, float, float, float]:
    """Union of CCF canonical bounds + every HUST neuron bbox + cortex bbox, padded."""
    return _compute_bounds_from_bboxes([n.bbox for n in neurons], cortex_bbox)


def _compute_bounds_from_bboxes(
    hust_bboxes: list[tuple[float, float, float, float, float, float]],
    cortex_bbox: tuple[float, float, float, float, float, float] | None,
) -> tuple[float, float, float, float, float, float]:
    """Same as _compute_bounds but takes raw bbox tuples (no _NeuronCtx)."""
    min_x, min_y, min_z, max_x, max_y, max_z = CCF_CANONICAL_BOUNDS
    boxes = list(hust_bboxes)
    if cortex_bbox:
        boxes.append(cortex_bbox)
    for b in boxes:
        min_x = min(min_x, b[0])
        min_y = min(min_y, b[1])
        min_z = min(min_z, b[2])
        max_x = max(max_x, b[3])
        max_y = max(max_y, b[4])
        max_z = max(max_z, b[5])
    p = BOUNDS_PADDING_UM
    return (
        min_x - p, min_y - p, min_z - p,
        max_x + p, max_y + p, max_z + p,
    )


def _mesh_bbox(
    coll: MuDMFeatureCollection,
) -> tuple[float, float, float, float, float, float]:
    """Bounding box over every coordinate in a MuDMFeatureCollection (TIN/PolyhedralSurface)."""
    min_x = min_y = min_z = float("inf")
    max_x = max_y = max_z = float("-inf")
    for f in coll.features:
        geom = f.geometry
        if geom is None or not getattr(geom, "coordinates", None):
            continue
        for face in geom.coordinates:
            for ring in face:
                for p in ring:
                    x, y, z = p[0], p[1], p[2]
                    if x < min_x:
                        min_x = x
                    if y < min_y:
                        min_y = y
                    if z < min_z:
                        min_z = z
                    if x > max_x:
                        max_x = x
                    if y > max_y:
                        max_y = y
                    if z > max_z:
                        max_z = z
    if min_x == float("inf"):
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return (min_x, min_y, min_z, max_x, max_y, max_z)


def _log_ccf_overruns(neurons: list[_NeuronCtx]) -> None:
    """Log each HUST neuron whose bbox exceeds the canonical CCF extent."""
    cmx, cmy, cmz, cMx, cMy, cMz = CCF_CANONICAL_BOUNDS
    for n in neurons:
        b = n.bbox
        overruns = []
        if b[0] < cmx:
            overruns.append(("x_min", cmx - b[0]))
        if b[1] < cmy:
            overruns.append(("y_min", cmy - b[1]))
        if b[2] < cmz:
            overruns.append(("z_min", cmz - b[2]))
        if b[3] > cMx:
            overruns.append(("x_max", b[3] - cMx))
        if b[4] > cMy:
            overruns.append(("y_max", b[4] - cMy))
        if b[5] > cMz:
            overruns.append(("z_max", b[5] - cMz))
        if overruns:
            pieces = ", ".join(
                f"{axis}:+{delta:.1f}µm" for axis, delta in overruns
            )
            print(
                f"[bounds] HUST neuron {n.record.neuron_id} "
                f"({n.record.neuron_name}) exceeds canonical CCF: {pieces}"
            )


def _log_ccf_overruns_from_records(records: list) -> None:
    """Log HUST neurons whose SWC bbox exceeds canonical CCF extent.

    Reads swc_bbox_* fields on NeuronRecord (no _NeuronCtx dependency).
    """
    cmx, cmy, cmz, cMx, cMy, cMz = CCF_CANONICAL_BOUNDS
    for r in records:
        if r.source != SOURCE_HUST:
            continue
        b = (r.swc_bbox_min_x, r.swc_bbox_min_y, r.swc_bbox_min_z,
             r.swc_bbox_max_x, r.swc_bbox_max_y, r.swc_bbox_max_z)
        overruns = []
        if b[0] < cmx:
            overruns.append(("x_min", cmx - b[0]))
        if b[1] < cmy:
            overruns.append(("y_min", cmy - b[1]))
        if b[2] < cmz:
            overruns.append(("z_min", cmz - b[2]))
        if b[3] > cMx:
            overruns.append(("x_max", b[3] - cMx))
        if b[4] > cMy:
            overruns.append(("y_max", b[4] - cMy))
        if b[5] > cMz:
            overruns.append(("z_max", b[5] - cMz))
        if overruns:
            pieces = ", ".join(
                f"{axis}:+{delta:.1f}µm" for axis, delta in overruns
            )
            print(
                f"[bounds] HUST neuron {r.neuron_id} ({r.neuron_name}) "
                f"exceeds canonical CCF: {pieces}",
                flush=True,
            )


def run_aux_only(
    *,
    data_dir: Path,
    tiles_dir: Path,
    cortex_obj: Path | None,
    scene_id: str = "hust-ccf",
) -> None:
    """Re-emit the three auxiliary JSONs (tilejson3d, features, pyramids)
    against an EXISTING tileset.json, without re-running the (10+ hour)
    FC build + Rust tiling.

    Reads the existing sidecar to know which neurons are HUST, then derives
    per-compartment world-space bboxes directly from raw SWC samples grouped
    by SWC type. This is cheap (~20 ms/neuron) and gives a slightly looser
    bbox than the mesh bbox (mesh bbox is centerline ± radius; this is
    sample positions ± max_radius), which only makes feature→tile lists a
    little more conservative — benign for the feature-index use case.
    """
    import datetime as _dt
    from mudm_tools.neuron_meta import load_sidecar
    from mudm_tools.swc import _parse_swc, SWC_SOMA, SWC_AXON, \
        SWC_BASAL_DENDRITE, SWC_APICAL_DENDRITE, SWC_TYPE_NAMES
    from mudm_tools.tiling3d.features_index import (
        write_features_index as _write_features_index,
    )

    data_dir = Path(data_dir)
    tiles_dir = Path(tiles_dir)
    scene_dir = tiles_dir / scene_id
    tdtiles_dir = scene_dir / "3dtiles"
    tileset_path = tdtiles_dir / "tileset.json"
    sidecar_path = tiles_dir / "neurons_meta.parquet"

    if not tileset_path.exists():
        raise FileNotFoundError(f"missing {tileset_path} — run the full build first")
    if not sidecar_path.exists():
        raise FileNotFoundError(f"missing {sidecar_path}")

    records = load_sidecar(sidecar_path)
    hust_records = [r for r in records.values() if r.source == SOURCE_HUST]
    print(f"[aux] {len(hust_records)} HUST neurons in sidecar", flush=True)

    # Derive per-compartment bboxes per neuron from raw SWC samples.
    features_catalog: dict[str, dict] = {}
    for i, r in enumerate(hust_records, 1):
        swc = data_dir / f"{r.neuron_name}.swc"
        if not swc.exists():
            print(f"[aux] {r.neuron_name}: SWC not found, skipping", flush=True)
            continue
        morph = _parse_swc(str(swc))
        by_type: dict[int, list] = {}
        for s in morph.tree:
            by_type.setdefault(s.type, []).append(s)
        max_r = max((s.r for s in morph.tree), default=1.0)
        for t, samples in by_type.items():
            if not samples:
                continue
            xs = [s.x for s in samples]; ys = [s.y for s in samples]; zs = [s.z for s in samples]
            bbox = (
                min(xs) - max_r, min(ys) - max_r, min(zs) - max_r,
                max(xs) + max_r, max(ys) + max_r, max(zs) + max_r,
            )
            compartment = SWC_TYPE_NAMES.get(t, f"type_{t}")
            key = f"{r.neuron_name}/{compartment}"
            properties = {
                "neuron_id": r.neuron_id,
                "neuron_name": r.neuron_name,
                "source": r.source,
                "archive": r.archive or "",
                "compartment": compartment,
                "species": r.species or "",
                "brain_region": r.brain_region_flat,
                "cell_type": r.cell_type_flat,
            }
            if r.surface_m is not None: properties["surface_m"] = r.surface_m
            if r.volume_m  is not None: properties["volume_m"]  = r.volume_m
            if r.length    is not None: properties["length"]    = r.length
            if r.n_bifs    is not None: properties["n_bifs"]    = r.n_bifs
            if r.n_branch  is not None: properties["n_branch"]  = r.n_branch
            features_catalog[key] = {"bbox": bbox, "properties": properties}
        if i % 20 == 0:
            print(f"[aux] {i}/{len(hust_records)} neurons cataloged", flush=True)

    # Cortex is rendered as a standalone overlay mesh loaded directly by
    # the viewer (see viewer3d/js/main.js::loadAtlasOverlay). We do NOT
    # declare it as a tiled feature here: doing so fragments the surface
    # across octree tiles. The cortex mesh nodes baked into existing
    # tiles remain as inert ballast — harmless because no feature selects
    # them so they stay at visible=false.

    print(f"[aux] {len(features_catalog)} feature entries total", flush=True)

    # features.json
    _write_features_index(tileset_path, features_catalog, tdtiles_dir / "features.json")
    print(f"[aux] wrote {tdtiles_dir / 'features.json'}", flush=True)

    # tilejson3d.json — match the shape the Rust write_tilejson3d produces.
    # We don't have a StreamingTileGenerator instance here; write directly.
    tileset = json.loads(tileset_path.read_text())
    root_bv = (tileset.get("root", {}).get("boundingVolume") or {}).get("box", [])
    bounds = None
    if len(root_bv) >= 12:
        cx, cy, cz = root_bv[0], root_bv[1], root_bv[2]
        hx = abs(root_bv[3]); hy = abs(root_bv[7]); hz = abs(root_bv[11])
        bounds = (cx - hx, cy - hy, cz - hz, cx + hx, cy + hy, cz + hz)
    else:
        bounds = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)

    # Infer max zoom from existing tileset tile URIs.
    max_zoom = 0
    def _scan_max_zoom(node):
        nonlocal max_zoom
        content = node.get("content")
        if content and "uri" in content:
            try:
                z = int(content["uri"].split("/", 1)[0])
                if z > max_zoom:
                    max_zoom = z
            except Exception:
                pass
        for c in node.get("children", []) or []:
            _scan_max_zoom(c)
    _scan_max_zoom(tileset.get("root", {}))

    tj = {
        "tilejson": "3.0.0",
        "tiles": ["{z}/{x}/{y}/{d}"],
        "name": "default",
        "minzoom": 0,
        "maxzoom": max_zoom,
        "bounds3d": list(bounds),
        "center3d": [
            (bounds[0] + bounds[3]) / 2.0,
            (bounds[1] + bounds[4]) / 2.0,
            (bounds[2] + bounds[5]) / 2.0,
            0.0,
        ],
        "vector_layers": [{
            "id": "default", "fields": {},
            "minzoom": 0, "maxzoom": max_zoom,
        }],
        # id_fields here is the "skip from color-by" list. We leave it empty
        # so every per-feature attribute (compartment, source, cell_type,
        # surface_m, etc.) is selectable in the Color By dropdown.
        "id_fields": [],
    }
    (tdtiles_dir / "tilejson3d.json").write_text(json.dumps(tj, indent=2))
    print(f"[aux] wrote {tdtiles_dir / 'tilejson3d.json'}", flush=True)

    # pyramids.json
    _update_pyramids_manifest(
        tiles_dir / "pyramids.json",
        scene_id=scene_id,
        label=f"HUST-Suzhou ({len(hust_records)} neurons) + CCF Cortex",
        n_features=len(features_catalog),
        max_zoom=max_zoom,
        tiles_dir=tdtiles_dir,
    )
    print(f"[aux] wrote {tiles_dir / 'pyramids.json'}", flush=True)


def _update_pyramids_manifest(
    manifest_path: Path,
    *,
    scene_id: str,
    label: str,
    n_features: int,
    max_zoom: int,
    tiles_dir: Path,
) -> None:
    """Update (or create) the top-level ``pyramids.json`` manifest.

    This file is read by the existing viewer's ``PyramidSelector``. We
    replace any existing entry with id==scene_id so re-running the build is
    idempotent; other pyramids (from previous runs or unrelated datasets
    sharing this tiles root) are preserved.
    """
    import os as _os

    tile_count = 0
    size_bytes = 0
    if tiles_dir.exists():
        for root, _, files in _os.walk(tiles_dir):
            for f in files:
                if f.endswith(".glb"):
                    tile_count += 1
                    size_bytes += (Path(root) / f).stat().st_size

    existing: list[dict] = []
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text())
            existing = [p for p in data.get("pyramids", []) if p.get("id") != scene_id]
        except Exception:
            existing = []

    existing.append({
        "id": scene_id,
        "label": label,
        "tiles": tile_count,
        "features": n_features,
        "max_zoom": max_zoom,
        "size_bytes": size_bytes,
    })
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"pyramids": existing}, indent=2)
    )


def _add_tin_feature(
    gen,
    feat: MuDMFeature,
    world_bounds: tuple[float, float, float, float, float, float],
) -> None:
    """Project a MuDMFeature(TIN) into [0,1]³ and feed the Rust generator.

    StreamingTileGenerator.add_feature expects normalized coordinates
    (xy in [0,1]², z in [0,1]) and a normalized per-feature bbox — see
    ``tests/test_tiling3d_parquet.py::_make_tin_feature`` and
    ``tests/test_tiling3d_3dtiles.py::test_parentid_extras_in_glb`` for
    the canonical shape. We unproject via the world_bounds supplied to
    generate_*.
    """
    geom = feat.geometry
    assert geom is not None and geom.type in ("TIN", "PolyhedralSurface")

    xmin, ymin, zmin, xmax, ymax, zmax = world_bounds
    dx = (xmax - xmin) if xmax != xmin else 1.0
    dy = (ymax - ymin) if ymax != ymin else 1.0
    dz = (zmax - zmin) if zmax != zmin else 1.0

    positions: list[float] = []
    positions_z: list[float] = []
    ring_lengths: list[int] = []
    fmin_x = fmin_y = fmin_z = float("inf")
    fmax_x = fmax_y = fmax_z = float("-inf")
    for face in geom.coordinates:
        for ring in face:
            ring_lengths.append(len(ring))
            for p in ring:
                wx, wy, wz = p[0], p[1], p[2]
                nx = (wx - xmin) / dx
                ny = (wy - ymin) / dy
                nz = (wz - zmin) / dz
                positions.append(nx)
                positions.append(ny)
                positions_z.append(nz)
                if nx < fmin_x:
                    fmin_x = nx
                if ny < fmin_y:
                    fmin_y = ny
                if nz < fmin_z:
                    fmin_z = nz
                if nx > fmax_x:
                    fmax_x = nx
                if ny > fmax_y:
                    fmax_y = ny
                if nz > fmax_z:
                    fmax_z = nz

    if not positions:
        return

    d: dict = {
        "geometry": positions,
        "geometry_z": positions_z,
        "type": 5,  # TIN
        "ring_lengths": ring_lengths,
        "minX": fmin_x, "minY": fmin_y, "minZ": fmin_z,
        "maxX": fmax_x, "maxY": fmax_y, "maxZ": fmax_z,
        "tags": dict(feat.properties or {}),
    }
    if feat.parentId is not None:
        d["parentId"] = feat.parentId
    if feat.id is not None:
        d["id"] = feat.id
    if getattr(feat, "featureClass", None) is not None:
        d["featureClass"] = feat.featureClass
    gen.add_feature(d)


def _write_skeletons(
    scene_dir: Path,
    hust_work: list[tuple[Path, int, str, str]],
    records_by_id: dict[int, "NeuronRecord"],
) -> None:
    """Write a Neuroglancer precomputed *skeleton* dataset for HUST neurons.

    Uses mudm_tools.neuroglancer.skeleton_writer (skeletons are authoritative
    for SWC data; generate_neuroglancer_multilod emits coarse meshes instead).
    Output: {scene_dir}/neuroglancer/skeletons/{info, <seg_id>, segment_properties/info}.

    hust_work is the per-neuron tuple list (path, id, name, archive) from
    run_build's pass 1. Re-parsing each SWC here is cheap (~20 ms/file).
    records_by_id maps neuron_id → NeuronRecord for segment property lookups.
    """
    from mudm_tools.neuroglancer.skeleton_writer import (
        build_skeleton_info,
        neuron_to_skeleton_binary,
    )
    from mudm_tools.neuroglancer.properties_writer import (
        write_segment_properties,
    )
    from mudm_tools.swc import _parse_swc

    sk_dir = scene_dir / "neuroglancer" / "skeletons"
    sk_dir.mkdir(parents=True, exist_ok=True)

    info = build_skeleton_info(
        include_radius=True,
        include_type=True,
        segment_properties="segment_properties",
    )
    (sk_dir / "info").write_text(
        json.dumps(info.to_info_dict(), indent=2)
    )

    # Build MuDMFeatures for segment_properties (one per neuron)
    prop_features: list[MuDMFeature] = []
    seg_ids: list[int] = []
    for path, neuron_id, _name, _archive in hust_work:
        morph = _parse_swc(str(path))
        binary = neuron_to_skeleton_binary(
            morph, include_radius=True, include_type=True,
        )
        (sk_dir / str(neuron_id)).write_bytes(binary)
        seg_ids.append(neuron_id)
        rec = records_by_id.get(neuron_id)
        if rec is None:
            continue
        prop_features.append(
            MuDMFeature(
                type="Feature",
                geometry=None,
                properties={
                    "neuron_id": rec.neuron_id,
                    "neuron_name": rec.neuron_name,
                    "source": rec.source,
                    "archive": rec.archive or "",
                    "brain_region": rec.brain_region_flat,
                    "cell_type": rec.cell_type_flat,
                },
            )
        )

    if prop_features:
        write_segment_properties(
            sk_dir / "segment_properties", prop_features, seg_ids,
        )


def run_build(
    *,
    data_dir: Path,
    tiles_dir: Path,
    cortex_obj: Path | None,
    scene_id: str = "hust-ccf",
) -> None:
    """Run the full cnn-nmo pipeline end-to-end.

    - Reads data_dir/*.swc + data_dir/metadata.json.
    - Builds per-neuron feature collections (parentId + tags).
    - Partitions by source.
    - For HUST: builds the hust-ccf scene (3D Tiles + Neuroglancer + Parquet).
    - Writes neurons_meta.parquet (all neurons, tiled or not).
    - Writes scene_manifest.json.
    """
    import datetime as _dt

    from mudm_tools._rs import StreamingTileGenerator
    from mudm_tools.neuron_meta import NeuronRecord, write_sidecar
    from mudm_tools.tiling3d.features_index import (
        write_features_index as _write_features_index,
    )
    from mudm_tools.swc import _parse_swc
    from mudm_tools.tiling3d.parquet_writer import (
        generate_parquet as _generate_parquet,
    )

    data_dir = Path(data_dir)
    tiles_dir = Path(tiles_dir)

    metadata_path = data_dir / "metadata.json"
    meta = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}

    iso_now = _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds")
    git_sha = _git_sha()

    records: list[NeuronRecord] = []
    # hust_work: per-neuron tuple used by pass 2 (FC streaming) and skeleton
    # writer. We do NOT hold feature collections here — that was the
    # ~35 GB memory issue in the batched design.
    hust_work: list[tuple[Path, int, str, str]] = []
    errors: list[dict] = []
    seen_ids: set[int] = set()

    all_swcs = sorted(data_dir.glob("*.swc"))
    total = len(all_swcs)

    # ==================================================================
    # Pass 1: parse every SWC, write sidecar records. Feature collection
    # generation is deferred to pass 2 (HUST only) so we don't hold 136
    # full meshes in memory at once.
    # ==================================================================

    print(f"[pass 1] parsing {total} SWCs + building sidecar records", flush=True)

    for i, swc in enumerate(all_swcs, 1):
        entry = meta.get(swc.name, {})
        neuron_md = entry.get("neuron") or {}
        morph_md = entry.get("morphometry") or {}

        note = neuron_md.get("note") or ""
        source = classify_source(note)
        neuron_id = int(neuron_md.get("neuron_id") or 0)
        neuron_name = neuron_md.get("neuron_name") or swc.stem

        if neuron_id == 0:
            msg = "missing neuron_id (got 0)"
            print(f"[{i:>3}/{total}] {swc.name}: skip ({msg})", flush=True)
            errors.append({"file": swc.name, "error": msg})
            continue
        if neuron_id in seen_ids:
            msg = f"duplicate neuron_id {neuron_id}"
            print(f"[{i:>3}/{total}] {swc.name}: skip ({msg})", flush=True)
            errors.append({"file": swc.name, "error": msg})
            continue
        seen_ids.add(neuron_id)

        # Always parse the SWC (fast — ~20ms) to populate sidecar fields.
        try:
            morph = _parse_swc(str(swc))
        except Exception as e:
            print(f"[{i:>3}/{total}] {swc.name}: parse failed ({e})", flush=True)
            errors.append({"file": swc.name, "error": f"parse: {e}"})
            continue

        n_nodes = len(morph.tree)
        n_compartments = len({s.type for s in morph.tree})
        raw_bbox = (
            min(s.x for s in morph.tree), min(s.y for s in morph.tree),
            min(s.z for s in morph.tree), max(s.x for s in morph.tree),
            max(s.y for s in morph.tree), max(s.z for s in morph.tree),
        )

        # Pass 1 uses the raw SWC bbox for every neuron (HUST or not). The
        # mesh-surface bbox for HUST is ≤1 µm different (clamped min_radius),
        # well within the 500 µm bounds padding, so raw is fine for scene
        # bounds and sidecar.
        bbox = raw_bbox
        print(
            f"[{i:>3}/{total}] {swc.name}: {source} ({n_nodes} nodes)",
            flush=True,
        )

        soma_ccf = None
        if source == SOURCE_HUST:
            root = next((s for s in morph.tree if s.parent == -1), None)
            if root is not None:
                soma_ccf = (root.x, root.y, root.z)

        record = NeuronRecord(
            neuron_id=neuron_id,
            neuron_name=neuron_name,
            archive=neuron_md.get("archive"),
            age_scale=neuron_md.get("age_scale"),
            gender=neuron_md.get("gender"),
            age_classification=neuron_md.get("age_classification"),
            species=neuron_md.get("species"),
            strain=neuron_md.get("strain"),
            scientific_name=neuron_md.get("scientific_name"),
            stain=neuron_md.get("stain"),
            protocol=neuron_md.get("protocol"),
            slicing_direction=neuron_md.get("slicing_direction"),
            reconstruction_software=neuron_md.get("reconstruction_software"),
            objective_type=neuron_md.get("objective_type"),
            original_format=neuron_md.get("original_format"),
            domain=neuron_md.get("domain"),
            attributes=neuron_md.get("attributes"),
            magnification=neuron_md.get("magnification"),
            upload_date=neuron_md.get("upload_date"),
            deposition_date=neuron_md.get("deposition_date"),
            shrinkage_reported=neuron_md.get("shrinkage_reported"),
            shrinkage_corrected=neuron_md.get("shrinkage_corrected"),
            slicing_thickness=neuron_md.get("slicing_thickness"),
            min_age=neuron_md.get("min_age"),
            max_age=neuron_md.get("max_age"),
            min_weight=neuron_md.get("min_weight"),
            max_weight=neuron_md.get("max_weight"),
            png_url=neuron_md.get("png_url"),
            physical_integrity=(
                neuron_md.get("physical_Integrity")
                or neuron_md.get("physical_integrity")
            ),
            note=note,
            brain_region=list(neuron_md.get("brain_region") or []),
            cell_type=list(neuron_md.get("cell_type") or []),
            experiment_condition=list(neuron_md.get("experiment_condition") or []),
            reference_pmid=list(neuron_md.get("reference_pmid") or []),
            reference_doi=list(neuron_md.get("reference_doi") or []),
            reported_value=neuron_md.get("reported_value"),
            reported_xy=neuron_md.get("reported_xy"),
            reported_z=neuron_md.get("reported_z"),
            corrected_value=neuron_md.get("corrected_value"),
            corrected_xy=neuron_md.get("corrected_xy"),
            corrected_z=neuron_md.get("corrected_z"),
            soma_surface=neuron_md.get("soma_surface"),
            surface=neuron_md.get("surface"),
            volume=neuron_md.get("volume"),
            surface_m=morph_md.get("surface"),
            volume_m=morph_md.get("volume"),
            length=morph_md.get("length"),
            n_stems=morph_md.get("n_stems"),
            n_bifs=morph_md.get("n_bifs"),
            n_branch=morph_md.get("n_branch"),
            width=morph_md.get("width"),
            height=morph_md.get("height"),
            depth=morph_md.get("depth"),
            diameter=morph_md.get("diameter"),
            eucDistance=morph_md.get("eucDistance"),
            pathDistance=morph_md.get("pathDistance"),
            branch_Order=morph_md.get("branch_Order"),
            contraction=morph_md.get("contraction"),
            fragmentation=morph_md.get("fragmentation"),
            partition_asymmetry=morph_md.get("partition_asymmetry"),
            pk_classic=morph_md.get("pk_classic"),
            bif_ampl_local=morph_md.get("bif_ampl_local"),
            bif_ampl_remote=morph_md.get("bif_ampl_remote"),
            fractal_Dim=morph_md.get("fractal_Dim"),
            n_nodes_m=morph_md.get("n_nodes") or morph_md.get("n_nodes_m"),
            soma_Surface_m=morph_md.get("soma_Surface"),
            neuron_name_m=morph_md.get("neuron_name"),
            source=source,
            in_ccf_frame=(source == SOURCE_HUST),
            n_nodes_swc=n_nodes,
            n_compartments_swc=n_compartments,
            swc_bbox_min_x=bbox[0],
            swc_bbox_min_y=bbox[1],
            swc_bbox_min_z=bbox[2],
            swc_bbox_max_x=bbox[3],
            swc_bbox_max_y=bbox[4],
            swc_bbox_max_z=bbox[5],
            soma_ccf_x=(soma_ccf[0] if soma_ccf else None),
            soma_ccf_y=(soma_ccf[1] if soma_ccf else None),
            soma_ccf_z=(soma_ccf[2] if soma_ccf else None),
            brain_region_flat="/".join(neuron_md.get("brain_region") or []),
            cell_type_flat="/".join(neuron_md.get("cell_type") or []),
            ingest_date=iso_now,
            ingest_git_sha=git_sha,
        )
        records.append(record)

        if source == SOURCE_HUST:
            hust_work.append(
                (swc, neuron_id, neuron_name, neuron_md.get("archive") or "")
            )

    tiles_dir.mkdir(parents=True, exist_ok=True)
    write_sidecar(records, tiles_dir / "neurons_meta.parquet")
    print(
        f"[pass 1] done: {len(records)} records, {len(hust_work)} HUST to tile, "
        f"{len(errors)} errors",
        flush=True,
    )

    # Nothing to tile — write errors and bail. The downstream tile writers
    # (StreamingTileGenerator.generate_3dtiles, etc.) require at least one
    # feature; running them with an empty hust_work raises OSError.
    if not hust_work:
        if errors:
            (tiles_dir / "build_errors.json").write_text(
                json.dumps(errors, indent=2)
            )
        return

    scene_dir = tiles_dir / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    cortex_bbox = None
    cortex_coll: MuDMFeatureCollection | None = None
    if cortex_obj and Path(cortex_obj).exists():
        cortex_coll = _parse_obj_to_feature_collection(Path(cortex_obj))
        for feat in cortex_coll.features:
            feat.parentId = "ccf_isocortex"
            feat.properties = {
                "source": "ccf_atlas",
                "compartment": "cortex",
                "neuron_id": "ccf_isocortex",
                "neuron_name": "CCF Isocortex",
                "archive": "AllenCCFv3",
                # Per-feature styling the viewer honors (data-driven, not
                # baked into the GLB): translucent atlas envelope at α=0.2
                # so neurons inside it remain visible.
                "opacity": 0.2,
                "color": "#a8b5c4",
            }
        cortex_bbox = _mesh_bbox(cortex_coll)

    # Bounds from HUST raw SWC bboxes (sidecar) + cortex + CCF canonical + pad.
    hust_bboxes = [
        (r.swc_bbox_min_x, r.swc_bbox_min_y, r.swc_bbox_min_z,
         r.swc_bbox_max_x, r.swc_bbox_max_y, r.swc_bbox_max_z)
        for r in records if r.source == SOURCE_HUST
    ]
    bounds = _compute_bounds_from_bboxes(hust_bboxes, cortex_bbox)
    _log_ccf_overruns_from_records(records)

    # StreamingTileGenerator: default constructor defaults are fine for the
    # cnn-nmo scene — see tests/test_tiling3d_parquet.py::_build_generator_with_features.
    # Features must arrive in normalized [0,1]^3 space; _add_tin_feature projects.
    MIN_ZOOM = 0
    MAX_ZOOM = 4
    gen = StreamingTileGenerator(min_zoom=MIN_ZOOM, max_zoom=MAX_ZOOM)

    # ==================================================================
    # Pass 2: build FC per HUST neuron, STREAM features to the Rust gen,
    # release. This keeps Python memory bounded at ~one FC at a time.
    # smooth_subdivisions=1 (not the default 3) reduces tube point density
    # by ~3x with acceptable visual quality — tube surfaces stay smooth.
    # ==================================================================

    print(f"[pass 2] building + streaming {len(hust_work)} HUST FCs", flush=True)

    # features_catalog: per-compartment feature catalog keyed by
    # "{neuron_name}/{compartment}". Used to emit features.json after
    # generate_3dtiles. Records world-space bbox + viewer-facing properties
    # so the existing ColorBy/FeatureSelector/InfoPanel can use per-compartment
    # granularity.
    features_catalog: dict[str, dict] = {}
    records_by_id = {r.neuron_id: r for r in records}

    def _catalog(feat: MuDMFeature) -> None:
        """Record a world-space feature in features_catalog."""
        geom = feat.geometry
        if geom is None or not getattr(geom, "coordinates", None):
            return
        min_x = min_y = min_z = float("inf")
        max_x = max_y = max_z = float("-inf")
        for face in geom.coordinates:
            for ring in face:
                for p in ring:
                    x, y, z = p[0], p[1], p[2]
                    if x < min_x: min_x = x
                    if y < min_y: min_y = y
                    if z < min_z: min_z = z
                    if x > max_x: max_x = x
                    if y > max_y: max_y = y
                    if z > max_z: max_z = z
        if min_x == float("inf"):
            return

        props = dict(feat.properties or {})
        nid = props.get("neuron_id")
        nm = props.get("neuron_name", "")
        compartment = props.get("compartment", "unknown")

        # Enrich properties from the sidecar record for better color-by /
        # InfoPanel UX. Skip for atlas features (nid is a string like
        # "ccf_isocortex" — no sidecar record).
        rec = records_by_id.get(nid) if isinstance(nid, int) else None
        if rec is not None:
            props.setdefault("species", rec.species or "")
            props.setdefault("brain_region", rec.brain_region_flat)
            props.setdefault("cell_type", rec.cell_type_flat)
            if rec.surface_m is not None: props.setdefault("surface_m", rec.surface_m)
            if rec.volume_m  is not None: props.setdefault("volume_m",  rec.volume_m)
            if rec.length    is not None: props.setdefault("length",    rec.length)
            if rec.n_bifs    is not None: props.setdefault("n_bifs",    rec.n_bifs)
            if rec.n_branch  is not None: props.setdefault("n_branch",  rec.n_branch)

        key = f"{nm}/{compartment}" if nm else f"{nid}/{compartment}"
        features_catalog[key] = {
            "bbox": (min_x, min_y, min_z, max_x, max_y, max_z),
            "properties": props,
        }

    for j, (swc, nid, nm, archive) in enumerate(hust_work, 1):
        t_fc0 = _dt.datetime.now()
        try:
            coll = build_feature_collection(
                swc_path=swc,
                neuron_id=nid,
                neuron_name=nm,
                archive=archive,
                source=SOURCE_HUST,
                smooth_subdivisions=1,
            )
        except Exception as e:
            print(f"[fc {j}/{len(hust_work)}] {swc.name}: FC failed ({e})",
                  flush=True)
            errors.append({"file": swc.name, "error": f"FC: {e}"})
            continue
        fc_secs = (_dt.datetime.now() - t_fc0).total_seconds()
        n_feats = len(coll.features)
        for f in coll.features:
            _catalog(f)
            _add_tin_feature(gen, f, bounds)
        print(
            f"[fc {j:>3}/{len(hust_work)}] {swc.name}: "
            f"{n_feats} feats in {fc_secs:.1f}s",
            flush=True,
        )
        # Drop the Python FC so its meshes can be GC'd before the next one
        del coll

    if cortex_coll:
        print(f"[fc +] cortex: {len(cortex_coll.features)} feats", flush=True)
        for f in cortex_coll.features:
            _catalog(f)
            _add_tin_feature(gen, f, bounds)
        # Keep cortex_coll alive — only used once, GC'd when run_build returns

    # 3D Tiles — Rust: (output_dir, world_bounds, layer_name?, compression, ...)
    tdtiles_dir = scene_dir / "3dtiles"
    gen.generate_3dtiles(
        str(tdtiles_dir),
        bounds,
        "default",
        "meshopt",
    )

    # tilejson3d.json — Rust emits canonical shape given world bounds + ids.
    # id_fields is the "skip from Color By" list in the viewer; we leave it
    # empty so every attribute (compartment, cell_type, surface_m, …) stays
    # colorable.
    gen.write_tilejson3d(
        str(tdtiles_dir / "tilejson3d.json"),
        bounds,
        "default",
        None,
        None,
        [],
    )

    # features.json — derived by intersecting per-compartment bboxes with
    # the emitted octree tiles (conservative: no false negatives).
    _write_features_index(
        tdtiles_dir / "tileset.json",
        features_catalog,
        tdtiles_dir / "features.json",
    )

    # Neuroglancer: precomputed skeletons at {scene}/neuroglancer/skeletons/.
    _write_skeletons(scene_dir, hust_work, records_by_id)

    # Parquet — module-level helper in tiling3d.parquet_writer.
    _generate_parquet(gen, scene_dir / "geom.parquet", bounds)

    # pyramids.json — top-level manifest; the existing PyramidSelector reads
    # this. One scene == one pyramid for cnn-nmo.
    _update_pyramids_manifest(
        tiles_dir / "pyramids.json",
        scene_id=scene_id,
        label=f"HUST-Suzhou ({len(hust_work)} neurons) + CCF Cortex",
        n_features=len(features_catalog),
        max_zoom=MAX_ZOOM,
        tiles_dir=tdtiles_dir,
    )

    if errors:
        (tiles_dir / "build_errors.json").write_text(
            json.dumps(errors, indent=2)
        )
