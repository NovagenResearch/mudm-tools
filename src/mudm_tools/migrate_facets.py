"""Migration-scoped transcoder: re-emit lean 2D tiles + a facet store from a
dataset's already-tiled ``features.parquet`` — without re-downloading the source.

**Chosen path: B (strip-tags-from-PBF), at the protobuf wire level.**

A naive Path B (``mapbox_vector_tile.decode`` -> drop ``m_*`` -> ``mapbox_vector_tile.encode``)
is *not* byte-faithful: the shapely-backed encoder dedupes redundant consecutive
vertices and re-winds rings, so on real tiles it silently changes the decoded
geometry (53/84 mibi-uterus tiles fail the fidelity gate even with
``check_winding_order=False``). Path A (encoding tile-local muDM binary geometry
straight to MVT) would re-derive the same command stream the original tiler
already produced — strictly more risk for zero benefit.

So we operate directly on the compiled MVT protobuf (``vector_tile_pb2``): parse
the tile, rebuild each layer's ``keys`` / ``values`` / ``features[].tags`` to drop
the faceted keys (and rename the join key — see below), and leave every
``feature.geometry`` command-integer list, ``feature.type``, ``feature.id`` and
``layer.extent`` completely untouched. The geometry is therefore byte-faithful by
construction; the fidelity gate (decode before+after, geometry signatures
identical) passes exactly.

The faceted marker set is **resolved ONCE** (apply the policy — include/exclude
globs and the explicit ``markers`` names, modality-agnostic: ``m_*`` for CODEX,
``g_*`` for CosMx/merscope — to the ACTUAL tag keys present) and that one set
drives BOTH the facet build AND the tile strip, so every faceted key is stripped
and nothing un-faceted is. The join value is read from the ``policy.key`` tag
(which may be ``cell`` or ``cell_id``) and NORMALIZED to ``cell_id`` in both the
facet store and the rewritten tiles, so the viewer always joins on ``cell_id``.
The attributes are extracted from ONLY the all-cells ``zoom=0`` partition (every
cell appears once there) — a ~7x memory cut over scanning all zoom partitions.

Public entry point:
    ``tile_from_features_parquet(dataset_dir, policy, *, markers) -> dict``
which rewrites ``dataset_dir/vectors/**/*.pbf`` (faceted keys dropped, join key
renamed to ``cell_id``), writes ``dataset_dir/facets/markers.parquet`` (via
:func:`mudm_tools.facets.emit_facet_store`), patches ``dataset_dir/metadata.json``
(adds the ``facets`` block with ``key="cell_id"``; bumps any layer whose
``id == "transcripts"`` ``min_zoom`` by +1), and returns a fidelity report
``{"tiles", "geometry_ok", "max_coord_delta"}``.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as pds

from mapbox_vector_tile import decode as _mvt_decode
from mapbox_vector_tile.Mapbox import vector_tile_pb2 as _vt

from .facets import FacetPolicy, _matches, emit_facet_store

_GZIP_MAGIC = b"\x1f\x8b"


# --------------------------------------------------------------------------- #
# PBF (de)compression helpers — preserve the original on-disk encoding.
# --------------------------------------------------------------------------- #
def _read_pbf(path: Path) -> tuple[bytes, bool]:
    """Return (protobuf_bytes, was_gzipped)."""
    raw = path.read_bytes()
    if raw[:2] == _GZIP_MAGIC:
        return gzip.decompress(raw), True
    return raw, False


def _write_pbf(path: Path, pbf: bytes, gzipped: bool) -> None:
    path.write_bytes(gzip.compress(pbf) if gzipped else pbf)


def _geom_signature(pbf: bytes) -> list[str]:
    """Order-independent signature of every feature's decoded geometry."""
    t = _mvt_decode(pbf)
    return sorted(
        json.dumps(f["geometry"], sort_keys=True) for layer in t.values() for f in layer["features"]
    )


# --------------------------------------------------------------------------- #
# Path B: protobuf-level tag strip (geometry untouched).
# --------------------------------------------------------------------------- #
def _strip_tags(pbf: bytes, drop_key, rename: dict[str, str] | None = None) -> bytes:
    """Drop tags whose key matches ``drop_key`` from every feature in every layer,
    optionally renaming surviving keys via ``rename`` (e.g. ``{"cell": "cell_id"}``).

    Rebuilds each layer's ``keys`` / ``values`` string tables (compacted to only
    the survivors, with renames applied) and rewrites ``feature.tags`` accordingly.
    ``feature.geometry`` is never touched, so decoded geometry is byte-faithful.

    A layer is rewritten if it has any faceted key to drop OR any key to rename;
    otherwise its bytes are left exactly as-is.
    """
    rename = rename or {}
    tile = _vt.tile()
    tile.ParseFromString(pbf)
    for layer in tile.layers:
        old_keys = list(layer.keys)
        old_vals = list(layer.values)

        drop_key_idx = {i for i, k in enumerate(old_keys) if drop_key(k)}
        has_rename = any(k in rename for k in old_keys)
        if not drop_key_idx and not has_rename:
            continue  # nothing faceted or renamed in this layer; leave bytes as-is

        # Compact the surviving key table (applying renames to the surviving names).
        key_remap: dict[int, int] = {}
        new_keys: list[str] = []
        for i, k in enumerate(old_keys):
            if i in drop_key_idx:
                continue
            key_remap[i] = len(new_keys)
            new_keys.append(rename.get(k, k))

        # Collect value indices still referenced by surviving tags, then compact.
        used_val: set[int] = set()
        for feat in layer.features:
            tags = list(feat.tags)
            for ki, vi in zip(tags[::2], tags[1::2]):
                if ki not in drop_key_idx:
                    used_val.add(vi)
        val_remap: dict[int, int] = {}
        new_vals = []
        for vi in sorted(used_val):
            val_remap[vi] = len(new_vals)
            new_vals.append(old_vals[vi])

        # Rewrite each feature's tag list with the remapped indices.
        for feat in layer.features:
            tags = list(feat.tags)
            new_tags: list[int] = []
            for ki, vi in zip(tags[::2], tags[1::2]):
                if ki in drop_key_idx:
                    continue
                new_tags.append(key_remap[ki])
                new_tags.append(val_remap[vi])
            del feat.tags[:]
            feat.tags.extend(new_tags)

        del layer.keys[:]
        layer.keys.extend(new_keys)
        del layer.values[:]
        layer.values.extend(new_vals)

    return tile.SerializeToString()


# --------------------------------------------------------------------------- #
# Read the all-cells (zoom=0) partition once: tag rows + the actual tag keys.
# --------------------------------------------------------------------------- #
def _read_all_cells_partition(parquet_dir: Path) -> tuple[list[dict], list[str]]:
    """Return ``(rows, tag_keys)`` from ONLY the ``zoom=0`` partition.

    ``zoom=0`` is the all-cells level: every cell appears exactly once there, so
    we never need to scan (and de-dup across) all zoom partitions. This is the
    MEMORY FIX — loading the full tags map across every zoom previously peaked at
    ~13.6 GB; reading just ``zoom=0`` cuts that ~7x.

    ``tag_keys`` is the union of keys observed across the partition's rows (the
    ACTUAL keys present, against which the facet policy is resolved).
    """
    dataset = pds.dataset(str(parquet_dir), format="parquet", partitioning="hive")
    # Restrict to the all-cells partition when the dataset is zoom-partitioned.
    if "zoom" in dataset.schema.names:
        table = dataset.to_table(columns=["tags"], filter=pds.field("zoom") == 0)
    else:  # not partitioned (e.g. a hand-built single-file fixture)
        table = dataset.to_table(columns=["tags"])

    rows = [dict(r) for r in table.column("tags").to_pylist()]
    tag_keys: list[str] = []
    seen_keys: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen_keys:
                seen_keys.add(k)
                tag_keys.append(k)
    return rows, tag_keys


# --------------------------------------------------------------------------- #
# Resolve the faceted marker set ONCE — used for BOTH the facet build and the
# tile strip, so every faceted key is stripped and nothing else is.
# --------------------------------------------------------------------------- #
def _resolve_facet_keys(policy: FacetPolicy, tag_keys: list[str], markers: list[str]) -> list[str]:
    """Apply the policy (include/exclude globs, keep_inline, key) to the ACTUAL
    tag keys present and return the single, authoritative faceted-key set.

    A key is faceted iff it is neither the join key nor kept inline nor excluded,
    AND it matches the include set. The include set is the policy's include globs
    UNIONED with the explicit ``markers`` names (exact, modality-agnostic: ``m_*``
    for CODEX, ``g_*`` for CosMx/merscope) — the caller passes both and they must
    agree, so we resolve against their union for robustness.
    """
    include = list(policy.include) + list(markers)
    resolved: list[str] = []
    for k in tag_keys:
        if k == policy.key or k in policy.keep_inline:
            continue
        if policy.exclude and _matches(k, policy.exclude):
            continue
        if k in markers or _matches(k, include):
            resolved.append(k)
    return resolved


# --------------------------------------------------------------------------- #
# Facet attributes from the all-cells partition (muDM tile-geometry `tags` map).
# --------------------------------------------------------------------------- #
def _facet_attrs_from_rows(
    rows: list[dict], policy: FacetPolicy, facet_keys: list[str]
) -> tuple[list[str], dict[str, np.ndarray]]:
    """From the all-cells rows return ``(cell_ids, {marker: float32 array})``
    aligned by cell, reading the join value from the ``policy.key`` tag.

    The join value (which may live under ``cell`` or ``cell_id``) becomes the
    NORMALIZED ``cell_id`` column written by :func:`emit_facet_store`. The
    ``zoom=0`` partition already holds each cell once; we still guard against
    accidental duplicates by keeping the first occurrence.
    """
    key = policy.key  # "cell_id" or "cell"
    cell_ids: list[str] = []
    seen: set[str] = set()
    values: dict[str, list[float]] = {m: [] for m in facet_keys}

    for tagd in rows:
        cid = tagd.get(key)
        if cid is None or cid in seen:
            continue
        seen.add(cid)
        cell_ids.append(str(cid))
        for m in facet_keys:
            raw = tagd.get(m, "")
            try:
                values[m].append(float(raw) if raw != "" else 0.0)
            except (TypeError, ValueError):
                values[m].append(0.0)

    attrs = {m: np.asarray(values[m], dtype="float32") for m in facet_keys}
    return cell_ids, attrs


def _patch_metadata(meta_path: Path, facets_block: dict) -> dict:
    """Insert the ``facets`` block and bump any ``transcripts`` layer min_zoom."""
    meta = json.loads(meta_path.read_text())
    meta["facets"] = facets_block
    vectors = meta.get("vectors") or {}
    for layer in vectors.get("layers", []):
        if layer.get("id") == "transcripts" and "min_zoom" in layer:
            layer["min_zoom"] = int(layer["min_zoom"]) + 1
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


# --------------------------------------------------------------------------- #
# Public entry point.
# --------------------------------------------------------------------------- #
def tile_from_features_parquet(
    dataset_dir: str, policy: FacetPolicy, *, markers: list[str]
) -> dict:
    """Re-emit lean tiles + a wide marker facet store for an already-tiled dataset.

    Rewrites ``<dataset_dir>/vectors/**/*.pbf`` (faceted keys dropped), writes
    ``<dataset_dir>/facets/markers.parquet``, patches ``metadata.json`` (adds the
    ``facets`` block; bumps any ``transcripts`` layer ``min_zoom`` by +1), and
    returns a fidelity report ``{"tiles", "geometry_ok", "max_coord_delta"}``.
    """
    ds = Path(dataset_dir)
    parquet_dir = ds / "parquet"
    if not parquet_dir.exists():
        parquet_dir = ds / "features.parquet"

    # 0) Read the all-cells (zoom=0) partition ONCE (memory fix) and learn the
    #    actual tag keys present.
    rows, tag_keys = _read_all_cells_partition(parquet_dir)

    # 1) Resolve the faceted marker set ONCE — the single source of truth for
    #    BOTH the facet build and the tile strip.
    facet_keys = _resolve_facet_keys(policy, tag_keys, markers)
    facet_set = set(facet_keys)

    # 2) Build the facet store from the all-cells rows. The join value is read
    #    from the policy.key tag (``cell`` or ``cell_id``); emit_facet_store
    #    always writes it as a NORMALIZED ``cell_id`` column.
    cell_ids, attrs = _facet_attrs_from_rows(rows, policy, facet_keys)
    facets_block = emit_facet_store(ds, cell_ids, attrs, policy)
    # The facets-block key is normalized to cell_id (matches the column above).
    facets_block["key"] = "cell_id"
    for asset in facets_block.get("assets", []):
        asset["key"] = "cell_id"

    # 3) Rewrite the tiles using the SAME resolved set: drop exactly the faceted
    #    keys, and rename the join key to ``cell_id`` so the viewer joins
    #    uniformly (no-op rename when policy.key is already ``cell_id``).
    rename = {} if policy.key == "cell_id" else {policy.key: "cell_id"}

    def drop_key(k: str) -> bool:
        return k in facet_set

    pbf_paths = sorted((ds / "vectors").rglob("*.pbf"))
    geometry_ok = True
    max_coord_delta = 0
    for p in pbf_paths:
        pbf, gz = _read_pbf(p)
        before_sig = _geom_signature(pbf)
        out = _strip_tags(pbf, drop_key, rename=rename)
        after_sig = _geom_signature(out)
        if before_sig != after_sig:
            geometry_ok = False
            max_coord_delta = max(max_coord_delta, 1)
        _write_pbf(p, out, gz)

    # 4) Patch metadata.json (facets block + transcript min_zoom bump).
    _patch_metadata(ds / "metadata.json", facets_block)

    return {
        "tiles": len(pbf_paths),
        "geometry_ok": geometry_ok,
        "max_coord_delta": int(max_coord_delta),
    }
