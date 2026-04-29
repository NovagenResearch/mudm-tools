#!/usr/bin/env python3
"""Inject a ``name`` field into each GLB node's ``extras`` to match what
the viewer's ``TileManager._loadTile`` expects.

Context
-------
The cnn-nmo build emits GLB nodes with extras like
``{neuron_name, compartment, neuron_id, source, archive, _parent_id}``.
The viewer's mesh→feature matcher at
``src/mudm_tools/viewers/viewer3d/js/TileManager.js:669-672`` reads
``props.name / props.acronym / props.instance / props.body_id`` in that
order and expects one of them to equal a key in ``features.json``.

Our features.json keys are ``{neuron_name}/{compartment}`` (plus
``CCF Isocortex/cortex`` for the atlas). This script rewrites each GLB
to add ``extras.name`` with the matching key so the viewer can match.

It's a post-processing fix. For future builds, update
``build_feature_collection`` to write ``name`` into properties directly;
then this script is never needed again.

Usage
-----
    uv run python scripts/patch_glb_name_extras.py \\
        --tiles-dir data/cnn-nmo/tiles/hust-ccf/3dtiles

GLB format recap
----------------
12-byte header: magic (4) + version (4) + total length (4)
Then ≥1 chunks: length (4) + type (4) + payload (padded to 4-byte boundary).
The first chunk is JSON (type 0x4E4F534A); the second, if present, is BIN
(type 0x004E4942). We only rewrite the JSON chunk's payload; the BIN chunk
is copied verbatim. Byte offsets inside the JSON that reference binary
buffers (bufferViews etc.) remain valid because we don't touch the BIN chunk.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

GLB_MAGIC = 0x46546C67           # "glTF"
JSON_CHUNK_TYPE = 0x4E4F534A     # "JSON"
BIN_CHUNK_TYPE = 0x004E4942      # "BIN\0"


def _read_chunks(data: bytes) -> tuple[dict, bytes, bytes]:
    """Parse a GLB into (json_obj, bin_chunk_raw, trailing).

    bin_chunk_raw includes length+type header + padded payload so it can be
    appended verbatim. trailing is anything after the last recognized chunk
    (should be empty for well-formed GLBs).
    """
    if len(data) < 12 or struct.unpack_from("<I", data, 0)[0] != GLB_MAGIC:
        raise ValueError("not a GLB: bad magic")
    version = struct.unpack_from("<I", data, 4)[0]
    if version != 2:
        raise ValueError(f"GLB version {version} unsupported")

    pos = 12
    json_obj = None
    bin_raw = b""
    while pos + 8 <= len(data):
        length, ctype = struct.unpack_from("<II", data, pos)
        chunk_start = pos + 8
        chunk_end = chunk_start + length
        if chunk_end > len(data):
            raise ValueError("chunk overflow")
        payload = data[chunk_start:chunk_end]
        if ctype == JSON_CHUNK_TYPE:
            text = payload.rstrip(b"\x20").rstrip(b"\x00")
            json_obj = json.loads(text)
        elif ctype == BIN_CHUNK_TYPE:
            bin_raw = data[pos:chunk_end]
        # Ignore unknown chunks per glTF spec.
        pos = chunk_end
    if json_obj is None:
        raise ValueError("GLB has no JSON chunk")
    trailing = data[pos:] if pos < len(data) else b""
    return json_obj, bin_raw, trailing


def _compose_name(extras: dict) -> str | None:
    """Compose a ``name`` matching our features.json key convention."""
    if not extras:
        return None
    # Atlas features use a specific composite key.
    if extras.get("source") == "ccf_atlas":
        nn = extras.get("neuron_name") or "CCF Isocortex"
        c = extras.get("compartment") or "cortex"
        return f"{nn}/{c}"
    nn = extras.get("neuron_name")
    c = extras.get("compartment")
    if nn and c:
        return f"{nn}/{c}"
    return None


def _patch_nodes(json_obj: dict) -> int:
    """Inject ``extras.name`` on every node where we can compose one.

    Returns the number of nodes modified.
    """
    n_patched = 0
    for node in json_obj.get("nodes", []) or []:
        extras = node.get("extras")
        if not isinstance(extras, dict):
            continue
        if "name" in extras and extras["name"]:
            # Already has a name — don't overwrite
            continue
        name = _compose_name(extras)
        if name is None:
            continue
        extras["name"] = name
        n_patched += 1
    return n_patched


def _pack_glb(json_obj: dict, bin_raw: bytes) -> bytes:
    """Rebuild a GLB from a patched JSON dict + untouched BIN chunk bytes."""
    # Serialize JSON as compact UTF-8, pad to 4-byte boundary with spaces.
    json_bytes = json.dumps(json_obj, separators=(",", ":")).encode("utf-8")
    pad = (4 - (len(json_bytes) % 4)) % 4
    json_payload = json_bytes + b"\x20" * pad

    # Assemble chunks
    json_chunk = struct.pack("<II", len(json_payload), JSON_CHUNK_TYPE) + json_payload
    total = 12 + len(json_chunk) + len(bin_raw)

    header = struct.pack("<III", GLB_MAGIC, 2, total)
    return header + json_chunk + bin_raw


def patch_file(path: str) -> tuple[str, int, int] | tuple[str, str]:
    """Worker: patch a single GLB in place.

    Returns (path, n_nodes, n_patched) on success, or (path, error_str).
    """
    try:
        p = Path(path)
        data = p.read_bytes()
        json_obj, bin_raw, trailing = _read_chunks(data)
        if trailing:
            return (path, f"trailing {len(trailing)} bytes after last chunk")
        n_nodes = len(json_obj.get("nodes", []) or [])
        n_patched = _patch_nodes(json_obj)
        if n_patched == 0:
            return (path, n_nodes, 0)
        new_bytes = _pack_glb(json_obj, bin_raw)
        # Atomic write: tmp + rename
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(new_bytes)
        tmp.replace(p)
        return (path, n_nodes, n_patched)
    except Exception as e:
        return (path, f"{type(e).__name__}: {e}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tiles-dir",
        type=Path,
        default=Path("data/cnn-nmo/tiles/hust-ccf/3dtiles"),
        help="Directory containing <z>/<x>/<y>/<d>.glb files",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of worker processes (default: 8)",
    )
    args = parser.parse_args()

    glbs = sorted(str(p) for p in args.tiles_dir.rglob("*.glb"))
    print(f"Patching {len(glbs)} GLB files in {args.tiles_dir}")

    ok = failed = 0
    total_nodes = total_patched = 0
    errors: list[tuple[str, str]] = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(patch_file, g) for g in glbs]
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if len(res) == 3:
                path, n_nodes, n_patched = res
                ok += 1
                total_nodes += n_nodes
                total_patched += n_patched
            else:
                path, err = res
                failed += 1
                errors.append((path, err))
            if i % 100 == 0 or i == len(glbs):
                print(f"  {i}/{len(glbs)} processed (ok={ok}, failed={failed})",
                      flush=True)

    print()
    print(f"Total GLBs : {len(glbs)}")
    print(f"OK         : {ok}")
    print(f"Failed     : {failed}")
    print(f"Nodes seen : {total_nodes}")
    print(f"Names added: {total_patched}")
    if errors:
        print("\nFirst 5 errors:")
        for p, e in errors[:5]:
            print(f"  {p}: {e}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
