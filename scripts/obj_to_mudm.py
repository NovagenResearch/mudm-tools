#!/usr/bin/env python3
"""Convert OBJ triangle meshes to MuDM TIN features with Allen CCF metadata.

Parses OBJ files (vertices + faces), converts to MuDM TIN geometry,
optionally fetches Allen CCF ontology from API and matches OBJ filenames
to brain regions. Outputs a MuDMFeatureCollection as JSON.

Usage:
    .venv/bin/python scripts/obj_to_microjson.py [OBJ_DIR_OR_FILES...] [OPTIONS]

Options:
    -o OUTPUT          Output file path (default: stdout)
    --ontology PATH    Local Allen CCF JSON (skip API fetch)
    --no-ontology      Skip ontology lookup entirely

Examples:
    # Single OBJ to stdout
    .venv/bin/python scripts/obj_to_microjson.py meshes/brain_region.obj

    # Directory of OBJs with ontology
    .venv/bin/python scripts/obj_to_microjson.py meshes/ -o brain.json

    # Without ontology lookup
    .venv/bin/python scripts/obj_to_microjson.py meshes/*.obj --no-ontology -o out.json
"""

import json
import sys
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

from mudm.model import (
    MuDMFeature,
    MuDMFeatureCollection,
    OntologyTerm,
    Vocabulary,
)
from mudm_tools.swc import _mesh_to_tin


# ---------------------------------------------------------------------------
# OBJ Parser
# ---------------------------------------------------------------------------

def parse_obj(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse an OBJ file into vertices and face indices.

    Args:
        path: Path to the OBJ file.

    Returns:
        Tuple of (vertices [N,3] float64, faces [M,3] uint32).
    """
    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            prefix = parts[0]

            if prefix == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

            elif prefix == "f":
                # Handle f v, f v/t, f v//n, f v/t/n
                face_verts = []
                for token in parts[1:]:
                    vi = token.split("/")[0]
                    face_verts.append(int(vi) - 1)  # OBJ is 1-based

                # Fan-triangulate if >3 vertices (quads, n-gons)
                for i in range(1, len(face_verts) - 1):
                    faces.append([face_verts[0], face_verts[i], face_verts[i + 1]])

    if not vertices:
        raise ValueError(f"No vertices found in {path}")
    if not faces:
        raise ValueError(f"No faces found in {path}")

    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.uint32)


# ---------------------------------------------------------------------------
# Allen CCF Ontology
# ---------------------------------------------------------------------------

_ALLEN_API_URL = "http://api.brain-map.org/api/v2/structure_graph_download/1.json"
_CACHE_DIR = Path.home() / ".cache" / "mudm"
_CACHE_FILE = _CACHE_DIR / "allen_ccf_ontology.json"


def _flatten_structures(node: dict, out: dict) -> None:
    """Recursively flatten the Allen CCF structure tree."""
    entry = {
        "id": node["id"],
        "name": node["name"],
        "acronym": node["acronym"],
        "parent_structure_id": node.get("parent_structure_id"),
        "structure_id_path": node.get("structure_id_path", ""),
        "color_hex_triplet": node.get("color_hex_triplet", ""),
    }
    out[node["name"].lower()] = entry
    out[node["acronym"].lower()] = entry
    out[str(node["id"])] = entry
    for child in node.get("children", []):
        _flatten_structures(child, out)


def fetch_allen_ontology(local_path: Optional[str] = None) -> dict[str, dict]:
    """Fetch or load the Allen CCF ontology as a flat lookup dict.

    Keys are lowercase name, lowercase acronym, and string ID.

    Args:
        local_path: If provided, load from this file instead of API/cache.

    Returns:
        Flat dict mapping keys to region info dicts.
    """
    if local_path:
        raw = json.loads(Path(local_path).read_text())
    elif _CACHE_FILE.exists():
        raw = json.loads(_CACHE_FILE.read_text())
    else:
        print("Fetching Allen CCF ontology from API...", file=sys.stderr)
        with urllib.request.urlopen(_ALLEN_API_URL, timeout=30) as resp:
            raw = json.loads(resp.read().decode())
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps(raw))
        print(f"Cached ontology to {_CACHE_FILE}", file=sys.stderr)

    # The API returns {"success": true, "id": 0, "msg": [...]}
    # msg[0] is the root structure node
    if isinstance(raw, dict) and "msg" in raw:
        roots = raw["msg"]
    elif isinstance(raw, list):
        roots = raw
    else:
        raise ValueError("Unexpected Allen CCF JSON structure")

    flat: dict[str, dict] = {}
    for root in roots:
        _flatten_structures(root, flat)
    return flat


# ---------------------------------------------------------------------------
# Region Matching
# ---------------------------------------------------------------------------

def match_region(filename_stem: str, ontology: dict[str, dict]) -> Optional[dict]:
    """Match an OBJ filename stem to an Allen CCF region.

    Tries: exact name (lowercased), exact acronym (lowercased),
    numeric part as Allen CCF ID.

    Returns:
        Region info dict or None.
    """
    key = filename_stem.lower()

    # Exact match on name or acronym
    if key in ontology:
        return ontology[key]

    # Try numeric part as CCF ID
    numeric = "".join(c for c in filename_stem if c.isdigit())
    if numeric and numeric in ontology:
        return ontology[numeric]

    return None


# ---------------------------------------------------------------------------
# Feature Builder
# ---------------------------------------------------------------------------

def obj_to_feature(
    obj_path: str,
    ontology: Optional[dict[str, dict]] = None,
) -> MuDMFeature:
    """Convert an OBJ file to a MuDMFeature with TIN geometry.

    Args:
        obj_path: Path to the OBJ file.
        ontology: Allen CCF ontology lookup (or None to skip).

    Returns:
        MuDMFeature with TIN geometry and properties.
    """
    vertices, faces = parse_obj(obj_path)
    tin = _mesh_to_tin(vertices, faces)

    stem = Path(obj_path).stem
    props: dict = {
        "mesh_name": stem,
        "source": Path(obj_path).name,
        "vertex_count": int(vertices.shape[0]),
        "face_count": int(faces.shape[0]),
    }
    feature_class = stem

    if ontology is not None:
        region = match_region(stem, ontology)
        if region:
            props["ccf_id"] = region["id"]
            props["name"] = region["name"]
            props["acronym"] = region["acronym"]
            if region["parent_structure_id"] is not None:
                props["parent_id"] = region["parent_structure_id"]
            if region["color_hex_triplet"]:
                props["color"] = f"#{region['color_hex_triplet']}"
            if region["structure_id_path"]:
                props["hierarchy_path"] = region["structure_id_path"]
            feature_class = region["acronym"]

    return MuDMFeature(
        type="Feature",
        geometry=tin,
        properties=props,
        featureClass=feature_class,
    )


# ---------------------------------------------------------------------------
# Collection + Vocabulary Builder
# ---------------------------------------------------------------------------

def build_collection(
    features: list[MuDMFeature],
    ontology: Optional[dict[str, dict]] = None,
) -> MuDMFeatureCollection:
    """Build a MuDMFeatureCollection with optional Allen CCF vocabulary.

    Args:
        features: List of MuDMFeature objects.
        ontology: Allen CCF ontology lookup (or None to skip vocabulary).

    Returns:
        MuDMFeatureCollection with vocabulary and summary properties.
    """
    total_verts = sum(f.properties.get("vertex_count", 0) for f in features)
    total_faces = sum(f.properties.get("face_count", 0) for f in features)

    coll_props = {
        "mesh_count": len(features),
        "total_vertices": total_verts,
        "total_faces": total_faces,
    }

    vocabs: Optional[dict[str, Vocabulary]] = None
    if ontology is not None:
        namespace = "http://api.brain-map.org/api/v2/data/Structure"
        terms: dict[str, OntologyTerm] = {}
        for feat in features:
            ccf_id = (feat.properties or {}).get("ccf_id")
            name = (feat.properties or {}).get("name")
            if ccf_id and name:
                terms[name] = OntologyTerm(
                    uri=f"{namespace}/{ccf_id}",
                    label=name,
                )
        if terms:
            vocabs = {
                "allen_ccf": Vocabulary(
                    namespace=namespace,
                    description="Allen Common Coordinate Framework brain regions",
                    terms=terms,
                ),
            }

    return MuDMFeatureCollection(
        type="FeatureCollection",
        features=features,
        properties=coll_props,
        vocabularies=vocabs,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _pop_flag(args: list[str], flag: str) -> Optional[str]:
    """Remove ``flag VALUE`` from *args*, return VALUE or None."""
    if flag in args:
        idx = args.index(flag)
        val = args[idx + 1]
        del args[idx : idx + 2]
        return val
    return None


def _pop_bool_flag(args: list[str], flag: str) -> bool:
    """Remove a boolean flag from *args*, return True if present."""
    if flag in args:
        args.remove(flag)
        return True
    return False


def main() -> None:
    args = sys.argv[1:]

    out_path = _pop_flag(args, "-o") or _pop_flag(args, "--output")
    ontology_path = _pop_flag(args, "--ontology")
    no_ontology = _pop_bool_flag(args, "--no-ontology")

    # Discover OBJ files
    obj_paths: list[Path] = []
    if not args:
        print("Usage: obj_to_microjson.py [OBJ_DIR_OR_FILES...] [OPTIONS]", file=sys.stderr)
        sys.exit(1)

    for arg in args:
        p = Path(arg)
        if p.is_dir():
            found = sorted(p.glob("*.obj"))
            if not found:
                print(f"No .obj files found in {p}/", file=sys.stderr)
                sys.exit(1)
            obj_paths.extend(found)
        elif p.is_file():
            obj_paths.append(p)
        else:
            print(f"Error: not found: {p}", file=sys.stderr)
            sys.exit(1)

    if not obj_paths:
        print("No OBJ files to process.", file=sys.stderr)
        sys.exit(1)

    # Fetch ontology
    ontology: Optional[dict[str, dict]] = None
    if not no_ontology:
        ontology = fetch_allen_ontology(ontology_path)
        print(f"Loaded {len(ontology)} ontology entries", file=sys.stderr)

    # Convert each OBJ
    features: list[MuDMFeature] = []
    for i, obj_path in enumerate(obj_paths, 1):
        print(
            f"[{i}/{len(obj_paths)}] {obj_path.name}",
            file=sys.stderr,
            end="",
            flush=True,
        )
        feat = obj_to_feature(str(obj_path), ontology)
        features.append(feat)
        verts = feat.properties.get("vertex_count", 0)
        faces = feat.properties.get("face_count", 0)
        print(f" — {verts:,} verts, {faces:,} faces", file=sys.stderr)

    # Build collection
    collection = build_collection(features, ontology)
    json_str = collection.model_dump_json(indent=2, exclude_none=True)

    if out_path:
        Path(out_path).write_text(json_str)
        size_kb = len(json_str) / 1024
        print(
            f"Wrote {out_path} ({size_kb:,.0f} KB)"
            f" — {len(features)} mesh(es)",
            file=sys.stderr,
        )
    else:
        print(json_str)


if __name__ == "__main__":
    main()
