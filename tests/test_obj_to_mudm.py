"""Tests for scripts/obj_to_microjson.py — OBJ → MuDM conversion."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add scripts/ to path so we can import the module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from mudm.model import MuDMFeatureCollection
from obj_to_mudm import (
    build_collection,
    fetch_allen_ontology,
    match_region,
    obj_to_feature,
    parse_obj,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_obj(content: str) -> str:
    """Write OBJ content to a temp file and return the path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".obj", delete=False)
    f.write(content)
    f.close()
    return f.name


# A simple ontology for testing (mimics flattened Allen CCF structure)
_MOCK_ONTOLOGY = {
    "primary motor area": {
        "id": 985,
        "name": "Primary motor area",
        "acronym": "MOp",
        "parent_structure_id": 500,
        "structure_id_path": "/997/8/567/688/695/315/500/985/",
        "color_hex_triplet": "1F9D5A",
    },
    "mop": {
        "id": 985,
        "name": "Primary motor area",
        "acronym": "MOp",
        "parent_structure_id": 500,
        "structure_id_path": "/997/8/567/688/695/315/500/985/",
        "color_hex_triplet": "1F9D5A",
    },
    "985": {
        "id": 985,
        "name": "Primary motor area",
        "acronym": "MOp",
        "parent_structure_id": 500,
        "structure_id_path": "/997/8/567/688/695/315/500/985/",
        "color_hex_triplet": "1F9D5A",
    },
    "hippocampal formation": {
        "id": 1089,
        "name": "Hippocampal formation",
        "acronym": "HPF",
        "parent_structure_id": 695,
        "structure_id_path": "/997/8/567/688/695/1089/",
        "color_hex_triplet": "7ED04B",
    },
    "hpf": {
        "id": 1089,
        "name": "Hippocampal formation",
        "acronym": "HPF",
        "parent_structure_id": 695,
        "structure_id_path": "/997/8/567/688/695/1089/",
        "color_hex_triplet": "7ED04B",
    },
    "1089": {
        "id": 1089,
        "name": "Hippocampal formation",
        "acronym": "HPF",
        "parent_structure_id": 695,
        "structure_id_path": "/997/8/567/688/695/1089/",
        "color_hex_triplet": "7ED04B",
    },
}


# ---------------------------------------------------------------------------
# parse_obj tests
# ---------------------------------------------------------------------------

class TestParseObj:
    def test_simple_triangle(self):
        """Single triangle — vertices and face indices parsed correctly."""
        obj = _write_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
        verts, faces = parse_obj(obj)

        assert verts.shape == (3, 3)
        assert faces.shape == (1, 3)
        np.testing.assert_array_equal(verts[0], [0, 0, 0])
        np.testing.assert_array_equal(verts[1], [1, 0, 0])
        np.testing.assert_array_equal(verts[2], [0, 1, 0])
        np.testing.assert_array_equal(faces[0], [0, 1, 2])  # 0-based

    def test_with_normals(self):
        """f v1//n1 v2//n2 v3//n3 format — normals ignored, vertices correct."""
        obj = _write_obj(
            "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
            "vn 0 0 1\nvn 0 0 1\nvn 0 0 1\n"
            "f 1//1 2//2 3//3\n"
        )
        verts, faces = parse_obj(obj)

        assert verts.shape == (3, 3)
        assert faces.shape == (1, 3)
        np.testing.assert_array_equal(faces[0], [0, 1, 2])

    def test_with_texcoords_and_normals(self):
        """f v/t/n format — texture coords and normals ignored."""
        obj = _write_obj(
            "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
            "vt 0 0\nvt 1 0\nvt 0 1\n"
            "vn 0 0 1\n"
            "f 1/1/1 2/2/1 3/3/1\n"
        )
        verts, faces = parse_obj(obj)

        assert faces.shape == (1, 3)
        np.testing.assert_array_equal(faces[0], [0, 1, 2])

    def test_quads(self):
        """Quad face → 2 triangles via fan triangulation."""
        obj = _write_obj(
            "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
            "f 1 2 3 4\n"
        )
        verts, faces = parse_obj(obj)

        assert verts.shape == (4, 3)
        assert faces.shape == (2, 3)
        np.testing.assert_array_equal(faces[0], [0, 1, 2])
        np.testing.assert_array_equal(faces[1], [0, 2, 3])

    def test_comments_and_ignored_lines(self):
        """Comments, mtllib, usemtl, group, object lines are skipped."""
        obj = _write_obj(
            "# comment\n"
            "mtllib material.mtl\n"
            "usemtl default\n"
            "o mesh_object\n"
            "g group1\n"
            "s 1\n"
            "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
            "f 1 2 3\n"
        )
        verts, faces = parse_obj(obj)

        assert verts.shape == (3, 3)
        assert faces.shape == (1, 3)

    def test_no_vertices_raises(self):
        """Empty OBJ raises ValueError."""
        obj = _write_obj("# empty\n")
        with pytest.raises(ValueError, match="No vertices found"):
            parse_obj(obj)

    def test_no_faces_raises(self):
        """OBJ with only vertices raises ValueError."""
        obj = _write_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\n")
        with pytest.raises(ValueError, match="No faces found"):
            parse_obj(obj)


# ---------------------------------------------------------------------------
# match_region tests
# ---------------------------------------------------------------------------

class TestMatchRegion:
    def test_match_by_name(self):
        """Exact name match (case-insensitive)."""
        region = match_region("Primary motor area", _MOCK_ONTOLOGY)
        assert region is not None
        assert region["id"] == 985

    def test_match_by_acronym(self):
        """Exact acronym match (case-insensitive)."""
        region = match_region("MOp", _MOCK_ONTOLOGY)
        assert region is not None
        assert region["id"] == 985

    def test_match_by_id(self):
        """Numeric part of filename matches Allen CCF ID."""
        region = match_region("region_985", _MOCK_ONTOLOGY)
        assert region is not None
        assert region["id"] == 985

    def test_no_match(self):
        """Unrecognized filename returns None."""
        region = match_region("unknown_region", _MOCK_ONTOLOGY)
        assert region is None


# ---------------------------------------------------------------------------
# obj_to_feature tests
# ---------------------------------------------------------------------------

class TestObjToFeature:
    def test_basic(self):
        """OBJ → MuDMFeature with TIN geometry and basic properties."""
        obj = _write_obj(
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\n"
            "f 1 2 3\nf 2 4 3\n"
        )
        feat = obj_to_feature(obj)

        assert feat.type == "Feature"
        assert feat.geometry.type == "TIN"
        assert len(feat.geometry.coordinates) == 2  # 2 triangles
        assert feat.properties["vertex_count"] == 4
        assert feat.properties["face_count"] == 2
        assert "mesh_name" in feat.properties
        assert "source" in feat.properties

    def test_with_ontology(self):
        """OBJ with ontology lookup attaches CCF metadata."""
        obj = _write_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
        # Rename temp file to match an ontology name
        p = Path(obj)
        named = p.parent / "MOp.obj"
        p.rename(named)

        feat = obj_to_feature(str(named), _MOCK_ONTOLOGY)

        assert feat.properties["ccf_id"] == 985
        assert feat.properties["name"] == "Primary motor area"
        assert feat.properties["acronym"] == "MOp"
        assert feat.properties["parent_id"] == 500
        assert feat.properties["color"] == "#1F9D5A"
        assert feat.featureClass == "MOp"


# ---------------------------------------------------------------------------
# build_collection tests
# ---------------------------------------------------------------------------

class TestBuildCollection:
    def test_collection_with_vocabulary(self):
        """Collection has vocabulary with OntologyTerms from matched features."""
        obj1 = _write_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
        p1 = Path(obj1)
        named1 = p1.parent / "MOp.obj"
        p1.rename(named1)

        obj2 = _write_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
        p2 = Path(obj2)
        named2 = p2.parent / "HPF.obj"
        p2.rename(named2)

        feat1 = obj_to_feature(str(named1), _MOCK_ONTOLOGY)
        feat2 = obj_to_feature(str(named2), _MOCK_ONTOLOGY)
        coll = build_collection([feat1, feat2], _MOCK_ONTOLOGY)

        assert coll.type == "FeatureCollection"
        assert len(coll.features) == 2
        assert coll.properties["mesh_count"] == 2
        assert coll.properties["total_vertices"] == 6
        assert coll.properties["total_faces"] == 2

        assert coll.vocabularies is not None
        assert "allen_ccf" in coll.vocabularies
        vocab = coll.vocabularies["allen_ccf"]
        assert "Primary motor area" in vocab.terms
        assert "Hippocampal formation" in vocab.terms
        term = vocab.terms["Primary motor area"]
        assert "985" in term.uri
        assert term.label == "Primary motor area"

    def test_collection_without_ontology(self):
        """Collection without ontology has no vocabulary."""
        obj = _write_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
        feat = obj_to_feature(obj)
        coll = build_collection([feat])

        assert coll.vocabularies is None
        assert coll.properties["mesh_count"] == 1

    def test_serialization_roundtrip(self):
        """Collection serializes to valid JSON and parses back."""
        obj = _write_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
        feat = obj_to_feature(obj, _MOCK_ONTOLOGY)
        coll = build_collection([feat], _MOCK_ONTOLOGY)

        json_str = coll.model_dump_json(indent=2, exclude_none=True)
        parsed = MuDMFeatureCollection.model_validate_json(json_str)
        assert len(parsed.features) == 1
        assert parsed.features[0].geometry.type == "TIN"


# ---------------------------------------------------------------------------
# Allen CCF ontology (offline test)
# ---------------------------------------------------------------------------

class TestAllenOntology:
    def test_fetch_from_local_file(self, tmp_path):
        """Loading a local Allen CCF JSON file works."""
        # Minimal Allen CCF structure
        data = {
            "success": True,
            "id": 0,
            "msg": [
                {
                    "id": 997,
                    "name": "root",
                    "acronym": "root",
                    "parent_structure_id": None,
                    "structure_id_path": "/997/",
                    "color_hex_triplet": "FFFFFF",
                    "children": [
                        {
                            "id": 8,
                            "name": "Basic cell groups and regions",
                            "acronym": "grey",
                            "parent_structure_id": 997,
                            "structure_id_path": "/997/8/",
                            "color_hex_triplet": "BFDAE3",
                            "children": [],
                        }
                    ],
                }
            ],
        }
        local = tmp_path / "ontology.json"
        local.write_text(json.dumps(data))

        ont = fetch_allen_ontology(str(local))
        assert "root" in ont
        assert "grey" in ont
        assert "997" in ont
        assert "8" in ont
        assert ont["grey"]["name"] == "Basic cell groups and regions"
