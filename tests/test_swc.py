"""Tests for NeuronMorphology model and SWC interoperability."""

import pytest
from pathlib import Path
from pydantic import ValidationError
from mudm_tools.swc import (
    SWCSample,
    NeuronMorphology,
    SWC_TYPE_NAMES,
    _parse_swc,
    microjson_to_swc,
    swc_to_feature_collection,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"
SAMPLE_SWC = FIXTURE_DIR / "sample_neuron.swc"


# ---------------------------------------------------------------------------
# SWCSample model tests
# ---------------------------------------------------------------------------


class TestSWCSample:
    def test_create(self):
        s = SWCSample(id=1, type=1, x=0.0, y=0.0, z=0.0, r=8.0, parent=-1)
        assert s.id == 1
        assert s.type == 1
        assert s.r == 8.0
        assert s.parent == -1

    def test_roundtrip(self):
        s = SWCSample(id=2, type=3, x=10.0, y=5.0, z=2.0, r=2.0, parent=1)
        data = s.model_dump()
        s2 = SWCSample.model_validate(data)
        assert s2.x == 10.0
        assert s2.parent == 1


# ---------------------------------------------------------------------------
# NeuronMorphology model tests
# ---------------------------------------------------------------------------


class TestNeuronMorphology:
    def _simple_tree(self):
        return [
            SWCSample(id=1, type=1, x=0, y=0, z=0, r=8.0, parent=-1),
            SWCSample(id=2, type=3, x=10, y=5, z=2, r=2.0, parent=1),
            SWCSample(id=3, type=3, x=20, y=10, z=5, r=1.5, parent=2),
        ]

    def test_create(self):
        nm = NeuronMorphology(type="NeuronMorphology", tree=self._simple_tree())
        assert nm.type == "NeuronMorphology"
        assert len(nm.tree) == 3

    def test_bbox3d(self):
        nm = NeuronMorphology(type="NeuronMorphology", tree=self._simple_tree())
        bb = nm.bbox3d()
        assert bb == (0.0, 0.0, 0.0, 20.0, 10.0, 5.0)

    def test_centroid3d(self):
        nm = NeuronMorphology(type="NeuronMorphology", tree=self._simple_tree())
        c = nm.centroid3d()
        assert c[0] == pytest.approx(10.0)
        assert c[1] == pytest.approx(5.0)
        assert c[2] == pytest.approx(7.0 / 3.0)

    def test_no_root_rejected(self):
        """At least one node must have parent == -1."""
        tree = [
            SWCSample(id=1, type=1, x=0, y=0, z=0, r=1, parent=2),
            SWCSample(id=2, type=3, x=1, y=1, z=1, r=1, parent=1),
        ]
        with pytest.raises(ValidationError, match="root"):
            NeuronMorphology(type="NeuronMorphology", tree=tree)

    def test_invalid_parent_rejected(self):
        """Non-root parent ids must reference existing nodes."""
        tree = [
            SWCSample(id=1, type=1, x=0, y=0, z=0, r=1, parent=-1),
            SWCSample(id=2, type=3, x=1, y=1, z=1, r=1, parent=99),
        ]
        with pytest.raises(ValidationError, match="parent"):
            NeuronMorphology(type="NeuronMorphology", tree=tree)

    def test_empty_tree_rejected(self):
        with pytest.raises(ValidationError):
            NeuronMorphology(type="NeuronMorphology", tree=[])

    def test_roundtrip_json(self):
        nm = NeuronMorphology(type="NeuronMorphology", tree=self._simple_tree())
        data = nm.model_dump()
        nm2 = NeuronMorphology.model_validate(data)
        assert len(nm2.tree) == 3
        assert nm2.tree[0].parent == -1


# ---------------------------------------------------------------------------
# SWC converter tests
# ---------------------------------------------------------------------------


class TestSWCConverters:
    def test_swc_to_microjson(self):
        from mudm_tools.swc import swc_to_microjson

        feature = swc_to_microjson(str(SAMPLE_SWC))
        assert feature.type == "Feature"
        # swc_to_microjson now returns TIN geometry
        assert feature.geometry.type == "TIN"
        assert len(feature.geometry.coordinates) > 0

    def test_microjson_to_swc(self):
        """microjson_to_swc accepts a NeuronMorphology directly."""
        morphology = _parse_swc(str(SAMPLE_SWC))
        swc_text = microjson_to_swc(morphology)
        lines = [line for line in swc_text.strip().split("\n") if not line.startswith("#")]
        assert len(lines) == 8
        # first data line: id=1, type=1, parent=-1
        parts = lines[0].split()
        assert parts[0] == "1"
        assert parts[1] == "1"
        assert parts[6] == "-1"

    def test_swc_roundtrip(self):
        """Parse SWC -> NeuronMorphology -> SWC text."""
        morphology = _parse_swc(str(SAMPLE_SWC))
        swc_text = microjson_to_swc(morphology)
        # Parse the generated SWC text back
        lines = [line for line in swc_text.strip().split("\n") if not line.startswith("#")]
        assert len(lines) == 8
        for line in lines:
            parts = line.split()
            assert len(parts) == 7

    def test_swc_to_linestring3d(self):
        from mudm_tools.swc import swc_to_linestring3d

        mls = swc_to_linestring3d(str(SAMPLE_SWC))
        assert mls.type == "MultiLineString"
        # 7 edges in the tree (8 nodes, each non-root has one parent edge)
        assert len(mls.coordinates) == 7
        # Each edge is a 2-point line
        for line in mls.coordinates:
            assert len(line) == 2
            assert len(line[0]) == 3  # 3D coordinates


# ---------------------------------------------------------------------------
# swc_to_tin converter tests
# ---------------------------------------------------------------------------


class TestSWCToTIN:
    def test_swc_to_tin_produces_tin(self):
        from mudm_tools.swc import swc_to_tin

        feature = swc_to_tin(str(SAMPLE_SWC))
        assert feature.type == "Feature"
        assert feature.geometry.type == "TIN"

    def test_swc_to_tin_valid_triangles(self):
        from mudm_tools.swc import swc_to_tin

        feature = swc_to_tin(str(SAMPLE_SWC))
        tin = feature.geometry
        assert len(tin.coordinates) > 0
        for face in tin.coordinates:
            # Each face: exactly 1 ring
            assert len(face) == 1
            ring = face[0]
            # Closed triangle: 4 positions (3 vertices + repeated first)
            assert len(ring) == 4
            # Ring is closed
            assert ring[0] == ring[3]
            # All positions are 3D
            for pos in ring:
                assert len(pos) == 3

    def test_swc_to_tin_to_glb(self):
        """TIN from SWC goes through existing glTF TIN path."""
        from mudm_tools.swc import swc_to_tin
        from mudm_tools.gltf.writer import to_glb

        feature = swc_to_tin(str(SAMPLE_SWC))
        glb_bytes = to_glb(feature)
        assert len(glb_bytes) > 0
        # GLB magic number
        assert glb_bytes[:4] == b"glTF"

    def test_swc_to_tin_smoothing(self):
        """Higher smooth_subdivisions increases face count."""
        from mudm_tools.swc import swc_to_tin

        f_no_smooth = swc_to_tin(str(SAMPLE_SWC), smooth_subdivisions=0)
        f_smooth = swc_to_tin(str(SAMPLE_SWC), smooth_subdivisions=3)
        assert len(f_smooth.geometry.coordinates) > len(f_no_smooth.geometry.coordinates)

    def test_swc_to_tin_quality(self):
        """mesh_quality < 1 decreases face count."""
        from mudm_tools.swc import swc_to_tin

        f_full = swc_to_tin(str(SAMPLE_SWC), smooth_subdivisions=3, mesh_quality=1.0)
        f_reduced = swc_to_tin(str(SAMPLE_SWC), smooth_subdivisions=3, mesh_quality=0.5)
        assert len(f_reduced.geometry.coordinates) < len(f_full.geometry.coordinates)


# ---------------------------------------------------------------------------
# swc_to_feature_collection tests
# ---------------------------------------------------------------------------


class TestSWCToFeatureCollection:
    def test_returns_feature_collection(self):
        coll = swc_to_feature_collection(str(SAMPLE_SWC))
        assert coll.type == "FeatureCollection"
        assert len(coll.features) > 0

    def test_one_feature_per_type(self):
        """Each SWC type present in the file gets its own Feature."""
        coll = swc_to_feature_collection(str(SAMPLE_SWC))
        compartments = [f.properties["compartment"] for f in coll.features]
        # No duplicates
        assert len(compartments) == len(set(compartments))

    def test_features_have_compartment_property(self):
        coll = swc_to_feature_collection(str(SAMPLE_SWC))
        for feat in coll.features:
            assert "compartment" in feat.properties
            assert feat.properties["compartment"] in SWC_TYPE_NAMES.values()

    def test_features_have_tin_geometry(self):
        coll = swc_to_feature_collection(str(SAMPLE_SWC))
        for feat in coll.features:
            assert feat.geometry.type == "TIN"
            assert len(feat.geometry.coordinates) > 0

    def test_features_have_feature_class(self):
        coll = swc_to_feature_collection(str(SAMPLE_SWC))
        for feat in coll.features:
            assert feat.featureClass == feat.properties["compartment"]

    def test_soma_present(self):
        """The sample SWC has soma (type 1)."""
        coll = swc_to_feature_collection(str(SAMPLE_SWC))
        compartments = {f.properties["compartment"] for f in coll.features}
        assert "soma" in compartments

    def test_collection_has_neuron_name(self):
        """Collection properties include the neuron name from filename."""
        coll = swc_to_feature_collection(str(SAMPLE_SWC))
        assert coll.properties is not None
        assert coll.properties["name"] == "sample_neuron"

    def test_neuromorpho_name_strips_cng(self):
        """NeuroMorpho .CNG.swc suffix is stripped from the name."""
        from mudm_tools.swc import _neuron_name_from_path

        assert _neuron_name_from_path("swcs/cnic_041.CNG.swc") == "cnic_041"
        assert _neuron_name_from_path("foo/bar.swc") == "bar"

    def test_explicit_name(self):
        """Explicit name overrides filename derivation."""
        coll = swc_to_feature_collection(str(SAMPLE_SWC), name="my_neuron")
        assert coll.properties["name"] == "my_neuron"

    def test_to_glb_roundtrip(self):
        """Feature collection can be exported to GLB."""
        from mudm_tools.gltf.writer import to_glb

        coll = swc_to_feature_collection(str(SAMPLE_SWC))
        glb_bytes = to_glb(coll)
        assert len(glb_bytes) > 0
        assert glb_bytes[:4] == b"glTF"


class TestFloatFormattedIds:
    def test_parse_float_ids(self):
        """SWCs with '1.000000'-style integer columns must parse (cnn-nmo convention)."""
        from mudm_tools.swc import _parse_swc

        fixture = Path(__file__).parent / "fixtures" / "sample_neuron_float_ids.swc"
        morph = _parse_swc(str(fixture))
        assert len(morph.tree) == 3
        assert morph.tree[0].id == 1
        assert morph.tree[0].type == 1
        assert morph.tree[0].parent == -1
        assert morph.tree[1].id == 2
        assert morph.tree[1].parent == 1
