"""Tests for ontology vocabulary support on MuDM models."""

import pytest
from geojson_pydantic import Point

from mudm.model import (
    MuDMFeature,
    MuDMFeatureCollection,
    OntologyTerm,
    Vocabulary,
)


# ---------------------------------------------------------------------------
# OntologyTerm basics
# ---------------------------------------------------------------------------

class TestOntologyTerm:
    def test_create_full(self):
        t = OntologyTerm(
            uri="http://purl.obolibrary.org/obo/CL_0000598",
            label="pyramidal neuron",
            description="A projection neuron",
        )
        assert t.uri == "http://purl.obolibrary.org/obo/CL_0000598"
        assert t.label == "pyramidal neuron"
        assert t.description == "A projection neuron"

    def test_create_minimal(self):
        t = OntologyTerm(uri="http://example.org/term/1")
        assert t.label is None
        assert t.description is None

    def test_roundtrip_json(self):
        t = OntologyTerm(
            uri="http://purl.obolibrary.org/obo/CL_0000598",
            label="pyramidal neuron",
        )
        data = t.model_dump()
        t2 = OntologyTerm(**data)
        assert t2.uri == t.uri
        assert t2.label == t.label


# ---------------------------------------------------------------------------
# Vocabulary basics
# ---------------------------------------------------------------------------

class TestVocabulary:
    def test_create(self):
        v = Vocabulary(
            namespace="http://purl.obolibrary.org/obo/CL_",
            terms={
                "pyramidal": OntologyTerm(
                    uri="http://purl.obolibrary.org/obo/CL_0000598",
                    label="pyramidal neuron",
                ),
            },
        )
        assert "pyramidal" in v.terms
        assert v.namespace == "http://purl.obolibrary.org/obo/CL_"

    def test_roundtrip_json(self):
        v = Vocabulary(
            namespace="http://purl.obolibrary.org/obo/CL_",
            description="Cell ontology",
            terms={
                "pyramidal": OntologyTerm(uri="http://purl.obolibrary.org/obo/CL_0000598"),
                "interneuron": OntologyTerm(uri="http://purl.obolibrary.org/obo/CL_0000099"),
            },
        )
        data = v.model_dump()
        v2 = Vocabulary(**data)
        assert set(v2.terms.keys()) == {"pyramidal", "interneuron"}

    def test_minimal_vocabulary(self):
        v = Vocabulary(terms={"a": OntologyTerm(uri="http://example.org/a")})
        assert v.namespace is None
        assert v.description is None


# ---------------------------------------------------------------------------
# Collection-level vocabularies
# ---------------------------------------------------------------------------

class TestCollectionVocabularies:
    def test_inline_vocabularies(self):
        fc = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[
                MuDMFeature(
                    type="Feature",
                    geometry=Point(type="Point", coordinates=(1.0, 2.0)),
                    properties={"cell_type": "pyramidal"},
                ),
            ],
            vocabularies={
                "cell_type": Vocabulary(
                    namespace="http://purl.obolibrary.org/obo/CL_",
                    terms={
                        "pyramidal": OntologyTerm(
                            uri="http://purl.obolibrary.org/obo/CL_0000598",
                            label="pyramidal neuron",
                        ),
                    },
                ),
            },
        )
        assert "cell_type" in fc.vocabularies
        assert fc.vocabularies["cell_type"].terms["pyramidal"].label == "pyramidal neuron"

    def test_uri_reference(self):
        fc = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[
                MuDMFeature(
                    type="Feature",
                    geometry=Point(type="Point", coordinates=(1.0, 2.0)),
                    properties={},
                ),
            ],
            vocabularies="https://neuromorpho.org/vocab/neuroscience-v1.json",
        )
        assert fc.vocabularies == "https://neuromorpho.org/vocab/neuroscience-v1.json"

    def test_backwards_compatible_none(self):
        fc = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[
                MuDMFeature(
                    type="Feature",
                    geometry=Point(type="Point", coordinates=(1.0, 2.0)),
                    properties={},
                ),
            ],
        )
        assert fc.vocabularies is None

    def test_json_roundtrip(self):
        fc = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[
                MuDMFeature(
                    type="Feature",
                    geometry=Point(type="Point", coordinates=(1.0, 2.0)),
                    properties={"cell_type": "pyramidal"},
                ),
            ],
            vocabularies={
                "cell_type": Vocabulary(
                    terms={
                        "pyramidal": OntologyTerm(uri="http://purl.obolibrary.org/obo/CL_0000598"),
                    },
                ),
            },
        )
        data = fc.model_dump()
        fc2 = MuDMFeatureCollection(**data)
        assert "cell_type" in fc2.vocabularies
        assert fc2.vocabularies["cell_type"].terms["pyramidal"].uri == "http://purl.obolibrary.org/obo/CL_0000598"


# ---------------------------------------------------------------------------
# Feature-level vocabularies (override)
# ---------------------------------------------------------------------------

class TestFeatureVocabularies:
    def test_feature_level_override(self):
        """Feature-level vocabularies should override collection-level for the same key."""
        collection_vocab = {
            "cell_type": Vocabulary(
                terms={
                    "pyramidal": OntologyTerm(uri="http://example.org/COLLECTION"),
                },
            ),
        }
        feature_vocab = {
            "cell_type": Vocabulary(
                terms={
                    "pyramidal": OntologyTerm(uri="http://example.org/FEATURE"),
                },
            ),
        }
        feat = MuDMFeature(
            type="Feature",
            geometry=Point(type="Point", coordinates=(1.0, 2.0)),
            properties={"cell_type": "pyramidal"},
            vocabularies=feature_vocab,
        )
        fc = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[feat],
            vocabularies=collection_vocab,
        )

        # Resolution: check feature first, then collection
        resolved = feat.vocabularies or fc.vocabularies
        assert resolved["cell_type"].terms["pyramidal"].uri == "http://example.org/FEATURE"

    def test_feature_uri_reference(self):
        feat = MuDMFeature(
            type="Feature",
            geometry=Point(type="Point", coordinates=(1.0, 2.0)),
            properties={},
            vocabularies="https://example.org/vocab.json",
        )
        assert feat.vocabularies == "https://example.org/vocab.json"

    def test_feature_no_vocabularies(self):
        feat = MuDMFeature(
            type="Feature",
            geometry=Point(type="Point", coordinates=(1.0, 2.0)),
            properties={},
        )
        assert feat.vocabularies is None


# ---------------------------------------------------------------------------
# Multiple vocabularies on same collection
# ---------------------------------------------------------------------------

class TestMultipleVocabularies:
    def test_multiple_property_vocabularies(self):
        fc = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[
                MuDMFeature(
                    type="Feature",
                    geometry=Point(type="Point", coordinates=(1.0, 2.0)),
                    properties={"cell_type": "pyramidal", "brain_region": "hippocampus_CA1"},
                ),
            ],
            vocabularies={
                "cell_type": Vocabulary(
                    namespace="http://purl.obolibrary.org/obo/CL_",
                    terms={
                        "pyramidal": OntologyTerm(uri="http://purl.obolibrary.org/obo/CL_0000598"),
                    },
                ),
                "brain_region": Vocabulary(
                    namespace="http://purl.obolibrary.org/obo/UBERON_",
                    terms={
                        "hippocampus_CA1": OntologyTerm(uri="http://purl.obolibrary.org/obo/UBERON_0003881"),
                    },
                ),
            },
        )
        assert len(fc.vocabularies) == 2
        assert fc.vocabularies["brain_region"].terms["hippocampus_CA1"].uri == "http://purl.obolibrary.org/obo/UBERON_0003881"
