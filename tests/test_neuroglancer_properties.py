"""Tests for Neuroglancer segment_properties writer."""

import json


from mudm.model import MuDMFeature
from mudm_tools.neuroglancer.properties_writer import (
    features_to_segment_properties,
    write_segment_properties,
)


def _make_feature(**props) -> MuDMFeature:
    """Helper to create a MuDMFeature with given properties."""
    return MuDMFeature(
        type="Feature",
        geometry={"type": "Point", "coordinates": [0, 0, 0]},
        properties=props,
    )


class TestFeaturesToSegmentProperties:
    def test_label_props(self):
        features = [
            _make_feature(name="neuron_a"),
            _make_feature(name="neuron_b"),
        ]
        result = features_to_segment_properties(features, [1, 2])
        assert result["@type"] == "neuroglancer_segment_properties"
        inline = result["inline"]
        assert inline["ids"] == ["1", "2"]
        assert len(inline["properties"]) == 1
        assert inline["properties"][0]["id"] == "name"
        assert inline["properties"][0]["type"] == "label"
        assert inline["properties"][0]["values"] == ["neuron_a", "neuron_b"]

    def test_numeric_props(self):
        features = [
            _make_feature(area=100.5),
            _make_feature(area=200.0),
        ]
        result = features_to_segment_properties(features, [1, 2])
        prop = result["inline"]["properties"][0]
        assert prop["type"] == "number"
        assert prop["values"] == [100.5, 200.0]

    def test_mixed_props(self):
        features = [
            _make_feature(name="a", area=10),
            _make_feature(name="b", area=20),
        ]
        result = features_to_segment_properties(features, [1, 2])
        props = result["inline"]["properties"]
        assert len(props) == 2
        ids = [p["id"] for p in props]
        assert "name" in ids
        assert "area" in ids

    def test_missing_prop_filled(self):
        features = [
            _make_feature(name="a", area=10),
            _make_feature(name="b"),
        ]
        result = features_to_segment_properties(features, [1, 2])
        area_prop = [p for p in result["inline"]["properties"] if p["id"] == "area"][0]
        # Missing value should be "" (empty string)
        assert area_prop["values"][1] == ""

    def test_empty_features(self):
        result = features_to_segment_properties([], [])
        assert result["inline"]["ids"] == []
        assert result["inline"]["properties"] == []


class TestWriteSegmentProperties:
    def test_creates_info(self, tmp_path):
        features = [_make_feature(name="neuron_1")]
        out = tmp_path / "seg_props"
        write_segment_properties(out, features, [1])
        assert (out / "info").exists()

    def test_info_json_valid(self, tmp_path):
        features = [_make_feature(name="n1"), _make_feature(name="n2")]
        out = tmp_path / "seg_props"
        write_segment_properties(out, features, [10, 20])
        info = json.loads((out / "info").read_text())
        assert info["@type"] == "neuroglancer_segment_properties"
        assert info["inline"]["ids"] == ["10", "20"]
