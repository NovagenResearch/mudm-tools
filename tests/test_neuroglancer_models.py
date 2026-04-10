"""Tests for Neuroglancer Pydantic config models."""

import pytest

from mudm_tools.neuroglancer.models import (
    AnnotationDimension,
    AnnotationInfo,
    AnnotationSpatialEntry,
    SegmentPropertiesInfo,
    SegmentPropertiesInline,
    SegmentPropertyField,
    SkeletonInfo,
    VertexAttributeInfo,
)


class TestSkeletonInfo:
    def test_minimal(self):
        info = SkeletonInfo()
        d = info.to_info_dict()
        assert d["@type"] == "neuroglancer_skeletons"
        assert "transform" not in d
        assert "vertex_attributes" not in d

    def test_with_transform(self):
        transform = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
        info = SkeletonInfo(transform=transform)
        d = info.to_info_dict()
        assert d["transform"] == transform

    def test_with_vertex_attributes(self):
        attrs = [
            VertexAttributeInfo(id="radius", data_type="float32"),
            VertexAttributeInfo(id="type", data_type="float32"),
        ]
        info = SkeletonInfo(vertex_attributes=attrs)
        d = info.to_info_dict()
        assert len(d["vertex_attributes"]) == 2
        assert d["vertex_attributes"][0]["id"] == "radius"

    def test_with_segment_properties(self):
        info = SkeletonInfo(segment_properties="seg_props")
        d = info.to_info_dict()
        assert d["segment_properties"] == "seg_props"


class TestVertexAttributeInfo:
    def test_defaults(self):
        va = VertexAttributeInfo(id="radius", data_type="float32")
        d = va.to_info_dict()
        assert d == {"id": "radius", "data_type": "float32", "num_components": 1}

    def test_multi_component(self):
        va = VertexAttributeInfo(id="color", data_type="uint8", num_components=3)
        d = va.to_info_dict()
        assert d["num_components"] == 3


class TestAnnotationInfo:
    def test_point(self):
        info = AnnotationInfo(
            annotation_type="POINT",
            upper_bound=[100.0, 100.0, 100.0],
        )
        d = info.to_info_dict()
        assert d["@type"] == "neuroglancer_annotations_v1"
        assert d["annotation_type"] == "POINT"
        assert "dimensions" in d
        assert d["lower_bound"] == [0.0, 0.0, 0.0]
        assert d["upper_bound"] == [100.0, 100.0, 100.0]

    def test_required_fields_present(self):
        info = AnnotationInfo(annotation_type="POINT")
        d = info.to_info_dict()
        assert d["properties"] == []
        assert d["relationships"] == []
        assert d["by_id"] == {"key": "by_id"}

    def test_line(self):
        info = AnnotationInfo(annotation_type="LINE")
        d = info.to_info_dict()
        assert d["annotation_type"] == "LINE"

    def test_with_spatial(self):
        spatial = AnnotationSpatialEntry(chunk_size=[100.0, 100.0, 100.0])
        info = AnnotationInfo(
            annotation_type="POINT",
            spatial=[spatial],
        )
        d = info.to_info_dict()
        assert len(d["spatial"]) == 1
        assert d["spatial"][0]["chunk_size"] == [100.0, 100.0, 100.0]
        assert d["spatial"][0]["key"] == "spatial0"


class TestAnnotationDimension:
    def test_defaults(self):
        dim = AnnotationDimension()
        d = dim.to_info_dict()
        assert "dimensions" in d
        assert d["dimensions"]["x"] == [1.0, "nm"]
        assert d["dimensions"]["y"] == [1.0, "nm"]
        assert d["dimensions"]["z"] == [1.0, "nm"]

    def test_custom(self):
        dim = AnnotationDimension(
            names=["x", "y", "z"],
            units=["um", "um", "um"],
            scales=[0.5, 0.5, 1.0],
        )
        d = dim.to_info_dict()
        assert d["dimensions"]["x"] == [0.5, "um"]


class TestSegmentPropertiesInfo:
    def test_minimal(self):
        info = SegmentPropertiesInfo()
        d = info.to_info_dict()
        assert d["@type"] == "neuroglancer_segment_properties"
        assert "inline" not in d

    def test_with_inline(self):
        inline = SegmentPropertiesInline(
            ids=["1", "2"],
            properties=[
                SegmentPropertyField(
                    id="label",
                    type="label",
                    values=["neuron_1", "neuron_2"],
                ),
                SegmentPropertyField(
                    id="area",
                    type="number",
                    values=[100.0, 200.0],
                ),
            ],
        )
        info = SegmentPropertiesInfo(inline=inline)
        d = info.to_info_dict()
        assert d["inline"]["ids"] == ["1", "2"]
        assert len(d["inline"]["properties"]) == 2
        assert d["inline"]["properties"][0]["type"] == "label"
        assert d["inline"]["properties"][1]["values"] == [100.0, 200.0]


class TestSegmentPropertyField:
    def test_label(self):
        f = SegmentPropertyField(id="name", type="label", values=["a", "b"])
        d = f.to_info_dict()
        assert d == {"id": "name", "type": "label", "values": ["a", "b"]}

    def test_number(self):
        f = SegmentPropertyField(id="area", type="number", values=[1.0, 2.0])
        d = f.to_info_dict()
        assert d["type"] == "number"
        assert d["values"] == [1.0, 2.0]

    def test_tags(self):
        f = SegmentPropertyField(
            id="tags",
            type="tags",
            tags=["soma", "axon"],
            values=[0, 1],
        )
        d = f.to_info_dict()
        assert d["tags"] == ["soma", "axon"]
