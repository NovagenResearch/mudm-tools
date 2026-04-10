"""Tests for Neuroglancer viewer state URL generation."""

import json
import urllib.parse

import pytest

from mudm_tools.neuroglancer.state import (
    build_annotation_layer,
    build_skeleton_layer,
    build_viewer_state,
    viewer_state_to_url,
)


class TestBuildSkeletonLayer:
    def test_structure(self):
        layer = build_skeleton_layer("neurons", "precomputed://http://host/skel")
        assert layer["type"] == "segmentation"
        assert layer["source"] == "precomputed://http://host/skel"
        assert layer["name"] == "neurons"
        assert layer["selectedAlpha"] == 0
        assert layer["notSelectedAlpha"] == 0

    def test_radius_shader_default(self):
        layer = build_skeleton_layer("n", "precomputed://x")
        assert "skeletonRendering" in layer
        assert "shader" in layer["skeletonRendering"]
        assert "getDataValue(0)" in layer["skeletonRendering"]["shader"]

    def test_radius_shader_disabled(self):
        layer = build_skeleton_layer("n", "precomputed://x", use_radius=False)
        assert "skeletonRendering" not in layer


class TestBuildAnnotationLayer:
    def test_structure(self):
        layer = build_annotation_layer("markers", "precomputed://http://host/ann")
        assert layer["type"] == "annotation"
        assert layer["source"] == "precomputed://http://host/ann"
        assert layer["name"] == "markers"


class TestBuildViewerState:
    def test_layers_only(self):
        layers = [build_skeleton_layer("n", "precomputed://x")]
        state = build_viewer_state(layers)
        assert "layers" in state
        assert len(state["layers"]) == 1
        assert state["layout"] == "3d"
        assert "navigation" not in state

    def test_with_position(self):
        layers = [build_skeleton_layer("n", "precomputed://x")]
        state = build_viewer_state(layers, position=[100.0, 200.0, 50.0])
        assert state["navigation"]["pose"]["position"]["voxelCoordinates"] == [
            100.0, 200.0, 50.0
        ]

    def test_multiple_layers(self):
        layers = [
            build_skeleton_layer("skel", "precomputed://a"),
            build_annotation_layer("ann", "precomputed://b"),
        ]
        state = build_viewer_state(layers)
        assert len(state["layers"]) == 2
        assert state["layers"][0]["type"] == "segmentation"
        assert state["layers"][1]["type"] == "annotation"


class TestViewerStateToUrl:
    def test_contains_base_url(self):
        state = {"layers": []}
        url = viewer_state_to_url(state)
        assert url.startswith("https://neuroglancer-demo.appspot.com/#!")

    def test_custom_base_url(self):
        state = {"layers": []}
        url = viewer_state_to_url(state, base_url="http://localhost:8080")
        assert url.startswith("http://localhost:8080/#!")

    def test_roundtrip(self):
        """The JSON in the URL fragment should decode back to the original state."""
        state = build_viewer_state(
            [build_skeleton_layer("test", "precomputed://http://host/data")],
            position=[1.0, 2.0, 3.0],
        )
        url = viewer_state_to_url(state)
        fragment = url.split("#!")[1]
        decoded = json.loads(urllib.parse.unquote(fragment))
        assert decoded["layers"][0]["name"] == "test"
        assert decoded["navigation"]["pose"]["position"]["voxelCoordinates"] == [
            1.0, 2.0, 3.0
        ]
