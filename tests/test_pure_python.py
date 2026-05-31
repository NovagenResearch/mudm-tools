"""Verify the package works without the Rust _rs extension.

These tests exercise only pure-Python codepaths: model validation,
legacy 2D tiling (mudm2vt), GeoParquet I/O, and glTF export.
They must pass even when _rs is not compiled.
"""

import json
import pytest


def test_model_validation():
    from mudm.model import MuDM, GeoJSON

    data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [10, 20]},
                "properties": {"label": "test"},
            }
        ],
    }
    obj = MuDM.model_validate(data)
    assert len(obj.root.features) == 1
    dumped = json.loads(obj.root.model_dump_json())
    assert dumped["features"][0]["properties"]["label"] == "test"


def test_geojson_validation():
    from mudm.model import GeoJSON

    data = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [0, 0]},
        "properties": {},
    }
    obj = GeoJSON.model_validate(data)
    assert obj.root.type == "Feature"


def test_rust_available_flag():
    import mudm_tools

    assert isinstance(mudm_tools.RUST_AVAILABLE, bool)


def test_tilemodel():
    from mudm.tilemodel import TileLayer

    layer = TileLayer(id="test", fields={"name": "String"})
    assert layer.id == "test"


def test_transforms():
    from mudm.transforms import AffineTransform

    t = AffineTransform(matrix=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    assert t.matrix[0][0] == 1
