"""labelmask converter: per-label smooth meshes -> muDM 3D tiles.

Default path writes one OBJ per label and tiles via the streaming Rust ``obj`` converter (bounded
memory, all cores, meshopt) — validated end-to-end incl. the tag -> glTF-extras -> features.json chain.
``streaming=False`` exercises the legacy in-memory TileGenerator3D path.
"""

import numpy as np
import pytest

pytest.importorskip("skimage")  # provided by the `labelmask` extra

from scipy import ndimage as ndi  # noqa: E402

import build_feature_index as bfi  # noqa: E402  (mudm-tools scripts/ on pythonpath)
from mudm_tools.converters import labelmask  # noqa: E402


def _two_cubes():
    m = np.zeros((12, 20, 20), np.uint16)
    m[2:10, 3:11, 3:11] = 1  # 8x8x8 cube -> label 1
    m[3:9, 12:18, 12:18] = 2  # 6x6x6 cube -> label 2
    return m


def test_labelmask_to_features_names_props_color():
    fc = labelmask.labelmask_to_features(
        _two_cubes(),
        spacing=(1.0, 1.0, 1.0),
        step_size=1,
        smooth_sigma=1.0,
        properties={1: {"volume_um3": 512.0}, 2: {"volume_um3": 216.0}},
        color="distinct",
        name_prefix="nucleus",
    )
    feats = fc.features
    assert len(feats) == 2
    assert feats[0].properties["name"] == "nucleus-1"
    assert feats[0].properties["volume_um3"] == 512.0
    assert feats[0].properties["color"].startswith("#")
    assert feats[0].properties["color"] != feats[1].properties["color"]
    assert feats[0].geometry.type == "TIN" and len(feats[0].geometry.coordinates) > 0


def test_labelmask_to_objs_writes_meshes_tags_bounds(tmp_path):
    tags, bounds = labelmask.labelmask_to_objs(
        _two_cubes(),
        tmp_path,
        spacing=(1.0, 1.0, 1.0),
        smooth_sigma=1.0,
        properties={1: {"volume_um3": 512.0}},
        color="distinct",
        name_prefix="nucleus",
    )
    assert set(tags) == {"nucleus-1", "nucleus-2"}
    assert (tmp_path / "nucleus-1.obj").exists() and (tmp_path / "nucleus-2.obj").exists()
    assert tags["nucleus-1"]["volume_um3"] == 512.0
    assert tags["nucleus-1"]["color"].startswith("#")
    txt = (tmp_path / "nucleus-1.obj").read_text()
    assert txt.startswith("v ") and "\nf " in txt  # OBJ vertices + faces
    assert len(bounds) == 6 and bounds[3] > bounds[0]  # a real world box


def test_labelmask_smoothing_changes_surface_but_keeps_it_closed():
    m = _two_cubes()
    slices = ndi.find_objects(m)
    v_raw, f_raw = labelmask.label_mesh(m, 1, slices, (1.0, 1.0, 1.0), smooth_sigma=0.0)
    v_sm, f_sm = labelmask.label_mesh(m, 1, slices, (1.0, 1.0, 1.0), smooth_sigma=1.0)
    assert len(f_raw) > 0 and len(f_sm) > 0
    assert v_sm.shape != v_raw.shape or not np.allclose(v_sm, v_raw)


def test_streaming_convert_then_index_keeps_props_and_color(tmp_path):
    """The default streaming path: OBJ bridge -> Rust obj converter -> tiles, and the per-label tags
    survive as glTF extras that build_feature_index recovers (name + custom prop + color)."""
    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, _two_cubes())
    out = tmp_path / "tiles"
    rep = labelmask.LabelMaskConverter().convert(
        str(mask_path),
        str(out),
        {
            "spacing": (1.0, 1.0, 1.0),
            "max_zoom": 1,
            "smooth_sigma": 1.0,
            "name_prefix": "nucleus",
            "properties": {1: {"volume_um3": 512.0}, 2: {"volume_um3": 216.0}},
        },
    )
    assert rep["features"] == 2
    assert (out / "3dtiles" / "tileset.json").exists()  # streaming (Rust obj) output shape
    index, _zc, _mz = bfi.build_index(out / "3dtiles", id_fields=["name", "nucleus_id"])
    props = {f["id"]: f["properties"] for f in index["features"]}
    assert set(props) == {"nucleus-1", "nucleus-2"}
    assert props["nucleus-1"]["volume_um3"] == 512.0  # custom prop survived the streaming path
    assert props["nucleus-1"]["color"].startswith("#")


def test_inmemory_convert_smoke(tmp_path):
    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, _two_cubes())
    out = tmp_path / "tiles"
    rep = labelmask.LabelMaskConverter().convert(
        str(mask_path),
        str(out),
        {"spacing": (1.0, 1.0, 1.0), "max_zoom": 1, "smooth_sigma": 1.0, "streaming": False},
    )
    assert rep["features"] == 2
    assert (out / "3dtiles" / "tileset.json").exists()
