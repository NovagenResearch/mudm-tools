"""DAPI morphology discovery across the Xenium output layouts 10x ships per XOA version.

Regression guard: XOA 2.0/3.0 bundles ship numbered per-channel files
(``morphology_focus/morphology_focus_0000.ome.tif``, channel 0 = DAPI) rather than the named
``ch0000_dapi.ome.tif`` (XOA 4.0). The converter previously only knew the named/single-file layouts and
silently produced an empty raster for the numbered layout.
"""

from __future__ import annotations

from mudm_tools.converters.xenium import _find_morphology_image


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_finds_named_dapi(tmp_path):
    p = _touch(tmp_path / "morphology_focus" / "ch0000_dapi.ome.tif")
    assert _find_morphology_image(tmp_path) == p


def test_finds_numbered_channel_zero(tmp_path):
    # XOA 2.0/3.0 layout: numbered per-channel files, channel 0 = DAPI.
    p = _touch(tmp_path / "morphology_focus" / "morphology_focus_0000.ome.tif")
    assert _find_morphology_image(tmp_path) == p


def test_finds_single_file(tmp_path):
    p = _touch(tmp_path / "morphology_focus.ome.tif")
    assert _find_morphology_image(tmp_path) == p


def test_absent_returns_none(tmp_path):
    assert _find_morphology_image(tmp_path) is None


def test_prefers_named_over_numbered(tmp_path):
    named = _touch(tmp_path / "morphology_focus" / "ch0000_dapi.ome.tif")
    _touch(tmp_path / "morphology_focus" / "morphology_focus_0000.ome.tif")
    assert _find_morphology_image(tmp_path) == named
