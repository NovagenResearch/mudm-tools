"""Tests for the overview poster generator (mudm_tools.viewers.overview).

The viewer's OverviewPanel drops each poster onto a world-space quad spanning the
octree-root bounds, so the world<->pixel mapping must be linear with a specific
orientation: column 0 = h_min (left), row 0 = v_max (top), axes NOT inverted.
"""
import json

import numpy as np
import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from mudm_tools.viewers.overview import render_overview  # noqa: E402


def _write_pyramid(tmp_path, color="#ff0000"):
    pyr = tmp_path / "pyr"
    pyr.mkdir()
    # 3D Tiles box = [cx,cy,cz, hx,0,0, 0,hy,0, 0,0,hz]; center (0,0,0), half-extent 10.
    box = [0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 10]
    (pyr / "tileset.json").write_text(json.dumps({"root": {"boundingVolume": {"box": box}}}))
    # legacy dict-format features.json: name -> properties (flywire uses this shape)
    (pyr / "features.json").write_text(
        json.dumps({"features": {"1": {"body_id": 1, "color": color}}})
    )
    meshes = tmp_path / "meshes"
    meshes.mkdir()
    # OBJ stem "1" matches body_id 1. Verts at center + two opposite corners.
    verts = [(0, 0, 0), (10, 10, 10), (-10, -10, -10)]
    (meshes / "1.obj").write_text(
        "\n".join(f"v {x} {y} {z}" for x, y, z in verts) + "\nf 1 2 3\n"
    )
    return meshes, pyr


def test_overview_structure_and_orientation(tmp_path):
    meshes, pyr = _write_pyramid(tmp_path)
    # supersample=1 → deterministic 1px-per-vertex placement (no AA blur).
    meta = render_overview(meshes, pyr, target_px=5, supersample=1, workers=1)

    # --- overview.json contract (consumed by OverviewPanel.loadPosters) ---
    assert set(meta["planes"]) == {"xy", "xz", "yz"}
    assert meta["planes"]["xy"] == "xy.png"
    assert meta["bounds"] == [-10, -10, -10, 10, 10, 10]
    assert (pyr / "overview" / "overview.json").is_file()
    for plane in ("xy", "xz", "yz"):
        assert (pyr / "overview" / f"{plane}.png").is_file()

    # --- orientation on the XY plane (h=x, v=y) ---
    img = np.asarray(Image.open(pyr / "overview" / "xy.png").convert("RGB"))
    assert img.shape[:2] == (5, 5)
    red = (255, 0, 0)
    assert tuple(img[2, 2]) == red          # center vertex -> center pixel
    assert tuple(img[0, 4]) == red          # (x=+10, y=+10) -> top-right (row0=v_max, col_max=h_max)
    assert tuple(img[4, 0]) == red          # (x=-10, y=-10) -> bottom-left
    assert tuple(img[0, 0]) != red          # empty corner -> background, not the feature color


def test_overview_accepts_inmemory_bounds(tmp_path):
    # When bounds are passed in, tileset.json is not required to derive them.
    meshes, pyr = _write_pyramid(tmp_path)
    (pyr / "tileset.json").unlink()
    meta = render_overview(
        meshes, pyr, target_px=5, supersample=1, workers=1,
        bounds=[-10, -10, -10, 10, 10, 10],
    )
    assert meta["bounds"] == [-10, -10, -10, 10, 10, 10]
    assert (pyr / "overview" / "xy.png").is_file()
