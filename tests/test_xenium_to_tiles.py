"""Tests for Xenium -> muDM tile conversion pipeline."""

import json
import numpy as np
import polars as pl
import pytest
from pathlib import Path

from PIL import Image

from scripts.xenium_to_tiles import boundaries_to_geojson, transcripts_to_geojson, read_um_per_px, generate_raster_tiles, generate_vector_tiles, write_metadata, convert_xenium


@pytest.fixture
def tmp_boundary_parquet(tmp_path):
    """Create a minimal cell_boundaries.parquet with 2 cells."""
    df = pl.DataFrame({
        "cell_id": ["cell_1", "cell_1", "cell_1", "cell_1",
                     "cell_2", "cell_2", "cell_2", "cell_2"],
        "vertex_x": [10.0, 30.0, 20.0, 10.0,
                     50.0, 70.0, 60.0, 50.0],
        "vertex_y": [20.0, 20.0, 40.0, 20.0,
                     60.0, 60.0, 80.0, 60.0],
    })
    path = tmp_path / "cell_boundaries.parquet"
    df.write_parquet(path)
    return path


def test_boundaries_to_geojson(tmp_boundary_parquet):
    fc = boundaries_to_geojson(tmp_boundary_parquet, id_column="cell_id")
    parsed = json.loads(fc)

    assert parsed["type"] == "FeatureCollection"
    assert len(parsed["features"]) == 2

    f0 = parsed["features"][0]
    assert f0["type"] == "Feature"
    assert f0["geometry"]["type"] == "Polygon"
    assert f0["properties"]["cell_id"] == "cell_1"
    coords = f0["geometry"]["coordinates"][0]
    assert len(coords) == 4  # 3 vertices + closed
    assert coords[0] == [10.0, 20.0]
    assert coords[-1] == coords[0]  # closed ring


@pytest.fixture
def tmp_transcript_parquet(tmp_path):
    """Create a minimal transcripts.parquet with 3 transcripts."""
    df = pl.DataFrame({
        "feature_name": ["TNC", "GAPDH", "TNC"],
        "x_location": [15.0, 55.0, 25.0],
        "y_location": [25.0, 65.0, 35.0],
    })
    path = tmp_path / "transcripts.parquet"
    df.write_parquet(path)
    return path


def test_transcripts_to_geojson(tmp_transcript_parquet):
    fc = transcripts_to_geojson(tmp_transcript_parquet)
    parsed = json.loads(fc)

    assert parsed["type"] == "FeatureCollection"
    assert len(parsed["features"]) == 3

    f0 = parsed["features"][0]
    assert f0["geometry"]["type"] == "Point"
    assert f0["geometry"]["coordinates"] == [15.0, 25.0]
    assert f0["properties"]["gene_name"] == "TNC"


@pytest.fixture
def tmp_experiment_xenium(tmp_path):
    """Create a minimal experiment.xenium JSON."""
    data = {"pixel_size": 0.2125}
    path = tmp_path / "experiment.xenium"
    path.write_text(json.dumps(data))
    return path


def test_read_um_per_px(tmp_experiment_xenium):
    result = read_um_per_px(tmp_experiment_xenium)
    assert result == pytest.approx(0.2125)


@pytest.fixture
def tmp_dapi_image(tmp_path):
    """Create a 512x512 grayscale test image."""
    rng = np.random.default_rng(42)
    img = rng.integers(0, 65535, size=(512, 512), dtype=np.uint16)
    import tifffile
    tifffile.imwrite(tmp_path / "dapi.tif", img)
    return tmp_path / "dapi.tif"


def test_generate_raster_tiles(tmp_dapi_image, tmp_path):
    output_dir = tmp_path / "raster_out"
    info = generate_raster_tiles(tmp_dapi_image, output_dir, tile_size=256)

    # 512x512 image with 256 tiles -> max_zoom=1 (2x2 tiles at max zoom)
    assert info["max_zoom"] == 1
    assert info["image_size_px"] == [512, 512]

    # Check zoom 1 tiles exist (2x2 grid)
    assert (output_dir / "1" / "0" / "0.png").exists()
    assert (output_dir / "1" / "1" / "0.png").exists()
    assert (output_dir / "1" / "0" / "1.png").exists()
    assert (output_dir / "1" / "1" / "1.png").exists()

    # Check zoom 0 tile exists (1x1)
    assert (output_dir / "0" / "0" / "0.png").exists()

    # Verify tiles are valid PNGs of correct size
    tile = Image.open(output_dir / "1" / "0" / "0.png")
    assert tile.size == (256, 256)


def test_generate_vector_tiles(tmp_boundary_parquet, tmp_path):
    output_dir = tmp_path / "vector_out"
    bounds = (0.0, 0.0, 100.0, 100.0)

    geojson_str = boundaries_to_geojson(tmp_boundary_parquet, id_column="cell_id")

    info = generate_vector_tiles(
        geojson_str=geojson_str,
        output_dir=output_dir,
        bounds=bounds,
        layer_name="cells",
        min_zoom=0,
        max_zoom=1,
    )

    assert info["feature_count"] == 2
    assert info["tile_count"] > 0
    # MVT tiles directory should exist
    assert (output_dir / "cells" / "0").exists()


def test_generate_transcript_vector_tiles(tmp_transcript_parquet, tmp_path):
    output_dir = tmp_path / "transcript_out"
    bounds = (0.0, 0.0, 100.0, 100.0)

    geojson_str = transcripts_to_geojson(tmp_transcript_parquet)

    info = generate_vector_tiles(
        geojson_str=geojson_str,
        output_dir=output_dir,
        bounds=bounds,
        layer_name="transcripts",
        min_zoom=0,
        max_zoom=1,
    )

    assert info["feature_count"] == 3
    assert info["tile_count"] > 0


def test_write_metadata(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    write_metadata(
        output_dir=tmp_path,
        name="test_dataset",
        um_per_px=0.2125,
        bounds_um=(0.0, 0.0, 100.0, 80.0),
        raster_info={"max_zoom": 3, "image_size_px": [512, 410]},
        vector_infos={
            "cells": {"feature_count": 10, "tile_count": 5, "type": "polygon"},
            "nuclei": {"feature_count": 10, "tile_count": 5, "type": "polygon"},
            "transcripts": {"feature_count": 100, "tile_count": 8, "type": "point"},
        },
        max_zoom=3,
    )

    meta = json.loads(metadata_path.read_text())
    assert meta["name"] == "test_dataset"
    assert meta["um_per_px"] == 0.2125
    assert len(meta["vectors"]["layers"]) == 3
    assert meta["raster"]["max_zoom"] == 3
    assert meta["parquet"]["path"] == "features.parquet"
    assert meta["parquet"]["partitioned"] is True


@pytest.fixture
def tmp_xenium_dataset(tmp_path, tmp_boundary_parquet, tmp_transcript_parquet, tmp_experiment_xenium):
    """Create a minimal Xenium-like dataset directory."""
    data_dir = tmp_path / "xenium_dataset"
    data_dir.mkdir()

    # Copy parquet files
    import shutil
    shutil.copy(tmp_boundary_parquet, data_dir / "cell_boundaries.parquet")
    shutil.copy(tmp_boundary_parquet, data_dir / "nucleus_boundaries.parquet")
    shutil.copy(tmp_transcript_parquet, data_dir / "transcripts.parquet")
    shutil.copy(tmp_experiment_xenium, data_dir / "experiment.xenium")

    # Create a small DAPI image
    import tifffile
    rng = np.random.default_rng(42)
    img = rng.integers(0, 65535, size=(256, 256), dtype=np.uint16)
    morph_dir = data_dir / "morphology_focus"
    morph_dir.mkdir()
    tifffile.imwrite(morph_dir / "ch0000_dapi.ome.tif", img)

    return data_dir


def test_convert_xenium_end_to_end(tmp_xenium_dataset, tmp_path):
    output_dir = tmp_path / "output"

    convert_xenium(
        data_dir=tmp_xenium_dataset,
        output_dir=output_dir,
        max_zoom=1,
        temp_dir=str(tmp_path / "tmp"),
    )

    # Check metadata.json
    meta_path = output_dir / "metadata.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["platform"] == "xenium"
    assert meta["um_per_px"] == 0.2125
    assert len(meta["vectors"]["layers"]) == 3

    # Check raster tiles
    assert (output_dir / "raster" / "0" / "0" / "0.png").exists()

    # Check combined MVT tiles
    assert (output_dir / "vectors").exists()

    # Check partitioned Parquet
    assert (output_dir / "features.parquet").exists()
    assert (output_dir / "features.parquet" / "zoom=0").exists()

    # Check TileJSON
    assert (output_dir / "vectors" / "metadata.json").exists()
