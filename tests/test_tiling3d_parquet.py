"""Tests for Parquet output (tile-centric ML-ready format).

Covers: generate_parquet(), read_parquet(), schema, row groups, mesh data,
tags, filtering, world coordinates, multi-feature, edge cases, compression.
"""

import struct
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

try:
    from mudm_tools._rs import StreamingTileGenerator
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

from mudm_tools.tiling3d.parquet_writer import generate_parquet
from mudm_tools.tiling3d.parquet_reader import read_parquet
from mudm_tools.tiling3d.parquet_prime import prime_parquet, deprime_parquet, repartition_parquet

pytestmark = pytest.mark.skipif(
    not RUST_AVAILABLE, reason="Rust extensions not compiled"
)


# ---------------------------------------------------------------------------
# Helpers (same patterns as test_tiling3d_feature_pbf3.py)
# ---------------------------------------------------------------------------

def _make_tin_feature(xy, z, ring_lengths, tags=None):
    """Build a TIN feature dict in normalized [0,1]³ space."""
    n = len(z)
    return {
        "geometry": xy,
        "geometry_z": z,
        "ring_lengths": ring_lengths,
        "type": 5,  # TIN
        "tags": tags or {},
        "minX": min(xy[i * 2] for i in range(n)),
        "minY": min(xy[i * 2 + 1] for i in range(n)),
        "minZ": min(z),
        "maxX": max(xy[i * 2] for i in range(n)),
        "maxY": max(xy[i * 2 + 1] for i in range(n)),
        "maxZ": max(z),
    }


def _make_point_feature(x, y, z, tags=None):
    """Build a Point3D feature dict in normalized [0,1]³ space."""
    return {
        "geometry": [x, y],
        "geometry_z": [z],
        "type": 1,
        "tags": tags or {},
        "minX": x, "minY": y, "minZ": z,
        "maxX": x, "maxY": y, "maxZ": z,
    }


def _make_line_feature(xy, z, tags=None):
    """Build a LineString3D feature dict in normalized [0,1]³ space."""
    n = len(z)
    return {
        "geometry": xy,
        "geometry_z": z,
        "type": 2,
        "tags": tags or {},
        "minX": min(xy[i * 2] for i in range(n)),
        "minY": min(xy[i * 2 + 1] for i in range(n)),
        "minZ": min(z),
        "maxX": max(xy[i * 2] for i in range(n)),
        "maxY": max(xy[i * 2 + 1] for i in range(n)),
        "maxZ": max(z),
    }


WORLD_BOUNDS = (0.0, 0.0, 0.0, 100.0, 200.0, 300.0)


def _build_generator_with_features(features, min_zoom=0, max_zoom=2):
    """Create a StreamingTileGenerator, add features, return (gen, fids)."""
    gen = StreamingTileGenerator(min_zoom=min_zoom, max_zoom=max_zoom)
    fids = []
    for feat in features:
        fid = gen.add_feature(feat)
        fids.append(fid)
    return gen, fids


def _make_dense_tin_feature(n_triangles=20):
    """Build a TIN feature with many triangles spread across [0.1, 0.9]³."""
    import random
    random.seed(42)
    xy = []
    z = []
    ring_lengths = []
    for _ in range(n_triangles):
        cx = random.uniform(0.15, 0.85)
        cy = random.uniform(0.15, 0.85)
        cz = random.uniform(0.15, 0.85)
        d = 0.05
        xy.extend([cx - d, cy - d, cx + d, cy - d, cx, cy + d])
        z.extend([cz - d, cz + d, cz])
        ring_lengths.append(3)
    return _make_tin_feature(xy, z, ring_lengths, tags={"name": "dense"})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParquetProducesFile:
    """Test that generate_parquet creates a valid Parquet file."""

    def test_file_created(self, tmp_path):
        """Parquet file is created on disk."""
        features = [
            _make_tin_feature(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [0.1, 0.2, 0.3],
                [3],
                tags={"name": "mesh_a"},
            ),
        ]
        gen, _ = _build_generator_with_features(features)
        out = tmp_path / "tiles.parquet"
        count = generate_parquet(gen, out, WORLD_BOUNDS)

        assert out.exists()
        assert count > 0

    def test_readable_by_pyarrow(self, tmp_path):
        """Output file is valid Parquet readable by PyArrow."""
        features = [
            _make_tin_feature(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [0.1, 0.2, 0.3],
                [3],
            ),
        ]
        gen, _ = _build_generator_with_features(features)
        out = tmp_path / "tiles.parquet"
        generate_parquet(gen, out, WORLD_BOUNDS)

        table = pq.read_table(str(out))
        assert table.num_rows > 0


class TestParquetSchema:
    """Test that the Parquet schema matches the spec."""

    def test_column_names(self, tmp_path):
        """All 9 expected columns present."""
        features = [
            _make_tin_feature(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [0.1, 0.2, 0.3],
                [3],
            ),
        ]
        gen, _ = _build_generator_with_features(features)
        out = tmp_path / "tiles.parquet"
        generate_parquet(gen, out, WORLD_BOUNDS)

        table = pq.read_table(str(out))
        expected = {"zoom", "tile_x", "tile_y", "tile_d", "feature_id",
                    "geom_type", "positions", "indices", "tags"}
        assert set(table.column_names) == expected

    def test_column_types(self, tmp_path):
        """Column types match the spec."""
        features = [
            _make_tin_feature(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [0.1, 0.2, 0.3],
                [3],
            ),
        ]
        gen, _ = _build_generator_with_features(features)
        out = tmp_path / "tiles.parquet"
        generate_parquet(gen, out, WORLD_BOUNDS)

        schema = pq.read_schema(str(out))
        assert schema.field("zoom").type == pa.uint8()
        assert schema.field("tile_x").type == pa.uint16()
        assert schema.field("tile_y").type == pa.uint16()
        assert schema.field("tile_d").type == pa.uint16()
        assert schema.field("feature_id").type == pa.uint32()
        assert schema.field("geom_type").type == pa.uint8()
        assert schema.field("positions").type == pa.large_binary()
        assert schema.field("indices").type == pa.large_binary()
        assert schema.field("tags").type == pa.map_(pa.utf8(), pa.utf8())

    def test_zoom_values(self, tmp_path):
        """Zoom column contains expected zoom levels."""
        features = [_make_dense_tin_feature(20)]
        gen, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=2)
        out = tmp_path / "tiles.parquet"
        generate_parquet(gen, out, WORLD_BOUNDS)

        table = pq.read_table(str(out))
        zoom_vals = set(table.column("zoom").to_pylist())
        assert zoom_vals == {0, 1, 2}


class TestParquetRowGroups:
    """Test row group structure (one per zoom level)."""

    def test_row_groups_per_zoom(self, tmp_path):
        """Number of row groups equals number of distinct zoom levels."""
        features = [_make_dense_tin_feature(20)]
        gen, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=2)
        out = tmp_path / "tiles.parquet"
        generate_parquet(gen, out, WORLD_BOUNDS)

        pf = pq.ParquetFile(str(out))
        assert pf.metadata.num_row_groups == 3  # zoom 0, 1, 2

    def test_predicate_pushdown(self, tmp_path):
        """Reading with zoom filter uses predicate pushdown."""
        features = [_make_dense_tin_feature(20)]
        gen, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=2)
        out = tmp_path / "tiles.parquet"
        total = generate_parquet(gen, out, WORLD_BOUNDS)

        # Read only zoom=2
        rows = read_parquet(out, zoom=2)
        assert len(rows) > 0
        assert all(r["zoom"] == 2 for r in rows)
        assert len(rows) < total


class TestParquetMeshData:
    """Test mesh positions and indices data format."""

    def test_positions_float32(self, tmp_path):
        """Positions column contains valid float32 triples."""
        features = [
            _make_tin_feature(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [0.1, 0.2, 0.3],
                [3],
            ),
        ]
        gen, _ = _build_generator_with_features(features)
        out = tmp_path / "tiles.parquet"
        generate_parquet(gen, out, WORLD_BOUNDS)

        rows = read_parquet(out, zoom=2)
        assert len(rows) > 0
        pos = rows[0]["positions"]
        assert pos.dtype == np.float32
        assert pos.ndim == 2
        assert pos.shape[1] == 3

    def test_indices_uint32(self, tmp_path):
        """Indices column contains valid uint32 triangle indices."""
        features = [
            _make_tin_feature(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [0.1, 0.2, 0.3],
                [3],
            ),
        ]
        gen, _ = _build_generator_with_features(features)
        out = tmp_path / "tiles.parquet"
        generate_parquet(gen, out, WORLD_BOUNDS)

        rows = read_parquet(out, zoom=2)
        assert len(rows) > 0
        idx = rows[0]["indices"]
        assert idx.dtype == np.uint32
        # Indices should reference valid positions
        n_verts = rows[0]["positions"].shape[0]
        assert all(i < n_verts for i in idx)

    def test_triangle_count(self, tmp_path):
        """TIN features produce triangle indices (multiples of 3)."""
        features = [
            _make_tin_feature(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                 0.6, 0.7, 0.8, 0.8, 0.7, 0.9],
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [3, 3],
            ),
        ]
        gen, _ = _build_generator_with_features(features)
        out = tmp_path / "tiles.parquet"
        generate_parquet(gen, out, WORLD_BOUNDS)

        rows = read_parquet(out, zoom=2)
        for row in rows:
            if row["geom_type"] == 5:
                assert len(row["indices"]) % 3 == 0


class TestParquetTags:
    """Test tag roundtrip through Parquet."""

    def test_string_tags(self, tmp_path):
        """String tag values survive roundtrip."""
        features = [
            _make_tin_feature(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [0.1, 0.2, 0.3],
                [3],
                tags={"species": "mouse", "region": "cortex"},
            ),
        ]
        gen, _ = _build_generator_with_features(features)
        out = tmp_path / "tiles.parquet"
        generate_parquet(gen, out, WORLD_BOUNDS)

        rows = read_parquet(out)
        tags = rows[0]["tags"]
        assert tags["species"] == "mouse"
        assert tags["region"] == "cortex"

    def test_numeric_tags_as_strings(self, tmp_path):
        """Numeric tag values are coerced to strings in Parquet."""
        features = [
            _make_tin_feature(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [0.1, 0.2, 0.3],
                [3],
                tags={"volume": 42.5, "neuron_id": 12345, "active": True},
            ),
        ]
        gen, _ = _build_generator_with_features(features)
        out = tmp_path / "tiles.parquet"
        generate_parquet(gen, out, WORLD_BOUNDS)

        rows = read_parquet(out)
        tags = rows[0]["tags"]
        assert tags["volume"] == "42.5"
        assert tags["neuron_id"] == "12345"
        assert tags["active"] == "true"


class TestParquetFiltering:
    """Test read_parquet filtering by zoom, feature_id, tile coords."""

    def test_filter_by_zoom(self, tmp_path):
        """Filtering by zoom returns only matching rows."""
        features = [_make_dense_tin_feature(20)]
        gen, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=2)
        out = tmp_path / "tiles.parquet"
        generate_parquet(gen, out, WORLD_BOUNDS)

        rows = read_parquet(out, zoom=0)
        assert len(rows) > 0
        assert all(r["zoom"] == 0 for r in rows)

    def test_filter_by_feature_id(self, tmp_path):
        """Filtering by feature_id returns only matching rows."""
        features = [
            _make_tin_feature(
                [0.1, 0.1, 0.2, 0.1, 0.15, 0.2],
                [0.1, 0.2, 0.15],
                [3],
                tags={"name": "a"},
            ),
            _make_tin_feature(
                [0.6, 0.6, 0.7, 0.6, 0.65, 0.7],
                [0.5, 0.6, 0.55],
                [3],
                tags={"name": "b"},
            ),
        ]
        gen, fids = _build_generator_with_features(features)
        out = tmp_path / "tiles.parquet"
        generate_parquet(gen, out, WORLD_BOUNDS)

        rows = read_parquet(out, feature_id=fids[1])
        assert len(rows) > 0
        assert all(r["feature_id"] == fids[1] for r in rows)

    def test_combined_filters(self, tmp_path):
        """Multiple filters can be combined."""
        features = [_make_dense_tin_feature(20)]
        gen, fids = _build_generator_with_features(features, min_zoom=0, max_zoom=2)
        out = tmp_path / "tiles.parquet"
        generate_parquet(gen, out, WORLD_BOUNDS)

        rows = read_parquet(out, zoom=2, feature_id=fids[0])
        assert len(rows) > 0
        assert all(r["zoom"] == 2 and r["feature_id"] == fids[0] for r in rows)


class TestParquetWorldCoords:
    """Test that positions are in world coordinate range."""

    def test_all_positions_within_bounds(self, tmp_path):
        """All mesh positions fall within world bounds (with small tolerance)."""
        features = [_make_dense_tin_feature(20)]
        gen, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=2)
        out = tmp_path / "tiles.parquet"
        generate_parquet(gen, out, WORLD_BOUNDS)

        xmin, ymin, zmin, xmax, ymax, zmax = WORLD_BOUNDS
        tol = 5.0  # small tolerance for clustering centroids near edges

        rows = read_parquet(out)
        for row in rows:
            pos = row["positions"]
            if pos.size == 0:
                continue
            assert pos[:, 0].min() >= xmin - tol, f"x below bounds in zoom {row['zoom']}"
            assert pos[:, 0].max() <= xmax + tol, f"x above bounds in zoom {row['zoom']}"
            assert pos[:, 1].min() >= ymin - tol, f"y below bounds in zoom {row['zoom']}"
            assert pos[:, 1].max() <= ymax + tol, f"y above bounds in zoom {row['zoom']}"
            assert pos[:, 2].min() >= zmin - tol, f"z below bounds in zoom {row['zoom']}"
            assert pos[:, 2].max() <= zmax + tol, f"z above bounds in zoom {row['zoom']}"


class TestParquetMultiFeature:
    """Test with multiple features."""

    def test_multiple_features_present(self, tmp_path):
        """All feature IDs appear in the Parquet output."""
        features = [
            _make_tin_feature(
                [0.1, 0.1, 0.2, 0.1, 0.15, 0.2],
                [0.1, 0.2, 0.15],
                [3],
                tags={"name": "alpha"},
            ),
            _make_tin_feature(
                [0.5, 0.5, 0.6, 0.5, 0.55, 0.6],
                [0.4, 0.5, 0.45],
                [3],
                tags={"name": "beta"},
            ),
            _make_tin_feature(
                [0.8, 0.8, 0.9, 0.8, 0.85, 0.9],
                [0.7, 0.8, 0.75],
                [3],
                tags={"name": "gamma"},
            ),
        ]
        gen, fids = _build_generator_with_features(features)
        out = tmp_path / "tiles.parquet"
        generate_parquet(gen, out, WORLD_BOUNDS)

        table = pq.read_table(str(out))
        present_fids = set(table.column("feature_id").to_pylist())
        assert set(fids) == present_fids

    def test_correct_tags_per_feature(self, tmp_path):
        """Each feature's rows have correct tags."""
        features = [
            _make_tin_feature(
                [0.1, 0.1, 0.2, 0.1, 0.15, 0.2],
                [0.1, 0.2, 0.15],
                [3],
                tags={"name": "alpha"},
            ),
            _make_tin_feature(
                [0.6, 0.6, 0.7, 0.6, 0.65, 0.7],
                [0.5, 0.6, 0.55],
                [3],
                tags={"name": "beta"},
            ),
        ]
        gen, fids = _build_generator_with_features(features)
        out = tmp_path / "tiles.parquet"
        generate_parquet(gen, out, WORLD_BOUNDS)

        for fid, expected_name in zip(fids, ["alpha", "beta"]):
            rows = read_parquet(out, feature_id=fid)
            assert len(rows) > 0
            assert all(r["tags"]["name"] == expected_name for r in rows)


class TestParquetEdgeCases:
    """Test edge cases."""

    def test_empty_generator(self, tmp_path):
        """Empty generator produces a valid empty Parquet file."""
        gen = StreamingTileGenerator(min_zoom=0, max_zoom=2)
        out = tmp_path / "tiles.parquet"
        count = generate_parquet(gen, out, WORLD_BOUNDS)

        assert count == 0
        assert out.exists()
        table = pq.read_table(str(out))
        assert table.num_rows == 0

    def test_point_features(self, tmp_path):
        """Point features produce valid Parquet rows with empty indices."""
        features = [
            _make_point_feature(0.5, 0.5, 0.5, tags={"label": "center"}),
        ]
        gen, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=0)
        out = tmp_path / "tiles.parquet"
        count = generate_parquet(gen, out, WORLD_BOUNDS)

        assert count > 0
        rows = read_parquet(out)
        point_rows = [r for r in rows if r["geom_type"] == 1]
        assert len(point_rows) > 0
        for r in point_rows:
            assert r["positions"].shape[1] == 3
            assert len(r["indices"]) == 0


class TestParquetCompression:
    """Test ZSTD compression produces smaller files."""

    def test_zstd_smaller_than_none(self, tmp_path):
        """ZSTD-compressed file is smaller than uncompressed."""
        features = [_make_dense_tin_feature(50)]
        gen1, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=3)
        out_zstd = tmp_path / "zstd.parquet"
        generate_parquet(gen1, out_zstd, WORLD_BOUNDS, compression="zstd")

        gen2, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=3)
        out_none = tmp_path / "none.parquet"
        generate_parquet(gen2, out_none, WORLD_BOUNDS, compression="none")

        assert out_zstd.stat().st_size < out_none.stat().st_size


# ---------------------------------------------------------------------------
# Streaming Parquet tests
# ---------------------------------------------------------------------------

class TestParquetStreaming:
    """Test streaming batch Parquet generation (single-file mode)."""

    def test_streaming_same_row_count_as_inmemory(self, tmp_path):
        """Streaming and in-memory modes produce the same row count."""
        features = [_make_dense_tin_feature(20)]

        gen1, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=2)
        out1 = tmp_path / "inmemory.parquet"
        count1 = generate_parquet(gen1, out1, WORLD_BOUNDS, batch_size=50_000)

        gen2, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=2)
        out2 = tmp_path / "streaming.parquet"
        count2 = generate_parquet(gen2, out2, WORLD_BOUNDS, batch_size=10)

        assert count1 == count2

    def test_streaming_small_batch_size(self, tmp_path):
        """batch_size=1 still produces valid output."""
        features = [
            _make_tin_feature(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [0.1, 0.2, 0.3],
                [3],
            ),
        ]
        gen, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=1)
        out = tmp_path / "tiny_batch.parquet"
        count = generate_parquet(gen, out, WORLD_BOUNDS, batch_size=1)

        assert count > 0
        table = pq.read_table(str(out))
        assert table.num_rows == count

    def test_streaming_empty_generator(self, tmp_path):
        """Empty generator in streaming mode produces valid empty file."""
        gen = StreamingTileGenerator(min_zoom=0, max_zoom=2)
        out = tmp_path / "empty_streaming.parquet"
        count = generate_parquet(gen, out, WORLD_BOUNDS, batch_size=100)

        assert count == 0
        assert out.exists()
        table = pq.read_table(str(out))
        assert table.num_rows == 0

    def test_streaming_row_groups_per_zoom(self, tmp_path):
        """Single-file streaming produces one row group per zoom."""
        features = [_make_dense_tin_feature(20)]
        gen, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=2)
        out = tmp_path / "streaming_rg.parquet"
        generate_parquet(gen, out, WORLD_BOUNDS, batch_size=5)

        pf = pq.ParquetFile(str(out))
        assert pf.metadata.num_row_groups == 3  # zoom 0, 1, 2

    def test_streaming_preserves_tags(self, tmp_path):
        """Tags survive roundtrip through streaming mode."""
        features = [
            _make_tin_feature(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [0.1, 0.2, 0.3],
                [3],
                tags={"species": "mouse", "region": "cortex"},
            ),
        ]
        gen, _ = _build_generator_with_features(features)
        out = tmp_path / "tags_streaming.parquet"
        generate_parquet(gen, out, WORLD_BOUNDS, batch_size=10)

        rows = read_parquet(out)
        assert len(rows) > 0
        assert rows[0]["tags"]["species"] == "mouse"
        assert rows[0]["tags"]["region"] == "cortex"

    def test_streaming_world_coords(self, tmp_path):
        """Positions from streaming mode fall within world bounds."""
        features = [_make_dense_tin_feature(20)]
        gen, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=2)
        out = tmp_path / "coords_streaming.parquet"
        generate_parquet(gen, out, WORLD_BOUNDS, batch_size=10)

        xmin, ymin, zmin, xmax, ymax, zmax = WORLD_BOUNDS
        tol = 5.0

        rows = read_parquet(out)
        for row in rows:
            pos = row["positions"]
            if pos.size == 0:
                continue
            assert pos[:, 0].min() >= xmin - tol
            assert pos[:, 0].max() <= xmax + tol
            assert pos[:, 1].min() >= ymin - tol
            assert pos[:, 1].max() <= ymax + tol
            assert pos[:, 2].min() >= zmin - tol
            assert pos[:, 2].max() <= zmax + tol


# ---------------------------------------------------------------------------
# Partitioned Parquet tests
# ---------------------------------------------------------------------------

class TestParquetPartitioned:
    """Test partitioned Parquet output (one file per zoom level)."""

    def test_partitioned_creates_directories(self, tmp_path):
        """Partitioned output creates zoom=N/ directories."""
        features = [_make_dense_tin_feature(20)]
        gen, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=2)
        out_dir = tmp_path / "partitioned"
        generate_parquet(gen, out_dir, WORLD_BOUNDS, partitioned=True, batch_size=10)

        for z in range(3):
            part = out_dir / f"zoom={z}" / "part_000.parquet"
            assert part.exists(), f"Missing partition for zoom={z}"

    def test_partitioned_readable_by_dataset(self, tmp_path):
        """pyarrow.dataset reads partitioned directory correctly."""
        features = [_make_dense_tin_feature(20)]
        gen, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=2)
        out_dir = tmp_path / "partitioned_read"
        total = generate_parquet(
            gen, out_dir, WORLD_BOUNDS, partitioned=True, batch_size=10,
        )

        rows = read_parquet(out_dir)
        assert len(rows) == total

    def test_partitioned_zoom_filter(self, tmp_path):
        """Zoom filter works via partition pruning."""
        features = [_make_dense_tin_feature(20)]
        gen, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=2)
        out_dir = tmp_path / "partitioned_filter"
        total = generate_parquet(
            gen, out_dir, WORLD_BOUNDS, partitioned=True, batch_size=10,
        )

        rows_z2 = read_parquet(out_dir, zoom=2)
        assert len(rows_z2) > 0
        assert all(r["zoom"] == 2 for r in rows_z2)
        assert len(rows_z2) < total

    def test_partitioned_same_data_as_single(self, tmp_path):
        """Partitioned and single-file modes produce same row count and feature IDs."""
        features = [
            _make_tin_feature(
                [0.1, 0.1, 0.2, 0.1, 0.15, 0.2],
                [0.1, 0.2, 0.15],
                [3],
                tags={"name": "a"},
            ),
            _make_tin_feature(
                [0.6, 0.6, 0.7, 0.6, 0.65, 0.7],
                [0.5, 0.6, 0.55],
                [3],
                tags={"name": "b"},
            ),
        ]

        gen1, fids1 = _build_generator_with_features(features, min_zoom=0, max_zoom=2)
        out_single = tmp_path / "single.parquet"
        count_single = generate_parquet(gen1, out_single, WORLD_BOUNDS, batch_size=10)

        gen2, fids2 = _build_generator_with_features(features, min_zoom=0, max_zoom=2)
        out_part = tmp_path / "partitioned_cmp"
        count_part = generate_parquet(
            gen2, out_part, WORLD_BOUNDS, partitioned=True, batch_size=10,
        )

        assert count_single == count_part

        rows_s = read_parquet(out_single)
        rows_p = read_parquet(out_part)
        fids_s = sorted(set(r["feature_id"] for r in rows_s))
        fids_p = sorted(set(r["feature_id"] for r in rows_p))
        assert fids_s == fids_p


# ---------------------------------------------------------------------------
# Streaming API low-level tests
# ---------------------------------------------------------------------------

class TestParquetStreamingAPI:
    """Test the low-level _init/_next/_close Parquet streaming API."""

    def test_init_and_next_batch(self, tmp_path):
        """Basic streaming lifecycle: init → next → close."""
        features = [
            _make_tin_feature(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [0.1, 0.2, 0.3],
                [3],
            ),
        ]
        gen, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=1)

        gen._init_parquet_stream()
        try:
            batch = gen._next_parquet_batch(1000, WORLD_BOUNDS)
            assert batch is not None
            assert batch["row_count"] > 0

            # Second call should return None (EOF)
            batch2 = gen._next_parquet_batch(1000, WORLD_BOUNDS)
            assert batch2 is None
        finally:
            gen._close_parquet_stream()

    def test_double_init_raises(self, tmp_path):
        """Calling _init_parquet_stream() twice without close raises RuntimeError."""
        features = [
            _make_tin_feature(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [0.1, 0.2, 0.3],
                [3],
            ),
        ]
        gen, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=1)
        gen._init_parquet_stream()
        try:
            with pytest.raises(RuntimeError):
                gen._init_parquet_stream()
        finally:
            gen._close_parquet_stream()

    def test_batch_without_init_raises(self, tmp_path):
        """Calling _next_parquet_batch() without init raises RuntimeError."""
        features = [
            _make_tin_feature(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [0.1, 0.2, 0.3],
                [3],
            ),
        ]
        gen, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=1)
        with pytest.raises(RuntimeError):
            gen._next_parquet_batch(100, WORLD_BOUNDS)


# ---------------------------------------------------------------------------
# Prime / Deprime tests
# ---------------------------------------------------------------------------

def _make_partitioned_pyramid(tmp_path, min_zoom=0, max_zoom=2):
    """Helper: generate a partitioned Parquet pyramid and return (dir, total_rows)."""
    features = [_make_dense_tin_feature(20)]
    gen, _ = _build_generator_with_features(features, min_zoom=min_zoom, max_zoom=max_zoom)
    out_dir = tmp_path / "pyramid"
    total = generate_parquet(gen, out_dir, WORLD_BOUNDS, partitioned=True, batch_size=10)
    return out_dir, total


class TestParquetPrimeDeprime:
    """Test prime_parquet() and deprime_parquet() Arrow IPC cache layer."""

    # --- prime_parquet ---

    def test_prime_creates_arrow_files(self, tmp_path):
        """prime_parquet creates .arrow sibling next to each .parquet file."""
        out_dir, _ = _make_partitioned_pyramid(tmp_path)
        count = prime_parquet(out_dir)

        assert count == 3  # zoom 0, 1, 2
        for z in range(3):
            assert (out_dir / f"zoom={z}" / "part_000.arrow").exists()

    def test_prime_returns_count(self, tmp_path):
        """prime_parquet returns the number of files written."""
        out_dir, _ = _make_partitioned_pyramid(tmp_path)
        assert prime_parquet(out_dir) == 3

    def test_prime_idempotent(self, tmp_path):
        """Calling prime_parquet twice overwrites without error."""
        out_dir, _ = _make_partitioned_pyramid(tmp_path)
        prime_parquet(out_dir)
        count = prime_parquet(out_dir)
        assert count == 3
        for z in range(3):
            assert (out_dir / f"zoom={z}" / "part_000.arrow").exists()

    def test_prime_preserves_parquet(self, tmp_path):
        """prime_parquet does not remove or modify original parquet files."""
        out_dir, _ = _make_partitioned_pyramid(tmp_path)
        sizes_before = {
            z: (out_dir / f"zoom={z}" / "part_000.parquet").stat().st_size
            for z in range(3)
        }
        prime_parquet(out_dir)
        for z in range(3):
            assert (out_dir / f"zoom={z}" / "part_000.parquet").exists()
            assert (out_dir / f"zoom={z}" / "part_000.parquet").stat().st_size == sizes_before[z]

    def test_prime_compression_lz4(self, tmp_path):
        """prime_parquet with compression='lz4' produces valid readable files."""
        out_dir, _ = _make_partitioned_pyramid(tmp_path)
        count = prime_parquet(out_dir, compression="lz4")
        assert count == 3
        # Verify files are readable
        import pyarrow.feather as feather
        for z in range(3):
            table = feather.read_table(str(out_dir / f"zoom={z}" / "part_000.arrow"))
            assert table.num_rows > 0

    def test_prime_compression_zstd(self, tmp_path):
        """prime_parquet with compression='zstd' produces valid files."""
        out_dir, _ = _make_partitioned_pyramid(tmp_path)
        count = prime_parquet(out_dir, compression="zstd")
        assert count == 3

    def test_prime_invalid_compression_raises(self, tmp_path):
        """prime_parquet raises ValueError for invalid compression."""
        out_dir, _ = _make_partitioned_pyramid(tmp_path)
        with pytest.raises(ValueError, match="compression"):
            prime_parquet(out_dir, compression="brotli")

    def test_prime_nonexistent_path_raises(self, tmp_path):
        """prime_parquet raises FileNotFoundError for missing path."""
        with pytest.raises(FileNotFoundError):
            prime_parquet(tmp_path / "nonexistent")

    def test_prime_file_not_dir_raises(self, tmp_path):
        """prime_parquet raises NotADirectoryError for a file path."""
        f = tmp_path / "file.txt"
        f.write_text("hi")
        with pytest.raises(NotADirectoryError):
            prime_parquet(f)

    # --- deprime_parquet ---

    def test_deprime_removes_arrow_files(self, tmp_path):
        """deprime_parquet removes all .arrow files."""
        out_dir, _ = _make_partitioned_pyramid(tmp_path)
        prime_parquet(out_dir)
        count = deprime_parquet(out_dir)

        assert count == 3
        for z in range(3):
            assert not (out_dir / f"zoom={z}" / "part_000.arrow").exists()

    def test_deprime_leaves_parquet_intact(self, tmp_path):
        """deprime_parquet does not touch .parquet files."""
        out_dir, _ = _make_partitioned_pyramid(tmp_path)
        prime_parquet(out_dir)
        deprime_parquet(out_dir)

        for z in range(3):
            assert (out_dir / f"zoom={z}" / "part_000.parquet").exists()

    def test_deprime_idempotent(self, tmp_path):
        """Calling deprime_parquet on already-deprimed directory returns 0."""
        out_dir, _ = _make_partitioned_pyramid(tmp_path)
        prime_parquet(out_dir)
        deprime_parquet(out_dir)
        count = deprime_parquet(out_dir)
        assert count == 0

    def test_deprime_nonexistent_path_raises(self, tmp_path):
        """deprime_parquet raises FileNotFoundError for missing path."""
        with pytest.raises(FileNotFoundError):
            deprime_parquet(tmp_path / "nonexistent")

    def test_deprime_file_not_dir_raises(self, tmp_path):
        """deprime_parquet raises NotADirectoryError for a file path."""
        f = tmp_path / "file.txt"
        f.write_text("hi")
        with pytest.raises(NotADirectoryError):
            deprime_parquet(f)

    # --- Reader auto-detection ---

    def test_reader_auto_detects_arrow(self, tmp_path):
        """read_parquet auto-detects Arrow IPC when primed."""
        out_dir, total = _make_partitioned_pyramid(tmp_path)
        prime_parquet(out_dir)

        rows = read_parquet(out_dir)
        assert len(rows) == total

    def test_reader_falls_back_to_parquet(self, tmp_path):
        """read_parquet reads parquet when no arrow files exist."""
        out_dir, total = _make_partitioned_pyramid(tmp_path)
        rows = read_parquet(out_dir)
        assert len(rows) == total

    def test_reader_zoom_filter_with_arrow(self, tmp_path):
        """Zoom filtering works with Arrow IPC format."""
        out_dir, total = _make_partitioned_pyramid(tmp_path)
        prime_parquet(out_dir)

        rows = read_parquet(out_dir, zoom=2)
        assert len(rows) > 0
        assert all(r["zoom"] == 2 for r in rows)
        assert len(rows) < total

    # --- Data integrity ---

    def test_arrow_reads_match_parquet_reads(self, tmp_path):
        """Arrow IPC reads produce identical data to Parquet reads."""
        out_dir, _ = _make_partitioned_pyramid(tmp_path)

        rows_pq = read_parquet(out_dir)
        prime_parquet(out_dir)
        rows_arrow = read_parquet(out_dir)

        assert len(rows_pq) == len(rows_arrow)

        # Sort both by (zoom, tile_x, tile_y, feature_id) for deterministic comparison
        key = lambda r: (r["zoom"], r["tile_x"], r["tile_y"], r["feature_id"])
        rows_pq.sort(key=key)
        rows_arrow.sort(key=key)

        for rp, ra in zip(rows_pq, rows_arrow):
            assert rp["zoom"] == ra["zoom"]
            assert rp["tile_x"] == ra["tile_x"]
            assert rp["tile_y"] == ra["tile_y"]
            assert rp["tile_d"] == ra["tile_d"]
            assert rp["feature_id"] == ra["feature_id"]
            assert rp["geom_type"] == ra["geom_type"]
            np.testing.assert_array_equal(rp["positions"], ra["positions"])
            np.testing.assert_array_equal(rp["indices"], ra["indices"])
            assert rp["tags"] == ra["tags"]

    # --- Full lifecycle ---

    def test_full_lifecycle_roundtrip(self, tmp_path):
        """generate -> read -> prime -> read -> deprime -> read all consistent."""
        out_dir, total = _make_partitioned_pyramid(tmp_path)

        # 1. Read from parquet
        rows_1 = read_parquet(out_dir)
        assert len(rows_1) == total

        # 2. Prime and read from arrow
        prime_parquet(out_dir)
        rows_2 = read_parquet(out_dir)
        assert len(rows_2) == total

        # 3. Deprime and read from parquet again
        deprime_parquet(out_dir)
        rows_3 = read_parquet(out_dir)
        assert len(rows_3) == total


# ---------------------------------------------------------------------------
# File splitting tests (writer-side rotation)
# ---------------------------------------------------------------------------

class TestParquetFileSplitting:
    """Test size-based file rotation in partitioned output."""

    def test_rotation_creates_multiple_parts(self, tmp_path):
        """With a tiny max_file_bytes, rotation creates multiple part files."""
        features = [_make_dense_tin_feature(50)]
        gen, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=2)
        out_dir = tmp_path / "split"
        generate_parquet(
            gen, out_dir, WORLD_BOUNDS,
            partitioned=True, batch_size=5, max_file_bytes=1,  # 1 byte → always rotate
        )

        # At least one zoom level should have >1 part file
        max_parts = 0
        for z in range(3):
            parts = sorted((out_dir / f"zoom={z}").glob("part_*.parquet"))
            max_parts = max(max_parts, len(parts))
        assert max_parts > 1

    def test_rotation_correct_naming(self, tmp_path):
        """Part files follow part_NNN.parquet naming."""
        features = [_make_dense_tin_feature(50)]
        gen, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=2)
        out_dir = tmp_path / "naming"
        generate_parquet(
            gen, out_dir, WORLD_BOUNDS,
            partitioned=True, batch_size=5, max_file_bytes=1,
        )

        import re
        for z in range(3):
            parts = sorted((out_dir / f"zoom={z}").glob("part_*.parquet"))
            for p in parts:
                assert re.fullmatch(r"part_\d{3}\.parquet", p.name), f"Bad name: {p.name}"

    def test_rotation_readable_by_dataset(self, tmp_path):
        """Split output is readable via read_parquet (Hive auto-discovery)."""
        features = [_make_dense_tin_feature(50)]
        gen, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=2)
        out_dir = tmp_path / "readable"
        total = generate_parquet(
            gen, out_dir, WORLD_BOUNDS,
            partitioned=True, batch_size=5, max_file_bytes=1,
        )

        rows = read_parquet(out_dir)
        assert len(rows) == total

    def test_rotation_zoom_filter_works(self, tmp_path):
        """Zoom filtering works with split files."""
        features = [_make_dense_tin_feature(50)]
        gen, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=2)
        out_dir = tmp_path / "filter"
        total = generate_parquet(
            gen, out_dir, WORLD_BOUNDS,
            partitioned=True, batch_size=5, max_file_bytes=1,
        )

        rows = read_parquet(out_dir, zoom=2)
        assert len(rows) > 0
        assert all(r["zoom"] == 2 for r in rows)
        assert len(rows) < total

    def test_rotation_same_data_as_unsplit(self, tmp_path):
        """Split and unsplit produce the same row count and feature IDs."""
        features = [_make_dense_tin_feature(30)]

        gen1, fids1 = _build_generator_with_features(features, min_zoom=0, max_zoom=2)
        out_big = tmp_path / "big"
        count_big = generate_parquet(
            gen1, out_big, WORLD_BOUNDS,
            partitioned=True, batch_size=10, max_file_bytes=10**18,  # effectively no split
        )

        gen2, fids2 = _build_generator_with_features(features, min_zoom=0, max_zoom=2)
        out_split = tmp_path / "split"
        count_split = generate_parquet(
            gen2, out_split, WORLD_BOUNDS,
            partitioned=True, batch_size=10, max_file_bytes=1,  # always split
        )

        assert count_big == count_split

        rows_big = read_parquet(out_big)
        rows_split = read_parquet(out_split)
        fids_big = sorted(set(r["feature_id"] for r in rows_big))
        fids_split = sorted(set(r["feature_id"] for r in rows_split))
        assert fids_big == fids_split

    def test_single_part_when_under_threshold(self, tmp_path):
        """Small data produces a single part_000.parquet per zoom."""
        features = [
            _make_tin_feature(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [0.1, 0.2, 0.3],
                [3],
            ),
        ]
        gen, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=1)
        out_dir = tmp_path / "single"
        generate_parquet(gen, out_dir, WORLD_BOUNDS, partitioned=True, batch_size=10)

        for z in range(2):
            parts = sorted((out_dir / f"zoom={z}").glob("part_*.parquet"))
            assert len(parts) == 1
            assert parts[0].name == "part_000.parquet"


# ---------------------------------------------------------------------------
# Repartition tests
# ---------------------------------------------------------------------------

class TestRepartitionParquet:
    """Test repartition_parquet() for splitting existing pyramids."""

    def test_splits_large_file(self, tmp_path):
        """repartition_parquet splits a single large file into multiple parts."""
        # Generate a pyramid with single data file per zoom (legacy naming)
        features = [_make_dense_tin_feature(50)]
        gen, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=2)
        out_dir = tmp_path / "pyramid"
        generate_parquet(gen, out_dir, WORLD_BOUNDS, partitioned=True, batch_size=10)

        # With max_file_bytes=1, every zoom should be split
        result = repartition_parquet(out_dir, max_file_bytes=1)

        assert len(result) == 3  # zoom 0, 1, 2
        assert any(n > 1 for n in result.values()), "Expected at least one zoom to split"

        # Verify all files are part_NNN.parquet
        import re
        for z in range(3):
            parts = sorted((out_dir / f"zoom={z}").glob("*.parquet"))
            assert len(parts) == result[z]
            for p in parts:
                assert re.fullmatch(r"part_\d{3}\.parquet", p.name)

    def test_idempotent(self, tmp_path):
        """Calling repartition_parquet twice produces same result."""
        features = [_make_dense_tin_feature(30)]
        gen, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=1)
        out_dir = tmp_path / "pyramid"
        generate_parquet(gen, out_dir, WORLD_BOUNDS, partitioned=True, batch_size=10)

        result1 = repartition_parquet(out_dir, max_file_bytes=1)
        result2 = repartition_parquet(out_dir, max_file_bytes=1)

        # Second call should find already-split files
        # Count should be the same (it re-reads and re-splits but result is equivalent)
        assert set(result1.keys()) == set(result2.keys())

    def test_renames_legacy_data_parquet(self, tmp_path):
        """Legacy data.parquet is renamed to part_000.parquet when under threshold."""
        features = [
            _make_tin_feature(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [0.1, 0.2, 0.3],
                [3],
            ),
        ]
        gen, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=0)
        out_dir = tmp_path / "legacy"
        out_dir.mkdir(parents=True)

        # Manually write a legacy data.parquet
        zoom_dir = out_dir / "zoom=0"
        zoom_dir.mkdir()
        from mudm_tools.tiling3d.parquet_writer import generate_parquet as gen_pq
        gen2, _ = _build_generator_with_features([
            _make_tin_feature(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [0.1, 0.2, 0.3],
                [3],
            ),
        ], min_zoom=0, max_zoom=0)

        # Write using old-style naming by creating a fake data.parquet
        import pyarrow as pa
        schema = pa.schema([
            pa.field("tile_x", pa.uint16()),
            pa.field("tile_y", pa.uint16()),
            pa.field("tile_d", pa.uint16()),
            pa.field("feature_id", pa.uint32()),
            pa.field("geom_type", pa.uint8()),
            pa.field("positions", pa.large_binary()),
            pa.field("indices", pa.large_binary()),
            pa.field("tags", pa.map_(pa.utf8(), pa.utf8())),
        ])
        table = pa.table({
            "tile_x": pa.array([0], type=pa.uint16()),
            "tile_y": pa.array([0], type=pa.uint16()),
            "tile_d": pa.array([0], type=pa.uint16()),
            "feature_id": pa.array([0], type=pa.uint32()),
            "geom_type": pa.array([5], type=pa.uint8()),
            "positions": pa.array([b"\x00" * 12], type=pa.large_binary()),
            "indices": pa.array([b"\x00" * 12], type=pa.large_binary()),
            "tags": pa.array([[(b"k", b"v")]], type=pa.map_(pa.utf8(), pa.utf8())),
        })
        pq.write_table(table, str(zoom_dir / "data.parquet"), compression="zstd")

        assert (zoom_dir / "data.parquet").exists()
        result = repartition_parquet(out_dir)

        # data.parquet should be gone, replaced by part_000.parquet
        assert not (zoom_dir / "data.parquet").exists()
        assert (zoom_dir / "part_000.parquet").exists()
        assert result[0] == 1

    def test_readable_after_repartition(self, tmp_path):
        """Data is readable via read_parquet after repartition."""
        features = [_make_dense_tin_feature(30)]
        gen, _ = _build_generator_with_features(features, min_zoom=0, max_zoom=2)
        out_dir = tmp_path / "pyramid"
        total = generate_parquet(gen, out_dir, WORLD_BOUNDS, partitioned=True, batch_size=10)

        repartition_parquet(out_dir, max_file_bytes=1)

        rows = read_parquet(out_dir)
        assert len(rows) == total

    def test_nonexistent_path_raises(self, tmp_path):
        """repartition_parquet raises FileNotFoundError for missing path."""
        with pytest.raises(FileNotFoundError):
            repartition_parquet(tmp_path / "nonexistent")

    def test_file_not_dir_raises(self, tmp_path):
        """repartition_parquet raises NotADirectoryError for a file path."""
        f = tmp_path / "file.txt"
        f.write_text("hi")
        with pytest.raises(NotADirectoryError):
            repartition_parquet(f)


def test_parentid_survives_parquet(tmp_path):
    """Top-level MuDMFeature.parentId must round-trip into Parquet tags as _parent_id."""
    feat = _make_tin_feature(
        [0.1, 0.1, 0.2, 0.1, 0.2, 0.2],
        [0.1, 0.1, 0.1],
        [3],
        tags={"color": "red"},
    )
    # Add the top-level muDM field that the Rust extractor should promote.
    feat["parentId"] = "neuron_17"

    gen, _ = _build_generator_with_features([feat])
    out = tmp_path / "neurons.parquet"
    generate_parquet(gen, out, WORLD_BOUNDS)

    table = pq.read_table(str(out))
    tags_col = table.column("tags").to_pylist()
    assert len(tags_col) > 0
    for tags in tags_col:
        kv = {k: v for k, v in tags}
        assert kv.get("_parent_id") == "neuron_17", f"got {kv}"
        assert kv.get("color") == "red"
