"""SpatiallyCoherentLoader: yields batches drawn from a single octree tile, with
adjacent-tile padding when a tile has fewer features than the requested batch size."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _build_synthetic_partitioned_parquet(tmp_path, *, zoom: int, n_features_per_tile: dict):
    """Build a tiny zoom={N}/tile_x={X}/tile_y={Y}/tile_d={D} hive-partitioned
    Parquet directory with n_features_per_tile[(x,y,d)] features in each tile.
    Each feature has a stub `feature_id` and a 9-float `positions` blob."""
    rows = []
    for (x, y, d), n in n_features_per_tile.items():
        for i in range(n):
            rows.append(
                {
                    "zoom": zoom,
                    "tile_x": x,
                    "tile_y": y,
                    "tile_d": d,
                    "feature_id": f"f_{x}_{y}_{d}_{i}",
                    "geom_type": 5,
                    "positions": np.zeros(9, dtype=np.float32).tobytes(),
                    "indices": np.zeros(3, dtype=np.uint32).tobytes(),
                    "ring_lengths": b"",
                    "tags": {"label": str((x + y + d) % 4)},  # 4 fake classes
                }
            )
    table = pa.Table.from_pylist(rows)
    out_dir = tmp_path / "parquet_partitioned"
    pq.write_to_dataset(
        table,
        root_path=str(out_dir),
        partition_cols=["zoom", "tile_x", "tile_y", "tile_d"],
    )
    return out_dir


def test_spatial_loader_yields_single_tile_batch(tmp_path):
    from mudm_tools.spatial_batching import SpatiallyCoherentLoader

    parquet_dir = _build_synthetic_partitioned_parquet(
        tmp_path,
        zoom=3,
        n_features_per_tile={(0, 0, 0): 8, (1, 0, 0): 8, (0, 1, 0): 8},
    )

    loader = SpatiallyCoherentLoader(
        parquet_dir=parquet_dir,
        zoom=3,
        batch_size=8,
        shuffle=False,
    )
    batches = list(loader)
    assert len(batches) == 3
    for batch in batches:
        # Every feature_id in a batch must come from the same (tile_x, tile_y, tile_d)
        tile_keys = {(b["tile_x"], b["tile_y"], b["tile_d"]) for b in batch}
        assert len(tile_keys) == 1, f"batch spans multiple tiles: {tile_keys}"


def test_spatial_loader_pads_undersized_tile(tmp_path):
    """If a tile has fewer features than batch_size, the loader pads by sampling
    from another tile (the test verifies the batch size is exactly batch_size)."""
    from mudm_tools.spatial_batching import SpatiallyCoherentLoader

    parquet_dir = _build_synthetic_partitioned_parquet(
        tmp_path,
        zoom=3,
        n_features_per_tile={(0, 0, 0): 4, (1, 0, 0): 4, (2, 0, 0): 8},
    )
    loader = SpatiallyCoherentLoader(
        parquet_dir=parquet_dir,
        zoom=3,
        batch_size=8,
        shuffle=False,
    )
    batches = list(loader)
    # Each batch is exactly batch_size; total features = 4+4+8 = 16, so 2 batches
    # with padding (the under-sized tiles are merged or padded).
    for batch in batches:
        assert len(batch) == 8


def test_spatial_loader_total_features_preserved(tmp_path):
    """Across all batches in one epoch, every feature_id appears exactly once
    (with padding sampled from outside the count, so we check a >= relationship)."""
    from mudm_tools.spatial_batching import SpatiallyCoherentLoader

    parquet_dir = _build_synthetic_partitioned_parquet(
        tmp_path,
        zoom=3,
        n_features_per_tile={(0, 0, 0): 8, (1, 0, 0): 8},
    )
    loader = SpatiallyCoherentLoader(
        parquet_dir=parquet_dir,
        zoom=3,
        batch_size=8,
        shuffle=False,
    )
    seen = set()
    for batch in loader:
        for b in batch:
            seen.add(b["feature_id"])
    # Every original feature should be visited at least once
    expected = {
        f"f_{x}_{y}_{d}_{i}" for (x, y, d), n in [((0, 0, 0), 8), ((1, 0, 0), 8)] for i in range(n)
    }
    assert expected.issubset(seen)


def test_spatial_loader_shuffles_tile_order(tmp_path):
    """With shuffle=True and a fixed seed, the order of tiles visited differs
    from natural sorted order across multiple epochs."""
    from mudm_tools.spatial_batching import SpatiallyCoherentLoader

    parquet_dir = _build_synthetic_partitioned_parquet(
        tmp_path,
        zoom=3,
        n_features_per_tile={(0, 0, 0): 4, (1, 0, 0): 4, (2, 0, 0): 4, (3, 0, 0): 4},
    )
    loader = SpatiallyCoherentLoader(
        parquet_dir=parquet_dir,
        zoom=3,
        batch_size=4,
        shuffle=True,
        seed=42,
    )
    epoch_a = [b[0]["tile_x"] for b in loader]
    epoch_b = [b[0]["tile_x"] for b in loader]
    # With seed=42, two consecutive iterations should produce the same shuffle...
    assert epoch_a == epoch_b
    # ...but different from the sorted order.
    assert epoch_a != sorted(epoch_a)
