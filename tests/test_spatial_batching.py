"""SpatiallyCoherentLoader: yields batches drawn from a single octree tile, with
adjacent-tile padding when a tile has fewer features than the requested batch size."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# The curriculum-benchmark helpers we exercise live under ``scripts/`` rather
# than the installed package, so make that path importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))


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
        table, root_path=str(out_dir),
        partition_cols=["zoom", "tile_x", "tile_y", "tile_d"],
    )
    return out_dir


def test_spatial_loader_yields_single_tile_batch(tmp_path):
    from mudm_tools.spatial_batching import SpatiallyCoherentLoader

    parquet_dir = _build_synthetic_partitioned_parquet(
        tmp_path, zoom=3,
        n_features_per_tile={(0, 0, 0): 8, (1, 0, 0): 8, (0, 1, 0): 8},
    )

    loader = SpatiallyCoherentLoader(
        parquet_dir=parquet_dir, zoom=3, batch_size=8, shuffle=False,
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
        tmp_path, zoom=3,
        n_features_per_tile={(0, 0, 0): 4, (1, 0, 0): 4, (2, 0, 0): 8},
    )
    loader = SpatiallyCoherentLoader(
        parquet_dir=parquet_dir, zoom=3, batch_size=8, shuffle=False,
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
        tmp_path, zoom=3,
        n_features_per_tile={(0, 0, 0): 8, (1, 0, 0): 8},
    )
    loader = SpatiallyCoherentLoader(
        parquet_dir=parquet_dir, zoom=3, batch_size=8, shuffle=False,
    )
    seen = set()
    for batch in loader:
        for b in batch:
            seen.add(b["feature_id"])
    # Every original feature should be visited at least once
    expected = {f"f_{x}_{y}_{d}_{i}" for (x, y, d), n in [((0,0,0),8),((1,0,0),8)] for i in range(n)}
    assert expected.issubset(seen)


def test_spatial_loader_shuffles_tile_order(tmp_path):
    """With shuffle=True and a fixed seed, the order of tiles visited differs
    from natural sorted order across multiple epochs."""
    from mudm_tools.spatial_batching import SpatiallyCoherentLoader

    parquet_dir = _build_synthetic_partitioned_parquet(
        tmp_path, zoom=3,
        n_features_per_tile={(0, 0, 0): 4, (1, 0, 0): 4, (2, 0, 0): 4, (3, 0, 0): 4},
    )
    loader = SpatiallyCoherentLoader(
        parquet_dir=parquet_dir, zoom=3, batch_size=4, shuffle=True, seed=42,
    )
    epoch_a = [b[0]["tile_x"] for b in loader]
    epoch_b = [b[0]["tile_x"] for b in loader]
    # With seed=42, two consecutive iterations should produce the same shuffle...
    assert epoch_a == epoch_b
    # ...but different from the sorted order.
    assert epoch_a != sorted(epoch_a)


# ---------------------------------------------------------------------------
# Train-only Parquet view helper (used by the curriculum-benchmark spatial-
# batch arm so its mini-batches see only train-split neurons, matching the
# baseline arm's pre-filtered-dataset semantics).
# ---------------------------------------------------------------------------


def _build_mixed_split_partitioned_parquet(tmp_path, *, zoom: int, body_ids_per_tile):
    """Synthetic hive-partitioned Parquet whose ``tags`` column is a struct
    containing ``body_id`` and ``cell_type`` -- mirroring the real hemibrain
    schema consumed by ``_rows_to_tensor_batch``.

    ``body_ids_per_tile`` maps ``(tile_x, tile_y, tile_d) -> list[str]``.
    """
    rows = []
    for (x, y, d), bids in body_ids_per_tile.items():
        for bid in bids:
            rows.append(
                {
                    "zoom": zoom,
                    "tile_x": x,
                    "tile_y": y,
                    "tile_d": d,
                    "feature_id": bid,
                    "geom_type": 5,
                    "positions": np.zeros(9, dtype=np.float32).tobytes(),
                    "indices": np.zeros(3, dtype=np.uint32).tobytes(),
                    "ring_lengths": b"",
                    "tags": {"body_id": bid, "cell_type": f"CT_{bid[:1]}"},
                }
            )
    table = pa.Table.from_pylist(rows)
    out_dir = tmp_path / "parquet_partitioned"
    pq.write_to_dataset(
        table, root_path=str(out_dir),
        partition_cols=["zoom", "tile_x", "tile_y", "tile_d"],
    )
    return out_dir


def test_train_only_parquet_view_contains_only_train_rows(tmp_path):
    """The helper must filter the source Parquet down to train-set ``body_id``
    rows only, drop tiles that retain zero rows, and preserve the hive
    partition structure (zoom={N}/tile_x={X}/tile_y={Y}/tile_d={D})."""
    from benchmark_curriculum import _write_train_only_parquet_view

    src = _build_mixed_split_partitioned_parquet(
        tmp_path,
        zoom=3,
        body_ids_per_tile={
            (0, 0, 0): ["t1", "t2", "v1"],   # mixed train/val
            (1, 0, 0): ["t3", "t4"],         # all train
            (0, 1, 0): ["v2", "x1"],         # zero train -> tile dropped
            (1, 1, 0): ["t5", "x2", "v3"],   # mixed all three splits
        },
    )

    train_id_set = {"t1", "t2", "t3", "t4", "t5"}
    out_dir = _write_train_only_parquet_view(
        source_parquet_dir=src,
        train_id_set=train_id_set,
        zoom=3,
    )

    try:
        out = Path(out_dir)
        # Hive partition layout preserved.
        assert (out / "zoom=3").is_dir()
        assert (out / "zoom=3" / "tile_x=0" / "tile_y=0" / "tile_d=0").is_dir()
        assert (out / "zoom=3" / "tile_x=1" / "tile_y=0" / "tile_d=0").is_dir()
        assert (out / "zoom=3" / "tile_x=1" / "tile_y=1" / "tile_d=0").is_dir()
        # Tile that contained no train rows must NOT appear.
        assert not (out / "zoom=3" / "tile_x=0" / "tile_y=1" / "tile_d=0").exists()

        # Every surviving row's body_id is in the train set.
        seen_bids: set[str] = set()
        for tile_dir in (out / "zoom=3").rglob("tile_d=*"):
            table = pq.read_table(tile_dir)
            tags_list = table.column("tags").to_pylist()
            for tags_val in tags_list:
                tags_dict = dict(tags_val) if isinstance(tags_val, list) else tags_val
                bid = tags_dict.get("body_id")
                assert bid in train_id_set, f"non-train body_id leaked: {bid!r}"
                seen_bids.add(bid)
        assert seen_bids == train_id_set
    finally:
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)


def test_train_only_parquet_view_is_consumable_by_spatial_loader(tmp_path):
    """The temp view's schema must be compatible with ``SpatiallyCoherentLoader``,
    and every batch must contain only train-set neurons (i.e. the loader is no
    longer responsible for filtering)."""
    from benchmark_curriculum import _write_train_only_parquet_view
    from mudm_tools.spatial_batching import SpatiallyCoherentLoader

    src = _build_mixed_split_partitioned_parquet(
        tmp_path,
        zoom=3,
        body_ids_per_tile={
            (0, 0, 0): [f"t{i}" for i in range(8)] + ["v0", "v1"],
            (1, 0, 0): [f"t{i}" for i in range(8, 16)],
            (0, 1, 0): ["v2", "v3", "v4"],   # zero train rows
        },
    )
    train_id_set = {f"t{i}" for i in range(16)}

    out_dir = _write_train_only_parquet_view(
        source_parquet_dir=src,
        train_id_set=train_id_set,
        zoom=3,
    )
    try:
        loader = SpatiallyCoherentLoader(
            parquet_dir=Path(out_dir), zoom=3, batch_size=8, shuffle=False,
        )
        batches = list(loader)
        # 16 train rows, batch_size=8 -> exactly 2 full batches, no carry drop.
        assert len(batches) == 2
        for batch in batches:
            assert len(batch) == 8
            for row in batch:
                tags_val = row["tags"]
                tags_dict = dict(tags_val) if isinstance(tags_val, list) else tags_val
                assert tags_dict["body_id"] in train_id_set
    finally:
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)
