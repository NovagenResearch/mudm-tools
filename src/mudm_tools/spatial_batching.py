"""Spatially-coherent batch loader.

Reads from a hive-partitioned Parquet directory whose layout is
zoom={N}/tile_x={X}/tile_y={Y}/tile_d={D}/*.parquet, and yields batches whose
every row comes from the same (tile_x, tile_y, tile_d) tile (with deterministic
merging from neighbour tiles when a tile is undersized for the batch).

This is the muDM-specific batching primitive that exercises the spatial
autocorrelation structure of the tiled corpus -- random sampling from across
the corpus loses that structure.

Padding policy
--------------
When a tile contains fewer rows than ``batch_size``, those rows are accumulated
into a ``carry`` buffer and emitted as a combined batch as soon as the carry
reaches ``batch_size``.  Such "carry" batches may therefore span multiple
adjacent tiles, but each batch is exactly ``batch_size`` rows.  Any trailing
partial carry at the end of an epoch is dropped (so up to ``batch_size - 1``
features may be skipped per epoch -- acceptable for ablation studies).

Tiles whose row count is at least ``batch_size`` are emitted as one or more
single-tile batches; only the leftover from such a tile (the modulo remainder)
spills into the carry buffer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pyarrow.parquet as pq


class SpatiallyCoherentLoader:
    """Yields lists-of-row-dicts batched by octree tile.

    Each batch is exactly ``batch_size`` rows.  If a tile has fewer rows than
    ``batch_size``, the loader accumulates them with rows from adjacent tiles
    (in the natural sort order, or shuffled order if ``shuffle=True``) and
    yields a combined batch when the carry reaches ``batch_size``.

    Parameters
    ----------
    parquet_dir : Path
        Hive-partitioned Parquet root.  Expected layout
        ``zoom={N}/tile_x={X}/tile_y={Y}/tile_d={D}/*.parquet``.
    zoom : int
        Zoom level to read from.
    batch_size : int
        Rows per batch.
    shuffle : bool
        If True, the tile visit order is shuffled at the start of each epoch
        and rows within a tile are also shuffled.
    seed : int | None
        Seed for the shuffle RNG.  Re-seeded on each ``__iter__`` call so the
        same seed produces identical shuffles across epochs (useful for
        reproducible benchmarks).
    """

    def __init__(
        self,
        parquet_dir: Path,
        *,
        zoom: int,
        batch_size: int,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self._parquet_dir = Path(parquet_dir)
        self._zoom = zoom
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._seed = seed
        self._tile_keys = self._discover_tile_keys()

    def _discover_tile_keys(self) -> list[tuple[int, int, int]]:
        """Walk the partitioned directory to find all (tile_x, tile_y, tile_d)
        keys at the configured zoom level."""
        zoom_dir = self._parquet_dir / f"zoom={self._zoom}"
        if not zoom_dir.exists():
            return []
        keys: list[tuple[int, int, int]] = []
        for tx_dir in sorted(zoom_dir.glob("tile_x=*")):
            try:
                tx = int(tx_dir.name.split("=", 1)[1])
            except ValueError:
                continue
            for ty_dir in sorted(tx_dir.glob("tile_y=*")):
                try:
                    ty = int(ty_dir.name.split("=", 1)[1])
                except ValueError:
                    continue
                for td_dir in sorted(ty_dir.glob("tile_d=*")):
                    try:
                        td = int(td_dir.name.split("=", 1)[1])
                    except ValueError:
                        continue
                    keys.append((tx, ty, td))
        return keys

    def _read_tile_rows(self, tx: int, ty: int, td: int) -> list[dict]:
        """Load all rows from one tile.  Adds back the partition columns since
        hive-style partitioning strips them from the file payload."""
        tile_dir = (
            self._parquet_dir
            / f"zoom={self._zoom}"
            / f"tile_x={tx}"
            / f"tile_y={ty}"
            / f"tile_d={td}"
        )
        table = pq.read_table(tile_dir)
        rows = table.to_pylist()
        for row in rows:
            row.setdefault("zoom", self._zoom)
            row.setdefault("tile_x", tx)
            row.setdefault("tile_y", ty)
            row.setdefault("tile_d", td)
        return rows

    @property
    def tile_keys(self) -> list[tuple[int, int, int]]:
        """Return the list of (tile_x, tile_y, tile_d) keys discovered at init."""
        return list(self._tile_keys)

    def __iter__(self) -> Iterator[list[dict]]:
        if not self._tile_keys:
            return

        # Re-seed every epoch so shuffles are reproducible across iterations.
        rng = np.random.default_rng(self._seed)
        order = list(self._tile_keys)
        if self._shuffle:
            rng.shuffle(order)

        # Pre-load all tiles' rows.  For a 38-brain corpus this is bounded by
        # memory; if memory becomes an issue, replace with on-demand loading.
        tile_rows: dict[tuple[int, int, int], list[dict]] = {
            key: self._read_tile_rows(*key) for key in order
        }

        # Accumulate undersized tiles' rows until they reach batch_size.
        carry: list[dict] = []
        for key in order:
            rows = tile_rows[key]
            if not rows:
                continue
            if self._shuffle:
                rng.shuffle(rows)
            if len(rows) >= self._batch_size:
                # Yield as many full-size batches as the tile supports.
                full_batches = len(rows) // self._batch_size
                for i in range(full_batches):
                    yield rows[i * self._batch_size : (i + 1) * self._batch_size]
                remainder_start = full_batches * self._batch_size
                if remainder_start < len(rows):
                    carry.extend(rows[remainder_start:])
            else:
                carry.extend(rows)
            while len(carry) >= self._batch_size:
                yield carry[: self._batch_size]
                carry = carry[self._batch_size :]

        # Trailing partial carry is dropped (documented in module docstring).
