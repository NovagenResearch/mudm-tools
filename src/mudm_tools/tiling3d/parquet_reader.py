"""Read tile-centric Parquet files for ML training / data loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _detect_format(root: Path) -> str:
    """Detect whether a partitioned directory has Arrow IPC siblings.

    Returns ``"ipc"`` if any ``zoom=*/data.arrow`` file exists, else
    ``"parquet"``.  Uses early exit — O(1) for the common case.
    """
    return "ipc" if next(root.glob("zoom=*/*.arrow"), None) else "parquet"


def read_parquet(
    path: str | Path,
    *,
    zoom: int | None = None,
    feature_id: int | None = None,
    tile_x: int | None = None,
    tile_y: int | None = None,
) -> list[dict]:
    """Read rows from a tiled mesh Parquet file.

    Returns a list of dicts with:
        zoom, tile_x, tile_y, tile_d, feature_id, geom_type,
        positions (np.float32 [N,3]), indices (np.uint32 [M]),
        tags (dict[str, str]).

    Uses PyArrow predicate pushdown for efficient filtering.
    When Arrow IPC siblings exist (created by :func:`prime_parquet`),
    reads from those instead for zero-copy memory-mapped access.

    Args:
        path: Path to the .parquet file or partitioned directory.
        zoom: Filter to this zoom level.
        feature_id: Filter to this feature ID.
        tile_x: Filter to this tile X coordinate.
        tile_y: Filter to this tile Y coordinate.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    path_obj = Path(path)
    if path_obj.is_dir():
        fmt = _detect_format(path_obj)
        if fmt == "ipc":
            # Build dataset from explicit arrow file list — ds.dataset() with
            # format="ipc" on a directory tries to open ALL files (including
            # .parquet) as IPC, which fails. partition_base_dir lets PyArrow
            # extract Hive partition keys (zoom=N) from the file paths.
            arrow_files = sorted(str(f) for f in path_obj.glob("zoom=*/*.arrow"))
            partitioning = ds.HivePartitioning(pa.schema([("zoom", pa.int32())]))
            dataset = ds.dataset(
                arrow_files,
                format="ipc",
                partitioning=partitioning,
                partition_base_dir=str(path_obj),
            )
        else:
            dataset = ds.dataset(str(path), format="parquet", partitioning="hive")
    else:
        dataset = ds.dataset(str(path), format="parquet")

    filters = []
    if zoom is not None:
        filters.append(pc.field("zoom") == zoom)
    if feature_id is not None:
        filters.append(pc.field("feature_id") == feature_id)
    if tile_x is not None:
        filters.append(pc.field("tile_x") == tile_x)
    if tile_y is not None:
        filters.append(pc.field("tile_y") == tile_y)

    combined = None
    for f in filters:
        combined = f if combined is None else (combined & f)

    table = dataset.to_table(filter=combined)

    rows = []
    for i in range(table.num_rows):
        pos_bytes = table.column("positions")[i].as_py()
        idx_bytes = table.column("indices")[i].as_py()

        positions = np.frombuffer(pos_bytes, dtype=np.float32).reshape(-1, 3)
        indices = np.frombuffer(idx_bytes, dtype=np.uint32)

        tag_map = table.column("tags")[i].as_py()
        tags = dict(tag_map) if tag_map else {}

        rows.append(
            {
                "zoom": table.column("zoom")[i].as_py(),
                "tile_x": table.column("tile_x")[i].as_py(),
                "tile_y": table.column("tile_y")[i].as_py(),
                "tile_d": table.column("tile_d")[i].as_py(),
                "feature_id": table.column("feature_id")[i].as_py(),
                "geom_type": table.column("geom_type")[i].as_py(),
                "positions": positions,
                "indices": indices,
                "tags": tags,
            }
        )

    return rows
