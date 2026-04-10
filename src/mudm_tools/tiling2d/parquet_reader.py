"""Read tile-centric 2D Parquet files for ML training / data loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _detect_format(root: Path) -> str:
    return "ipc" if next(root.glob("zoom=*/*.arrow"), None) else "parquet"


def read_parquet(
    path: str | Path,
    *,
    zoom: int | None = None,
    feature_id: int | None = None,
    tile_x: int | None = None,
    tile_y: int | None = None,
) -> list[dict]:
    """Read rows from a tiled 2D Parquet file.

    Returns a list of dicts with:
        zoom, tile_x, tile_y, feature_id, geom_type,
        positions (np.float32 [N,2]), indices (np.uint32 [M]),
        ring_lengths (list[int]), tags (dict[str, str]).

    Uses PyArrow predicate pushdown for efficient filtering.

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

        positions = np.frombuffer(pos_bytes, dtype=np.float32).reshape(-1, 2)
        indices = np.frombuffer(idx_bytes, dtype=np.uint32)

        # ring_lengths: list<uint32>
        rl_val = table.column("ring_lengths")[i].as_py()
        ring_lengths = list(rl_val) if rl_val else []

        tag_map = table.column("tags")[i].as_py()
        tags = dict(tag_map) if tag_map else {}

        rows.append(
            {
                "zoom": table.column("zoom")[i].as_py(),
                "tile_x": table.column("tile_x")[i].as_py(),
                "tile_y": table.column("tile_y")[i].as_py(),
                "feature_id": table.column("feature_id")[i].as_py(),
                "geom_type": table.column("geom_type")[i].as_py(),
                "positions": positions,
                "indices": indices,
                "ring_lengths": ring_lengths,
                "tags": tags,
            }
        )

    return rows
