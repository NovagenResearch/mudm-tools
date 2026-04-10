"""Prime / deprime / repartition partitioned 2D Parquet pyramids.

Same functionality as ``tiling3d.parquet_prime`` adapted for the 2D schema.
"""

from __future__ import annotations

import re
from pathlib import Path

import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq

_DEFAULT_MAX_FILE_BYTES = 500_000_000


def prime_parquet(
    path: str | Path,
    *,
    compression: str = "uncompressed",
) -> int:
    """Convert each partition's Parquet files to sibling Arrow IPC files.

    Args:
        path: Root directory of a Hive-partitioned Parquet pyramid.
        compression: Arrow IPC compression (default "uncompressed").

    Returns:
        Number of Arrow IPC files written.
    """
    allowed = {"uncompressed", "lz4", "zstd"}
    if compression not in allowed:
        raise ValueError(
            f"compression must be one of {sorted(allowed)}, got {compression!r}"
        )

    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root}")

    count = 0
    for pq_file in sorted(root.glob("zoom=*/*.parquet")):
        table = pq.read_table(str(pq_file))
        if "zoom" in table.column_names:
            table = table.drop_columns(["zoom"])
        arrow_file = pq_file.with_suffix(".arrow")
        feather.write_feather(table, str(arrow_file), compression=compression)
        count += 1

    return count


def deprime_parquet(path: str | Path) -> int:
    """Remove all Arrow IPC siblings from a partitioned Parquet pyramid.

    Args:
        path: Root directory of a Hive-partitioned Parquet pyramid.

    Returns:
        Number of Arrow IPC files deleted.
    """
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root}")

    count = 0
    for arrow_file in sorted(root.glob("zoom=*/*.arrow")):
        arrow_file.unlink()
        count += 1

    return count


def repartition_parquet(
    path: str | Path,
    *,
    max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES,
    compression: str = "zstd",
    compression_level: int = 3,
) -> dict[int, int]:
    """Split oversized partition files into smaller ``part_NNN.parquet`` files.

    Args:
        path: Root directory of a Hive-partitioned Parquet pyramid.
        max_file_bytes: Maximum uncompressed binary bytes per output file.
        compression: Parquet compression codec (default "zstd").
        compression_level: Compression level (default 3).

    Returns:
        Dict mapping zoom level to number of output parts.
    """
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root}")

    result: dict[int, int] = {}

    for zoom_dir in sorted(root.glob("zoom=*")):
        if not zoom_dir.is_dir():
            continue
        m = re.fullmatch(r"zoom=(\d+)", zoom_dir.name)
        if not m:
            continue
        zoom = int(m.group(1))

        pq_files = sorted(zoom_dir.glob("*.parquet"))
        if not pq_files:
            continue

        all_named_ok = all(re.fullmatch(r"part_\d{3}\.parquet", f.name) for f in pq_files)
        all_small = all(f.stat().st_size <= max_file_bytes for f in pq_files)
        if all_named_ok and all_small:
            result[zoom] = len(pq_files)
            continue

        tables = []
        for f in pq_files:
            tables.append(pq.read_table(str(f)))
        table = pa.concat_tables(tables)
        if "zoom" in table.column_names:
            table = table.drop_columns(["zoom"])

        kwargs: dict = {"compression": compression}
        if compression not in ("none", "NONE", None):
            kwargs["compression_level"] = compression_level

        table = table.combine_chunks()
        schema = table.schema
        part_idx = 0
        cum_bytes = 0
        writer: pq.ParquetWriter | None = None
        tmp_files: list[Path] = []

        def _open_writer() -> pq.ParquetWriter:
            nonlocal part_idx
            tmp_path = zoom_dir / f".tmp_part_{part_idx:03d}.parquet"
            tmp_files.append(tmp_path)
            return pq.ParquetWriter(str(tmp_path), schema, **kwargs)

        total_binary = _estimate_binary_bytes(table)
        if table.num_rows > 0 and total_binary > 0:
            bytes_per_row = total_binary / table.num_rows
            rows_per_file = max(1, int(max_file_bytes / bytes_per_row))
            chunk_size = max(1, rows_per_file // 2)
        else:
            chunk_size = max(1, table.num_rows)

        for start in range(0, table.num_rows, chunk_size):
            chunk = table.slice(start, min(chunk_size, table.num_rows - start))
            chunk_bytes = _estimate_binary_bytes(chunk)

            if writer is not None and cum_bytes > 0 and cum_bytes + chunk_bytes > max_file_bytes:
                writer.close()
                part_idx += 1
                cum_bytes = 0
                writer = None

            if writer is None:
                writer = _open_writer()

            writer.write_table(chunk)
            cum_bytes += chunk_bytes

        if writer is not None:
            writer.close()

        for f in pq_files:
            f.unlink()
        for f in zoom_dir.glob("*.arrow"):
            f.unlink()

        for tmp_path in tmp_files:
            final_name = tmp_path.name.replace(".tmp_", "")
            tmp_path.rename(zoom_dir / final_name)

        result[zoom] = part_idx + 1

    return result


def _estimate_binary_bytes(data: pa.Table | pa.RecordBatch) -> int:
    total = 0
    for col in data.columns:
        if not pa.types.is_large_binary(col.type):
            continue
        chunks = col.chunks if hasattr(col, "chunks") else [col]
        for chunk in chunks:
            bufs = chunk.buffers()
            if len(bufs) >= 3 and bufs[2] is not None:
                total += bufs[2].size
    return total
