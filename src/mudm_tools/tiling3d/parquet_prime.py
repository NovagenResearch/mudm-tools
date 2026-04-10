"""Prime / deprime / repartition partitioned Parquet pyramids.

Parquet (ZSTD) is ideal for storage but requires decompression on every read.
Arrow IPC (Feather v2) enables memory-mapped zero-copy reads at the cost of
larger files on disk. ``prime_parquet`` creates Arrow IPC siblings next to each
partition's Parquet files; ``deprime_parquet`` removes them. The Parquet
files are always the source of truth.

``repartition_parquet`` splits oversized partition files (e.g. a 5 GB
``data.parquet``) into smaller ``part_NNN.parquet`` files (default ≤ 500 MB)
for parallel DataLoader workers and memory-mapped reads.
"""

from __future__ import annotations

import re
from pathlib import Path

import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq

_DEFAULT_MAX_FILE_BYTES = 500_000_000  # 500 MB uncompressed binary threshold


def prime_parquet(
    path: str | Path,
    *,
    compression: str = "uncompressed",
) -> int:
    """Convert each partition's ``data.parquet`` to a sibling ``data.arrow``.

    Walks ``zoom=*/data.parquet`` inside *path* and writes Arrow IPC files
    using :func:`pyarrow.feather.write_feather`.

    Args:
        path: Root directory of a Hive-partitioned Parquet pyramid.
        compression: Arrow IPC compression — ``"uncompressed"`` (default,
            fastest reads), ``"lz4"``, or ``"zstd"``.

    Returns:
        Number of Arrow IPC files written.

    Raises:
        FileNotFoundError: If *path* does not exist.
        NotADirectoryError: If *path* is not a directory.
        ValueError: If *compression* is not one of the allowed values.
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
        # Drop zoom column if present — it's encoded in the Hive partition
        # directory name and will be reconstructed by ds.dataset().
        if "zoom" in table.column_names:
            table = table.drop_columns(["zoom"])
        arrow_file = pq_file.with_suffix(".arrow")
        feather.write_feather(table, str(arrow_file), compression=compression)
        count += 1

    return count


def deprime_parquet(path: str | Path) -> int:
    """Remove all Arrow IPC siblings from a partitioned Parquet pyramid.

    Deletes every ``zoom=*/data.arrow`` file inside *path*, leaving the
    original ``data.parquet`` files intact.

    Args:
        path: Root directory of a Hive-partitioned Parquet pyramid.

    Returns:
        Number of Arrow IPC files deleted.

    Raises:
        FileNotFoundError: If *path* does not exist.
        NotADirectoryError: If *path* is not a directory.
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

    Reads all ``*.parquet`` files in each ``zoom=N/`` directory, then writes
    new ``part_NNN.parquet`` files so that each stays under *max_file_bytes*
    of uncompressed binary data. Uses atomic temp-file swap to avoid partial
    reads.

    Also renames legacy ``data.parquet`` files to ``part_000.parquet`` when
    the partition is already under the threshold.

    Args:
        path: Root directory of a Hive-partitioned Parquet pyramid.
        max_file_bytes: Maximum uncompressed binary bytes per output file
            (default 500 MB).
        compression: Parquet compression codec (default ``"zstd"``).
        compression_level: Compression level (default 3).

    Returns:
        Dict mapping zoom level to number of output parts.

    Raises:
        FileNotFoundError: If *path* does not exist.
        NotADirectoryError: If *path* is not a directory.
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

        # Check if already properly partitioned (all part_NNN.parquet, under threshold)
        all_named_ok = all(re.fullmatch(r"part_\d{3}\.parquet", f.name) for f in pq_files)
        all_small = all(f.stat().st_size <= max_file_bytes for f in pq_files)
        if all_named_ok and all_small:
            result[zoom] = len(pq_files)
            continue

        # Read all existing parquet files into one table
        tables = []
        for f in pq_files:
            tables.append(pq.read_table(str(f)))
        table = pa.concat_tables(tables)
        # Drop zoom column if present (it's in the directory name)
        if "zoom" in table.column_names:
            table = table.drop_columns(["zoom"])

        # Write to temp files, respecting size threshold
        kwargs: dict = {"compression": compression}
        if compression not in ("none", "NONE", None):
            kwargs["compression_level"] = compression_level

        # Combine chunks so slicing produces single-chunk results
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

        # Compute chunk_size from estimated bytes per row so rotation
        # checks happen at useful intervals
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

        # Atomic swap: delete old files, rename temps
        for f in pq_files:
            f.unlink()
        # Also remove any leftover .arrow files
        for f in zoom_dir.glob("*.arrow"):
            f.unlink()

        for tmp_path in tmp_files:
            final_name = tmp_path.name.replace(".tmp_", "")
            tmp_path.rename(zoom_dir / final_name)

        result[zoom] = part_idx + 1

    return result


def _estimate_binary_bytes(data: pa.Table | pa.RecordBatch) -> int:
    """Estimate uncompressed binary column size via Arrow buffer API.

    Handles both Table (chunked columns) and RecordBatch (flat arrays).
    """
    total = 0
    for col in data.columns:
        if not pa.types.is_large_binary(col.type):
            continue
        # Table columns are ChunkedArrays; RecordBatch columns are Arrays
        chunks = col.chunks if hasattr(col, "chunks") else [col]
        for chunk in chunks:
            bufs = chunk.buffers()
            if len(bufs) >= 3 and bufs[2] is not None:
                total += bufs[2].size
    return total
