"""Write tile-centric Parquet files for ML training from StreamingTileGenerator.

Supports three modes:
- In-memory (legacy): loads all fragments, builds one table. O(all) memory.
- Single-file streaming: batch reads, one row group per zoom. O(batch + zoom buffers).
- Partitioned streaming: batch reads, one file per zoom. O(batch) strict.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_DEFAULT_MAX_FILE_BYTES = 500_000_000  # 500 MB uncompressed binary threshold
_DEFAULT_MAX_BATCH_BYTES = 2_000_000_000  # 2 GB per-batch memory budget


def generate_parquet(
    generator,
    output_path: str | Path,
    world_bounds: tuple[float, float, float, float, float, float],
    *,
    compression: str = "zstd",
    compression_level: int = 3,
    batch_size: int = 50_000,
    partitioned: bool = False,
    max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES,
    max_batch_bytes: int | None = None,
    parallel: bool = True,
    io_threads: int | None = None,
) -> int:
    """Generate a Parquet file from a StreamingTileGenerator.

    Each row represents one fragment (feature x tile intersection) with
    world-coordinate mesh data stored as raw binary columns.

    Args:
        generator: A StreamingTileGenerator with fragments already added.
        output_path: Path for the output .parquet file (or directory if partitioned).
        world_bounds: World bounding box (xmin, ymin, zmin, xmax, ymax, zmax).
        compression: Parquet compression codec (default "zstd").
        compression_level: Compression level (default 3).
        batch_size: Number of fragments to process per batch (streaming mode).
        partitioned: If True, write partitioned output (one file per zoom level).
        max_batch_bytes: Byte budget per batch. Stops reading fragments once
            cumulative in-memory size exceeds this threshold. When None
            (default), the budget is derived from the generator's resolved
            memory ceiling (``_get_max_memory``) and capped at the historical
            2 GB default — so the default behavior is unchanged on hosts whose
            ceiling exceeds 2 GB, and tightened (never loosened) on smaller
            hosts. Pass an explicit value to override entirely.
        parallel: If True (and the generator supports it), use the parallel
            shard reader for batch decoding. Falls back to the serial path
            transparently when the extension lacks ``_next_parquet_batch_parallel``.
            Output is byte-identical modulo row order. Default True (verified
            equivalent + ~3.6x faster on real Hemibrain shards, 2026-05-29);
            pass parallel=False or set MUDM_IO_THREADS=1 for an exact serial path.
        io_threads: Optional I/O concurrency for the parallel reader. When set
            (and supported), applied via the generator's ``_set_io_threads``.

    Returns:
        Number of rows written.
    """
    has_streaming = hasattr(generator, "_init_parquet_stream")

    if not has_streaming:
        return _generate_parquet_inmemory(
            generator,
            output_path,
            world_bounds,
            compression=compression,
            compression_level=compression_level,
        )

    # Resolve the per-batch byte budget. An explicit value always wins; the
    # None default derives from the generator's resolved memory ceiling
    # (WS-0) but is capped at the historical 2 GB default so default behavior
    # is preserved on large-RAM hosts and only tightened on smaller ones.
    if max_batch_bytes is None:
        derived = None
        if hasattr(generator, "_get_max_memory"):
            ceiling = generator._get_max_memory()
            if ceiling and ceiling > 0:
                # Per-batch budget derives from the path ceiling; the Rust
                # batch reader already folds a ~3x ZSTD expansion internally,
                # so the budget handed in is the decoded-bytes target.
                derived = ceiling
        max_batch_bytes = (
            min(_DEFAULT_MAX_BATCH_BYTES, derived)
            if derived is not None
            else _DEFAULT_MAX_BATCH_BYTES
        )

    effective_parallel = parallel and hasattr(generator, "_next_parquet_batch_parallel")
    if io_threads is not None and hasattr(generator, "_set_io_threads"):
        generator._set_io_threads(io_threads)

    if partitioned:
        # Fully-native Rust path: chunked parallel read + transform + parallel
        # per-zoom ZSTD write, all GIL-free in one call (no dict bridge, no
        # Python-side rotation). Gated on parallel + zstd + extension support;
        # falls back transparently otherwise.
        _use_native = (
            effective_parallel
            and compression == "zstd"
            and hasattr(generator, "generate_parquet_native_partitioned")
        )
        if _use_native:
            return generator.generate_parquet_native_partitioned(
                str(output_path),
                world_bounds,
                compression,
                compression_level,
                max_batch_bytes,
                max_file_bytes,
            )
        return _generate_parquet_partitioned_streaming(
            generator,
            output_path,
            world_bounds,
            compression=compression,
            compression_level=compression_level,
            batch_size=batch_size,
            max_file_bytes=max_file_bytes,
            max_batch_bytes=max_batch_bytes,
            parallel=effective_parallel,
        )

    return _generate_parquet_single_streaming(
        generator,
        output_path,
        world_bounds,
        compression=compression,
        compression_level=compression_level,
        batch_size=batch_size,
        max_batch_bytes=max_batch_bytes,
        parallel=effective_parallel,
    )


# ---------------------------------------------------------------------------
# In-memory path (legacy / fallback for old Rust builds)
# ---------------------------------------------------------------------------


def _generate_parquet_inmemory(
    generator,
    output_path: str | Path,
    world_bounds: tuple[float, ...],
    *,
    compression: str = "zstd",
    compression_level: int = 3,
) -> int:
    """Original in-memory path: loads all fragments at once."""
    data = generator._collect_parquet_data(world_bounds)
    row_count = data["row_count"]

    if row_count == 0:
        schema = _parquet_schema()
        table = pa.table({f.name: pa.array([], type=f.type) for f in schema})
        pq.write_table(table, str(output_path), compression=compression)
        return 0

    table = _dict_to_table(data)

    # Sort by zoom for row-group splitting
    sort_indices = pa.compute.sort_indices(table, sort_keys=[("zoom", "ascending")])
    table = table.take(sort_indices)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    zoom_col = table.column("zoom").to_pylist()
    zoom_levels = sorted(set(zoom_col))

    writer = pq.ParquetWriter(
        str(output_path),
        table.schema,
        **_writer_kwargs(compression, compression_level),
    )
    try:
        for z in zoom_levels:
            mask = pa.compute.equal(table.column("zoom"), z)
            writer.write_table(table.filter(mask))
    finally:
        writer.close()

    return row_count


# ---------------------------------------------------------------------------
# Single-file streaming path
# ---------------------------------------------------------------------------


def _generate_parquet_single_streaming(
    generator,
    output_path: str | Path,
    world_bounds: tuple[float, ...],
    *,
    compression: str = "zstd",
    compression_level: int = 3,
    batch_size: int = 50_000,
    max_batch_bytes: int = _DEFAULT_MAX_BATCH_BYTES,
    parallel: bool = False,
) -> int:
    """Streaming path: single file, one row group per zoom level.

    Memory: O(batch) for processing + O(all rows) for zoom buffers.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generator._init_parquet_stream()

    try:
        zoom_batches: dict[int, list[pa.RecordBatch]] = defaultdict(list)
        total_rows = 0

        while True:
            data = (
                generator._next_parquet_batch_parallel(batch_size, world_bounds, max_batch_bytes)
                if parallel
                else generator._next_parquet_batch(batch_size, world_bounds, max_batch_bytes)
            )
            if data is None:
                break

            batch = _dict_to_record_batch(data)
            total_rows += batch.num_rows

            # Partition by zoom and buffer
            zoom_col = batch.column("zoom").to_pylist()
            zoom_set = sorted(set(zoom_col))
            for z in zoom_set:
                mask = pa.compute.equal(batch.column("zoom"), z)
                zoom_batches[z].append(batch.filter(mask))

        if total_rows == 0:
            schema = _parquet_schema()
            table = pa.table({f.name: pa.array([], type=f.type) for f in schema})
            pq.write_table(table, str(output_path), compression=compression)
            return 0

        # Write one row group per zoom
        schema = _parquet_schema()
        writer = pq.ParquetWriter(
            str(output_path),
            schema,
            **_writer_kwargs(compression, compression_level),
        )
        try:
            for z in sorted(zoom_batches.keys()):
                table = pa.Table.from_batches(zoom_batches[z], schema=schema)
                writer.write_table(table)
        finally:
            writer.close()

    finally:
        generator._close_parquet_stream()

    return total_rows


# ---------------------------------------------------------------------------
# Partitioned streaming path (TB-scale)
# ---------------------------------------------------------------------------


def _estimate_binary_bytes(batch: pa.RecordBatch) -> int:
    """Estimate uncompressed binary column size via Arrow buffer API — O(1)."""
    total = 0
    for col in batch.columns:
        if pa.types.is_large_binary(col.type):
            bufs = col.buffers()
            if len(bufs) >= 3 and bufs[2] is not None:
                total += bufs[2].size
    return total


class _RotatingWriter:
    """Parquet writer that rotates to a new file when size threshold is exceeded.

    Files are named ``part_000.parquet``, ``part_001.parquet``, etc.
    """

    def __init__(
        self,
        directory: Path,
        schema: pa.Schema,
        writer_kwargs: dict,
        max_file_bytes: int,
    ):
        self._dir = directory
        self._dir.mkdir(parents=True, exist_ok=True)
        self._schema = schema
        self._kwargs = writer_kwargs
        self._max = max_file_bytes
        self._idx = 0
        self._cum_bytes = 0
        self._writer: pq.ParquetWriter | None = None

    def _open(self) -> pq.ParquetWriter:
        path = self._dir / f"part_{self._idx:03d}.parquet"
        return pq.ParquetWriter(str(path), self._schema, **self._kwargs)

    def write(self, batch: pa.RecordBatch) -> None:
        batch_bytes = _estimate_binary_bytes(batch)

        # Rotate if current file would exceed threshold (never rotate on first write)
        if (
            self._writer is not None
            and self._cum_bytes > 0
            and self._cum_bytes + batch_bytes > self._max
        ):
            self._writer.close()
            self._idx += 1
            self._cum_bytes = 0
            self._writer = None

        if self._writer is None:
            self._writer = self._open()

        _write_table_safe(self._writer, batch, self._schema)
        self._cum_bytes += batch_bytes

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def _generate_parquet_partitioned_streaming(
    generator,
    output_path: str | Path,
    world_bounds: tuple[float, ...],
    *,
    compression: str = "zstd",
    compression_level: int = 3,
    batch_size: int = 50_000,
    max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES,
    max_batch_bytes: int = _DEFAULT_MAX_BATCH_BYTES,
    parallel: bool = False,
) -> int:
    """Partitioned streaming: files per zoom with size-based rotation.

    Output: {output_path}/zoom=N/part_000.parquet, part_001.parquet, etc.
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator._init_parquet_stream()

    schema = _parquet_schema_no_zoom()
    writers: dict[int, _RotatingWriter] = {}
    total_rows = 0
    kwargs = _writer_kwargs(compression, compression_level)

    import queue as _queue
    import threading as _threading

    # Prefer the Rust-native Arrow path (builds the RecordBatch in Rust via the
    # Arrow C-Data interface — no GIL-held dict bridge, no pa.array(list) copy).
    _use_arrow = parallel and hasattr(generator, "_next_parquet_batch_arrow")

    def _read_next():
        if _use_arrow:
            return generator._next_parquet_batch_arrow(batch_size, world_bounds, max_batch_bytes)
        return (
            generator._next_parquet_batch_parallel(batch_size, world_bounds, max_batch_bytes)
            if parallel
            else generator._next_parquet_batch(batch_size, world_bounds, max_batch_bytes)
        )

    def _consume(data) -> None:
        nonlocal total_rows
        # data is already a pyarrow.RecordBatch on the Arrow path; else a dict.
        batch = data if _use_arrow else _dict_to_record_batch(data)
        total_rows += batch.num_rows
        # Partition by zoom → write immediately
        for z in sorted(set(batch.column("zoom").to_pylist())):
            mask = pa.compute.equal(batch.column("zoom"), z)
            # Drop zoom column (it's encoded in the directory name)
            z_batch_no_zoom = batch.filter(mask).drop_columns(["zoom"])
            if z not in writers:
                writers[z] = _RotatingWriter(
                    output_dir / f"zoom={z}",
                    schema,
                    kwargs,
                    max_file_bytes,
                )
            writers[z].write(z_batch_no_zoom)

    try:
        if parallel:
            # PIPELINED consolidation. A background reader thread calls
            # _next_parquet_batch_parallel (which releases the GIL during the Rust
            # shard read) while the main thread converts+writes the PREVIOUS batch.
            # So the read of chunk N+1 overlaps the GIL-held convert/write of chunk
            # N — eliminating the un-pipelined loop's ~75%-idle-disk stall (the disk
            # is otherwise idle whenever the main thread is in dict/Arrow/ZSTD work).
            # A single producer + single consumer over a FIFO queue preserves batch
            # order, so output stays byte-identical (modulo within-tile row order)
            # to the serial path. Memory: bounded at ~(queue maxsize + 2) batches
            # (the serial loop holds 1); shrink max_batch_bytes if peak RSS matters.
            q: _queue.Queue = _queue.Queue(maxsize=2)
            sentinel = object()
            box: dict = {}

            def _producer() -> None:
                try:
                    while True:
                        d = _read_next()
                        if d is None:
                            break
                        q.put(d)
                except BaseException as exc:  # surface to the consumer thread
                    box["err"] = exc
                finally:
                    q.put(sentinel)

            reader = _threading.Thread(target=_producer, name="parquet-reader", daemon=True)
            reader.start()
            while True:
                d = q.get()
                if d is sentinel:
                    break
                _consume(d)
            reader.join()
            if "err" in box:
                raise box["err"]
        else:
            while True:
                d = _read_next()
                if d is None:
                    break
                _consume(d)
    finally:
        for w in writers.values():
            w.close()
        generator._close_parquet_stream()

    return total_rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dict_to_record_batch(data: dict) -> pa.RecordBatch:
    """Convert the Python dict from _next_parquet_batch/_collect_parquet_data to a RecordBatch."""
    zoom_arr = pa.array(list(data["zoom"]), type=pa.uint8())
    tx_arr = pa.array(list(data["tile_x"]), type=pa.uint16())
    ty_arr = pa.array(list(data["tile_y"]), type=pa.uint16())
    td_arr = pa.array(list(data["tile_d"]), type=pa.uint16())
    fid_arr = pa.array(list(data["feature_id"]), type=pa.uint32())
    gt_arr = pa.array(list(data["geom_type"]), type=pa.uint8())
    pos_arr = pa.array(list(data["positions"]), type=pa.large_binary())
    idx_arr = pa.array(list(data["indices"]), type=pa.large_binary())

    tag_arrays = []
    for tag_list in data["tags"]:
        keys = [k for k, _ in tag_list]
        vals = [v for _, v in tag_list]
        tag_arrays.append(
            pa.MapArray.from_arrays(
                [0, len(keys)],
                pa.array(keys, type=pa.utf8()),
                pa.array(vals, type=pa.utf8()),
            )
        )
    if tag_arrays:
        tags_arr = pa.concat_arrays(tag_arrays)
    else:
        tags_arr = pa.array([], type=pa.map_(pa.utf8(), pa.utf8()))

    return pa.RecordBatch.from_arrays(
        [zoom_arr, tx_arr, ty_arr, td_arr, fid_arr, gt_arr, pos_arr, idx_arr, tags_arr],
        schema=_parquet_schema(),
    )


def _dict_to_table(data: dict) -> pa.Table:
    """Convert the Python dict to a full Table (for in-memory path)."""
    batch = _dict_to_record_batch(data)
    return pa.Table.from_batches([batch])


def _parquet_schema() -> pa.Schema:
    """Return the canonical Parquet schema for tiled mesh data."""
    return pa.schema(
        [
            pa.field("zoom", pa.uint8()),
            pa.field("tile_x", pa.uint16()),
            pa.field("tile_y", pa.uint16()),
            pa.field("tile_d", pa.uint16()),
            pa.field("feature_id", pa.uint32()),
            pa.field("geom_type", pa.uint8()),
            pa.field("positions", pa.large_binary()),
            pa.field("indices", pa.large_binary()),
            pa.field("tags", pa.map_(pa.utf8(), pa.utf8())),
        ]
    )


def _parquet_schema_no_zoom() -> pa.Schema:
    """Schema without zoom column (for partitioned output where zoom is in directory name)."""
    return pa.schema(
        [
            pa.field("tile_x", pa.uint16()),
            pa.field("tile_y", pa.uint16()),
            pa.field("tile_d", pa.uint16()),
            pa.field("feature_id", pa.uint32()),
            pa.field("geom_type", pa.uint8()),
            pa.field("positions", pa.large_binary()),
            pa.field("indices", pa.large_binary()),
            pa.field("tags", pa.map_(pa.utf8(), pa.utf8())),
        ]
    )


def _writer_kwargs(compression: str, compression_level: int) -> dict:
    """Build kwargs dict for ParquetWriter."""
    kwargs: dict = {"compression": compression}
    if compression not in ("none", "NONE", None):
        kwargs["compression_level"] = compression_level
    return kwargs


# Arrow row groups have a 2 GB limit on binary columns (int32 offsets internally).
_MAX_ROW_GROUP_BYTES = 1_500_000_000  # 1.5 GB — safe margin


def _write_table_safe(
    writer: pq.ParquetWriter,
    batch: pa.RecordBatch,
    schema: pa.Schema,
) -> None:
    """Write a RecordBatch as one or more row groups, respecting the 2 GB limit."""
    # Estimate binary column size — O(1) via Arrow buffer API
    total_bytes = _estimate_binary_bytes(batch)

    if total_bytes <= _MAX_ROW_GROUP_BYTES or batch.num_rows <= 1:
        writer.write_table(pa.Table.from_batches([batch], schema=schema))
        return

    # Split in half recursively
    mid = batch.num_rows // 2
    _write_table_safe(writer, batch.slice(0, mid), schema)
    _write_table_safe(writer, batch.slice(mid), schema)
