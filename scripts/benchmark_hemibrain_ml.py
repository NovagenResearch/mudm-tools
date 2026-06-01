#!/usr/bin/env python3
"""Benchmark: PointNet classification on Hemibrain neuron meshes.

Compares two data-loading strategies:
  1. Parquet tiles (muDM tiled format) -- read via PyArrow
  2. Raw OBJ files -- parsed line-by-line with numpy

Both loaders produce identical (neuron, label) pairs so the only
variable is the I/O path.  A lightweight PointNet classifier is
trained for each loader, and timing / accuracy metrics are collected.

Usage:
    uv run python scripts/benchmark_hemibrain_ml.py \
        --parquet-dir data/hemibrain/tiles/hemibrain/parquet_partitioned \
        --obj-dir data/hemibrain/meshes \
        --metadata data/hemibrain/metadata.json \
        --epochs 50 --n-points 1024 --batch-size 32 --seed 42 \
        --output results/hemibrain_ml_benchmark.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
import sys
import time
import tracemalloc
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow.feather as feather
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import ttest_rel
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class ParquetMeshDataset(Dataset):
    """Load neuron point clouds from Parquet tiles (muDM format).

    Reads ``parquet_dir/zoom=<zoom>/*.parquet``, deduplicates by
    ``feature_id`` (each neuron may span multiple tiles), and extracts
    packed float32 positions from the ``positions`` binary column.
    """

    def __init__(
        self,
        parquet_dir: str | Path,
        zoom: int,
        n_points: int,
        label_map: dict[str, int],
        *,
        feature_ids: list[str] | None = None,
    ) -> None:
        self.n_points = n_points
        self.label_map = label_map

        parquet_dir = Path(parquet_dir)
        zoom_dir = parquet_dir / f"zoom={zoom}"
        if not zoom_dir.exists():
            raise FileNotFoundError(f"Zoom directory not found: {zoom_dir}")

        # Read all Parquet files at this zoom level
        dataset = ds.dataset(zoom_dir, format="parquet")
        table = dataset.to_table(columns=["feature_id", "positions", "tags"])

        # Convert feature_ids to a set for fast lookup
        feature_id_set = set(feature_ids) if feature_ids is not None else None

        # Batch extract to Python once (faster than per-row .as_py())
        tags_list = table.column("tags").to_pylist()
        positions_list = table.column("positions").to_pylist()

        # Aggregate all tiles per neuron: each neuron may span multiple
        # tiles at this zoom level, so we concatenate position arrays
        neuron_positions: dict[str, list[np.ndarray]] = {}
        neuron_labels: dict[str, int] = {}

        for tags_val, pos_bytes in zip(tags_list, positions_list):
            if tags_val is None or pos_bytes is None:
                continue
            tags_dict = dict(tags_val) if isinstance(tags_val, list) else tags_val

            body_id = tags_dict.get("body_id")
            if body_id is None:
                continue

            if feature_id_set is not None and body_id not in feature_id_set:
                continue

            # Only need to resolve label once per neuron
            if body_id not in neuron_labels:
                cell_type = tags_dict.get("cell_type")
                if cell_type is None or cell_type not in label_map:
                    continue
                neuron_labels[body_id] = label_map[cell_type]

            positions = np.frombuffer(pos_bytes, dtype=np.float32).reshape(-1, 3)
            if len(positions) == 0:
                continue

            if body_id not in neuron_positions:
                neuron_positions[body_id] = []
            neuron_positions[body_id].append(positions)

        # Merge all tile fragments into single arrays
        self.samples: list[tuple[np.ndarray, int]] = []
        for body_id, pos_list in neuron_positions.items():
            if body_id not in neuron_labels:
                continue
            merged = np.concatenate(pos_list, axis=0)
            self.samples.append((merged, neuron_labels[body_id]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        positions, label = self.samples[idx]
        pts = _sample_and_normalise(positions, self.n_points)
        return pts, label


class OBJMeshDataset(Dataset):
    """Load neuron point clouds from raw OBJ files.

    Each ``.obj`` file in *obj_dir* represents one neuron; the filename
    stem is the body ID used to look up the cell type in *metadata_path*.
    """

    def __init__(
        self,
        obj_dir: str | Path,
        metadata_path: str | Path,
        n_points: int,
        label_map: dict[str, int],
        *,
        feature_ids: list[str] | None = None,
    ) -> None:
        self.n_points = n_points
        self.label_map = label_map

        obj_dir = Path(obj_dir)
        metadata_path = Path(metadata_path)

        # Build body_id -> cellType mapping from metadata
        meta_lookup: dict[str, str] = {}
        if metadata_path.exists():
            raw = json.loads(metadata_path.read_text())
            for neuron in raw.get("neurons", []):
                bid = str(neuron["bodyId"])
                ct = neuron.get("cellType")
                if ct:
                    meta_lookup[bid] = ct

        # Pre-load all vertex data into memory (same as ParquetMeshDataset)
        # so training speed is identical — the benchmark isolates load time
        feature_id_set = set(feature_ids) if feature_ids is not None else None
        self.samples: list[tuple[np.ndarray, int]] = []
        for obj_path in sorted(obj_dir.glob("*.obj")):
            bid = obj_path.stem
            if feature_id_set is not None and bid not in feature_id_set:
                continue
            ct = meta_lookup.get(bid)
            if ct is None or ct not in label_map:
                continue
            positions = _parse_obj_vertices(obj_path)
            if len(positions) == 0:
                continue
            self.samples.append((positions, label_map[ct]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        positions, label = self.samples[idx]
        pts = _sample_and_normalise(positions, self.n_points)
        return pts, label


class ArrowIPCMeshDataset(Dataset):
    """Load neuron point clouds from Arrow IPC (Feather v2) files.

    Expects a single ``.arrow`` file produced by converting the Parquet
    zoom directory via ``pyarrow.feather.write_feather()``.
    """

    def __init__(
        self,
        arrow_path: str | Path,
        n_points: int,
        label_map: dict[str, int],
        *,
        feature_ids: list[str] | None = None,
    ) -> None:
        self.n_points = n_points
        self.label_map = label_map

        table = feather.read_table(Path(arrow_path))
        feature_id_set = set(feature_ids) if feature_ids is not None else None

        tags_list = table.column("tags").to_pylist()
        positions_list = table.column("positions").to_pylist()

        neuron_positions: dict[str, list[np.ndarray]] = {}
        neuron_labels: dict[str, int] = {}

        for tags_val, pos_bytes in zip(tags_list, positions_list):
            if tags_val is None or pos_bytes is None:
                continue
            tags_dict = dict(tags_val) if isinstance(tags_val, list) else tags_val
            body_id = tags_dict.get("body_id")
            if body_id is None:
                continue
            if feature_id_set is not None and body_id not in feature_id_set:
                continue
            if body_id not in neuron_labels:
                cell_type = tags_dict.get("cell_type")
                if cell_type is None or cell_type not in label_map:
                    continue
                neuron_labels[body_id] = label_map[cell_type]
            positions = np.frombuffer(pos_bytes, dtype=np.float32).reshape(-1, 3)
            if len(positions) == 0:
                continue
            if body_id not in neuron_positions:
                neuron_positions[body_id] = []
            neuron_positions[body_id].append(positions)

        self.samples: list[tuple[np.ndarray, int]] = []
        for body_id, pos_list in neuron_positions.items():
            if body_id not in neuron_labels:
                continue
            merged = np.concatenate(pos_list, axis=0)
            self.samples.append((merged, neuron_labels[body_id]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        positions, label = self.samples[idx]
        pts = _sample_and_normalise(positions, self.n_points)
        return pts, label


def convert_parquet_to_arrow_ipc(
    parquet_dir: Path, zoom: int, output_path: Path,
) -> float:
    """Convert a Parquet zoom partition to Arrow IPC (Feather v2).

    Returns the conversion time in seconds.
    """
    zoom_dir = parquet_dir / f"zoom={zoom}"
    t0 = time.perf_counter()
    dataset = ds.dataset(zoom_dir, format="parquet")
    table = dataset.to_table(columns=["feature_id", "positions", "tags"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feather.write_feather(table, output_path)
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_obj_vertices(path: Path) -> np.ndarray:
    """Parse vertex positions from an OBJ file (``v x y z`` lines)."""
    verts: list[list[float]] = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(verts, dtype=np.float32) if verts else np.zeros((1, 3), dtype=np.float32)


def _sample_and_normalise(positions: np.ndarray, n_points: int) -> torch.Tensor:
    """Randomly sample *n_points* and normalise to unit sphere."""
    n = len(positions)
    if n == 0:
        return torch.zeros(n_points, 3)
    if n >= n_points:
        idx = np.random.choice(n, n_points, replace=False)
    else:
        idx = np.random.choice(n, n_points, replace=True)
    pts = positions[idx].copy()

    # Centre to origin
    centroid = pts.mean(axis=0)
    pts -= centroid

    # Scale to unit sphere
    max_dist = np.linalg.norm(pts, axis=1).max()
    if max_dist > 0:
        pts /= max_dist

    return torch.from_numpy(pts)


# ---------------------------------------------------------------------------
# PointNet-lite model
# ---------------------------------------------------------------------------


class PointNet(nn.Module):
    """Lightweight PointNet classifier.

    Architecture: shared MLP (3->64->128->256) with BatchNorm and ReLU,
    global max pooling, then FC 256->128->num_classes with dropout.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        # Shared MLP (applied per-point via Conv1d)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        # Classifier head
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, N, 3) point clouds.

        Returns:
            (B, num_classes) logits.
        """
        # (B, N, 3) -> (B, 3, N) for Conv1d
        x = x.transpose(1, 2)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        # Global max pool: (B, 256, N) -> (B, 256)
        x = x.max(dim=2)[0]
        return self.fc(x)


# ---------------------------------------------------------------------------
# Training / evaluation loop
# ---------------------------------------------------------------------------


def _train_and_evaluate(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    num_classes: int,
    *,
    epochs: int,
    batch_size: int,
    lr: float = 1e-3,
    patience: int = 10,
    device: torch.device | None = None,
) -> dict:
    """Train PointNet and return metrics dict."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- DataLoaders (num_workers=0 for fair comparison) ---
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    # --- Time to first batch ---
    t0 = time.perf_counter()
    _first_batch = next(iter(train_loader))
    time_to_first_batch_ms = (time.perf_counter() - t0) * 1000.0
    del _first_batch

    # --- Model, optimizer, loss ---
    model = PointNet(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # --- Training loop ---
    best_val_loss = float("inf")
    epochs_no_improve = 0
    history: list[dict] = []
    epoch_times: list[float] = []
    data_load_times: list[float] = []

    tracemalloc.start()

    for epoch in range(1, epochs + 1):
        # -- Train --
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        epoch_data_time = 0.0
        epoch_start = time.perf_counter()

        t_data_start = time.perf_counter()
        for points, labels in train_loader:
            epoch_data_time += time.perf_counter() - t_data_start

            points = points.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            logits = model(points)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * labels.size(0)
            train_count += labels.size(0)

            t_data_start = time.perf_counter()

        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)
        data_load_times.append(epoch_data_time)

        train_loss = train_loss_sum / max(train_count, 1)

        # -- Validate --
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_count = 0
        with torch.no_grad():
            for points, labels in val_loader:
                points = points.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)
                logits = model(points)
                loss = criterion(logits, labels)
                val_loss_sum += loss.item() * labels.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_count += labels.size(0)

        val_loss = val_loss_sum / max(val_count, 1)
        val_acc = val_correct / max(val_count, 1)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_accuracy": round(val_acc, 5),
        })

        print(
            f"  Epoch {epoch:3d}/{epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_acc={val_acc:.4f}  "
            f"val_loss={val_loss:.4f}",
            flush=True,
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}", flush=True)
                break

    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # --- Test evaluation (using best model) ---
    model.load_state_dict(best_state)
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for points, labels in test_loader:
            points = points.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            logits = model(points)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    test_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / max(len(all_labels), 1)
    test_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {
        "test_accuracy": round(test_acc, 5),
        "test_f1_macro": round(test_f1, 5),
        "time_to_first_batch_ms": round(time_to_first_batch_ms, 2),
        "mean_epoch_time_s": round(float(np.mean(epoch_times)), 4),
        "mean_data_load_time_s": round(float(np.mean(data_load_times)), 4),
        "peak_memory_mb": round(peak_mem / (1024 * 1024), 2),
        "training_history": history,
    }


# ---------------------------------------------------------------------------
# Dataset preparation helpers
# ---------------------------------------------------------------------------


def _build_label_map_and_splits(
    metadata_path: Path,
    obj_dir: Path,
    min_instances: int = 10,
    max_classes: int = 0,
    seed: int = 42,
) -> tuple[dict[str, int], list[str], list[str], list[str], list[str]]:
    """Build label map and 80/10/10 stratified splits.

    Returns:
        (label_map, all_feature_ids, train_ids, val_ids, test_ids)
        where label_map maps cell_type -> int.
    """
    raw = json.loads(metadata_path.read_text())
    neurons = raw.get("neurons", [])

    # Map body_id -> cell_type (only neurons with OBJ files)
    available_objs = {p.stem for p in obj_dir.glob("*.obj")}
    bid_to_ct: dict[str, str] = {}
    for n in neurons:
        bid = str(n["bodyId"])
        ct = n.get("cellType")
        if ct and bid in available_objs:
            bid_to_ct[bid] = ct

    # Filter to cell types with >= min_instances
    ct_counts = Counter(bid_to_ct.values())
    valid_types = {ct for ct, count in ct_counts.items() if count >= min_instances}

    # Optionally limit to top N most common cell types
    if max_classes > 0 and len(valid_types) > max_classes:
        top_n = [ct for ct, _ in ct_counts.most_common(max_classes)
                 if ct in valid_types]
        valid_types = set(top_n[:max_classes])

    filtered = {bid: ct for bid, ct in bid_to_ct.items() if ct in valid_types}

    # Stable label map (sorted)
    sorted_types = sorted(valid_types)
    label_map = {ct: i for i, ct in enumerate(sorted_types)}

    # Stratified split
    bids = sorted(filtered.keys())
    labels = [filtered[bid] for bid in bids]

    # First split: 80% train, 20% temp
    train_ids, temp_ids, train_labels, temp_labels = train_test_split(
        bids, labels, test_size=0.2, stratify=labels, random_state=seed,
    )
    # Second split: 50/50 of temp -> 10% val, 10% test
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.5, stratify=temp_labels, random_state=seed,
    )

    return label_map, bids, train_ids, val_ids, test_ids


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def _parse_seeds(value: str) -> list[int]:
    """Parse a comma-separated seed list to ``list[int]``.

    Raises ``argparse.ArgumentTypeError`` if any entry is not a valid integer.
    """
    if not value or not value.strip():
        raise argparse.ArgumentTypeError("--seeds must contain at least one integer")
    out: list[int] = []
    for tok in value.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid integer in --seeds: {tok!r}"
            ) from exc
    if not out:
        raise argparse.ArgumentTypeError("--seeds must contain at least one integer")
    return out


def run_benchmark(args: argparse.Namespace) -> dict:
    """Run the full benchmark across multiple seeds."""

    parquet_dir = Path(args.parquet_dir)
    obj_dir = Path(args.obj_dir)
    metadata_path = Path(args.metadata)
    # ``args.seeds`` is parsed by argparse via ``_parse_seeds`` and is already
    # a list[int].  Fall back to the legacy ``--seed`` / ``--num-seeds`` form
    # only when the parsed list is empty (should never happen with type=).
    if args.seeds:
        seeds = list(args.seeds)
    else:
        seeds = [args.seed + i for i in range(args.num_seeds)]

    # --quick: shrink the workload for smoke tests.  Two epochs and a tiny
    # cell-type cap keeps the run under a couple of minutes while still
    # exercising the full code path (load -> train -> evaluate).
    if args.quick:
        args.epochs = min(args.epochs, 2)
        if args.max_classes == 0:
            args.max_classes = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build label map and splits using the first seed
    label_map, all_ids, train_ids, val_ids, test_ids = _build_label_map_and_splits(
        metadata_path, obj_dir, min_instances=10,
        max_classes=args.max_classes, seed=seeds[0],
    )
    num_classes = len(label_map)
    print(f"Classes: {num_classes}")
    print(f"Samples: {len(all_ids)} total "
          f"({len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test)")

    # Convert Parquet to Arrow IPC once (priming step)
    arrow_ipc_path = parquet_dir / "zoom3_primed.arrow"
    print("\n--- Converting Parquet to Arrow IPC (one-time priming) ---", flush=True)
    arrow_convert_time = convert_parquet_to_arrow_ipc(parquet_dir, zoom=3, output_path=arrow_ipc_path)
    print(f"  Arrow IPC priming: {arrow_convert_time:.1f}s", flush=True)

    # Collect per-run results
    parquet_runs: list[dict] = []
    arrow_runs: list[dict] = []
    obj_runs: list[dict] = []
    # Nested per-seed view: per_seed_results[seed][loader_key] = {accuracy,
    # macro_f1, load_time_s, epoch_time_s} -- used by the multi-seed
    # aggregation block below.
    per_seed_results: dict[int, dict[str, dict[str, float]]] = {}

    for run_idx, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"Run {run_idx + 1}/{len(seeds)}  (seed={seed})")
        print(f"{'='*60}")

        # Set seeds for reproducibility (Python random, NumPy, Torch, CUDA)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        per_seed_results[seed] = {}

        # Re-split with this seed (keeps same label_map but shuffles splits)
        _, _, train_ids, val_ids, test_ids = _build_label_map_and_splits(
            metadata_path, obj_dir, min_instances=10,
            max_classes=args.max_classes, seed=seed,
        )

        # --- Parquet loader ---
        print(f"\n--- Parquet loader (seed={seed}) ---", flush=True)
        t_load = time.perf_counter()
        train_pq = ParquetMeshDataset(
            parquet_dir, zoom=3, n_points=args.n_points,
            label_map=label_map, feature_ids=train_ids,
        )
        val_pq = ParquetMeshDataset(
            parquet_dir, zoom=3, n_points=args.n_points,
            label_map=label_map, feature_ids=val_ids,
        )
        test_pq = ParquetMeshDataset(
            parquet_dir, zoom=3, n_points=args.n_points,
            label_map=label_map, feature_ids=test_ids,
        )
        pq_load_time = time.perf_counter() - t_load
        print(f"  Loaded: {len(train_pq)}/{len(val_pq)}/{len(test_pq)} "
              f"(train/val/test) in {pq_load_time:.1f}s", flush=True)
        pq_metrics = _train_and_evaluate(
            train_pq, val_pq, test_pq, num_classes,
            epochs=args.epochs, batch_size=args.batch_size, device=device,
        )
        pq_metrics["dataset_load_time_s"] = round(pq_load_time, 2)
        parquet_runs.append(pq_metrics)
        per_seed_results[seed]["parquet"] = {
            "accuracy": float(pq_metrics["test_accuracy"]),
            "macro_f1": float(pq_metrics["test_f1_macro"]),
            "load_time_s": float(pq_load_time),
            "epoch_time_s": float(pq_metrics["mean_epoch_time_s"]),
        }
        del train_pq, val_pq, test_pq

        # --- OBJ loader ---
        print(f"\n--- OBJ loader (seed={seed}) ---", flush=True)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        t_load = time.perf_counter()
        train_obj = OBJMeshDataset(
            obj_dir, metadata_path, n_points=args.n_points,
            label_map=label_map, feature_ids=train_ids,
        )
        val_obj = OBJMeshDataset(
            obj_dir, metadata_path, n_points=args.n_points,
            label_map=label_map, feature_ids=val_ids,
        )
        test_obj = OBJMeshDataset(
            obj_dir, metadata_path, n_points=args.n_points,
            label_map=label_map, feature_ids=test_ids,
        )
        obj_load_time = time.perf_counter() - t_load
        print(f"  Loaded: {len(train_obj)}/{len(val_obj)}/{len(test_obj)} "
              f"(train/val/test) in {obj_load_time:.1f}s", flush=True)
        obj_metrics = _train_and_evaluate(
            train_obj, val_obj, test_obj, num_classes,
            epochs=args.epochs, batch_size=args.batch_size, device=device,
        )
        obj_metrics["dataset_load_time_s"] = round(obj_load_time, 2)
        obj_runs.append(obj_metrics)
        per_seed_results[seed]["raw_obj"] = {
            "accuracy": float(obj_metrics["test_accuracy"]),
            "macro_f1": float(obj_metrics["test_f1_macro"]),
            "load_time_s": float(obj_load_time),
            "epoch_time_s": float(obj_metrics["mean_epoch_time_s"]),
        }
        del train_obj, val_obj, test_obj

        # --- Arrow IPC loader ---
        print(f"\n--- Arrow IPC loader (seed={seed}) ---", flush=True)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        t_load = time.perf_counter()
        train_arrow = ArrowIPCMeshDataset(
            arrow_ipc_path, n_points=args.n_points,
            label_map=label_map, feature_ids=train_ids,
        )
        val_arrow = ArrowIPCMeshDataset(
            arrow_ipc_path, n_points=args.n_points,
            label_map=label_map, feature_ids=val_ids,
        )
        test_arrow = ArrowIPCMeshDataset(
            arrow_ipc_path, n_points=args.n_points,
            label_map=label_map, feature_ids=test_ids,
        )
        arrow_load_time = time.perf_counter() - t_load
        print(f"  Loaded: {len(train_arrow)}/{len(val_arrow)}/{len(test_arrow)} "
              f"(train/val/test) in {arrow_load_time:.1f}s", flush=True)
        arrow_metrics = _train_and_evaluate(
            train_arrow, val_arrow, test_arrow, num_classes,
            epochs=args.epochs, batch_size=args.batch_size, device=device,
        )
        arrow_metrics["dataset_load_time_s"] = round(arrow_load_time, 2)
        arrow_runs.append(arrow_metrics)
        per_seed_results[seed]["arrow_ipc"] = {
            "accuracy": float(arrow_metrics["test_accuracy"]),
            "macro_f1": float(arrow_metrics["test_f1_macro"]),
            "load_time_s": float(arrow_load_time),
            "epoch_time_s": float(arrow_metrics["mean_epoch_time_s"]),
        }
        del train_arrow, val_arrow, test_arrow
        gc.collect()

    # --- Aggregate across runs (legacy shape, kept for backward compat) ---
    def _aggregate(runs: list[dict]) -> dict:
        """Compute mean +/- std for each metric across runs."""
        keys = [
            "test_accuracy", "test_f1_macro", "time_to_first_batch_ms",
            "mean_epoch_time_s", "mean_data_load_time_s", "peak_memory_mb",
            "dataset_load_time_s",
        ]
        agg: dict = {}
        for k in keys:
            vals = [r[k] for r in runs]
            agg[k] = round(float(np.mean(vals)), 5)
            agg[f"{k}_std"] = round(float(np.std(vals)), 5)
        # Include the full training history from the first run
        agg["training_history"] = runs[0]["training_history"]
        return agg

    hardware_str = (
        f"{platform.processor() or platform.machine()} / "
        f"{'CUDA ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )
    hardware = {
        "description": hardware_str,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_cores": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": (
            torch.cuda.device_count() if torch.cuda.is_available() else 0
        ),
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }

    # --- Multi-seed summary statistics (B1 schema) -----------------------
    def _seed_stats(loader_key: str) -> dict:
        """Aggregate per-seed metrics for *loader_key* into mean/SD/per-seed."""
        per_seed = [per_seed_results[s][loader_key] for s in seeds]
        accs = [r["accuracy"] for r in per_seed]
        f1s = [r["macro_f1"] for r in per_seed]
        loads = [r["load_time_s"] for r in per_seed]
        epochs = [r["epoch_time_s"] for r in per_seed]
        return {
            "accuracy_mean": float(np.mean(accs)),
            # Sample SD (ddof=1); 0.0 when only one seed.
            "accuracy_sd": (
                float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
            ),
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_sd": (
                float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0
            ),
            "load_time_mean_s": float(np.mean(loads)),
            "load_time_sd_s": (
                float(np.std(loads, ddof=1)) if len(loads) > 1 else 0.0
            ),
            "epoch_time_mean_s": float(np.mean(epochs)),
            "epoch_time_sd_s": (
                float(np.std(epochs, ddof=1)) if len(epochs) > 1 else 0.0
            ),
            "per_seed_accuracy": [float(x) for x in accs],
            "per_seed_macro_f1": [float(x) for x in f1s],
            "per_seed_load_time_s": [float(x) for x in loads],
            "per_seed_epoch_time_s": [float(x) for x in epochs],
        }

    parquet_stats = _seed_stats("parquet")
    arrow_stats = _seed_stats("arrow_ipc")
    raw_obj_stats = _seed_stats("raw_obj")

    # Paired t-test: Parquet vs raw OBJ accuracy (one paired observation
    # per seed).  Only computable with >= 2 seeds.
    pq_accs = [per_seed_results[s]["parquet"]["accuracy"] for s in seeds]
    obj_accs = [per_seed_results[s]["raw_obj"]["accuracy"] for s in seeds]
    if len(seeds) >= 2:
        t_stat, p_val = ttest_rel(pq_accs, obj_accs)
        # ttest_rel returns NaN when both arrays are identical; coerce to 0/1.
        if np.isnan(t_stat) or np.isnan(p_val):
            paired_t_test = {
                "t_statistic": 0.0,
                "p_value": 1.0,
                "note": "identical samples; t-statistic undefined, p set to 1.0",
            }
        else:
            paired_t_test = {
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
            }
    else:
        paired_t_test = {
            "t_statistic": None,
            "p_value": None,
            "note": "fewer than 2 seeds; t-test not computed",
        }

    results = {
        "benchmark_id": "B1_hemibrain_5seed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware,
        "config": {
            "parquet_dir": str(parquet_dir),
            "obj_dir": str(obj_dir),
            "metadata": str(metadata_path),
            "epochs": args.epochs,
            "n_points": args.n_points,
            "batch_size": args.batch_size,
            "max_classes": args.max_classes,
            "quick": bool(args.quick),
            "seeds": seeds,
        },
        "dataset": "hemibrain_v1.2.1",
        "n_points": args.n_points,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "num_seeds": len(seeds),
        "seeds": seeds,
        "num_classes": num_classes,
        "num_train": len(train_ids),
        "num_val": len(val_ids),
        "num_test": len(test_ids),
        "arrow_ipc_convert_time_s": round(arrow_convert_time, 2),
        # B1 schema: per-loader mean / SD / per-seed lists.
        "parquet": parquet_stats,
        "arrow_ipc": arrow_stats,
        "raw_obj": raw_obj_stats,
        # Legacy schema retained for callers that still read these keys.
        "parquet_legacy": _aggregate(parquet_runs),
        "arrow_ipc_legacy": _aggregate(arrow_runs),
        "obj": _aggregate(obj_runs),
        "paired_t_test_parquet_vs_raw": paired_t_test,
    }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark PointNet on Hemibrain: Parquet vs OBJ loading",
    )
    parser.add_argument(
        "--parquet-dir", type=str,
        default="data/hemibrain/tiles/hemibrain/parquet_partitioned",
        help="Path to partitioned Parquet directory",
    )
    parser.add_argument(
        "--obj-dir", type=str,
        default="data/hemibrain/meshes",
        help="Path to directory of OBJ mesh files",
    )
    parser.add_argument(
        "--metadata", type=str,
        default="data/hemibrain/metadata.json",
        help="Path to metadata.json with neuron info",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--n-points", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base seed (legacy; only used when --seeds is the empty string)",
    )
    parser.add_argument(
        "--seeds", type=_parse_seeds, default=_parse_seeds("42,123,456,789,1024"),
        help=(
            "Comma-separated seed list (e.g. '42,123,456,789,1024'). "
            "Default: 42,123,456,789,1024 -- the canonical B1 5-seed sweep."
        ),
    )
    parser.add_argument(
        "--num-seeds", type=int, default=5,
        help="Number of seeds when --seeds not provided (legacy)",
    )
    parser.add_argument(
        "--max-classes", type=int, default=0,
        help="Limit to top N cell types (0=all)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help=(
            "Smoke-test mode: cap epochs to 2 and limit max_classes to 3. "
            "Intended only for sanity checks / CI."
        ),
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help=(
            "Path to write JSON results. If omitted, the result dict is "
            "printed to stdout."
        ),
    )
    args = parser.parse_args()

    results = run_benchmark(args)

    # Write output (or print to stdout when --output is omitted).
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults written to {output_path}")
    else:
        print(json.dumps(results, indent=2))

    # Print summary (B1 schema)
    print(f"\n{'='*60}")
    print(f"SUMMARY (mean +/- SD over {results['num_seeds']} seed(s))")
    print(f"{'='*60}")
    for loader, label in (("parquet", "PARQUET"), ("arrow_ipc", "ARROW_IPC"),
                          ("raw_obj", "RAW_OBJ")):
        r = results[loader]
        print(f"\n  {label}:")
        print(f"    Test accuracy:    {r['accuracy_mean']:.4f} +/- {r['accuracy_sd']:.4f}")
        print(f"    Test F1 (macro):  {r['macro_f1_mean']:.4f} +/- {r['macro_f1_sd']:.4f}")
        print(f"    Load time (s):    {r['load_time_mean_s']:.3f}")
        print(f"    Epoch time (s):   {r['epoch_time_mean_s']:.3f}")

    tt = results["paired_t_test_parquet_vs_raw"]
    print("\n  Paired t-test (parquet vs raw_obj accuracy):")
    print(f"    t = {tt['t_statistic']}, p = {tt['p_value']}")


if __name__ == "__main__":
    main()
