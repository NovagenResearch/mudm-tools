#!/usr/bin/env python3
"""Benchmark: PointNet-2D cell-type classification on CoNSeP dataset.

Compares two data-loading strategies:
  1. Parquet tiles (muDM tiled format) -- read via PyArrow
  2. Raw GeoJSON files -- parsed with json

Both loaders produce identical (polygon_points, label) pairs so the only
variable is the I/O path.  A lightweight PointNet-2D classifier is trained
for each loader, and timing / accuracy metrics are collected.

Usage::

    uv run python scripts/benchmark_consep_ml.py \
        --parquet-path data/consep/tiles.parquet \
        --geojson-dir data/consep/geojson \
        --output results/consep_ml_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import tracemalloc
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))


# Cell types used for classification (matching download_consep.py)
# Skip "other" (type 1) — too heterogeneous
LABEL_NAMES = [
    "inflammatory",       # type 2
    "epithelial",         # type 3
    "dysplastic_malignant",  # type 4
    "fibroblast",         # type 5
    "muscle",             # type 6
    "endothelial",        # type 7
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _sample_polygon_boundary(
    polygon: np.ndarray, n_points: int
) -> np.ndarray:
    """Sample N evenly-spaced points along polygon perimeter.

    Args:
        polygon: (M, 2) array of polygon vertices (x, y).
        n_points: Number of points to sample.

    Returns:
        (n_points, 2) array of sampled points.
    """
    if len(polygon) < 2:
        return np.zeros((n_points, 2), dtype=np.float32)

    # Compute cumulative arc length along the boundary
    diffs = np.diff(polygon, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cum_lengths = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cum_lengths[-1]

    if total_length < 1e-8:
        return np.tile(polygon[0], (n_points, 1)).astype(np.float32)

    # Sample at evenly-spaced arc-length positions
    target_lengths = np.linspace(0, total_length, n_points, endpoint=False)
    sampled = np.zeros((n_points, 2), dtype=np.float32)

    for i, t in enumerate(target_lengths):
        # Find which segment this falls in
        idx = np.searchsorted(cum_lengths, t, side="right") - 1
        idx = min(idx, len(polygon) - 2)
        idx = max(idx, 0)

        # Interpolate within segment
        seg_start = cum_lengths[idx]
        seg_len = seg_lengths[idx]
        if seg_len > 0:
            frac = (t - seg_start) / seg_len
        else:
            frac = 0.0
        sampled[i] = polygon[idx] + frac * (polygon[idx + 1] - polygon[idx])

    return sampled


def _normalise_points(points: np.ndarray) -> np.ndarray:
    """Centre to origin and scale to unit range."""
    centroid = points.mean(axis=0)
    pts = points - centroid
    max_range = np.abs(pts).max()
    if max_range > 0:
        pts /= max_range
    return pts.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset: Parquet tiles
# ---------------------------------------------------------------------------


class ParquetCellDataset(Dataset):
    """Load cell polygons from Parquet tiles (muDM 2D format).

    Reads the Parquet file at zoom=0 (full resolution), extracts
    polygon vertices from the positions column, and cell type from tags.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        n_points: int,
        label_map: dict[str, int],
        *,
        feature_ids: set[int] | None = None,
    ) -> None:
        import pyarrow.dataset as ds
        import pyarrow.compute as pc

        self.n_points = n_points
        self.label_map = label_map

        parquet_path = Path(parquet_path)
        if parquet_path.is_dir():
            dataset = ds.dataset(str(parquet_path), format="parquet", partitioning="hive")
        else:
            dataset = ds.dataset(str(parquet_path), format="parquet")

        # Read zoom=0 for full-resolution polygons. Deduplicate by feature_id.
        # Use predicate pushdown for zoom filter.
        zoom_filter = pc.field("zoom") == 0
        table = dataset.to_table(
            columns=["feature_id", "positions", "tags"],
            filter=zoom_filter,
        )

        seen: set[int] = set()
        self.samples: list[tuple[np.ndarray, int]] = []

        for i in range(table.num_rows):
            fid = int(table.column("feature_id")[i].as_py())

            if feature_ids is not None and fid not in feature_ids:
                continue
            if fid in seen:
                continue
            seen.add(fid)

            # Extract cell type from tags map
            tags_val = table.column("tags")[i].as_py()
            if tags_val is None:
                continue
            cell_type = tags_val.get("cell_type")
            if cell_type is None or cell_type not in label_map:
                continue

            # Decode positions: packed float32 xy pairs
            pos_bytes = table.column("positions")[i].as_py()
            positions = np.frombuffer(pos_bytes, dtype=np.float32).reshape(-1, 2)
            if len(positions) < 3:
                continue

            self.samples.append((positions, label_map[cell_type]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        positions, label = self.samples[idx]
        pts = _sample_polygon_boundary(positions, self.n_points)
        pts = _normalise_points(pts)
        return torch.from_numpy(pts), label


# ---------------------------------------------------------------------------
# Dataset: Raw GeoJSON files
# ---------------------------------------------------------------------------


class GeoJSONCellDataset(Dataset):
    """Load cell polygons from raw GeoJSON files.

    Each .geojson file is a FeatureCollection where each Feature is a
    cell with Polygon geometry and properties including cell_type.
    """

    def __init__(
        self,
        geojson_dir: str | Path,
        n_points: int,
        label_map: dict[str, int],
        *,
        feature_ids: set[int] | None = None,
    ) -> None:
        self.n_points = n_points
        self.label_map = label_map

        geojson_dir = Path(geojson_dir)
        gj_files = sorted(geojson_dir.glob("*.geojson"))

        self.samples: list[tuple[list, int]] = []  # (coordinates, label)
        global_fid = 0

        for gj_path in gj_files:
            data = json.loads(gj_path.read_text())
            for feat in data.get("features", []):
                if feature_ids is not None and global_fid not in feature_ids:
                    global_fid += 1
                    continue

                props = feat.get("properties", {})
                cell_type = props.get("cell_type")
                if cell_type is None or cell_type not in label_map:
                    global_fid += 1
                    continue

                geom = feat.get("geometry", {})
                coords = geom.get("coordinates", [[]])
                # Take outer ring
                ring = coords[0] if coords else []
                if len(ring) < 3:
                    global_fid += 1
                    continue

                self.samples.append((ring, label_map[cell_type]))
                global_fid += 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        ring, label = self.samples[idx]
        polygon = np.array(ring, dtype=np.float32)
        pts = _sample_polygon_boundary(polygon, self.n_points)
        pts = _normalise_points(pts)
        return torch.from_numpy(pts), label


# ---------------------------------------------------------------------------
# PointNet-2D model
# ---------------------------------------------------------------------------


class PointNet2D(nn.Module):
    """PointNet variant for 2D polygon classification.

    Architecture: shared MLP via Conv1d (2->64->128->128),
    global max pooling, then FC 128->64->num_classes.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(2, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, N, 2) point sets.

        Returns:
            (B, num_classes) logits.
        """
        # (B, N, 2) -> (B, 2, N) for Conv1d
        x = x.transpose(1, 2)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        # Global max pool: (B, 128, N) -> (B, 128)
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
    """Train PointNet-2D and return metrics dict."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    # Time to first batch
    t0 = time.perf_counter()
    _first_batch = next(iter(train_loader))
    time_to_first_batch_ms = (time.perf_counter() - t0) * 1000.0
    del _first_batch

    model = PointNet2D(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None
    epoch_times: list[float] = []
    data_load_times: list[float] = []

    tracemalloc.start()

    for epoch in range(1, epochs + 1):
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

        # Validate
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

        if epoch % 5 == 0 or epoch == 1:
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

    # Test evaluation using best model
    if best_state is not None:
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
    }


# ---------------------------------------------------------------------------
# Build label map and splits from GeoJSON
# ---------------------------------------------------------------------------


def _build_label_map_and_splits(
    geojson_dir: Path,
    min_instances: int = 5,
    seed: int = 42,
) -> tuple[dict[str, int], list[int], list[int], list[int], list[int]]:
    """Build label map and 80/10/10 stratified splits.

    Returns:
        (label_map, all_fids, train_fids, val_fids, test_fids)
    """
    gj_files = sorted(geojson_dir.glob("*.geojson"))

    # Collect all features with global feature IDs
    fid_to_type: dict[int, str] = {}
    global_fid = 0

    for gj_path in gj_files:
        data = json.loads(gj_path.read_text())
        for feat in data.get("features", []):
            ct = feat.get("properties", {}).get("cell_type")
            if ct and ct in LABEL_NAMES:
                fid_to_type[global_fid] = ct
            global_fid += 1

    # Filter to types with >= min_instances
    type_counts = Counter(fid_to_type.values())
    valid_types = {ct for ct, count in type_counts.items() if count >= min_instances}
    filtered = {fid: ct for fid, ct in fid_to_type.items() if ct in valid_types}

    # Stable label map
    sorted_types = sorted(valid_types)
    label_map = {ct: i for i, ct in enumerate(sorted_types)}

    # Stratified split
    fids = sorted(filtered.keys())
    labels = [filtered[fid] for fid in fids]

    # 80/20 split first
    train_fids, temp_fids, train_labels, temp_labels = train_test_split(
        fids, labels, test_size=0.2, stratify=labels, random_state=seed,
    )
    # 50/50 of temp -> 10% val, 10% test
    val_fids, test_fids = train_test_split(
        temp_fids, test_size=0.5, stratify=temp_labels, random_state=seed,
    )

    return label_map, fids, train_fids, val_fids, test_fids


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_benchmark(args: argparse.Namespace) -> dict:
    """Run the benchmark: Parquet vs GeoJSON loading."""

    parquet_path = Path(args.parquet_path)
    geojson_dir = Path(args.geojson_dir)
    seed = args.seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build label map and splits
    label_map, all_fids, train_fids, val_fids, test_fids = _build_label_map_and_splits(
        geojson_dir, min_instances=5, seed=seed,
    )
    num_classes = len(label_map)
    print(f"Classes: {num_classes} — {list(label_map.keys())}")
    print(f"Samples: {len(all_fids)} total "
          f"({len(train_fids)} train / {len(val_fids)} val / {len(test_fids)} test)")

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_fid_set = set(train_fids)
    val_fid_set = set(val_fids)
    test_fid_set = set(test_fids)

    # --- Parquet loader ---
    print(f"\n{'='*60}")
    print("Parquet loader")
    print(f"{'='*60}")

    if not parquet_path.exists():
        print(f"  WARNING: Parquet file not found: {parquet_path}")
        print("  Run: uv run python scripts/download_consep.py --output-dir data/consep --tile")
        parquet_metrics = None
    else:
        t_load = time.perf_counter()
        train_pq = ParquetCellDataset(
            parquet_path, n_points=args.n_points,
            label_map=label_map, feature_ids=train_fid_set,
        )
        val_pq = ParquetCellDataset(
            parquet_path, n_points=args.n_points,
            label_map=label_map, feature_ids=val_fid_set,
        )
        test_pq = ParquetCellDataset(
            parquet_path, n_points=args.n_points,
            label_map=label_map, feature_ids=test_fid_set,
        )
        load_time = time.perf_counter() - t_load
        print(f"  Loaded: {len(train_pq)}/{len(val_pq)}/{len(test_pq)} "
              f"(train/val/test) in {load_time:.2f}s")

        np.random.seed(seed)
        torch.manual_seed(seed)
        parquet_metrics = _train_and_evaluate(
            train_pq, val_pq, test_pq, num_classes,
            epochs=args.epochs, batch_size=args.batch_size, device=device,
        )
        parquet_metrics["dataset_load_time_s"] = round(load_time, 4)

    # --- GeoJSON loader ---
    print(f"\n{'='*60}")
    print("GeoJSON loader")
    print(f"{'='*60}")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    t_load = time.perf_counter()
    train_gj = GeoJSONCellDataset(
        geojson_dir, n_points=args.n_points,
        label_map=label_map, feature_ids=train_fid_set,
    )
    val_gj = GeoJSONCellDataset(
        geojson_dir, n_points=args.n_points,
        label_map=label_map, feature_ids=val_fid_set,
    )
    test_gj = GeoJSONCellDataset(
        geojson_dir, n_points=args.n_points,
        label_map=label_map, feature_ids=test_fid_set,
    )
    load_time = time.perf_counter() - t_load
    print(f"  Loaded: {len(train_gj)}/{len(val_gj)}/{len(test_gj)} "
          f"(train/val/test) in {load_time:.2f}s")

    np.random.seed(seed)
    torch.manual_seed(seed)
    geojson_metrics = _train_and_evaluate(
        train_gj, val_gj, test_gj, num_classes,
        epochs=args.epochs, batch_size=args.batch_size, device=device,
    )
    geojson_metrics["dataset_load_time_s"] = round(load_time, 4)

    # --- Results ---
    hardware = (
        f"{platform.processor() or platform.machine()} / "
        f"{'CUDA ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )

    results = {
        "dataset": "CoNSeP",
        "reference": "Graham et al., Medical Image Analysis 2019",
        "num_cells": len(all_fids),
        "num_classes": num_classes,
        "class_names": list(label_map.keys()),
        "n_points": args.n_points,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "seed": seed,
        "num_train": len(train_fids),
        "num_val": len(val_fids),
        "num_test": len(test_fids),
        "hardware": hardware,
        "geojson": geojson_metrics,
    }
    if parquet_metrics is not None:
        results["parquet"] = parquet_metrics

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark PointNet-2D on CoNSeP: Parquet vs GeoJSON loading",
    )
    parser.add_argument(
        "--parquet-path", type=str,
        default="data/consep/tiles.parquet",
        help="Path to tiled Parquet file",
    )
    parser.add_argument(
        "--geojson-dir", type=str,
        default="data/consep/geojson",
        help="Path to directory of GeoJSON files",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-points", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=str,
        default="results/consep_ml_benchmark.json",
        help="Path to write JSON results",
    )
    args = parser.parse_args()

    results = run_benchmark(args)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {output_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for loader in ("parquet", "geojson"):
        if loader not in results:
            continue
        r = results[loader]
        print(f"\n  {loader.upper()}:")
        print(f"    Test accuracy:       {r['test_accuracy']:.4f}")
        print(f"    Test F1 (macro):     {r['test_f1_macro']:.4f}")
        print(f"    Time to 1st batch:   {r['time_to_first_batch_ms']:.1f} ms")
        print(f"    Mean epoch time:     {r['mean_epoch_time_s']:.3f} s")
        print(f"    Mean data load time: {r['mean_data_load_time_s']:.3f} s")
        print(f"    Peak memory:         {r['peak_memory_mb']:.1f} MB")
        print(f"    Dataset load time:   {r['dataset_load_time_s']:.3f} s")


if __name__ == "__main__":
    main()
