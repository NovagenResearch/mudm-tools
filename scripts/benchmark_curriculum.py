#!/usr/bin/env python3
"""Benchmark: Zoom-level curriculum training with Parquet predicate pushdown.

Demonstrates that Parquet's partitioned layout enables zoom-level curriculum
training -- a training pattern impossible with raw file formats.  Switching
zoom level is ONE LINE: just read from a different ``zoom=N`` directory.
With raw OBJ files you would need to pre-generate simplified versions at
each resolution, which the muDM pipeline has already done and stored in
Parquet.

**Curriculum strategy** (50 epochs total):
  - Phase 1: 10 epochs on zoom=0 (coarsest, ~1.7% of original faces)
  - Phase 2: 10 epochs on zoom=1
  - Phase 3: 10 epochs on zoom=2
  - Phase 4: 20 epochs on zoom=3 (full resolution)

**Baseline**: 50 epochs on zoom=3 only (standard training).

Usage:
    uv run python scripts/benchmark_curriculum.py \\
        --parquet-dir data/hemibrain/tiles/hemibrain/parquet_partitioned \\
        --metadata data/hemibrain/metadata.json \\
        --output results/curriculum_benchmark.json \\
        --plot results/curriculum_training.pdf
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Import model, dataset, and helpers from the main ML benchmark script.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_hemibrain_ml import (  # noqa: E402
    ArrowIPCMeshDataset,
    ParquetMeshDataset,
    PointNet,
    _build_label_map_and_splits,
    _sample_and_normalise,
    convert_parquet_to_arrow_ipc,
)
from mudm_tools.spatial_batching import SpatiallyCoherentLoader  # noqa: E402

# ---------------------------------------------------------------------------
# Curriculum phases
# ---------------------------------------------------------------------------

CURRICULUM_PHASES = [
    {"zoom": 0, "epochs": 10},
    {"zoom": 1, "epochs": 10},
    {"zoom": 2, "epochs": 10},
    {"zoom": 3, "epochs": 20},
]

BASELINE_ZOOM = 3
TOTAL_EPOCHS = sum(p["epochs"] for p in CURRICULUM_PHASES)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _make_loader(
    parquet_dir: Path,
    zoom: int,
    n_points: int,
    label_map: dict[str, int],
    feature_ids: list[str],
    batch_size: int,
    shuffle: bool,
    *,
    use_arrow: bool = False,
    arrow_dir: Path | None = None,
) -> DataLoader:
    if use_arrow and arrow_dir is not None:
        arrow_path = arrow_dir / f"zoom_{zoom}.arrow"
        ds = ArrowIPCMeshDataset(
            arrow_path, n_points=n_points,
            label_map=label_map, feature_ids=feature_ids,
        )
    else:
        ds = ParquetMeshDataset(
            parquet_dir, zoom=zoom, n_points=n_points,
            label_map=label_map, feature_ids=feature_ids,
        )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Return (loss, accuracy) on *loader*."""
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for points, labels in loader:
            points = points.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            logits = model(points)
            loss_sum += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def _test_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Return (accuracy, macro-F1) on *loader*."""
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for points, labels in loader:
            points = points.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            logits = model(points)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / max(len(all_labels), 1)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, f1


# ---------------------------------------------------------------------------
# Curriculum training
# ---------------------------------------------------------------------------


def train_curriculum(
    parquet_dir: Path,
    label_map: dict[str, int],
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    n_points: int,
    batch_size: int,
    device: torch.device,
    *,
    use_arrow: bool = False,
    arrow_dir: Path | None = None,
) -> dict:
    """Run curriculum training: coarse-to-fine zoom progression."""
    num_classes = len(label_map)
    model = PointNet(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Validation loader -- always at zoom=3 (full resolution) for fair
    # comparison against the baseline.
    val_loader = _make_loader(
        parquet_dir, zoom=BASELINE_ZOOM, n_points=n_points,
        label_map=label_map, feature_ids=val_ids,
        batch_size=batch_size, shuffle=False,
        use_arrow=use_arrow, arrow_dir=arrow_dir,
    )

    history: list[dict] = []
    global_epoch = 0
    t_start = time.perf_counter()

    for phase in CURRICULUM_PHASES:
        zoom = phase["zoom"]
        phase_epochs = phase["epochs"]

        print(f"\n--- Curriculum phase: zoom={zoom}, {phase_epochs} epochs ---")
        train_loader = _make_loader(
            parquet_dir, zoom=zoom, n_points=n_points,
            label_map=label_map, feature_ids=train_ids,
            batch_size=batch_size, shuffle=True,
            use_arrow=use_arrow, arrow_dir=arrow_dir,
        )
        print(f"  Train samples at zoom={zoom}: {len(train_loader.dataset)}")

        for ep in range(1, phase_epochs + 1):
            global_epoch += 1
            model.train()
            train_loss_sum = 0.0
            train_count = 0

            for points, labels in train_loader:
                points = points.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                optimizer.zero_grad()
                logits = model(points)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * labels.size(0)
                train_count += labels.size(0)

            train_loss = train_loss_sum / max(train_count, 1)
            val_loss, val_acc = _evaluate(model, val_loader, criterion, device)

            history.append({
                "epoch": global_epoch,
                "zoom": zoom,
                "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5),
                "val_accuracy": round(val_acc, 5),
            })

            print(
                f"  Epoch {global_epoch:3d}/{TOTAL_EPOCHS} (z={zoom})  "
                f"train_loss={train_loss:.4f}  "
                f"val_acc={val_acc:.4f}",
                flush=True,
            )

    total_time = time.perf_counter() - t_start

    # Test evaluation
    test_loader = _make_loader(
        parquet_dir, zoom=BASELINE_ZOOM, n_points=n_points,
        label_map=label_map, feature_ids=test_ids,
        batch_size=batch_size, shuffle=False,
        use_arrow=use_arrow, arrow_dir=arrow_dir,
    )
    test_acc, test_f1 = _test_metrics(model, test_loader, device)

    return {
        "phases": CURRICULUM_PHASES,
        "test_accuracy": round(test_acc, 5),
        "test_f1_macro": round(test_f1, 5),
        "total_time_s": round(total_time, 2),
        "history": history,
    }


# ---------------------------------------------------------------------------
# Baseline training
# ---------------------------------------------------------------------------


def _write_train_only_parquet_view(
    *,
    source_parquet_dir: Path,
    train_id_set: set[str],
    zoom: int,
) -> Path:
    """Materialise a temporary hive-partitioned Parquet directory containing
    only rows whose ``tags["body_id"]`` is in ``train_id_set``.

    The output preserves the source's
    ``zoom={N}/tile_x={X}/tile_y={Y}/tile_d={D}/part-0.parquet`` layout so a
    ``SpatiallyCoherentLoader`` pointed at the result discovers the same tile
    keys (minus tiles that lose all their rows).  The non-partition columns
    are kept verbatim, so the loader's row dicts have the same shape as when
    it reads the source directly.

    The caller owns cleanup (e.g. wrap in ``contextlib.ExitStack`` or
    ``tempfile.TemporaryDirectory``).

    Why this exists: the spatial-batch arm of ``benchmark_curriculum.py`` must
    feed the loader train-only data so its mini-batches stay full-size of
    train rows -- matching the baseline arm's pre-filtered-dataset semantics.
    Filtering inside ``_rows_to_tensor_batch`` shrinks mixed-split tiles'
    effective batch size and skews the comparison.
    """
    out_dir = Path(tempfile.mkdtemp(prefix="mudm_spatial_train_"))
    zoom_dir_in = source_parquet_dir / f"zoom={zoom}"
    if not zoom_dir_in.is_dir():
        return out_dir

    zoom_dir_out = out_dir / f"zoom={zoom}"
    for tx_dir in sorted(zoom_dir_in.glob("tile_x=*")):
        tx_name = tx_dir.name
        for ty_dir in sorted(tx_dir.glob("tile_y=*")):
            ty_name = ty_dir.name
            for td_dir in sorted(ty_dir.glob("tile_d=*")):
                td_name = td_dir.name
                table = pq.read_table(td_dir)
                if table.num_rows == 0:
                    continue
                # ``tags`` may surface as either a struct or a map<string,string>
                # depending on how the source Parquet was written; handle both.
                tags_col = table.column("tags").to_pylist()
                mask: list[bool] = []
                for tags_val in tags_col:
                    if tags_val is None:
                        mask.append(False)
                        continue
                    tags_dict = (
                        dict(tags_val) if isinstance(tags_val, list) else tags_val
                    )
                    bid = tags_dict.get("body_id")
                    mask.append(bid is not None and bid in train_id_set)
                filtered = table.filter(pa.array(mask, type=pa.bool_()))
                if filtered.num_rows == 0:
                    continue
                out_td = zoom_dir_out / tx_name / ty_name / td_name
                out_td.mkdir(parents=True, exist_ok=True)
                pq.write_table(filtered, out_td / "part-0.parquet")
    return out_dir


def _rows_to_tensor_batch(
    rows: list[dict],
    n_points: int,
    label_map: dict[str, int],
    feature_id_set: set[str],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Aggregate a list of tile rows into a (points, labels) tensor batch.

    Each row carries a packed ``positions`` blob and a ``tags`` dict containing
    ``body_id`` and ``cell_type``.  Rows belonging to the same neuron (same
    ``body_id``) are concatenated, then sampled/normalised to ``n_points`` and
    stacked into a (B, n_points, 3) tensor.  Rows whose body_id is not in
    ``feature_id_set`` or whose cell_type is not in ``label_map`` are skipped.
    Returns ``None`` if no usable neurons remain.
    """
    by_body: dict[str, list[np.ndarray]] = {}
    label_by_body: dict[str, int] = {}
    for row in rows:
        tags_val = row.get("tags")
        pos_bytes = row.get("positions")
        if tags_val is None or pos_bytes is None:
            continue
        tags_dict = dict(tags_val) if isinstance(tags_val, list) else tags_val
        body_id = tags_dict.get("body_id")
        if body_id is None or body_id not in feature_id_set:
            continue
        cell_type = tags_dict.get("cell_type")
        if cell_type is None or cell_type not in label_map:
            continue
        positions = np.frombuffer(pos_bytes, dtype=np.float32).reshape(-1, 3)
        if len(positions) == 0:
            continue
        by_body.setdefault(body_id, []).append(positions)
        label_by_body[body_id] = label_map[cell_type]

    if not by_body:
        return None

    pts_list: list[torch.Tensor] = []
    label_list: list[int] = []
    for body_id, frags in by_body.items():
        merged = np.concatenate(frags, axis=0)
        pts_list.append(_sample_and_normalise(merged, n_points))
        label_list.append(label_by_body[body_id])
    return torch.stack(pts_list, dim=0), torch.tensor(label_list, dtype=torch.long)


def train_spatial_batch(
    parquet_dir: Path,
    label_map: dict[str, int],
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    n_points: int,
    batch_size: int,
    device: torch.device,
    *,
    seed: int,
    use_arrow: bool = False,
    arrow_dir: Path | None = None,
) -> dict:
    """Run spatially-coherent batching: each mini-batch comes from a single
    octree tile (with carry-merging when a tile is undersized).

    Mirrors the baseline arm exactly except for the sampler: same model, same
    optimizer, same epoch count.  Validation and test loaders are unchanged
    (random-shuffled at full resolution) so test-set metrics are comparable.
    """
    num_classes = len(label_map)
    model = PointNet(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_id_set = set(train_ids)

    # Materialise a train-only view of the partitioned Parquet directory so
    # the spatially-coherent loader yields full-size batches of train rows --
    # matching the baseline arm, whose ``ParquetMeshDataset`` is itself
    # constructed with ``feature_ids=train_ids`` and therefore filters
    # *before* batching (see ``ParquetMeshDataset.__init__`` in
    # ``benchmark_hemibrain_ml.py``).  Without this, mixed-split tiles would
    # produce shrunken batches and skew the comparison.
    val_loader = _make_loader(
        parquet_dir, zoom=BASELINE_ZOOM, n_points=n_points,
        label_map=label_map, feature_ids=val_ids,
        batch_size=batch_size, shuffle=False,
        use_arrow=use_arrow, arrow_dir=arrow_dir,
    )

    history: list[dict] = []
    t_start = time.perf_counter()

    with contextlib.ExitStack() as stack:
        train_only_dir = _write_train_only_parquet_view(
            source_parquet_dir=parquet_dir,
            train_id_set=train_id_set,
            zoom=BASELINE_ZOOM,
        )
        stack.callback(lambda: shutil.rmtree(train_only_dir, ignore_errors=True))

        train_loader = SpatiallyCoherentLoader(
            parquet_dir=train_only_dir,
            zoom=BASELINE_ZOOM,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
        )

        print(f"\n--- Spatial batch: zoom={BASELINE_ZOOM}, {TOTAL_EPOCHS} epochs ---")
        print(f"  Tiles discovered (train-only view): {len(train_loader.tile_keys)}")

        for epoch in range(1, TOTAL_EPOCHS + 1):
            model.train()
            train_loss_sum = 0.0
            train_count = 0

            for tile_rows in train_loader:
                # ``train_id_set`` is still passed defensively, but the loader
                # is now reading from a train-only view so every row already
                # has ``body_id in train_id_set`` -- this filter is a no-op for
                # the body_id check and only enforces ``cell_type in label_map``.
                batch = _rows_to_tensor_batch(tile_rows, n_points, label_map, train_id_set)
                if batch is None:
                    continue
                points, labels = batch
                points = points.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                optimizer.zero_grad()
                logits = model(points)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * labels.size(0)
                train_count += labels.size(0)

            train_loss = train_loss_sum / max(train_count, 1)
            val_loss, val_acc = _evaluate(model, val_loader, criterion, device)

            history.append({
                "epoch": epoch,
                "val_accuracy": round(val_acc, 5),
                "val_loss": round(val_loss, 5),
                "train_loss": round(train_loss, 5),
            })

            print(
                f"  Epoch {epoch:3d}/{TOTAL_EPOCHS}  "
                f"train_loss={train_loss:.4f}  "
                f"val_acc={val_acc:.4f}",
                flush=True,
            )

    total_time = time.perf_counter() - t_start

    test_loader = _make_loader(
        parquet_dir, zoom=BASELINE_ZOOM, n_points=n_points,
        label_map=label_map, feature_ids=test_ids,
        batch_size=batch_size, shuffle=False,
        use_arrow=use_arrow, arrow_dir=arrow_dir,
    )
    test_acc, test_f1 = _test_metrics(model, test_loader, device)

    return {
        "zoom": BASELINE_ZOOM,
        "epochs": TOTAL_EPOCHS,
        "test_accuracy": round(test_acc, 5),
        "test_f1_macro": round(test_f1, 5),
        "total_time_s": round(total_time, 2),
        "history": history,
    }


def train_baseline(
    parquet_dir: Path,
    label_map: dict[str, int],
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    n_points: int,
    batch_size: int,
    device: torch.device,
    *,
    use_arrow: bool = False,
    arrow_dir: Path | None = None,
) -> dict:
    """Run baseline training: all epochs on zoom=3."""
    num_classes = len(label_map)
    model = PointNet(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_loader = _make_loader(
        parquet_dir, zoom=BASELINE_ZOOM, n_points=n_points,
        label_map=label_map, feature_ids=train_ids,
        batch_size=batch_size, shuffle=True,
        use_arrow=use_arrow, arrow_dir=arrow_dir,
    )
    val_loader = _make_loader(
        parquet_dir, zoom=BASELINE_ZOOM, n_points=n_points,
        label_map=label_map, feature_ids=val_ids,
        batch_size=batch_size, shuffle=False,
        use_arrow=use_arrow, arrow_dir=arrow_dir,
    )

    print(f"\n--- Baseline: zoom={BASELINE_ZOOM}, {TOTAL_EPOCHS} epochs ---")
    print(f"  Train samples: {len(train_loader.dataset)}")

    history: list[dict] = []
    t_start = time.perf_counter()

    for epoch in range(1, TOTAL_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for points, labels in train_loader:
            points = points.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            logits = model(points)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * labels.size(0)
            train_count += labels.size(0)

        train_loss = train_loss_sum / max(train_count, 1)
        val_loss, val_acc = _evaluate(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch,
            "val_accuracy": round(val_acc, 5),
            "val_loss": round(val_loss, 5),
            "train_loss": round(train_loss, 5),
        })

        print(
            f"  Epoch {epoch:3d}/{TOTAL_EPOCHS}  "
            f"train_loss={train_loss:.4f}  "
            f"val_acc={val_acc:.4f}",
            flush=True,
        )

    total_time = time.perf_counter() - t_start

    # Test evaluation
    test_loader = _make_loader(
        parquet_dir, zoom=BASELINE_ZOOM, n_points=n_points,
        label_map=label_map, feature_ids=test_ids,
        batch_size=batch_size, shuffle=False,
        use_arrow=use_arrow, arrow_dir=arrow_dir,
    )
    test_acc, test_f1 = _test_metrics(model, test_loader, device)

    return {
        "zoom": BASELINE_ZOOM,
        "epochs": TOTAL_EPOCHS,
        "test_accuracy": round(test_acc, 5),
        "test_f1_macro": round(test_f1, 5),
        "total_time_s": round(total_time, 2),
        "history": history,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(aggregated: dict, plot_path: Path) -> None:
    """Save accuracy-vs-epoch comparison plot as PDF.

    *aggregated* has keys ``curriculum_mean``, ``curriculum_std``,
    ``baseline_mean``, ``baseline_std`` — each a list of per-epoch values.
    Falls back to single-seed format (``curriculum``/``baseline`` dicts with
    ``history``) when those keys are absent.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))

    epochs = list(range(1, TOTAL_EPOCHS + 1))

    def _plot_arm(prefix: str, marker: str, label: str) -> None:
        mean = aggregated.get(f"{prefix}_mean")
        std = aggregated.get(f"{prefix}_std")
        if not mean:
            return
        mean_arr = np.array(mean)
        std_arr = np.array(std) if std else np.zeros_like(mean_arr)
        ax.plot(epochs, mean_arr, f"{marker}-", markersize=3, label=label)
        ax.fill_between(epochs, mean_arr - std_arr, mean_arr + std_arr, alpha=0.2)

    if (
        "curriculum_mean" in aggregated
        or "baseline_mean" in aggregated
        or "spatial_batch_mean" in aggregated
    ):
        _plot_arm("curriculum", "o", "Curriculum (zoom 0→3)")
        _plot_arm("baseline", "s", "Baseline (zoom 3 only)")
        _plot_arm("spatial_batch", "^", "Spatial batch (single-tile)")
    else:
        # Single-seed fallback
        curriculum = aggregated.get("curriculum")
        baseline = aggregated.get("baseline")
        if curriculum is not None:
            cur_epochs = [h["epoch"] for h in curriculum["history"]]
            cur_acc = [h["val_accuracy"] for h in curriculum["history"]]
            ax.plot(cur_epochs, cur_acc, "o-", markersize=3, label="Curriculum (zoom 0→3)")
        if baseline is not None:
            base_epochs = [h["epoch"] for h in baseline["history"]]
            base_acc = [h["val_accuracy"] for h in baseline["history"]]
            ax.plot(base_epochs, base_acc, "s-", markersize=3, label="Baseline (zoom 3 only)")

    # Shade curriculum phases
    colors = ["#e0f0ff", "#c0e0ff", "#a0d0ff", "#80c0ff"]
    epoch_offset = 0
    for i, phase in enumerate(CURRICULUM_PHASES):
        ax.axvspan(
            epoch_offset + 0.5,
            epoch_offset + phase["epochs"] + 0.5,
            alpha=0.25,
            color=colors[i],
            label=f"zoom={phase['zoom']}" if i < 4 else None,
        )
        epoch_offset += phase["epochs"]

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Curriculum vs. Baseline Training (Hemibrain PointNet)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0.5, TOTAL_EPOCHS + 0.5)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(plot_path), dpi=150)
    plt.close(fig)
    print(f"Plot saved to {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


DEFAULT_SEEDS = [42, 123, 456, 789, 1024]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curriculum training demo: zoom-level progression via Parquet",
    )
    parser.add_argument(
        "--parquet-dir",
        type=str,
        default="data/hemibrain/tiles/hemibrain/parquet_partitioned",
        help="Path to partitioned Parquet directory (with zoom=N subdirs)",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/hemibrain/metadata.json",
        help="Path to metadata.json with neuron info",
    )
    parser.add_argument(
        "--obj-dir",
        type=str,
        default="data/hemibrain/meshes",
        help="Path to OBJ mesh directory (for label map / split building)",
    )
    parser.add_argument("--n-points", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42, help="Single seed (ignored if --seeds given)")
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds, e.g. 42,123,456,789,1024",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=None,
        help="Number of seeds to use (auto-generates from default list)",
    )
    parser.add_argument(
        "--use-arrow",
        action="store_true",
        help="Use Arrow IPC (Feather v2) instead of Parquet for data loading",
    )
    parser.add_argument(
        "--arms",
        type=str,
        default="curriculum,baseline",
        help=(
            "Comma-separated list of arms to run. "
            "Available arms: 'curriculum' (zoom 0->3 progression), "
            "'baseline' (zoom=3 only with random sampler), "
            "'spatial_batch' (zoom=3, mini-batches drawn from a single octree "
            "tile via SpatiallyCoherentLoader -- otherwise identical to "
            "baseline). Default: curriculum,baseline."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/curriculum_benchmark.json",
        help="Path to write JSON results",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="results/curriculum_training.pdf",
        help="Path to save training curve plot (PDF)",
    )
    args = parser.parse_args()

    parquet_dir = Path(args.parquet_dir)
    metadata_path = Path(args.metadata)
    obj_dir = Path(args.obj_dir)

    # --- Resolve seed list ---
    if args.seeds is not None:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    elif args.num_seeds is not None:
        seeds = DEFAULT_SEEDS[: args.num_seeds]
    else:
        seeds = [args.seed]

    # --- Resolve arm list ---
    valid_arms = {"curriculum", "baseline", "spatial_batch"}
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = set(arms) - valid_arms
    if unknown:
        parser.error(
            f"Unknown arm(s): {sorted(unknown)}. "
            f"Valid arms: {sorted(valid_arms)}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_name = "Arrow IPC" if args.use_arrow else "Parquet"
    print(f"Device: {device}")
    print(f"Data loader: {loader_name}")
    print(f"Seeds: {seeds}")
    print(f"Arms: {arms}")
    print(f"Curriculum phases: {CURRICULUM_PHASES}")
    print(f"Baseline: {TOTAL_EPOCHS} epochs on zoom={BASELINE_ZOOM}")

    # --- Arrow IPC conversion (all zoom levels) ---
    arrow_dir: Path | None = None
    if args.use_arrow:
        arrow_dir = parquet_dir.parent / "arrow_ipc_curriculum"
        arrow_dir.mkdir(parents=True, exist_ok=True)
        for zoom in range(4):  # zoom 0-3
            arrow_path = arrow_dir / f"zoom_{zoom}.arrow"
            if arrow_path.exists():
                print(f"  Arrow IPC zoom={zoom} already exists: {arrow_path}")
            else:
                t0 = time.perf_counter()
                convert_parquet_to_arrow_ipc(parquet_dir, zoom=zoom, output_path=arrow_path)
                print(f"  Converted zoom={zoom} → Arrow IPC in {time.perf_counter() - t0:.1f}s")

    # --- Per-seed results ---
    all_curriculum: list[dict] = []
    all_baseline: list[dict] = []
    all_spatial_batch: list[dict] = []

    for si, seed in enumerate(seeds):
        print(f"\n{'#'*60}")
        print(f"SEED {seed} ({si+1}/{len(seeds)})")
        print(f"{'#'*60}")

        # Build label map and stratified splits (same as main benchmark)
        label_map, all_ids, train_ids, val_ids, test_ids = _build_label_map_and_splits(
            metadata_path, obj_dir, min_instances=10, seed=seed,
        )
        if si == 0:
            print(f"Classes: {len(label_map)}")
            print(
                f"Samples: {len(all_ids)} total "
                f"({len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test)"
            )

        # --- Curriculum ---
        if "curriculum" in arms:
            _set_seeds(seed)
            print(f"\n{'='*60}")
            print("CURRICULUM TRAINING")
            print(f"{'='*60}")
            cur = train_curriculum(
                parquet_dir, label_map, train_ids, val_ids, test_ids,
                n_points=args.n_points, batch_size=args.batch_size, device=device,
                use_arrow=args.use_arrow, arrow_dir=arrow_dir,
            )
            cur["seed"] = seed
            all_curriculum.append(cur)

        # --- Baseline ---
        if "baseline" in arms:
            _set_seeds(seed)
            print(f"\n{'='*60}")
            print("BASELINE TRAINING")
            print(f"{'='*60}")
            base = train_baseline(
                parquet_dir, label_map, train_ids, val_ids, test_ids,
                n_points=args.n_points, batch_size=args.batch_size, device=device,
                use_arrow=args.use_arrow, arrow_dir=arrow_dir,
            )
            base["seed"] = seed
            all_baseline.append(base)

        # --- Spatial batch ---
        if "spatial_batch" in arms:
            # Mirrors the baseline arm but uses SpatiallyCoherentLoader instead
            # of PyTorch's default random sampler. All other settings (epochs,
            # model, optimizer, evaluation) are identical to baseline.
            _set_seeds(seed)
            print(f"\n{'='*60}")
            print("SPATIAL BATCH TRAINING")
            print(f"{'='*60}")
            spat = train_spatial_batch(
                parquet_dir, label_map, train_ids, val_ids, test_ids,
                n_points=args.n_points, batch_size=args.batch_size, device=device,
                seed=seed,
                use_arrow=args.use_arrow, arrow_dir=arrow_dir,
            )
            spat["seed"] = seed
            all_spatial_batch.append(spat)

    # --- Aggregate ---
    def _agg_block(records: list[dict], prefix: str) -> dict:
        if not records:
            return {}
        accs = [r["test_accuracy"] for r in records]
        f1s = [r["test_f1_macro"] for r in records]
        return {
            f"{prefix}_test_accuracy_mean": round(float(np.mean(accs)), 5),
            f"{prefix}_test_accuracy_std": round(float(np.std(accs)), 5),
            f"{prefix}_test_f1_mean": round(float(np.mean(f1s)), 5),
            f"{prefix}_test_f1_std": round(float(np.std(f1s)), 5),
        }

    def _per_epoch_arrays(records: list[dict]) -> tuple[list[float], list[float]]:
        if not records:
            return [], []
        matrix = np.array([
            [h["val_accuracy"] for h in r["history"]] for r in records
        ])
        return (
            np.mean(matrix, axis=0).round(5).tolist(),
            np.std(matrix, axis=0).round(5).tolist(),
        )

    aggregate: dict = {}
    aggregate.update(_agg_block(all_curriculum, "curriculum"))
    aggregate.update(_agg_block(all_baseline, "baseline"))
    aggregate.update(_agg_block(all_spatial_batch, "spatial_batch"))

    cur_mean, cur_std = _per_epoch_arrays(all_curriculum)
    base_mean, base_std = _per_epoch_arrays(all_baseline)
    spat_mean, spat_std = _per_epoch_arrays(all_spatial_batch)

    per_seed: dict = {}
    if all_curriculum:
        per_seed["curriculum"] = all_curriculum
    if all_baseline:
        per_seed["baseline"] = all_baseline
    if all_spatial_batch:
        per_seed["spatial_batch"] = all_spatial_batch

    results = {
        "seeds": seeds,
        "arms": arms,
        "loader": loader_name,
        "per_seed": per_seed,
        "aggregate": aggregate,
        # Per-epoch mean/std for plotting
        "curriculum_mean": cur_mean,
        "curriculum_std": cur_std,
        "baseline_mean": base_mean,
        "baseline_std": base_std,
        "spatial_batch_mean": spat_mean,
        "spatial_batch_std": spat_std,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {output_path}")

    # --- Plot ---
    plot_path = Path(args.plot)
    plot_results(results, plot_path)

    # --- Summary ---
    agg = results["aggregate"]
    print(f"\n{'='*60}")
    print(f"SUMMARY ({len(seeds)} seed{'s' if len(seeds) > 1 else ''})")
    print(f"{'='*60}")
    arm_labels = {
        "curriculum": "Curriculum",
        "baseline": "Baseline  ",
        "spatial_batch": "SpatialBatch",
    }
    for arm in arms:
        prefix = arm
        label = arm_labels.get(arm, arm)
        if f"{prefix}_test_accuracy_mean" in agg:
            print(
                f"  {label} test accuracy:  "
                f"{agg[prefix + '_test_accuracy_mean']:.4f} ± "
                f"{agg[prefix + '_test_accuracy_std']:.4f}"
            )
            print(
                f"  {label} test F1 (macro):"
                f"{agg[prefix + '_test_f1_mean']:.4f} ± "
                f"{agg[prefix + '_test_f1_std']:.4f}"
            )
            print()


if __name__ == "__main__":
    main()
