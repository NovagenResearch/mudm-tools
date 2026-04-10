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
import json
import sys
import time
from pathlib import Path

import numpy as np
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
    convert_parquet_to_arrow_ipc,
)

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

    if "curriculum_mean" in aggregated:
        cur_mean = np.array(aggregated["curriculum_mean"])
        cur_std = np.array(aggregated["curriculum_std"])
        base_mean = np.array(aggregated["baseline_mean"])
        base_std = np.array(aggregated["baseline_std"])

        ax.plot(epochs, cur_mean, "o-", markersize=3, label="Curriculum (zoom 0→3)")
        ax.fill_between(epochs, cur_mean - cur_std, cur_mean + cur_std, alpha=0.2)
        ax.plot(epochs, base_mean, "s-", markersize=3, label="Baseline (zoom 3 only)")
        ax.fill_between(epochs, base_mean - base_std, base_mean + base_std, alpha=0.2)
    else:
        # Single-seed fallback
        curriculum = aggregated["curriculum"]
        baseline = aggregated["baseline"]
        cur_epochs = [h["epoch"] for h in curriculum["history"]]
        cur_acc = [h["val_accuracy"] for h in curriculum["history"]]
        ax.plot(cur_epochs, cur_acc, "o-", markersize=3, label="Curriculum (zoom 0→3)")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_name = "Arrow IPC" if args.use_arrow else "Parquet"
    print(f"Device: {device}")
    print(f"Data loader: {loader_name}")
    print(f"Seeds: {seeds}")
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

    # --- Aggregate ---
    cur_accs = [r["test_accuracy"] for r in all_curriculum]
    cur_f1s = [r["test_f1_macro"] for r in all_curriculum]
    base_accs = [r["test_accuracy"] for r in all_baseline]
    base_f1s = [r["test_f1_macro"] for r in all_baseline]

    # Per-epoch val accuracy arrays (seeds x epochs)
    cur_val_matrix = np.array([
        [h["val_accuracy"] for h in r["history"]] for r in all_curriculum
    ])
    base_val_matrix = np.array([
        [h["val_accuracy"] for h in r["history"]] for r in all_baseline
    ])

    results = {
        "seeds": seeds,
        "loader": loader_name,
        "per_seed": {
            "curriculum": all_curriculum,
            "baseline": all_baseline,
        },
        "aggregate": {
            "curriculum_test_accuracy_mean": round(float(np.mean(cur_accs)), 5),
            "curriculum_test_accuracy_std": round(float(np.std(cur_accs)), 5),
            "curriculum_test_f1_mean": round(float(np.mean(cur_f1s)), 5),
            "curriculum_test_f1_std": round(float(np.std(cur_f1s)), 5),
            "baseline_test_accuracy_mean": round(float(np.mean(base_accs)), 5),
            "baseline_test_accuracy_std": round(float(np.std(base_accs)), 5),
            "baseline_test_f1_mean": round(float(np.mean(base_f1s)), 5),
            "baseline_test_f1_std": round(float(np.std(base_f1s)), 5),
        },
        # Per-epoch mean/std for plotting
        "curriculum_mean": np.mean(cur_val_matrix, axis=0).round(5).tolist(),
        "curriculum_std": np.std(cur_val_matrix, axis=0).round(5).tolist(),
        "baseline_mean": np.mean(base_val_matrix, axis=0).round(5).tolist(),
        "baseline_std": np.std(base_val_matrix, axis=0).round(5).tolist(),
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
    print(f"  Curriculum test accuracy:  {agg['curriculum_test_accuracy_mean']:.4f} ± {agg['curriculum_test_accuracy_std']:.4f}")
    print(f"  Curriculum test F1 (macro):{agg['curriculum_test_f1_mean']:.4f} ± {agg['curriculum_test_f1_std']:.4f}")
    print()
    print(f"  Baseline test accuracy:    {agg['baseline_test_accuracy_mean']:.4f} ± {agg['baseline_test_accuracy_std']:.4f}")
    print(f"  Baseline test F1 (macro):  {agg['baseline_test_f1_mean']:.4f} ± {agg['baseline_test_f1_std']:.4f}")


if __name__ == "__main__":
    main()
