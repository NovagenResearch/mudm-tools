"""Smoke test that benchmark_hemibrain_ml.py runs in 5-seed mode and emits
mean +/- SD with a paired t-test."""
import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
HEMIBRAIN_DATA = ROOT / "data" / "hemibrain"


@pytest.mark.skipif(
    not HEMIBRAIN_DATA.exists(), reason="Hemibrain dataset not available locally"
)
def test_benchmark_emits_multiseed_stats(tmp_path):
    out = tmp_path / "result.json"
    subprocess.run(
        [
            "uv", "run", "python",
            str(ROOT / "scripts" / "benchmark_hemibrain_ml.py"),
            "--seeds", "42,123",
            "--quick",
            "--output", str(out),
        ],
        check=True,
        cwd=ROOT,
    )
    data = json.loads(out.read_text())

    assert "seeds" in data
    assert data["seeds"] == [42, 123]

    for loader_key in ("parquet", "arrow_ipc", "raw_obj"):
        assert loader_key in data, f"missing loader: {loader_key}"
        ld = data[loader_key]
        for stat_key in (
            "accuracy_mean", "accuracy_sd", "macro_f1_mean", "macro_f1_sd",
            "load_time_mean_s", "epoch_time_mean_s",
        ):
            assert stat_key in ld, f"missing {loader_key}.{stat_key}"
        assert "per_seed_accuracy" in ld
        assert len(ld["per_seed_accuracy"]) == 2  # one per seed

    assert "paired_t_test_parquet_vs_raw" in data
    tt = data["paired_t_test_parquet_vs_raw"]
    assert "t_statistic" in tt
    assert "p_value" in tt
    assert isinstance(tt["p_value"], (int, float))

    assert "benchmark_id" in data and data["benchmark_id"] == "B1_hemibrain_5seed"
    assert "timestamp" in data
    assert "hardware" in data
