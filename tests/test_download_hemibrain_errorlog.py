"""WS-D Task D.4: download_hemibrain.tile_streaming surfaces the error-log summary.

The Python driver must:
  1. call ``gen._set_run_dir(str(pyramid_dir))`` on each generator before its
     ``generate_*`` call, so the Rust ErrorCollector streams ``errors.jsonl`` +
     ``run_summary.json`` under ``pyramid_dir``;
  2. after each phase, read ``<pyramid_dir>/run_summary.json`` (if present) and
     ``logging.info`` the ok/fail counts + the errors.jsonl path.

This drives ``tile_streaming`` on a tiny clean fixture and asserts a
``run_summary.json`` with honest counts lands under ``pyramid_dir`` and that the
summary is surfaced via ``logging.info``.
"""

import importlib.util
import json
import logging
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent

try:
    import mudm_tools._rs  # noqa: F401

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


def _load_download_hemibrain():
    """Import scripts/download_hemibrain.py as a module (not on sys.path)."""
    sys.path.insert(0, str(_ROOT / "src"))
    sys.path.insert(0, str(_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "download_hemibrain", str(_ROOT / "scripts" / "download_hemibrain.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_tri_obj(path: Path, base: float) -> None:
    """Write a tiny valid single-triangle OBJ at an offset."""
    b = base
    path.write_text(f"v {b} {b} {b}\n" f"v {b + 1} {b} {b}\n" f"v {b} {b + 1} {b}\n" f"f 1 2 3\n")


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not compiled")
def test_tile_streaming_writes_and_surfaces_run_summary(tmp_path, caplog):
    dh = _load_download_hemibrain()

    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir()
    # Two tiny valid neurons named by body id (so _build_tags works).
    _write_tri_obj(mesh_dir / "1001.obj", 1.0)
    _write_tri_obj(mesh_dir / "1002.obj", 5.0)

    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "neurons": [
                    {"bodyId": 1001, "type": "X", "instance": "x_1001"},
                    {"bodyId": 1002, "type": "Y", "instance": "y_1002"},
                ]
            }
        )
    )

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    with caplog.at_level(logging.INFO):
        results = dh.tile_streaming(
            mesh_dir,
            metadata_path,
            output_dir,
            max_zoom=1,
            skip_3dtiles=False,
            skip_pbf3=True,  # avoid pbf3 / feature_pbf3 phases
            pyramid_name="hemibrain",
        )

    pyramid_dir = output_dir / "hemibrain"

    # A run_summary.json must have been written under pyramid_dir by at least one
    # phase whose Rust side wires the ErrorCollector (ingest / 3dtiles).
    summary_path = pyramid_dir / "run_summary.json"
    assert summary_path.exists(), "run_summary.json must be written under pyramid_dir"

    summary = json.loads(summary_path.read_text())
    # Honest counts: clean fixture → no failures.
    assert summary["fail"] == 0
    assert summary["fatal"] is False
    assert summary["ok"] >= 1
    assert "phase" in summary and "elapsed_s" in summary

    # Clean run → errors.jsonl (if created) is empty.
    errlog = pyramid_dir / "errors.jsonl"
    if errlog.exists():
        assert [ln for ln in errlog.read_text().splitlines() if ln.strip()] == []

    # The summary must be surfaced via logging.info (ok/fail + log path).
    msgs = "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.INFO)
    assert "ok=" in msgs and "fail=" in msgs, f"summary not logged: {msgs!r}"

    # Geometry still produced (driver did not break the pipeline).
    assert results["parquet_rows"] >= 1
