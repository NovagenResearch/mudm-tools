"""Emit-order resilience for ``download_flywire.tile_meshopt``.

Secondary bug from ``.claude/specs/2026-06-09-neuroglancer-oom-139k-bug.md``: the
driver emitted Neuroglancer (memory-heavy, can OOM at high feature counts)
*before* ``build_index`` wrote the viewer-critical index (``features.json`` /
``tilejson3d.json`` / ``pyramids.json``), with no ``try/except``. So an NG crash
left a pile of 3D-Tiles GLBs with **no index** — unloadable by the viewer.

Contract under test: an NG failure must NOT propagate, and the durable index must
be written regardless (i.e. the index is produced before — and independent of —
the NG step).
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent

try:
    import mudm_tools._rs  # noqa: F401

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


def _load_download_flywire():
    """Import scripts/download_flywire.py as a module (not on sys.path)."""
    sys.path.insert(0, str(_ROOT / "src"))
    sys.path.insert(0, str(_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "download_flywire", str(_ROOT / "scripts" / "download_flywire.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_tri_obj(path: Path, base: float) -> None:
    """Write a tiny valid single-triangle OBJ at an offset."""
    b = base
    path.write_text(f"v {b} {b} {b}\nv {b + 1} {b} {b}\nv {b} {b + 1} {b}\nf 1 2 3\n")


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not compiled")
def test_index_written_even_when_neuroglancer_raises(tmp_path, monkeypatch):
    df = _load_download_flywire()
    import mudm_tools._rs as rs

    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir()
    # Two tiny valid neurons named by body id (so _build_tags works with {}).
    _write_tri_obj(mesh_dir / "1001.obj", 1.0)
    _write_tri_obj(mesh_dir / "1002.obj", 5.0)

    output_dir = tmp_path / "flywire-test"
    output_dir.mkdir()

    # Simulate an NG OOM/crash: the Neuroglancer step raises. (PyO3 type allows
    # class-level setattr; monkeypatch restores it afterward.)
    def _boom(self, *args, **kwargs):
        raise RuntimeError("simulated NG OOM")

    monkeypatch.setattr(rs.StreamingTileGenerator, "generate_neuroglancer_multilod", _boom)

    # The run must NOT propagate the NG failure.
    df.tile_meshopt(
        mesh_dir,
        {},
        output_dir,
        max_zoom=1,
        emit_neuroglancer=True,
        emit_parquet=False,
    )

    # ...and the durable, viewer-critical index must exist regardless.
    assert (output_dir / "features.json").exists(), "features.json missing — pyramid un-indexed"
    assert (output_dir / "tilejson3d.json").exists(), "tilejson3d.json missing"
    assert (output_dir.parent / "pyramids.json").exists(), "pyramids.json manifest missing"
    # The 3D-Tiles geometry produced before NG is intact.
    assert (output_dir / "3dtiles").is_dir()


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not compiled")
def test_happy_path_writes_index_then_neuroglancer(tmp_path):
    """Reorder must not break the success path: index AND Neuroglancer present."""
    df = _load_download_flywire()

    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir()
    _write_tri_obj(mesh_dir / "1001.obj", 1.0)
    _write_tri_obj(mesh_dir / "1002.obj", 5.0)

    output_dir = tmp_path / "flywire-test"
    output_dir.mkdir()

    df.tile_meshopt(
        mesh_dir,
        {},
        output_dir,
        max_zoom=1,
        emit_neuroglancer=True,
        emit_parquet=False,
    )

    assert (output_dir / "features.json").exists()
    assert (output_dir / "tilejson3d.json").exists()
    assert (output_dir.parent / "pyramids.json").exists()
    assert (output_dir / "3dtiles").is_dir()
    # NG ran after the index and produced its info manifest.
    assert (output_dir / "neuroglancer" / "info").exists()
