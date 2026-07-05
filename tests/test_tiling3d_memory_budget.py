"""TileGenerator3D derives its worker count from a memory budget (cap MEMORY, not workers).

Guards the memory governor added after an 80-worker fork Pool exhausted RAM on a large volume: with
``max_memory_bytes`` set, parallelism self-sizes under the budget; unset keeps legacy cpu_count.
"""

import os

import mudm_tools.tiling3d.generator3d as G
from mudm_tools.tiling3d.generator3d import TileGenerator3D
from mudm_tools.tiling3d.octree import OctreeConfig


def _gen(**kw):
    return TileGenerator3D(OctreeConfig(max_zoom=1), output_format="3dtiles", **kw)


def test_no_budget_is_legacy_cpu_count():
    assert _gen()._effective_workers() == (os.cpu_count() or 1)


def test_explicit_workers_without_budget_honored():
    assert _gen(workers=3)._effective_workers() == 3


def test_tiny_budget_forces_serial():
    # A budget below the process's own RSS leaves no room to fan out.
    assert _gen(workers=16, max_memory_bytes=1)._effective_workers() == 1


def test_huge_budget_allows_requested():
    assert _gen(workers=8, max_memory_bytes=10**15)._effective_workers() == 8


def test_budget_caps_well_below_requested():
    baseline = G._process_rss_bytes()
    assert baseline > 0
    per_worker = max(G._MIN_PER_WORKER_BYTES, int(baseline * G._WORKER_COW_FRACTION))
    budget = baseline + 3 * per_worker  # room for ~3 workers, far below the 64 requested
    n = _gen(workers=64, max_memory_bytes=budget)._effective_workers()
    assert 1 <= n <= 5  # the cap engaged (nowhere near 64)
