"""Round-trip latency instrumentation for the AL harness.

The muDM arm reads feature tensors directly from Parquet each round.
The re-extract baseline re-runs a feature extractor from raw data each round
because the AL policy may change which features are in scope (e.g., zoom level).
Latency is measured from uncertainty-computed → oracle-returned."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterator


@dataclass
class RoundTripLatencyRecorder:
    _samples: list[float] = field(default_factory=list)
    _uncertainty_t: float = 0.0
    _oracle_t: float = 0.0

    def mark_uncertainty_computed(self) -> None:
        self._uncertainty_t = perf_counter()

    def mark_oracle_returned(self) -> None:
        self._oracle_t = perf_counter()

    @contextlib.contextmanager
    def round_scope(self) -> Iterator[None]:
        self._uncertainty_t = 0.0
        self._oracle_t = 0.0
        yield
        if self._uncertainty_t and self._oracle_t:
            self._samples.append(self._oracle_t - self._uncertainty_t)

    def samples(self) -> list[float]:
        return list(self._samples)

    def summary(self) -> dict:
        import statistics

        if not self._samples:
            return {"n": 0}
        return {
            "n": len(self._samples),
            "mean_s": statistics.mean(self._samples),
            "stdev_s": statistics.stdev(self._samples) if len(self._samples) > 1 else 0.0,
            "median_s": statistics.median(self._samples),
        }
