"""Synthetic oracle for reproducible AL experiments.

Looks up ground-truth labels from a pre-built dict. Refuses queries whose feature_id
is in the held-out test set (to prevent trivial leakage)."""
from __future__ import annotations

from typing import Mapping, Optional, Set, Sequence


class SyntheticOracle:
    def __init__(
        self,
        ground_truth: Mapping[str, int],
        test_set: Optional[Set[str]] = None,
    ) -> None:
        self._truth = dict(ground_truth)
        self._test_set: Set[str] = set(test_set or ())
        self._queries_made: int = 0

    def query(self, feature_id: str) -> int:
        """Return label for an unlabeled feature_id. Raises ValueError if feature_id
        is in the held-out test_set — that would leak test labels into training."""
        if feature_id in self._test_set:
            raise ValueError(
                f"feature_id {feature_id!r} is in the held-out test_set; refusing query "
                "to prevent leakage"
            )
        self._queries_made += 1
        return self._truth[feature_id]

    def reveal_test_labels(self, test_ids: Sequence[str]) -> list[int]:
        """Return labels for evaluation. Raises if any id is not in test_set."""
        bad = [fid for fid in test_ids if fid not in self._test_set]
        if bad:
            raise ValueError(f"feature_ids not in test_set: {bad[:5]}{'...' if len(bad) > 5 else ''}")
        return [self._truth[fid] for fid in test_ids]

    @property
    def queries_made(self) -> int:
        return self._queries_made
