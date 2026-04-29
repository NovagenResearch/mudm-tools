"""Three uncertainty scorers: margin and entropy. Higher = more uncertain."""
import numpy as np
import pytest

from mudm_tools.active_learning.uncertainty import margin, entropy


def test_margin_peaks_at_half_half():
    probs = np.array([[0.5, 0.5], [0.9, 0.1]])
    u = margin(probs)
    assert u[0] > u[1]
    assert u[0] == pytest.approx(1.0)  # 1 - |0.5 - 0.5| = 1
    assert u[1] == pytest.approx(0.2)  # 1 - |0.9 - 0.1| = 0.2


def test_entropy_peaks_at_uniform():
    probs = np.array([[0.5, 0.5], [0.99, 0.01]])
    e = entropy(probs)
    assert e[0] > e[1]


def test_entropy_handles_zero_probabilities():
    probs = np.array([[1.0, 0.0]])
    e = entropy(probs)
    assert e[0] == pytest.approx(0.0, abs=1e-10)


def test_margin_handles_three_class():
    probs = np.array([[0.4, 0.4, 0.2], [0.7, 0.2, 0.1]])
    u = margin(probs)
    # First row: 1 - (0.4 - 0.4) = 1.0; second: 1 - (0.7 - 0.2) = 0.5
    assert u[0] == pytest.approx(1.0)
    assert u[1] == pytest.approx(0.5)
