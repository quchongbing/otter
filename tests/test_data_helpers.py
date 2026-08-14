from __future__ import annotations

import numpy as np
import pytest

from otter.data.helpers import trapz_integral


def test_trapz_integral_has_fixed_reduction_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_numpy_dispatch(*args: object, **kwargs: object) -> float:
        raise AssertionError("trapz_integral must not dispatch to NumPy")

    monkeypatch.setattr(np, "trapezoid", reject_numpy_dispatch)
    x = np.array([0.0, 0.25, 1.0, 2.0])
    y = np.array([1.0, -0.5, 2.0, 0.25])
    expected = float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))

    assert trapz_integral(y, x) == expected


def test_trapz_integral_validates_one_dimensional_inputs() -> None:
    with pytest.raises(ValueError, match="equal one-dimensional arrays"):
        trapz_integral(np.ones((2, 2)), np.arange(4.0))
