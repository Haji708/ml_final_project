"""Member 1's regression metrics only; classification belongs to Member 4.

The shared evaluate.py can re-export rmse, mae and r2 from here. Keeping this
module independent means installation never replaces Member 4's work.
"""
from __future__ import annotations

import numpy as np


def _regression_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    arrays = []
    for name, values in (("y_true", y_true), ("y_pred", y_pred)):
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 and not (array.ndim == 2 and array.shape[1] == 1):
            raise ValueError(f"{name} must be 1D or an (n, 1) column")
        array = array.reshape(-1)
        if not len(array) or not np.isfinite(array).all():
            raise ValueError(f"{name} must be nonempty and finite")
        arrays.append(array)
    if arrays[0].shape != arrays[1].shape:
        raise ValueError("Actual values and predictions must have the same shape")
    return arrays[0], arrays[1]


def rmse(y_true, y_pred) -> float:
    """Root mean squared error; penalizes large mistakes more strongly."""
    actual, predicted = _regression_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean absolute error, in the same units as the supplied targets."""
    actual, predicted = _regression_arrays(y_true, y_pred)
    return float(np.mean(np.abs(actual - predicted)))


def r2(y_true, y_pred) -> float:
    """R-squared; NaN for one row; sklearn-style finite constant-target policy."""
    actual, predicted = _regression_arrays(y_true, y_pred)
    if len(actual) < 2:
        return float("nan")
    residual = float(np.sum((actual - predicted) ** 2))
    total = float(np.sum((actual - actual.mean()) ** 2))
    if total == 0.0:
        return 1.0 if residual == 0.0 else 0.0
    return 1.0 - residual / total


def regression_scores(y_true, y_pred) -> dict[str, float]:
    """Convenience wrapper used by the Member 1 experiment runner."""
    return {"RMSE": rmse(y_true, y_pred), "MAE": mae(y_true, y_pred), "R2": r2(y_true, y_pred)}
