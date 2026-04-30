"""
drug_shortage_forecaster.utils.metrics
----------------------------------------
Forecast evaluation metrics: RMSE, MAE, MAPE.
"""

import numpy as np
import pandas as pd
from typing import Union

ArrayLike = Union[np.ndarray, pd.Series, list]


def _align(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(y_true, pd.Series) and isinstance(y_pred, pd.Series):
        combined = pd.concat([y_true, y_pred], axis=1).dropna()
        a, b = combined.iloc[:, 0].values, combined.iloc[:, 1].values
    else:
        a = np.asarray(y_true, dtype=float)
        b = np.asarray(y_pred, dtype=float)
        if a.shape != b.shape:
            raise ValueError(
                f"Shapes do not match: {a.shape} vs {b.shape}"
            )
        mask = np.isfinite(a) & np.isfinite(b)
        a, b = a[mask], b[mask]
    if len(a) == 0:
        raise ValueError("No finite values remain after alignment")
    return a, b


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root Mean Squared Error."""
    a, b = _align(y_true, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean Absolute Error."""
    a, b = _align(y_true, y_pred)
    return float(np.mean(np.abs(a - b)))


def mape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean Absolute Percentage Error (in %)."""
    a, b = _align(y_true, y_pred)
    if np.any(a == 0):
        raise ValueError("y_true contains zeros; MAPE is undefined")
    return float(np.mean(np.abs((a - b) / a)) * 100)
