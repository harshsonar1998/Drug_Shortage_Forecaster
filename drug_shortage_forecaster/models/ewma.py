"""
drug_shortage_forecaster.models.ewma
--------------------------------------
Exponentially Weighted Moving Average (EWMA) volatility model
applied to the monthly shortage log-change signal.

Variance recursion:
    σ²_t = λ · σ²_{t-1} + (1 − λ) · r²_{t-1}

For monthly data λ = 0.8 is a reasonable default (faster adaptation
than the daily 0.94 used in RiskMetrics).
"""

import numpy as np
import pandas as pd


class EWMAVolModel:
    """EWMA Volatility forecaster for monthly shortage signals.

    Parameters
    ----------
    lam : float, optional
        Decay factor λ ∈ (0, 1).  Default ``0.8`` for monthly data.

    Attributes
    ----------
    lam : float
    forecasts_ : pd.Series or None
        Annualized volatility forecasts; available after :meth:`fit`.
    """

    def __init__(self, lam: float = 0.8) -> None:
        if not (0 < lam < 1):
            raise ValueError(f"lam must be in (0, 1); got {lam}")
        self.lam = lam
        self.forecasts_: pd.Series | None = None

    def fit(self, signal: pd.Series) -> pd.Series:
        """Compute EWMA variance and return annualized vol forecasts.

        Parameters
        ----------
        signal : pd.Series
            Monthly log-change shortage series.

        Returns
        -------
        pd.Series
            Annualized EWMA volatility (√12 scaled).

        Raises
        ------
        TypeError
            If *signal* is not a ``pd.Series``.
        ValueError
            If fewer than 2 observations.
        """
        if not isinstance(signal, pd.Series):
            raise TypeError("signal must be a pandas Series")
        if len(signal) < 2:
            raise ValueError("Need at least 2 observations")

        r = signal.values.astype(float)
        n = len(r)
        var = np.empty(n)
        var[0] = r[0] ** 2
        for t in range(1, n):
            var[t] = self.lam * var[t - 1] + (1 - self.lam) * r[t - 1] ** 2

        self.forecasts_ = pd.Series(
            np.sqrt(var * 12), index=signal.index, name="EWMA_vol"
        )
        return self.forecasts_

    def predict(self, signal: pd.Series) -> pd.Series:
        """Alias for :meth:`fit`."""
        return self.fit(signal)
