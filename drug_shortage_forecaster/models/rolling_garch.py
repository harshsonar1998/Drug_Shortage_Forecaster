"""
drug_shortage_forecaster.models.rolling_garch
----------------------------------------------
Rolling-window GARCH(1,1) model for monthly shortage signals.
Re-estimates parameters on each sliding window and produces
one-step-ahead variance forecasts.
"""

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _garch_neg_loglik(params: np.ndarray, returns: np.ndarray) -> float:
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e10
    n = len(returns)
    var = np.empty(n)
    var[0] = np.var(returns) if np.var(returns) > 0 else 1e-8
    for t in range(1, n):
        var[t] = omega + alpha * returns[t - 1] ** 2 + beta * var[t - 1]
        if var[t] <= 0:
            return 1e10
    log_lik = -0.5 * np.sum(np.log(var) + returns ** 2 / var)
    return -log_lik


def _fit_garch(returns: np.ndarray) -> tuple[float, float, float]:
    sample_var = max(np.var(returns), 1e-8)
    x0 = np.array([sample_var * 0.05, 0.08, 0.85])
    bounds = [(1e-8, None), (1e-6, 0.5), (1e-6, 0.999)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(
            _garch_neg_loglik, x0, args=(returns,),
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-9},
        )
    return tuple(result.x) if result.success else (sample_var * 0.05, 0.08, 0.85)


class RollingGARCHModel:
    """Rolling-window GARCH(1,1) for monthly shortage signals.

    Parameters
    ----------
    window : int, optional
        Estimation window in months (default ``24`` ≈ 2 years).

    Attributes
    ----------
    window : int
    forecasts_ : pd.Series or None
    """

    def __init__(self, window: int = 24) -> None:
        if window < 12:
            raise ValueError("window must be at least 12 months for GARCH estimation")
        self.window = window
        self.forecasts_: pd.Series | None = None

    def fit(self, signal: pd.Series) -> pd.Series:
        """Fit rolling GARCH and return one-step-ahead forecasts.

        Parameters
        ----------
        signal : pd.Series
            Monthly log-change shortage series.

        Returns
        -------
        pd.Series
            Annualized one-step-ahead GARCH(1,1) volatility forecasts.

        Raises
        ------
        TypeError
            If *signal* is not a ``pd.Series``.
        ValueError
            If fewer than window + 1 observations.
        """
        if not isinstance(signal, pd.Series):
            raise TypeError("signal must be a pandas Series")
        if len(signal) < self.window + 1:
            raise ValueError(
                f"Need at least {self.window + 1} observations; got {len(signal)}"
            )

        r = signal.values.astype(float)
        n = len(r)
        vols, dates = [], []

        for i in range(self.window, n):
            w = r[i - self.window: i]
            omega, alpha, beta = _fit_garch(w)
            var = np.empty(self.window)
            var[0] = max(np.var(w), 1e-8)
            for t in range(1, self.window):
                var[t] = omega + alpha * w[t - 1] ** 2 + beta * var[t - 1]
            sigma2_next = omega + alpha * w[-1] ** 2 + beta * var[-1]
            vols.append(np.sqrt(max(sigma2_next, 0) * 12))
            dates.append(signal.index[i])

        self.forecasts_ = pd.Series(vols, index=dates, name="GARCH_vol")
        return self.forecasts_

    def predict(self, signal: pd.Series) -> pd.Series:
        """Alias for :meth:`fit`."""
        return self.fit(signal)
