"""
drug_shortage_forecaster.models.historical
-------------------------------------------
Rolling Historical Volatility model applied to the monthly shortage
log-change signal.
"""

import numpy as np
import pandas as pd


class HistoricalVolModel:
    """Rolling Historical Volatility (HV) forecaster.

    Computes the rolling standard deviation of the shortage log-change
    series, annualized by √12 (monthly data).

    Parameters
    ----------
    window : int, optional
        Look-back window in months (default ``6``).

    Attributes
    ----------
    window : int
    forecasts_ : pd.Series or None
        Fitted forecasts; available after calling :meth:`fit`.
    """

    def __init__(self, window: int = 6) -> None:
        if window < 2:
            raise ValueError("window must be at least 2")
        self.window = window
        self.forecasts_: pd.Series | None = None

    def fit(self, signal: pd.Series) -> pd.Series:
        """Fit the model and return annualized volatility forecasts.

        Parameters
        ----------
        signal : pd.Series
            Monthly log-change shortage series.

        Returns
        -------
        pd.Series
            Annualized rolling volatility (√12 scaled).

        Raises
        ------
        TypeError
            If *signal* is not a ``pd.Series``.
        ValueError
            If fewer observations than *window*.
        """
        if not isinstance(signal, pd.Series):
            raise TypeError("signal must be a pandas Series")
        if len(signal) < self.window:
            raise ValueError(
                f"Need at least {self.window} observations; got {len(signal)}"
            )
        self.forecasts_ = signal.rolling(self.window).std() * np.sqrt(12)
        self.forecasts_ = self.forecasts_.dropna()
        return self.forecasts_

    def predict(self, signal: pd.Series) -> pd.Series:
        """Alias for :meth:`fit`."""
        return self.fit(signal)
