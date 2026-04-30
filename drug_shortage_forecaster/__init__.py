"""
drug_shortage_forecaster
========================
A Python package for detecting and forecasting drug supply shortage risk
using FDA public shortage data and volatility modeling.

Modules
-------
data        : Fetch and process FDA drug shortage records
models      : Volatility models (Historical, EWMA, GARCH)
alerts      : Risk detector that flags high-volatility drugs
utils       : Evaluation metrics and plotting helpers
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from drug_shortage_forecaster.data.fetcher import fetch_shortage_data
from drug_shortage_forecaster.data.processor import build_shortage_series, list_drugs
from drug_shortage_forecaster.models.historical import HistoricalVolModel
from drug_shortage_forecaster.models.ewma import EWMAVolModel
from drug_shortage_forecaster.models.rolling_garch import RollingGARCHModel
from drug_shortage_forecaster.alerts.detector import RiskDetector
from drug_shortage_forecaster.utils.metrics import rmse, mae, mape
from drug_shortage_forecaster.utils.plotting import plot_shortage_volatility

__all__ = [
    "fetch_shortage_data",
    "build_shortage_series",
    "list_drugs",
    "HistoricalVolModel",
    "EWMAVolModel",
    "RollingGARCHModel",
    "RiskDetector",
    "rmse",
    "mae",
    "mape",
    "plot_shortage_volatility",
]
