"""tests/test_utils.py — Tests for metrics and plotting"""

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drug_shortage_forecaster.utils.metrics import rmse, mae, mape
from drug_shortage_forecaster.utils.plotting import (
    plot_shortage_volatility, plot_risk_distribution
)


@pytest.fixture
def vol_data():
    idx = pd.date_range("2020-01-31", periods=30, freq="ME")
    signal = pd.Series(np.random.randn(30) * 0.2, index=idx)
    hv = pd.Series(np.random.uniform(0.1, 0.5, 30), index=idx)
    ewma = pd.Series(np.random.uniform(0.1, 0.5, 30), index=idx)
    return signal, {"HV": hv, "EWMA": ewma}


class TestRMSE:
    def test_perfect(self):
        y = [1.0, 2.0, 3.0]
        assert rmse(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        assert rmse([0.0, 0.0], [3.0, 4.0]) == pytest.approx(3.5355, rel=1e-3)

    def test_returns_float(self):
        assert isinstance(rmse([1.0, 2.0], [1.1, 2.1]), float)

    def test_mismatched_raises(self):
        with pytest.raises(ValueError):
            rmse([1, 2, 3], [1, 2])

    def test_all_nan_raises(self):
        with pytest.raises(ValueError, match="No finite"):
            rmse(pd.Series([np.nan]), pd.Series([1.0]))


class TestMAE:
    def test_perfect(self):
        assert mae([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.0)

    def test_known_value(self):
        assert mae([1.0, 3.0], [2.0, 2.0]) == pytest.approx(1.0)

    def test_returns_float(self):
        assert isinstance(mae([1.0], [2.0]), float)


class TestMAPE:
    def test_perfect(self):
        assert mape([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.0)

    def test_known_value(self):
        assert mape([100.0, 200.0], [110.0, 200.0]) == pytest.approx(5.0)

    def test_raises_on_zero(self):
        with pytest.raises(ValueError, match="zeros"):
            mape([0.0, 1.0], [1.0, 1.0])


class TestPlotShortageVolatility:
    def test_returns_figure(self, vol_data):
        signal, forecasts = vol_data
        fig = plot_shortage_volatility(signal, forecasts, drug_name="TEST")
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_raises_on_non_series_signal(self, vol_data):
        _, forecasts = vol_data
        with pytest.raises(TypeError):
            plot_shortage_volatility([1, 2, 3], forecasts)

    def test_raises_on_non_dict_forecasts(self, vol_data):
        signal, _ = vol_data
        with pytest.raises(TypeError):
            plot_shortage_volatility(signal, [1, 2])

    def test_raises_on_empty_forecasts(self, vol_data):
        signal, _ = vol_data
        with pytest.raises(ValueError):
            plot_shortage_volatility(signal, {})

    def test_saves_file(self, vol_data, tmp_path):
        signal, forecasts = vol_data
        out = str(tmp_path / "test.png")
        plot_shortage_volatility(signal, forecasts, save_path=out)
        import os
        assert os.path.exists(out)
        plt.close("all")


class TestPlotRiskDistribution:
    @pytest.fixture
    def results_df(self):
        return pd.DataFrame({
            "drug_name":   ["A", "B", "C", "D"],
            "risk_level":  ["HIGH", "HIGH", "MEDIUM", "LOW"],
            "current_vol": [2.0, 1.8, 0.9, 0.3],
        })

    def test_returns_figure(self, results_df):
        fig = plot_risk_distribution(results_df)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_raises_on_non_dataframe(self):
        with pytest.raises(TypeError):
            plot_risk_distribution("not a df")
