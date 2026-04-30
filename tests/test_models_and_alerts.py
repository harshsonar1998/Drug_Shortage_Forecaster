"""tests/test_models_and_alerts.py — Tests for models and RiskDetector"""

import numpy as np
import pandas as pd
import pytest

from drug_shortage_forecaster.models.historical import HistoricalVolModel
from drug_shortage_forecaster.models.ewma import EWMAVolModel
from drug_shortage_forecaster.models.rolling_garch import RollingGARCHModel
from drug_shortage_forecaster.alerts.detector import RiskDetector


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def signal_50():
    np.random.seed(42)
    idx = pd.date_range("2019-01-31", periods=50, freq="ME")
    return pd.Series(np.random.randn(50) * 0.3, index=idx)


@pytest.fixture
def signal_30():
    np.random.seed(7)
    idx = pd.date_range("2020-01-31", periods=30, freq="ME")
    return pd.Series(np.random.randn(30) * 0.2, index=idx)


@pytest.fixture
def shortage_df():
    """Minimal shortage DataFrame for detector tests."""
    rows = []
    for drug in ["AMOXICILLIN", "INSULIN", "LIDOCAINE", "MORPHINE"]:
        for i in range(4):
            rows.append({
                "drug_name": drug,
                "generic_name": drug.lower(),
                "status": "active",
                "shortage_start": pd.Timestamp(f"201{i+5}-01-01"),
                "shortage_end":   pd.Timestamp(f"202{i}-12-31"),
                "reason": "Test",
                "dosage_form": "Tablet",
            })
    return pd.DataFrame(rows)


# ── HistoricalVolModel ────────────────────────────────────────────────────────

class TestHistoricalVolModel:

    def test_returns_series(self, signal_50):
        out = HistoricalVolModel(window=6).fit(signal_50)
        assert isinstance(out, pd.Series)

    def test_no_nan(self, signal_50):
        out = HistoricalVolModel(window=6).fit(signal_50)
        assert not out.isna().any()

    def test_all_positive(self, signal_50):
        out = HistoricalVolModel(window=6).fit(signal_50)
        assert (out > 0).all()

    def test_correct_length(self, signal_50):
        out = HistoricalVolModel(window=6).fit(signal_50)
        assert len(out) == len(signal_50) - 6 + 1

    def test_predict_alias(self, signal_50):
        m = HistoricalVolModel(window=6)
        pd.testing.assert_series_equal(m.fit(signal_50), m.predict(signal_50))

    def test_raises_on_non_series(self):
        with pytest.raises(TypeError):
            HistoricalVolModel().fit(np.random.randn(30))

    def test_raises_on_window_lt_2(self):
        with pytest.raises(ValueError):
            HistoricalVolModel(window=1)

    def test_raises_on_too_few_obs(self, signal_30):
        with pytest.raises(ValueError, match="at least"):
            HistoricalVolModel(window=100).fit(signal_30)

    def test_annualized_by_sqrt12(self, signal_50):
        """Output should equal rolling std * sqrt(12)."""
        window = 6
        out = HistoricalVolModel(window=window).fit(signal_50)
        expected = signal_50.rolling(window).std() * np.sqrt(12)
        expected = expected.dropna()
        np.testing.assert_allclose(out.values, expected.values, rtol=1e-10)


# ── EWMAVolModel ──────────────────────────────────────────────────────────────

class TestEWMAVolModel:

    def test_returns_series(self, signal_50):
        out = EWMAVolModel().fit(signal_50)
        assert isinstance(out, pd.Series)

    def test_same_length_as_input(self, signal_50):
        out = EWMAVolModel().fit(signal_50)
        assert len(out) == len(signal_50)

    def test_all_positive(self, signal_50):
        out = EWMAVolModel().fit(signal_50)
        assert (out > 0).all()

    def test_no_nan(self, signal_50):
        out = EWMAVolModel().fit(signal_50)
        assert not out.isna().any()

    def test_predict_alias(self, signal_50):
        m = EWMAVolModel()
        pd.testing.assert_series_equal(m.fit(signal_50), m.predict(signal_50))

    def test_raises_invalid_lambda(self):
        with pytest.raises(ValueError):
            EWMAVolModel(lam=0)
        with pytest.raises(ValueError):
            EWMAVolModel(lam=1.0)

    def test_raises_on_non_series(self):
        with pytest.raises(TypeError):
            EWMAVolModel().fit([0.1, 0.2, 0.3])

    def test_raises_on_single_obs(self):
        with pytest.raises(ValueError, match="at least 2"):
            EWMAVolModel().fit(pd.Series([0.1]))

    def test_higher_lambda_smoother(self, signal_50):
        high = EWMAVolModel(lam=0.95).fit(signal_50)
        low  = EWMAVolModel(lam=0.50).fit(signal_50)
        assert high.std() < low.std()

    def test_index_preserved(self, signal_50):
        out = EWMAVolModel().fit(signal_50)
        pd.testing.assert_index_equal(out.index, signal_50.index)


# ── RollingGARCHModel ─────────────────────────────────────────────────────────

class TestRollingGARCHModel:

    @pytest.fixture
    def long_signal(self):
        np.random.seed(99)
        idx = pd.date_range("2010-01-31", periods=60, freq="ME")
        return pd.Series(np.random.randn(60) * 0.25, index=idx)

    def test_returns_series(self, long_signal):
        out = RollingGARCHModel(window=24).fit(long_signal)
        assert isinstance(out, pd.Series)

    def test_output_length(self, long_signal):
        window = 24
        out = RollingGARCHModel(window=window).fit(long_signal)
        assert len(out) == len(long_signal) - window

    def test_all_positive(self, long_signal):
        out = RollingGARCHModel(window=24).fit(long_signal)
        assert (out > 0).all()

    def test_no_nan(self, long_signal):
        out = RollingGARCHModel(window=24).fit(long_signal)
        assert not out.isna().any()

    def test_predict_alias(self, long_signal):
        m = RollingGARCHModel(window=24)
        pd.testing.assert_series_equal(m.fit(long_signal), m.predict(long_signal))

    def test_raises_window_too_small(self):
        with pytest.raises(ValueError, match="at least 12"):
            RollingGARCHModel(window=5)

    def test_raises_on_non_series(self):
        with pytest.raises(TypeError):
            RollingGARCHModel(window=12).fit(np.random.randn(50))

    def test_raises_on_too_few_obs(self):
        short = pd.Series(np.random.randn(10) * 0.1)
        with pytest.raises(ValueError, match="Need at least"):
            RollingGARCHModel(window=24).fit(short)


# ── RiskDetector ──────────────────────────────────────────────────────────────

class TestRiskDetector:

    def test_scan_returns_dataframe(self, shortage_df):
        det = RiskDetector(min_records=3)
        out = det.scan(shortage_df)
        assert isinstance(out, pd.DataFrame)

    def test_scan_stores_results(self, shortage_df):
        det = RiskDetector(min_records=3)
        det.scan(shortage_df)
        assert det.results_ is not None

    def test_expected_columns(self, shortage_df):
        det = RiskDetector(min_records=3)
        out = det.scan(shortage_df)
        for col in ["drug_name", "current_vol", "risk_level", "n_shortages"]:
            assert col in out.columns

    def test_risk_levels_valid(self, shortage_df):
        det = RiskDetector(min_records=3)
        out = det.scan(shortage_df)
        assert out["risk_level"].isin(["HIGH", "MEDIUM", "LOW"]).all()

    def test_sorted_descending_vol(self, shortage_df):
        det = RiskDetector(min_records=3)
        out = det.scan(shortage_df)
        assert (out["current_vol"].diff().dropna() <= 0).all()

    def test_filter_by_risk_high(self, shortage_df):
        det = RiskDetector(min_records=3)
        det.scan(shortage_df)
        high = det.filter_by_risk("HIGH")
        assert (high["risk_level"] == "HIGH").all()

    def test_filter_by_risk_raises_before_scan(self):
        det = RiskDetector()
        with pytest.raises(ValueError, match="Call scan"):
            det.filter_by_risk("HIGH")

    def test_filter_by_risk_invalid_level(self, shortage_df):
        det = RiskDetector(min_records=3)
        det.scan(shortage_df)
        with pytest.raises(ValueError, match="HIGH, MEDIUM, or LOW"):
            det.filter_by_risk("CRITICAL")

    def test_raises_on_non_dataframe(self):
        det = RiskDetector()
        with pytest.raises(TypeError):
            det.scan("not a df")

    def test_raises_invalid_thresholds(self):
        with pytest.raises(ValueError, match="high_threshold must be greater"):
            RiskDetector(high_threshold=0.5, med_threshold=0.8)

    def test_raises_invalid_ewma_lam(self):
        with pytest.raises(ValueError):
            RiskDetector(ewma_lam=0)

    def test_empty_df_returns_empty_results(self):
        empty = pd.DataFrame(columns=[
            "drug_name","generic_name","status",
            "shortage_start","shortage_end","reason","dosage_form"
        ])
        det = RiskDetector()
        out = det.scan(empty)
        assert out.empty

    def test_classify_thresholds(self):
        det = RiskDetector(high_threshold=2.0, med_threshold=1.0)
        assert det._classify(2.5) == "HIGH"
        assert det._classify(1.5) == "MEDIUM"
        assert det._classify(0.5) == "LOW"
