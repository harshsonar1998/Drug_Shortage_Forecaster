"""
drug_shortage_forecaster.alerts.detector
-----------------------------------------
Risk detector — scans all drugs and classifies as LOW / MEDIUM / HIGH
based on EWMA volatility of the monthly shortage posting signal.
"""

from __future__ import annotations
import pandas as pd
from typing import Optional

from drug_shortage_forecaster.data.processor import build_shortage_series, list_drugs
from drug_shortage_forecaster.models.ewma import EWMAVolModel
from drug_shortage_forecaster.models.historical import HistoricalVolModel


class RiskDetector:
    """Scan FDA shortage data and classify drugs by shortage-volatility risk.

    Parameters
    ----------
    high_threshold : float  Volatility above which a drug is HIGH risk (default 1.5)
    med_threshold  : float  Volatility above which a drug is MEDIUM risk (default 0.5)
    min_records    : int    Minimum shortage records required (default 2)
    ewma_lam       : float  EWMA decay factor (default 0.8)
    hv_window      : int    Historical vol window in months (default 6)
    """

    def __init__(
        self,
        high_threshold: float = 1.5,
        med_threshold:  float = 0.5,
        min_records:    int   = 2,
        ewma_lam:       float = 0.8,
        hv_window:      int   = 6,
    ) -> None:
        if high_threshold <= med_threshold:
            raise ValueError("high_threshold must be greater than med_threshold")
        if not (0 < ewma_lam < 1):
            raise ValueError("ewma_lam must be in (0, 1)")
        if hv_window < 2:
            raise ValueError("hv_window must be at least 2")
        if min_records < 1:
            raise ValueError("min_records must be >= 1")

        self.high_threshold = high_threshold
        self.med_threshold  = med_threshold
        self.min_records    = min_records
        self.ewma_lam       = ewma_lam
        self.hv_window      = hv_window
        self.results_: Optional[pd.DataFrame] = None

    def scan(self, df: pd.DataFrame, progress: bool = False) -> pd.DataFrame:
        """Scan all drugs and return a risk summary table."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        drugs = list_drugs(df, min_records=self.min_records)
        rows  = []

        for i, drug in enumerate(drugs):
            if progress:
                print(f"  Scanning {i+1}/{len(drugs)}: {drug}  ", end="\r")
            row = self._assess_drug(df, drug)
            if row:
                rows.append(row)

        if progress:
            print()

        if not rows:
            self.results_ = pd.DataFrame(columns=[
                "drug_name", "current_vol", "risk_level",
                "n_shortages", "model_used"
            ])
            return self.results_

        result = pd.DataFrame(rows).sort_values("current_vol", ascending=False).reset_index(drop=True)
        self.results_ = result
        return self.results_

    def _assess_drug(self, df: pd.DataFrame, drug: str) -> Optional[dict]:
        try:
            signal = build_shortage_series(df, drug)
        except ValueError:
            return None

        vol_value: float
        model_used: str

        try:
            ewma = EWMAVolModel(lam=self.ewma_lam)
            forecast   = ewma.fit(signal)
            vol_value  = float(forecast.iloc[-1])
            model_used = f"EWMA(λ={self.ewma_lam})"
        except Exception:
            try:
                hv = HistoricalVolModel(window=self.hv_window)
                forecast   = hv.fit(signal)
                vol_value  = float(forecast.iloc[-1])
                model_used = f"HV(w={self.hv_window})"
            except Exception:
                return None

        drug_df     = df[df["drug_name"].str.upper().str.strip() == drug]
        n_shortages = len(drug_df)

        return {
            "drug_name":   drug,
            "current_vol": round(vol_value, 4),
            "risk_level":  self._classify(vol_value),
            "n_shortages": n_shortages,
            "model_used":  model_used,
        }

    def _classify(self, vol: float) -> str:
        if vol > self.high_threshold:  return "HIGH"
        if vol > self.med_threshold:   return "MEDIUM"
        return "LOW"

    def filter_by_risk(self, level: str) -> pd.DataFrame:
        if self.results_ is None:
            raise ValueError("Call scan() before filter_by_risk()")
        level = level.upper()
        if level not in ("HIGH", "MEDIUM", "LOW"):
            raise ValueError(f"level must be HIGH, MEDIUM, or LOW; got '{level}'")
        return self.results_[self.results_["risk_level"] == level].copy()
