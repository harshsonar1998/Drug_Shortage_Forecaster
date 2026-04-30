"""tests/test_processor.py — Unit tests for data.processor"""

import numpy as np
import pandas as pd
import pytest

from drug_shortage_forecaster.data.processor import (
    build_shortage_series, build_activity_counts, list_drugs
)


def _make_df(drugs_and_dates):
    """Helper: build a minimal shortage DataFrame matching the real FDA structure."""
    rows = []
    for drug, posting_date in drugs_and_dates:
        rows.append({
            "drug_name":            drug.upper(),
            "status":               "Current",
            "initial_posting_date": pd.Timestamp(posting_date),
            "shortage_reason":      "Test reason",
            "therapeutic_category": "Anti-Infective",
            "dosage_form":          "Tablet",
            "company_name":         "Test Co",
            "availability":         "Limited",
        })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_df():
    return _make_df([
        ("AMOXICILLIN", "2019-01-01"),
        ("AMOXICILLIN", "2019-06-01"),
        ("AMOXICILLIN", "2020-01-01"),
        ("AMOXICILLIN", "2020-06-01"),
        ("AMOXICILLIN", "2021-01-01"),
        ("INSULIN",     "2020-01-01"),
        ("INSULIN",     "2020-06-01"),
        ("INSULIN",     "2021-01-01"),
        ("INSULIN",     "2022-01-01"),
        ("RARE_DRUG",   "2021-01-01"),  # only 1 record
    ])


class TestListDrugs:

    def test_returns_list(self, sample_df):
        result = list_drugs(sample_df)
        assert isinstance(result, list)

    def test_filters_by_min_records(self, sample_df):
        result = list_drugs(sample_df, min_records=3)
        assert "AMOXICILLIN" in result
        assert "RARE_DRUG" not in result

    def test_sorted_alphabetically(self, sample_df):
        result = list_drugs(sample_df, min_records=2)
        assert result == sorted(result)

    def test_empty_df_returns_empty(self):
        empty = pd.DataFrame(columns=["drug_name"])
        assert list_drugs(empty) == []

    def test_raises_on_non_dataframe(self):
        with pytest.raises(TypeError):
            list_drugs("not a df")

    def test_raises_on_min_records_zero(self, sample_df):
        with pytest.raises(ValueError):
            list_drugs(sample_df, min_records=0)


class TestBuildShortageSeries:

    def test_returns_series(self, sample_df):
        s = build_shortage_series(sample_df, "AMOXICILLIN")
        assert isinstance(s, pd.Series)

    def test_no_nan(self, sample_df):
        s = build_shortage_series(sample_df, "AMOXICILLIN")
        assert not s.isna().any()

    def test_length_gt_1(self, sample_df):
        s = build_shortage_series(sample_df, "AMOXICILLIN")
        assert len(s) >= 2

    def test_case_insensitive(self, sample_df):
        s1 = build_shortage_series(sample_df, "amoxicillin")
        s2 = build_shortage_series(sample_df, "AMOXICILLIN")
        pd.testing.assert_series_equal(s1, s2)

    def test_raises_on_unknown_drug(self, sample_df):
        with pytest.raises(ValueError, match="No shortage records"):
            build_shortage_series(sample_df, "FAKE_DRUG_XYZ")

    def test_raises_on_non_dataframe(self):
        with pytest.raises(TypeError):
            build_shortage_series("not a df", "AMOXICILLIN")

    def test_custom_date_range(self, sample_df):
        s = build_shortage_series(sample_df, "AMOXICILLIN",
                                  start="2020-01-01", end="2021-06-01")
        assert s.index.min() >= pd.Timestamp("2020-01-01")

    def test_index_is_datetime(self, sample_df):
        s = build_shortage_series(sample_df, "AMOXICILLIN")
        assert pd.api.types.is_datetime64_any_dtype(s.index)


class TestBuildActivityCounts:

    def test_returns_series(self, sample_df):
        c = build_activity_counts(sample_df, "AMOXICILLIN")
        assert isinstance(c, pd.Series)

    def test_all_non_negative(self, sample_df):
        c = build_activity_counts(sample_df, "AMOXICILLIN")
        assert (c >= 0).all()

    def test_unknown_drug_returns_empty(self, sample_df):
        c = build_activity_counts(sample_df, "FAKE_DRUG")
        assert c.empty

    def test_raises_on_non_dataframe(self):
        with pytest.raises(TypeError):
            build_activity_counts([], "AMOXICILLIN")

    def test_active_months_nonzero(self, sample_df):
        c = build_activity_counts(sample_df, "AMOXICILLIN")
        assert (c > 0).any()
