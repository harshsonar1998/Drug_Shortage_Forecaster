"""tests/test_fetcher.py — Unit tests for data.fetcher"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from drug_shortage_forecaster.data.fetcher import (
    fetch_shortage_data, _parse_records, _safe_str, _safe_date
)


def _mock_response(records, total=None):
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {
        "results": records,
        "meta": {"results": {"total": total or len(records)}}
    }
    return mock


SAMPLE_RECORDS = [
    {
        "generic_name": "Amoxicillin",
        "status": "Current",
        "initial_posting_date": "01/01/2022",
        "update_date": "06/01/2022",
        "shortage_reason": "Manufacturing delay",
        "therapeutic_category": ["Anti-Infective"],
        "dosage_form": "Tablet",
        "company_name": "Sandoz Inc.",
        "availability": "Limited",
        "openfda": {"brand_name": ["AMOXICILLIN 500MG"]},
    },
    {
        "generic_name": "Insulin Glargine",
        "status": "Resolved",
        "initial_posting_date": "03/01/2021",
        "update_date": "09/01/2021",
        "shortage_reason": "Supply disruption",
        "therapeutic_category": ["Endocrine"],
        "dosage_form": "Injection",
        "company_name": "Sanofi",
        "availability": "Available",
        "openfda": {},
    },
]


class TestFetchShortageData:

    def test_returns_dataframe(self):
        with patch("drug_shortage_forecaster.data.fetcher.requests.get") as mock_get:
            mock_get.return_value = _mock_response(SAMPLE_RECORDS)
            df = fetch_shortage_data(limit=10)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self):
        with patch("drug_shortage_forecaster.data.fetcher.requests.get") as mock_get:
            mock_get.return_value = _mock_response(SAMPLE_RECORDS)
            df = fetch_shortage_data(limit=10)
        for col in ["drug_name", "status", "initial_posting_date", "shortage_reason"]:
            assert col in df.columns

    def test_drug_names_uppercase(self):
        with patch("drug_shortage_forecaster.data.fetcher.requests.get") as mock_get:
            mock_get.return_value = _mock_response(SAMPLE_RECORDS)
            df = fetch_shortage_data(limit=10)
        assert df["drug_name"].str.isupper().all()

    def test_raises_on_limit_zero(self):
        with pytest.raises(ValueError, match="limit must be"):
            fetch_shortage_data(limit=0)

    def test_raises_on_negative_limit(self):
        with pytest.raises(ValueError):
            fetch_shortage_data(limit=-5)

    def test_empty_result_on_404(self):
        mock = MagicMock()
        mock.status_code = 404
        with patch("drug_shortage_forecaster.data.fetcher.requests.get", return_value=mock):
            df = fetch_shortage_data(limit=10)
        assert df.empty

    def test_raises_on_persistent_500(self):
        mock = MagicMock()
        mock.status_code = 500
        with patch("drug_shortage_forecaster.data.fetcher.requests.get", return_value=mock):
            with pytest.raises(RuntimeError):
                fetch_shortage_data(limit=10, retries=1)

    def test_status_filter_passed_as_search_param(self):
        with patch("drug_shortage_forecaster.data.fetcher.requests.get") as mock_get:
            mock_get.return_value = _mock_response([])
            fetch_shortage_data(limit=10, status="Current", retries=1)
        call_kwargs = mock_get.call_args
        params = call_kwargs[1]["params"] if call_kwargs[1] else call_kwargs[0][1]
        assert "search" in params
        assert "Current" in params["search"]

    def test_respects_limit(self):
        many = SAMPLE_RECORDS * 10
        with patch("drug_shortage_forecaster.data.fetcher.requests.get") as mock_get:
            mock_get.return_value = _mock_response(many, total=20)
            df = fetch_shortage_data(limit=3)
        assert len(df) <= 3


class TestParseRecords:

    def test_empty_input_returns_empty_df(self):
        df = _parse_records([])
        assert df.empty

    def test_missing_dates_still_creates_row(self):
        rec = [{"generic_name": "TestDrug", "status": "Current"}]
        df = _parse_records(rec)
        assert len(df) == 1
        assert df.iloc[0]["drug_name"] == "TESTDRUG"

    def test_drug_name_from_generic_name(self):
        rec = [{"generic_name": "Amoxicillin", "status": "Current"}]
        df = _parse_records(rec)
        assert df.iloc[0]["drug_name"] == "AMOXICILLIN"

    def test_therapeutic_category_list_joined(self):
        rec = [{"generic_name": "TestDrug", "therapeutic_category": ["Cat A", "Cat B"]}]
        df = _parse_records(rec)
        assert "Cat A" in df.iloc[0]["therapeutic_category"]

    def test_returns_dataframe(self):
        df = _parse_records(SAMPLE_RECORDS)
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self):
        df = _parse_records(SAMPLE_RECORDS)
        assert len(df) == 2


class TestHelpers:

    def test_safe_str_none(self):
        assert _safe_str(None) == ""

    def test_safe_str_strips(self):
        assert _safe_str("  hello  ") == "hello"

    def test_safe_date_none(self):
        assert pd.isna(_safe_date(None))

    def test_safe_date_valid(self):
        result = _safe_date("2022-01-15")
        assert result == pd.Timestamp("2022-01-15")

    def test_safe_date_invalid(self):
        assert pd.isna(_safe_date("not-a-date"))
