"""
drug_shortage_forecaster.data.fetcher
--------------------------------------
Fetches drug shortage records from the FDA openFDA API.

FDA API confirmed field structure:
  - generic_name         : drug name
  - status               : "Current" | "Resolved" | "Discontinued"
  - initial_posting_date : MM/DD/YYYY
  - update_date          : MM/DD/YYYY
  - shortage_reason      : reason string
  - therapeutic_category : list
  - dosage_form          : string
  - company_name         : manufacturer
  - availability         : supply note
"""

import time
import requests
import pandas as pd
from typing import Optional

FDA_BASE_URL = "https://api.fda.gov/drug/shortages.json"
_DEFAULT_LIMIT = 100


def fetch_shortage_data(
    limit: int = 1000,
    status: Optional[str] = None,
    retries: int = 3,
    pause: float = 0.5,
) -> pd.DataFrame:
    """Fetch drug shortage records from the FDA openFDA API."""
    if limit < 1:
        raise ValueError(f"limit must be >= 1; got {limit}")

    records = []
    skip = 0
    per_request = min(_DEFAULT_LIMIT, limit)

    while len(records) < limit:
        params = {"limit": per_request, "skip": skip}
        if status:
            params["search"] = f'status:"{status}"'

        response = _get_with_retry(FDA_BASE_URL, params=params, retries=retries)
        data = response.json()
        results = data.get("results", [])
        if not results:
            break

        records.extend(results)
        skip += len(results)
        total = data.get("meta", {}).get("results", {}).get("total", 0)
        if skip >= total:
            break
        if len(records) < limit:
            time.sleep(pause)

    return _parse_records(records[:limit])


def _get_with_retry(url, params, retries):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return _empty_response()
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"FDA API returned {r.status_code} after {retries} attempts.")
        except requests.exceptions.RequestException as exc:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"Network error: {exc}") from exc
    return _empty_response()


class _empty_response:
    status_code = 200
    def json(self):
        return {"results": [], "meta": {"results": {"total": 0}}}


def _parse_records(records):
    rows = []
    for rec in records:
        tc = rec.get("therapeutic_category", "")
        if isinstance(tc, list):
            tc = ", ".join(tc)
        openfda = rec.get("openfda") or {}
        brand = openfda.get("brand_name", [""])
        brand = brand[0] if brand else ""

        rows.append({
            "drug_name":            _safe_str(rec.get("generic_name", "")),
            "brand_name":           _safe_str(brand),
            "status":               _safe_str(rec.get("status", "Unknown")),
            "initial_posting_date": _safe_date(rec.get("initial_posting_date")),
            "update_date":          _safe_date(rec.get("update_date")),
            "shortage_reason":      _safe_str(rec.get("shortage_reason", "")),
            "therapeutic_category": _safe_str(tc),
            "dosage_form":          _safe_str(rec.get("dosage_form", "")),
            "company_name":         _safe_str(rec.get("company_name", "")),
            "availability":         _safe_str(rec.get("availability", "")),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "drug_name", "brand_name", "status", "initial_posting_date",
            "update_date", "shortage_reason", "therapeutic_category",
            "dosage_form", "company_name", "availability"
        ])

    df = pd.DataFrame(rows)
    df["drug_name"] = df["drug_name"].str.upper().str.strip()
    df = df[df["drug_name"].str.len() > 0]
    return df.reset_index(drop=True)


def _safe_str(val):
    if val is None:
        return ""
    return str(val).strip()


def _safe_date(val):
    if not val:
        return pd.NaT
    try:
        return pd.to_datetime(val)
    except Exception:
        return pd.NaT
