# 💊 Drug Shortage Early Warning System

> **ORIE 5270 Final Project — Data Streams, Cornell University**
> Detects and forecasts drug supply shortage risk using live FDA data and volatility modelling (EWMA + Historical Volatility + GARCH).

## 🌐 Live Interactive Dashboard

👉 **[Open Dashboard](https://harshsonar1998.github.io/Drug_Shortage_Forecaster/)**

No installation needed — opens in any browser and fetches live FDA data directly. No Python, no servers.

---

## Table of Contents

1. [Purpose](#1-purpose)
2. [Dataset](#2-dataset)
3. [Installation](#3-installation)
4. [Running the Streamlit App](#4-running-the-streamlit-app)
5. [Running the CLI Script](#5-running-the-cli-script)
6. [Running Tests](#6-running-tests)
7. [Project Structure](#7-project-structure)
8. [API Reference](#8-api-reference)

---

## 1. Purpose

This package monitors the FDA drug shortage database and classifies every drug as **HIGH / MEDIUM / LOW** shortage risk.

**How it works:**
1. Fetches live shortage records from the FDA openFDA API — no key required
2. For each drug, builds a monthly shortage-activity time series
3. Computes month-over-month log-changes, analogous to stock log-returns
4. Applies EWMA and Historical Volatility models to measure how unpredictable each drug's shortage pattern is
5. Classifies and ranks every drug by its current volatility score

**Why this matters:** High volatility = erratic shortage behaviour = higher risk of an unexpected stockout. Hospital supply chain teams can use this to prioritise procurement before a shortage peaks — the same way financial risk managers flag volatile assets before a market event.

---

## 2. Dataset

**Source:** FDA openFDA Drug Shortages API

| Property | Value |
|----------|-------|
| URL | https://api.fda.gov/drug/shortages.json |
| Authentication | None required |
| Format | JSON |
| Coverage | All FDA-reported drug shortages |
| Update frequency | Continuously updated by FDA |
| Key fields used | generic_name, status, initial_posting_date, shortage_reason, therapeutic_category, company_name |

No manual download needed — data is fetched live every time you run the app or open the dashboard.

---

## 3. Installation

### Prerequisites
- Python >= 3.10
- pip

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/harshsonar1998/Drug_Shortage_Forecaster.git
cd Drug_Shortage_Forecaster

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install dev extras (for running tests)
pip install pytest pytest-cov
```

---

## 4. Running the Streamlit App

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

The Streamlit app provides the same functionality as the HTML dashboard but runs locally with Python. It includes a sidebar for data fetch size, model parameters, and risk thresholds, a Risk Overview tab with a colour-coded drug risk table and downloadable CSV, a Drug Analysis tab with volatility chart and shortage history per drug, and a Raw Data tab to browse and filter all FDA records.

---

## 5. Running the CLI Script

```bash
# Full scan of all drugs
python scripts\run_alert_scan.py

# Fetch more records
python scripts\run_alert_scan.py --limit 500

# Filter to current shortages only
python scripts\run_alert_scan.py --status Current

# Analyse one specific drug
python scripts\run_alert_scan.py --drug AMOXICILLIN

# Save results to CSV
python scripts\run_alert_scan.py --output risk_report.csv
```

---

## 6. Running Tests

```bash
# Run all tests from the project root
python -m pytest tests -v

# With coverage report
python -m pytest tests --cov=drug_shortage_forecaster --cov-report=term-missing
```

Target coverage: >= 80%

Test files:
- tests/test_fetcher.py — FDA API client and parser
- tests/test_processor.py — Time series builder
- tests/test_models_and_alerts.py — HV, EWMA, GARCH models and RiskDetector
- tests/test_utils.py — Metrics (RMSE, MAE, MAPE) and plotting

---

## 7. Project Structure

```
Drug_Shortage_Forecaster/
├── index.html                          <- Live interactive dashboard (open in browser)
├── app.py                              <- Streamlit dashboard (requires Python)
├── requirements.txt                    <- Python dependencies
├── pyproject.toml                      <- Package metadata
├── README.md
│
├── drug_shortage_forecaster/           <- Main Python package
│   ├── __init__.py                     <- Public API surface
│   ├── data/
│   │   ├── fetcher.py                  <- FDA openFDA API client
│   │   └── processor.py               <- Monthly time-series builder
│   ├── models/
│   │   ├── historical.py              <- Rolling Historical Volatility model
│   │   ├── ewma.py                    <- EWMA (RiskMetrics) model
│   │   └── rolling_garch.py           <- Rolling GARCH(1,1) model
│   ├── alerts/
│   │   └── detector.py                <- Risk classifier (HIGH/MEDIUM/LOW)
│   └── utils/
│       ├── metrics.py                 <- RMSE, MAE, MAPE
│       └── plotting.py                <- Matplotlib chart helpers
│
├── tests/
│   ├── __init__.py
│   ├── test_fetcher.py
│   ├── test_processor.py
│   ├── test_models_and_alerts.py
│   └── test_utils.py
│
└── scripts/
    └── run_alert_scan.py              <- CLI entry point
```

---

## 8. API Reference

### Data

| Function | Description |
|----------|-------------|
| fetch_shortage_data(limit, status) | Fetch shortage records from FDA openFDA API |
| build_shortage_series(df, drug_name) | Build monthly log-change signal for one drug |
| build_activity_counts(df, drug_name) | Raw monthly shortage posting counts |
| list_drugs(df, min_records) | List drugs with enough records to model |

### Models

All models expose .fit(signal) and .predict(signal) returning a pd.Series of annualised volatility.

| Class | Key parameter | Description |
|-------|--------------|-------------|
| HistoricalVolModel(window=6) | window - months | Rolling std dev, sqrt(12) annualised |
| EWMAVolModel(lam=0.8) | lam - decay factor | EWMA variance recursion |
| RollingGARCHModel(window=24) | window - months | GARCH(1,1) re-estimated on sliding window |

### Alerts

| Class / Method | Description |
|----------------|-------------|
| RiskDetector(high_threshold, med_threshold) | Configure risk thresholds |
| detector.scan(df) | Scan all drugs, return full risk table as pd.DataFrame |
| detector.filter_by_risk("HIGH") | Filter results by risk level |

### Metrics

| Function | Description |
|----------|-------------|
| rmse(y_true, y_pred) | Root Mean Squared Error |
| mae(y_true, y_pred) | Mean Absolute Error |
| mape(y_true, y_pred) | Mean Absolute Percentage Error (%) |
