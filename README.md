# 💊 Drug Shortage Early Warning System

> **ORIE 5270 Final Project — Data Streams, Cornell University**  
> Detects and forecasts drug supply shortage risk using live FDA data and volatility modelling (EWMA + Historical Volatility + GARCH).

🌐 **Live Demo:** `https://drug-shortage-alert.streamlit.app` *(after deployment — see Step 6)*

---

## Table of Contents

1. [Purpose](#1-purpose)
2. [Dataset](#2-dataset)
3. [Installation](#3-installation)
4. [Running the Dashboard](#4-running-the-dashboard)
5. [Running the CLI Script](#5-running-the-cli-script)
6. [Running Tests](#6-running-tests)
7. [Deploy to Streamlit Cloud (Public URL)](#7-deploy-to-streamlit-cloud-public-url)
8. [Project Structure](#8-project-structure)
9. [API Reference](#9-api-reference)

---

## 1. Purpose

This package monitors the FDA drug shortage database and classifies every drug as **🔴 HIGH / 🟡 MEDIUM / 🟢 LOW** shortage risk.

**How it works:**
1. Fetches live shortage records from the FDA openFDA API (no key required)
2. For each drug, builds a monthly "shortage activity" time series
3. Computes month-over-month log-changes (analogous to stock log-returns)
4. Applies EWMA and Historical Volatility models to measure how *unpredictable* each drug's shortage pattern is
5. Classifies and ranks drugs by their current volatility reading

**Why this matters:** High volatility = erratic shortage behaviour = higher risk of an unexpected stockout. Hospital supply chain teams can use this to prioritise procurement before a shortage peaks.

---

## 2. Dataset

**Source:** [FDA openFDA Drug Shortages API](https://open.fda.gov/apis/drug/shortages/)

| Property | Value |
|----------|-------|
| URL | `https://api.fda.gov/drug/shortages.json` |
| Auth | None required |
| Format | JSON |
| Coverage | All FDA-reported drug shortages |
| Update frequency | Continuously updated by FDA |

No manual download needed — data is fetched live every time you run the app.

---

## 3. Installation

### Prerequisites
- Python ≥ 3.10
- pip

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/drug-shortage-forecaster.git
cd drug-shortage-forecaster

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install dev extras (for tests)
pip install pytest pytest-cov
```

---

## 4. Running the Dashboard

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

**What you'll see:**
- Sidebar to configure data fetch size, model parameters, and risk thresholds
- **Risk Overview tab** — colour-coded table of all drugs + downloadable CSV
- **Drug Deep Dive tab** — search any drug, see its volatility chart and shortage history
- **Raw Data tab** — browse and filter all FDA records

---

## 5. Running the CLI Script

```bash
# Full scan of all drugs
python scripts\run_alert_scan.py

# Fetch more records
python scripts\run_alert_scan.py --limit 500

# Filter to active shortages only
python scripts\run_alert_scan.py --status active

# Analyse one specific drug
python scripts\run_alert_scan.py --drug AMOXICILLIN

# Save results to CSV
python scripts\run_alert_scan.py --output risk_report.csv
```

---

## 6. Running Tests

```bash
# Run all tests (from project root)
python -m pytest tests -v

# With coverage report
python -m pytest tests --cov=drug_shortage_forecaster --cov-report=term-missing
```

Target coverage: **≥ 80%**

---

## 7. Deploy to Streamlit Cloud (Public URL)

Anyone in the world can use your app — no installation needed. Takes ~10 minutes.

### Step-by-step

**Step 1 — Push to GitHub**
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/drug-shortage-forecaster.git
git push -u origin main
```

**Step 2 — Go to Streamlit Cloud**
- Visit [share.streamlit.io](https://share.streamlit.io)
- Sign in with your GitHub account

**Step 3 — Deploy**
- Click **"New app"**
- Select your repository: `drug-shortage-forecaster`
- Branch: `main`
- Main file path: `app.py`
- Click **"Deploy"**

**Step 4 — Get your URL**
- Streamlit builds and deploys automatically (~2 minutes)
- You get a permanent public URL like:  
  `https://drug-shortage-alert.streamlit.app`
- Share this URL in your README and GitHub repo

> **Note:** `requirements.txt` in the repo root tells Streamlit Cloud what to install. It is already included.

---

## 8. Project Structure

```
drug-shortage-forecaster/
├── app.py                              ← Streamlit dashboard (run this)
├── requirements.txt                    ← Dependencies for Streamlit Cloud
├── pyproject.toml                      ← Package metadata
├── README.md
│
├── drug_shortage_forecaster/           ← Main Python package
│   ├── __init__.py                     ← Public API
│   ├── data/
│   │   ├── fetcher.py                  ← FDA API client
│   │   └── processor.py               ← Time series builder
│   ├── models/
│   │   ├── historical.py              ← Rolling HV model
│   │   ├── ewma.py                    ← EWMA model
│   │   └── rolling_garch.py           ← GARCH(1,1) model
│   ├── alerts/
│   │   └── detector.py                ← Risk classifier
│   └── utils/
│       ├── metrics.py                 ← RMSE, MAE, MAPE
│       └── plotting.py                ← Charts
│
├── tests/
│   ├── test_fetcher.py
│   ├── test_processor.py
│   ├── test_models_and_alerts.py
│   └── test_utils.py
│
└── scripts/
    └── run_alert_scan.py              ← CLI entry point
```

---

## 9. API Reference

### Data

| Function | Description |
|----------|-------------|
| `fetch_shortage_data(limit, status)` | Fetch shortage records from FDA API |
| `build_shortage_series(df, drug_name)` | Build monthly log-change signal for one drug |
| `build_activity_counts(df, drug_name)` | Raw monthly active-shortage counts |
| `list_drugs(df, min_records)` | List drugs with enough records to model |

### Models

| Class | Key parameter | Description |
|-------|--------------|-------------|
| `HistoricalVolModel(window=6)` | months | Rolling std dev, √12 annualized |
| `EWMAVolModel(lam=0.8)` | decay factor | EWMA variance recursion |
| `RollingGARCHModel(window=24)` | months | GARCH(1,1) on sliding window |

### Alerts

| Class | Description |
|-------|-------------|
| `RiskDetector(high_threshold, med_threshold)` | Scans all drugs and classifies HIGH/MEDIUM/LOW |
| `detector.scan(df)` | Returns full risk table as DataFrame |
| `detector.filter_by_risk("HIGH")` | Filter results by risk level |

### Metrics

| Function | Description |
|----------|-------------|
| `rmse(y_true, y_pred)` | Root Mean Squared Error |
| `mae(y_true, y_pred)` | Mean Absolute Error |
| `mape(y_true, y_pred)` | Mean Absolute Percentage Error (%) |
