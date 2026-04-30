"""
app.py
-------
Streamlit dashboard for the Drug Shortage Early Warning System.

Run locally:
    streamlit run app.py

Deploy free:
    Push to GitHub → connect at share.streamlit.io
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drug_shortage_forecaster.data.fetcher import fetch_shortage_data
from drug_shortage_forecaster.data.processor import (
    build_shortage_series, build_activity_counts, list_drugs
)
from drug_shortage_forecaster.models.historical import HistoricalVolModel
from drug_shortage_forecaster.models.ewma import EWMAVolModel
from drug_shortage_forecaster.alerts.detector import RiskDetector
from drug_shortage_forecaster.utils.plotting import (
    plot_shortage_volatility, plot_risk_distribution
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Drug Shortage Early Warning",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .risk-high   { background:#fde8e8; border-left:4px solid #DD4444;
                   padding:10px 14px; border-radius:6px; font-weight:600; }
    .risk-medium { background:#fef6e4; border-left:4px solid #F5A623;
                   padding:10px 14px; border-radius:6px; font-weight:600; }
    .risk-low    { background:#e8f5e9; border-left:4px solid #2CA02C;
                   padding:10px 14px; border-radius:6px; font-weight:600; }
    .metric-card { background:#f0f4ff; border-radius:8px;
                   padding:12px 18px; text-align:center; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/"
             "FDA_logo_2016.svg/200px-FDA_logo_2016.svg.png", width=100)
    st.title("⚙️ Settings")

    st.markdown("### Data")
    n_records = st.slider("Records to fetch from FDA", 200, 2000, 500, step=100)
    status_filter = st.selectbox(
        "Shortage status", ["All", "active", "resolved", "discontinued"]
    )

    st.markdown("### Models")
    ewma_lam  = st.slider("EWMA decay (λ)", 0.5, 0.99, 0.80, step=0.01)
    hv_window = st.slider("HV window (months)", 3, 18, 6, step=1)

    st.markdown("### Risk Thresholds")
    high_thresh = st.slider("HIGH risk threshold", 1.0, 3.0, 1.5, step=0.1)
    med_thresh  = st.slider("MEDIUM risk threshold", 0.3, 1.4, 0.75, step=0.05)

    st.markdown("### Filters")
    min_records = st.slider("Min shortage records per drug", 2, 10, 3)

    run_scan = st.button("🔍 Run Scan", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption("Data: [FDA openFDA API](https://open.fda.gov/apis/drug/shortages/)")
    st.caption("Built for ORIE 5270 — Cornell University")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("💊 Drug Shortage Early Warning System")
st.markdown(
    "Monitors FDA drug shortage data and forecasts which drugs are at **elevated risk** "
    "of supply disruption using volatility modelling (EWMA + Historical Volatility)."
)
st.markdown("---")

# ── Session state ─────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "results" not in st.session_state:
    st.session_state.results = None

# ── Fetch + Scan ──────────────────────────────────────────────────────────────
if run_scan:
    with st.spinner("📡 Fetching data from FDA API..."):
        try:
            status_arg = None if status_filter == "All" else status_filter
            df = fetch_shortage_data(limit=n_records, status=status_arg)
            st.session_state.df = df
        except Exception as e:
            st.error(f"Failed to fetch FDA data: {e}")
            st.stop()

    with st.spinner("🔬 Running volatility scan across all drugs..."):
        try:
            detector = RiskDetector(
                high_threshold=high_thresh,
                med_threshold=med_thresh,
                min_records=min_records,
                ewma_lam=ewma_lam,
                hv_window=hv_window,
            )
            results = detector.scan(st.session_state.df)
            st.session_state.results = results
            st.success(f"✅ Scan complete — {len(results)} drugs analysed.")
        except Exception as e:
            st.error(f"Scan failed: {e}")
            st.stop()

# ── Main content ──────────────────────────────────────────────────────────────
if st.session_state.results is not None:
    results = st.session_state.results
    df      = st.session_state.df

    # ── Summary metrics ──────────────────────────────────────────────────────
    n_high   = (results["risk_level"] == "HIGH").sum()
    n_med    = (results["risk_level"] == "MEDIUM").sum()
    n_low    = (results["risk_level"] == "LOW").sum()
    n_total  = len(results)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Drugs Scanned", n_total)
    c2.metric("🔴 HIGH Risk",   n_high)
    c3.metric("🟡 MEDIUM Risk", n_med)
    c4.metric("🟢 LOW Risk",    n_low)
    st.markdown("---")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊 Risk Overview", "🔍 Drug Deep Dive", "📥 Raw Data"])

    # ── Tab 1: Risk Overview ─────────────────────────────────────────────────
    with tab1:
        col_left, col_right = st.columns([1.2, 1])

        with col_left:
            st.subheader("Drug Risk Table")
            risk_filter = st.multiselect(
                "Filter by risk level",
                ["HIGH", "MEDIUM", "LOW"],
                default=["HIGH", "MEDIUM", "LOW"],
            )
            show = results[results["risk_level"].isin(risk_filter)].copy()

            # Colour-code risk column
            def _style_risk(val):
                colors = {"HIGH": "#fde8e8", "MEDIUM": "#fef6e4", "LOW": "#e8f5e9"}
                return f"background-color: {colors.get(val, 'white')}"

            styled = show.style.map(_style_risk, subset=["risk_level"])
            st.dataframe(styled, use_container_width=True, height=420)

            csv = show.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download CSV", csv,
                file_name="drug_shortage_risk.csv", mime="text/csv"
            )

        with col_right:
            st.subheader("Risk Distribution")
            fig_dist = plot_risk_distribution(results)
            st.pyplot(fig_dist, use_container_width=True)
            plt.close(fig_dist)

            st.subheader("Top 10 Highest Risk Drugs")
            top10 = results.head(10)[["drug_name", "current_vol", "risk_level"]]
            for _, row in top10.iterrows():
                lvl = row["risk_level"]
                css = {"HIGH": "risk-high", "MEDIUM": "risk-medium", "LOW": "risk-low"}[lvl]
                emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[lvl]
                st.markdown(
                    f'<div class="{css}">'
                    f'{emoji} <b>{row["drug_name"]}</b> — vol: {row["current_vol"]:.3f}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.write("")

    # ── Tab 2: Drug Deep Dive ─────────────────────────────────────────────────
    with tab2:
        st.subheader("🔍 Individual Drug Analysis")

        available_drugs = list_drugs(df, min_records=min_records)
        if not available_drugs:
            st.warning("No drugs with sufficient records. Try lowering 'Min shortage records'.")
        else:
            search = st.text_input("Search drug name", "")
            filtered_drugs = [d for d in available_drugs
                              if search.upper() in d] if search else available_drugs

            selected = st.selectbox("Select a drug", filtered_drugs)

            if selected:
                # Build signal
                try:
                    signal = build_shortage_series(df, selected)
                    counts = build_activity_counts(df, selected)
                except ValueError as e:
                    st.error(str(e))
                    st.stop()

                # Risk badge
                drug_row = results[results["drug_name"] == selected]
                if not drug_row.empty:
                    lvl = drug_row.iloc[0]["risk_level"]
                    vol = drug_row.iloc[0]["current_vol"]
                    emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[lvl]
                    css   = {"HIGH": "risk-high", "MEDIUM": "risk-medium", "LOW": "risk-low"}[lvl]
                    st.markdown(
                        f'<div class="{css}">'
                        f'{emoji} Risk Level: <b>{lvl}</b> &nbsp;|&nbsp; '
                        f'Current Volatility: <b>{vol:.3f}</b>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.write("")

                # Fit models
                forecasts = {}
                try:
                    hv = HistoricalVolModel(window=hv_window)
                    forecasts[f"HV(w={hv_window})"] = hv.fit(signal)
                except Exception:
                    pass
                try:
                    ewma = EWMAVolModel(lam=ewma_lam)
                    forecasts[f"EWMA(λ={ewma_lam})"] = ewma.fit(signal)
                except Exception:
                    pass

                if forecasts:
                    fig = plot_shortage_volatility(signal, forecasts, drug_name=selected)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    st.warning("Not enough data to fit volatility models for this drug.")

                # Shortage history table
                st.subheader("Shortage History")
                drug_detail = df[
                    df["drug_name"].str.upper().str.strip() == selected
                ][["status", "shortage_start", "shortage_end", "reason", "dosage_form"]]
                st.dataframe(drug_detail.reset_index(drop=True), use_container_width=True)

    # ── Tab 3: Raw Data ───────────────────────────────────────────────────────
    with tab3:
        st.subheader("📋 Raw FDA Shortage Records")
        st.caption(f"{len(df):,} records fetched from FDA openFDA API")
        col_search, col_status = st.columns([2, 1])
        with col_search:
            raw_search = st.text_input("Filter by drug name", key="raw_search")
        with col_status:
            raw_status = st.selectbox(
                "Filter by status", ["All"] + list(df["status"].unique()), key="raw_status"
            )

        raw_show = df.copy()
        if raw_search:
            raw_show = raw_show[
                raw_show["drug_name"].str.upper().str.contains(raw_search.upper(), na=False)
            ]
        if raw_status != "All":
            raw_show = raw_show[raw_show["status"] == raw_status]

        st.dataframe(raw_show.reset_index(drop=True), use_container_width=True, height=500)
        csv_raw = raw_show.to_csv(index=False).encode()
        st.download_button("⬇️ Download Raw CSV", csv_raw,
                           file_name="fda_shortage_raw.csv", mime="text/csv")

else:
    # Welcome screen before first scan
    st.info(
        "👈 **Configure settings** in the sidebar and click **Run Scan** to begin.\n\n"
        "The app will:\n"
        "1. Fetch live shortage records from the FDA API\n"
        "2. Build a monthly shortage-activity signal for each drug\n"
        "3. Apply EWMA + Historical Volatility models\n"
        "4. Classify each drug as 🔴 HIGH / 🟡 MEDIUM / 🟢 LOW risk\n"
        "5. Display interactive charts and a downloadable risk report"
    )

    st.markdown("### How it works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📡 Live FDA Data**\n\nPulls directly from the FDA openFDA API — no manual downloads needed.")
    with col2:
        st.markdown("**📈 Volatility Modelling**\n\nApplies EWMA and Historical Volatility to the shortage signal, the same math used in finance for risk detection.")
    with col3:
        st.markdown("**🚨 Automatic Alerts**\n\nClassifies drugs into HIGH / MEDIUM / LOW risk tiers based on their current volatility reading.")
