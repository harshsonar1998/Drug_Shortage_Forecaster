#!/usr/bin/env python3
"""
scripts/run_alert_scan.py
--------------------------
Command-line drug shortage risk scan.

Usage
-----
    python scripts/run_alert_scan.py
    python scripts/run_alert_scan.py --limit 500 --status active
    python scripts/run_alert_scan.py --drug AMOXICILLIN
"""

import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import drug_shortage_forecaster as dsf


def main():
    parser = argparse.ArgumentParser(description="Drug Shortage Early Warning — CLI")
    parser.add_argument("--limit",  type=int, default=300, help="Records to fetch (default 300)")
    parser.add_argument("--status", default=None, help="Filter: active / resolved / discontinued")
    parser.add_argument("--drug",   default=None, help="Analyse a single drug by name")
    parser.add_argument("--output", default=None, help="Save risk table to CSV path")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  💊 Drug Shortage Early Warning System")
    print(f"{'='*60}")

    print(f"\n[1/3] Fetching FDA shortage data (limit={args.limit})...")
    df = dsf.fetch_shortage_data(limit=args.limit, status=args.status)
    print(f"      {len(df):,} records fetched.")

    if args.drug:
        print(f"\n[2/3] Analysing {args.drug.upper()}...")
        try:
            signal = dsf.build_shortage_series(df, args.drug)
            hv     = dsf.HistoricalVolModel(window=6).fit(signal)
            ewma   = dsf.EWMAVolModel(lam=0.8).fit(signal)
            print(f"      HV vol  (latest): {hv.iloc[-1]:.4f}")
            print(f"      EWMA vol (latest): {ewma.iloc[-1]:.4f}")
            dsf.plot_shortage_volatility(
                signal,
                {f"HV(w=6)": hv, "EWMA(λ=0.8)": ewma},
                drug_name=args.drug,
                save_path=f"{args.drug.replace(' ','_')}_volatility.png",
            )
            print(f"      Chart saved.")
        except ValueError as e:
            print(f"      Error: {e}")
        return

    print(f"\n[2/3] Running risk scan across all drugs...")
    detector = dsf.RiskDetector(min_records=3)
    results  = detector.scan(df, progress=True)
    print(f"      {len(results)} drugs assessed.\n")

    print(f"[3/3] Results\n")
    n_high = (results["risk_level"] == "HIGH").sum()
    n_med  = (results["risk_level"] == "MEDIUM").sum()
    n_low  = (results["risk_level"] == "LOW").sum()
    print(f"  🔴 HIGH Risk   : {n_high}")
    print(f"  🟡 MEDIUM Risk : {n_med}")
    print(f"  🟢 LOW Risk    : {n_low}")

    print(f"\n  Top 10 Highest-Risk Drugs:\n")
    print(f"  {'Drug':<30} {'Volatility':>10}  {'Risk':<8}  {'Shortages':>9}")
    print("  " + "-"*62)
    for _, row in results.head(10).iterrows():
        emoji = {"HIGH":"🔴","MEDIUM":"🟡","LOW":"🟢"}[row["risk_level"]]
        print(f"  {row['drug_name']:<30} {row['current_vol']:>10.4f}  "
              f"{emoji} {row['risk_level']:<6}  {row['n_shortages']:>9}")

    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\n  Results saved → {args.output}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
