"""
drug_shortage_forecaster.utils.plotting
-----------------------------------------
Plotting helpers for shortage volatility visualisation.
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional


def plot_shortage_volatility(
    signal: pd.Series,
    forecasts: dict[str, pd.Series],
    drug_name: str = "",
    figsize: tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot shortage log-change signal and volatility forecasts.

    Parameters
    ----------
    signal : pd.Series
        Monthly log-change shortage series (the raw signal).
    forecasts : dict[str, pd.Series]
        Model name → volatility forecast series.
    drug_name : str, optional
        Drug name for the chart title.
    figsize : tuple, optional
        Figure dimensions in inches.
    save_path : str or None, optional
        If given, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not isinstance(signal, pd.Series):
        raise TypeError("signal must be a pandas Series")
    if not isinstance(forecasts, dict):
        raise TypeError("forecasts must be a dict")
    if not forecasts:
        raise ValueError("forecasts must contain at least one entry")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                    gridspec_kw={"height_ratios": [1, 1.5]})

    # Top panel: raw signal
    ax1.bar(signal.index, signal.values, color="#4C72B0", alpha=0.6, width=20)
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Log-Change\n(Shortage Signal)", fontsize=9)
    ax1.set_title(
        f"Drug Shortage Volatility — {drug_name.upper()}" if drug_name
        else "Drug Shortage Volatility",
        fontsize=13, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3)

    # Bottom panel: volatility forecasts
    colors = ["#DD4444", "#F5A623", "#2CA02C", "#9467BD"]
    for (name, series), color in zip(forecasts.items(), colors):
        ax2.plot(series.index, series.values, linewidth=1.8,
                 label=name, color=color)

    ax2.set_ylabel("Annualised Volatility", fontsize=9)
    ax2.set_xlabel("Date", fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=30, ha="right")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_risk_distribution(results: pd.DataFrame, figsize: tuple = (7, 4)) -> plt.Figure:
    """Bar chart of HIGH / MEDIUM / LOW drug counts.

    Parameters
    ----------
    results : pd.DataFrame
        Output from :meth:`RiskDetector.scan`.
    figsize : tuple, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not isinstance(results, pd.DataFrame):
        raise TypeError("results must be a pandas DataFrame")

    counts = results["risk_level"].value_counts().reindex(
        ["HIGH", "MEDIUM", "LOW"], fill_value=0
    )
    colors = ["#DD4444", "#F5A623", "#2CA02C"]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(val), ha="center", va="bottom", fontweight="bold")

    ax.set_title("Drugs by Shortage Risk Level", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Drugs")
    ax.set_ylim(0, counts.max() + 3)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
