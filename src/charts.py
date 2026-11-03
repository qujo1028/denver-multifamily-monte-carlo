"""Charts saved as PNGs in outputs/charts/."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write files only, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

# Palette: one blue for the main series, orange only for comparisons.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_SOFT, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_SOFT,
    "axes.titlecolor": INK, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": INK_SOFT, "ytick.color": INK_SOFT, "font.size": 10,
    "legend.frameon": False,
})


def _save(fig, out_dir, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _bins(*series, n=60):
    """Shared bin edges covering the middle 99.8% of outcomes, so a few extreme
    scenarios don't squash the chart. Every scenario is still in the CSV."""
    lo = min(s.quantile(0.001) for s in series)
    hi = max(s.quantile(0.999) for s in series)
    return np.linspace(lo, hi, n + 1)


def _percentile_lines(ax, values, fmt, y_text):
    """Dashed lines at P5 / median / P95, each labeled at the top."""
    for label, q in [("P5", 0.05), ("Median", 0.50), ("P95", 0.95)]:
        x = values.quantile(q)
        ax.axvline(x, color=INK, linestyle="--", linewidth=1.2)
        ax.text(x, y_text, f" {label}\n {fmt(x)}", color=INK, fontsize=9, va="top")


def irr_histogram(df, out_dir):
    irr = df["levered_irr_fixed"]
    fig, ax = plt.subplots(figsize=(9, 5))
    counts, _, _ = ax.hist(irr, bins=_bins(irr), color=BLUE, edgecolor=SURFACE, linewidth=0.5)
    _percentile_lines(ax, irr, lambda v: f"{v:.1%}", counts.max() * 1.12)
    ax.set_ylim(0, counts.max() * 1.15)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title(f"Levered IRR, fixed-rate loan ({len(df):,} simulations)")
    ax.set_xlabel("5-year levered IRR")
    ax.set_ylabel("Number of scenarios")
    ax.text(0.99, 0.80, f"P(IRR < 0%) = {(irr < 0).mean():.1%}", transform=ax.transAxes,
            ha="right", color=INK_SOFT)
    return _save(fig, out_dir, "irr_histogram.png")


def equity_multiple_histogram(df, out_dir):
    em = df["equity_multiple_fixed"]
    fig, ax = plt.subplots(figsize=(9, 5))
    counts, _, _ = ax.hist(em, bins=_bins(em), color=BLUE, edgecolor=SURFACE, linewidth=0.5)
    _percentile_lines(ax, em, lambda v: f"{v:.2f}x", counts.max() * 1.12)
    ax.axvline(1.0, color=ORANGE, linewidth=2)
    ax.text(1.0, counts.max() * 0.80, " 1.0x =\n money back", color=INK, fontsize=9, va="top")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.set_title(f"Equity multiple, fixed-rate loan ({len(df):,} simulations)")
    ax.set_xlabel("Equity multiple (total distributions / equity invested)")
    ax.set_ylabel("Number of scenarios")
    return _save(fig, out_dir, "equity_multiple_histogram.png")


def tornado_chart(tornado_df, out_dir):
    t = tornado_df.iloc[::-1]  # biggest swing on top
    base = t["base_irr"].iloc[0]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = range(len(t))

    # Each bar runs from base IRR to the IRR at that input's P10 / P90.
    low_left = t[["irr_at_low", "irr_at_high"]].min(axis=1)
    high_right = t[["irr_at_low", "irr_at_high"]].max(axis=1)
    ax.barh(y, base - low_left, left=low_left, color=ORANGE, height=0.6, label="Downside")
    ax.barh(y, high_right - base, left=base, color=BLUE, height=0.6, label="Upside")
    ax.axvline(base, color=INK, linewidth=1.2)

    def fmt_input(v):
        return f"{v:.2%}"

    labels = [f"{d}\n({fmt_input(lo)} to {fmt_input(hi)})"
              for d, lo, hi in zip(t["Driver"], t["low_input"], t["high_input"])]
    ax.set_yticks(list(y), labels)
    for yi, lo, hi in zip(y, low_left, high_right):
        ax.text(lo, yi, f"{lo:.1%} ", va="center", ha="right", fontsize=9, color=INK)
        ax.text(hi, yi, f" {hi:.1%}", va="center", ha="left", fontsize=9, color=INK)

    pad = (high_right.max() - low_left.min()) * 0.15
    ax.set_xlim(low_left.min() - pad, high_right.max() + pad)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.set_title(f"What drives levered IRR? (base case {base:.1%}, inputs at P10 / P90)")
    ax.set_xlabel("Levered IRR, fixed-rate loan")
    ax.legend(loc="lower right")
    return _save(fig, out_dir, "tornado.png")


def fixed_vs_floating(df, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = _bins(df["levered_irr_fixed"], df["levered_irr_floating"], n=70)
    for col, color, label in [("levered_irr_fixed", BLUE, "Fixed 6.25%"),
                              ("levered_irr_floating", ORANGE, "Floating (random rate)")]:
        ax.hist(df[col], bins=bins, histtype="step", linewidth=2, color=color,
                label=f"{label}: median {df[col].median():.1%}, P5 {df[col].quantile(0.05):.1%}")
    ax.axvline(0, color=INK_SOFT, linewidth=1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title("Fixed vs. floating-rate debt: same deal, wider risk")
    ax.set_xlabel("5-year levered IRR")
    ax.set_ylabel("Number of scenarios")
    ax.legend(loc="upper left")
    return _save(fig, out_dir, "fixed_vs_floating.png")


def make_all(df, tornado_df, out_dir):
    return [irr_histogram(df, out_dir), equity_multiple_histogram(df, out_dir),
            tornado_chart(tornado_df, out_dir), fixed_vs_floating(df, out_dir)]
