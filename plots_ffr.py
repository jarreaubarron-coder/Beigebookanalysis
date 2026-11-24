"""Figures comparing Beige Book sentiment with the fed funds rate, and out-of-sample model results.

Colours: validated reference categorical palette (blue / orange / aqua), light surface.
No dual axes: each measure gets its own panel on a shared time axis.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import add_recessions
from features import build
from rates import load_effective, load_target

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def style(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(axis="y", color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.label.set_color(MUTED)


def fig_sentiment_vs_rate(path="out/figs/sentiment_vs_ffr.png"):
    d = build(start="1970-01-01")
    eff, tgt = load_effective(), load_target()
    fig, axs = plt.subplots(3, 1, figsize=(11, 9), sharex=True, facecolor="#fcfcfb",
                            gridspec_kw={"height_ratios": [1.2, 1, 1]})
    # panel 1: rate
    ax = axs[0]
    ax.plot(eff.index, eff.values, color=BLUE, lw=1.6, label="Effective rate (monthly avg)")
    ax.plot(tgt.index, tgt.values, color=ORANGE, lw=1.6, label="Announced target (upper bound)")
    ax.set_ylabel("Fed funds rate, %")
    ax.set_title("Beige Book sentiment vs. the fed funds rate", loc="left", color=INK, fontsize=13, pad=10)
    ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=MUTED)
    # panel 2: smoothed sentiment level, rolling-normalised
    ax = axs[1]
    z = d["s_dist_z"]
    ax.axhline(0, color=GRID, lw=1)
    ax.plot(z.index, z.rolling(4, min_periods=1).mean(), color=AQUA, lw=1.6)
    ax.set_ylabel("District sentiment\n(z vs trailing 5y, 4-release avg)")
    # panel 3: next-meeting change
    ax = axs[2]
    c = d["chg"].dropna()
    c = c[c.index >= "1983-01-01"]
    ax.bar(c.index, c.values * 100, width=25, color=ORANGE)
    ax.set_ylabel("Target change after\nthe release, bp")
    for ax in axs:
        style(ax)
        add_recessions(ax)
    axs[2].set_xlim(pd.Timestamp("1970-01-01"), pd.Timestamp("2020-12-31"))
    axs[2].text(0.0, -0.22, "Shaded: NBER recessions. Target change only defined from 1983 (no announced target earlier).",
                transform=axs[2].transAxes, color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def fig_oos(path="out/figs/model_oos.png"):
    p = pd.read_csv("out/csv/predictions.csv", index_col=0, parse_dates=True)
    fig, axs = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, facecolor="#fcfcfb")
    ax = axs[0]
    ax.bar(p.index, p["chg"] * 100, width=25, color=GRID if False else "#b9b8b2", label="Actual change")
    ax.plot(p.index, p["ridge_g_lean_act"] * 100, color=BLUE, lw=1.4,
            label="Predicted (ridge: policy state + directional tone, ZLB-gated)")
    ax.set_ylabel("Target change, bp")
    ax.set_title("Out-of-sample predictions (walk-forward, refit at every release)", loc="left",
                 color=INK, fontsize=13, pad=10)
    ax.legend(frameon=False, loc="lower left", fontsize=9, labelcolor=MUTED)
    ax = axs[1]
    sq0 = p["chg"] ** 2
    for col, color, lab in [("ridge_state", ORANGE, "Policy state only"),
                            ("ridge_g_both", BLUE, "State + VADER/TextBlob/Flair sentiment (ZLB-gated)"),
                            ("ridge_g_lean_act", AQUA, "State + directional activity & price tone (ZLB-gated)")]:
        gain = (sq0 - (p["chg"] - p[col]) ** 2).cumsum() * 1e4
        ax.plot(gain.index, gain.values, color=color, lw=1.6, label=lab)
    ax.axhline(0, color=MUTED, lw=1)
    ax.axvspan(pd.Timestamp("2008-12-16"), pd.Timestamp("2015-12-15"), color="#b9b8b2", alpha=0.25, lw=0)
    ax.text(pd.Timestamp("2009-03-01"), 2000, "Zero lower bound", color=MUTED, fontsize=8)
    ax.set_ylabel("Cumulative squared-error\nadvantage over 'hold' (bp²)")
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=MUTED)
    for ax in axs:
        style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    fig_sentiment_vs_rate()
    fig_oos()
