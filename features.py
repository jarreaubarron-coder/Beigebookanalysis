"""Build one row per Beige Book release: sentiment features + the FOMC outcome that follows.

Timing: the Beige Book is published ~2 weeks before an FOMC meeting. For each release at
date D we record the target in effect at D (known to the predictor) and the target in effect
HORIZON days later (capped before the next release) -- that captures the upcoming decision.
All features use only information available at D (rolling windows look backwards only).
"""
import numpy as np
import pandas as pd

from analysis import make_dataset, NORM
from rates import load_effective, load_target, target_on

HORIZON = 21          # days after release in which the next meeting's decision lands
HOLD_BAND = 0.10      # |change| <= 10bp counts as "hold"
ROLL = 40             # ~5 years of releases for rolling normalisation
DISTRICTS = ["at", "bo", "ch", "cl", "da", "kc", "mi", "ny", "ph", "ri", "sf", "sl"]


def sentiment_by_release():
    """Composite sentiment (mean of 3 normalised scorers) for national summary, district mean, dispersion."""
    df = make_dataset(printing=False)
    df["comp"] = df[NORM].mean(axis=1)
    wide = df.pivot(index="date", columns="region", values="comp").sort_index()
    out = pd.DataFrame(index=wide.index)
    out["s_nat"] = wide["su"]
    out["s_dist"] = wide[DISTRICTS].mean(axis=1)
    out["s_disp"] = wide[DISTRICTS].std(axis=1)
    # per-scorer national-summary scores, kept for the comparison plots
    for col in NORM:
        out[col + "_su"] = df[df.region == "su"].set_index("date")[col]
    return out.sort_index()


NEW = ["lm_net", "dir_act", "dir_price"]
ZLB_LEVEL = 0.25
GATED = ["s_nat_z", "s_dist_z", "d_s_nat", "d_s_dist", "s_disp_z", "dir_act_z", "dir_price_z"]


def lm_by_release():
    """District-mean of the LM / directional scores (lmscore.py), plus national-summary LM tone, per release date."""
    lm = pd.read_csv("out/csv/lm_scores.csv")
    dates = pd.read_csv("out/csv/dates.csv")
    lm = lm.merge(dates, on=["year", "month"])
    lm["date"] = pd.to_datetime(lm[["year", "month", "day"]])
    out = pd.DataFrame(index=sorted(lm["date"].unique()))
    dist = lm[lm.region.isin(DISTRICTS)]
    for col in NEW:
        out[col] = dist.groupby("date")[col].mean()
    out["lm_net_su"] = lm[lm.region == "su"].set_index("date")["lm_net"]
    return out


def build(start="1983-01-01"):
    s = sentiment_by_release().join(lm_by_release())
    for col in NEW:
        mu = s[col].rolling(ROLL, min_periods=12).mean().shift(1)
        sd = s[col].rolling(ROLL, min_periods=12).std().shift(1)
        s[col + "_z"] = (s[col] - mu) / sd
        s["d_" + col] = s[col].diff()
    eff = load_effective()
    tgt = load_target()

    # Rolling (backward-looking) z-scores remove the long-run drift in the scorers (e.g. TextBlob).
    for col in ["s_nat", "s_dist"]:
        mu = s[col].rolling(ROLL, min_periods=12).mean().shift(1)
        sd = s[col].rolling(ROLL, min_periods=12).std().shift(1)
        s[col + "_z"] = (s[col] - mu) / sd
        s["d_" + col] = s[col].diff()              # change vs previous release
        s[col + "_ma3"] = s[col].rolling(3).mean()
    s["s_disp_z"] = (s["s_disp"] - s["s_disp"].rolling(ROLL, min_periods=12).mean().shift(1)) / \
        s["s_disp"].rolling(ROLL, min_periods=12).std().shift(1)

    dates = s.index
    nxt = pd.Series(dates[1:].append(pd.DatetimeIndex([dates[-1] + pd.Timedelta(days=60)])), index=dates)
    after = np.minimum(dates + pd.Timedelta(days=HORIZON), nxt - pd.Timedelta(days=1))

    s["rate_now"] = target_on(tgt, dates).values
    s["rate_next"] = target_on(tgt, after).values
    # Zero lower bound: target (upper bound) at or below 25bp -> policy cannot respond to sentiment by cutting,
    # and held at the bound for years regardless of it. Gate sentiment features off in that regime.
    s["zlb"] = (s["rate_now"] <= ZLB_LEVEL).astype(float)
    for col in GATED:
        s[col + "_g"] = s[col] * (1 - s["zlb"])
    # effective rate (monthly avg) for the pre-target era / descriptive plots
    s["effective"] = eff.reindex(dates.to_period("M").to_timestamp()).values
    s["chg"] = s["rate_next"] - s["rate_now"]
    s["label"] = pd.cut(s["chg"], [-np.inf, -HOLD_BAND, HOLD_BAND, np.inf], labels=["cut", "hold", "hike"])

    # lagged policy state (information available at D)
    s["prev_chg"] = s["chg"].shift(1)
    s["prev_chg2"] = s["chg"].shift(2)
    s["rate_chg_12m"] = s["rate_now"] - target_on(tgt, dates - pd.Timedelta(days=365)).values

    s = s.loc[start:].dropna(subset=["rate_now", "rate_next"])
    return s


if __name__ == "__main__":
    d = build()
    d.to_csv("out/csv/features.csv")
    print(d[["s_nat", "s_dist_z", "rate_now", "rate_next", "chg", "label"]].tail(12))
    print(d["label"].value_counts())
    print(len(d), d.index.min(), d.index.max())
