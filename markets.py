"""S&P 500 and the Beige Book.

Part 1 (event study): does Beige Book tone, or our model's predicted rate decision, predict the S&P 500 return on the
FOMC announcement day that follows the release?
Part 2 (performance): buy-and-hold S&P 500 vs two simple rules that use only information available at each release.

Inputs: Data/SPX_daily.csv (Yahoo ^GSPC), Data/FOMC_dates.csv (fomc.py), out/csv/features.csv, out/csv/predictions.csv.
Returns are close-to-close log returns in percent. The index is price-only (no dividends).

Pre-declared, to limit forking paths:
  event study primary test  : announcement-day return ~ s_dist_z (tone; the original three-scorer district average)
  secondary tests            : dir_act_z, dir_price_z, predicted change, surprise (= actual - predicted), actual change
  timing rules               : R1 invested if s_dist_z > 0, R2 invested if the model predicts no cut (predicted change >= 0);
                               otherwise in cash at the effective fed funds rate. 10bp cost per switch. No tuning.
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

from predict import block_bootstrap
from rates import load_effective

warnings.filterwarnings("ignore")
EVENT_START, EVENT_END = "2000-02-01", "2020-06-30"   # FOMC issues a statement after every meeting from Feb 2000
MODEL = "ridge_g_lean_act"
COST = 0.0010


def load_spx():
    s = pd.read_csv("Data/SPX_daily.csv", parse_dates=["date"]).set_index("date")["close"]
    ret = 100 * np.log(s / s.shift(1))
    return s, ret.dropna()


def build_events(ret):
    """One row per release matched to the next scheduled FOMC meeting."""
    meet = pd.read_csv("Data/FOMC_dates.csv", parse_dates=["date"])
    meet = meet[meet["scheduled"]]["date"].reset_index(drop=True)
    rel = pd.read_csv("out/csv/dates.csv")
    rel = pd.to_datetime(rel[["year", "month", "day"]]).sort_values().reset_index(drop=True)
    feat = pd.read_csv("out/csv/features.csv", index_col=0, parse_dates=True)
    pred = pd.read_csv("out/csv/predictions.csv", index_col=0, parse_dates=True)
    days = ret.index
    cs = ret.cumsum()

    def td_on_or_after(d):
        i = days.searchsorted(d)
        return days[i] if i < len(days) else None

    rows = []
    for r in rel:
        w = meet[(meet > r + pd.Timedelta(days=7)) & (meet <= r + pd.Timedelta(days=25))]
        if not len(w) or r not in feat.index or r not in pred.index:
            continue
        d = td_on_or_after(w.iloc[0])
        r_td = td_on_or_after(r)
        i = days.get_loc(d)
        if d is None or r_td is None or i < 2:
            continue
        prev = days[i - 1]
        plc = td_on_or_after(d - pd.Timedelta(days=7))
        rows.append({"release": r, "meeting": w.iloc[0], "day": d,
                     "r0": ret.loc[d], "r_pre": cs.loc[prev] - cs.loc[r_td] if prev > r_td else 0.0,
                     "r_next": ret.iloc[i + 1] if i + 1 < len(days) else np.nan,
                     "r_placebo": ret.loc[plc] if plc is not None and plc != d else np.nan,
                     **{c: feat.loc[r, c] for c in ["s_dist_z", "s_nat_z", "dir_act_z", "dir_price_z", "lm_net_z"]},
                     "pred_bp": 100 * pred.loc[r, MODEL], "actual_bp": 100 * pred.loc[r, "chg"]})
    ev = pd.DataFrame(rows)
    ev["surprise_bp"] = ev["actual_bp"] - ev["pred_bp"]
    ev = ev[(ev["day"] >= EVENT_START) & (ev["day"] <= EVENT_END)].reset_index(drop=True)
    ev["crisis"] = (((ev["day"] >= "2008-09-01") & (ev["day"] <= "2009-06-30")) | (ev["day"] >= "2020-01-01")).astype(int)
    return ev


def ols(y, x, name):
    X = sm.add_constant(x)
    m = sm.OLS(y, X, missing="drop").fit(cov_type="HC3")
    return m


def perm_p(y, x, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    ok = ~(np.isnan(y) | np.isnan(x))
    y, x = y[ok], x[ok]
    xc = x - x.mean()
    b = (xc * (y - y.mean())).sum() / (xc ** 2).sum()
    bs = np.empty(B)
    for i in range(B):
        xp = rng.permutation(xc)
        bs[i] = (xp * (y - y.mean())).sum() / (xp ** 2).sum()
    return float((np.abs(bs) >= abs(b)).mean())


def event_study(ev):
    z = lambda s: (s - s.mean()) / s.std()
    ev = ev.copy()
    for c in ["s_dist_z", "dir_act_z", "dir_price_z", "pred_bp", "surprise_bp", "actual_bp"]:
        ev[c + "_s"] = z(ev[c])
    samples = {"2000-2020 (all)": ev, "excl. 2008-09 crisis & 2020": ev[ev["crisis"] == 0]}
    specs = [("tone: s_dist_z (PRIMARY)", "s_dist_z_s"), ("tone: dir_act_z", "dir_act_z_s"), ("tone: dir_price_z", "dir_price_z_s"),
             ("model-predicted change", "pred_bp_s"), ("actual change", "actual_bp_s"), ("surprise (actual - predicted)", "surprise_bp_s")]
    out = []
    for sname, d in samples.items():
        for label, xv in specs:
            m = ols(d["r0"], d[xv], label)
            out.append({"sample": sname, "outcome": "announcement-day return", "regressor": label, "n": int(m.nobs),
                        "coef_pct_per_sd": m.params[xv], "robust_se": m.bse[xv], "p": m.pvalues[xv], "r2": m.rsquared,
                        "spearman": spearmanr(d[xv], d["r0"]).statistic})
        # tone controlling for the market move since the release (tone co-moves with recent stock returns)
        X = sm.add_constant(d[["s_dist_z_s", "r_pre"]])
        m = sm.OLS(d["r0"], X).fit(cov_type="HC3")
        out.append({"sample": sname, "outcome": "announcement-day return", "regressor": "tone, controlling for return since release",
                    "n": int(m.nobs), "coef_pct_per_sd": m.params["s_dist_z_s"], "robust_se": m.bse["s_dist_z_s"],
                    "p": m.pvalues["s_dist_z_s"], "r2": m.rsquared, "spearman": np.nan})
        # placebo: same regression on a day a week earlier, and the day after the announcement
        for col, lab in [("r_placebo", "PLACEBO: day 7 days before"), ("r_next", "day after announcement")]:
            dd = d.dropna(subset=[col])
            m = ols(dd[col], dd["s_dist_z_s"], lab)
            out.append({"sample": sname, "outcome": lab, "regressor": "tone: s_dist_z", "n": int(m.nobs),
                        "coef_pct_per_sd": m.params["s_dist_z_s"], "robust_se": m.bse["s_dist_z_s"], "p": m.pvalues["s_dist_z_s"],
                        "r2": m.rsquared, "spearman": spearmanr(dd["s_dist_z_s"], dd[col]).statistic})
        # magnitude of the move
        m = ols(d["r0"].abs(), d["surprise_bp_s"].abs(), "abs")
        out.append({"sample": sname, "outcome": "absolute announcement-day return", "regressor": "|surprise|", "n": int(m.nobs),
                    "coef_pct_per_sd": m.params["surprise_bp_s"], "robust_se": m.bse["surprise_bp_s"], "p": m.pvalues["surprise_bp_s"],
                    "r2": m.rsquared, "spearman": spearmanr(d["surprise_bp_s"].abs(), d["r0"].abs()).statistic})
    res = pd.DataFrame(out)
    res["perm_p"] = np.nan
    for sname, d in samples.items():
        mask = (res["sample"] == sname) & (res["regressor"] == "tone: s_dist_z (PRIMARY)")
        res.loc[mask, "perm_p"] = perm_p(d["r0"].values, d["s_dist_z"].values)
    return res, ev


def timing(ev_all, spx, ret):
    """Buy-and-hold vs two rules, release to release. Position set from info at the release, entered at the next close."""
    eff = load_effective()
    pred = pd.read_csv("out/csv/predictions.csv", index_col=0, parse_dates=True)
    feat = pd.read_csv("out/csv/features.csv", index_col=0, parse_dates=True)
    rel = pred.index[pred.index.isin(feat.index)]
    days = ret.index
    entry = [days[days.searchsorted(r) + 1] for r in rel if days.searchsorted(r) + 1 < len(days)]
    rel = rel[:len(entry)]
    cash = pd.Series([eff.reindex([d.to_period("M").to_timestamp()]).ffill().iloc[0] / 100 / 252 for d in days], index=days)
    sig = {"Buy and hold S&P 500": pd.Series(1, index=rel),
           "R1: tone rule (invested if tone > 0)": (feat.loc[rel, "s_dist_z"] > 0).astype(int),
           "R2: model rule (invested unless a cut is predicted)": (pred.loc[rel, MODEL] >= 0).astype(int)}
    start, end = entry[0], entry[-1]
    window = days[(days > start) & (days <= end)]
    curves, rows, perper, sub = {}, [], {}, []
    simple = np.exp(ret / 100) - 1
    periods = {"full": (None, None), "1995-2007": (None, "2007-12-31"), "2008-2020": ("2008-01-01", None),
               "2003-2007": ("2003-01-01", "2007-09-30"), "2009-2019": ("2009-01-01", "2019-12-31")}
    for name, w in sig.items():
        pos = pd.Series(0.0, index=window)
        for i, r in enumerate(rel[:-1]):
            seg = (window > entry[i]) & (window <= entry[i + 1])
            pos[seg] = w.iloc[i]
        sw = pos.diff().abs().fillna(0)
        daily = pos * simple.loc[window] + (1 - pos) * cash.loc[window] - sw * COST
        for pname, (a, b) in periods.items():                 # same metrics over sub-periods
            d = daily.loc[a:b]
            lvl = (1 + d).cumprod()
            yrs_p = (d.index[-1] - d.index[0]).days / 365.25
            sub.append({"period": pname, "strategy": name, "cagr_pct": 100 * (lvl.iloc[-1] ** (1 / yrs_p) - 1),
                        "vol_pct": 100 * d.std() * np.sqrt(252), "sharpe": (d - cash.loc[d.index]).mean() / d.std() * np.sqrt(252),
                        "max_drawdown_pct": 100 * (lvl / lvl.cummax() - 1).min(), "pct_days_invested": 100 * pos.loc[a:b].mean(),
                        "switches": int(sw.loc[a:b].sum())})
        curves[name] = 100 * (1 + daily).cumprod()
        yrs = (window[-1] - window[0]).days / 365.25
        exc = daily - cash.loc[window]
        dd = curves[name] / curves[name].cummax() - 1
        rows.append({"strategy": name, "total_return_pct": curves[name].iloc[-1] - 100,
                     "cagr_pct": 100 * ((curves[name].iloc[-1] / 100) ** (1 / yrs) - 1), "vol_pct": 100 * daily.std() * np.sqrt(252),
                     "sharpe": exc.mean() / daily.std() * np.sqrt(252), "max_drawdown_pct": 100 * dd.min(),
                     "pct_days_invested": 100 * pos.mean(), "switches": int(sw.sum())})
        perper[name] = np.array([np.log1p(daily[(window > entry[i]) & (window <= entry[i + 1])]).sum() for i in range(len(rel) - 1)]) * 100
    tab = pd.DataFrame(rows).set_index("strategy")
    bh = perper["Buy and hold S&P 500"]
    ci = {}
    for name in list(sig)[1:]:
        m, c, p = block_bootstrap(perper[name] - bh)
        ci[name] = (m, c[0], c[1], p)
    return tab, pd.DataFrame(curves), ci, (window[0], window[-1]), pd.DataFrame(sub)


def plots(ev, curves):
    from plots_ffr import BLUE, ORANGE, AQUA, INK, MUTED, style
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor="#fcfcfb")
    cols = {"Buy and hold S&P 500": INK, "R1: tone rule (invested if tone > 0)": BLUE,
            "R2: model rule (invested unless a cut is predicted)": ORANGE}
    for name, c in cols.items():
        ax.plot(curves.index, curves[name], color=c, lw=1.8, label=name)
    ax.set_yscale("log")
    ax.set_yticks([50, 100, 200, 300, 400, 600])
    ax.get_yaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_ylabel("Growth of 100, log scale")
    ax.set_title("S&P 500 (price index) vs. two Beige Book rules", loc="left", color=INK, fontsize=13, pad=10)
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=MUTED)
    style(ax)
    fig.tight_layout()
    fig.savefig("out/figs/spx_vs_rules.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), facecolor="#fcfcfb", sharey=True)
    for ax, (col, xl) in zip(axs, [("s_dist_z", "Beige Book tone at release (z-score)"),
                                   ("surprise_bp", "Surprise: actual minus model-predicted change (bp)")]):
        ax.scatter(ev[col], ev["r0"], s=22, color=BLUE, alpha=0.75, edgecolor="white", lw=0.5)
        b = np.polyfit(ev[col], ev["r0"], 1)
        xs = np.linspace(ev[col].min(), ev[col].max(), 50)
        ax.plot(xs, np.polyval(b, xs), color=ORANGE, lw=1.8)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlabel(xl)
        style(ax)
    axs[0].set_ylabel("S&P 500 return on announcement day, %")
    fig.suptitle("Announcement-day S&P 500 returns, scheduled FOMC meetings 2000-2020", x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig("out/figs/fomc_day_returns.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    spx, ret = load_spx()
    ev = build_events(ret)
    ev.to_csv("out/csv/event_study.csv", index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
    print(f"{len(ev)} scheduled FOMC announcement days, {ev['day'].min().date()} to {ev['day'].max().date()}")
    other = ret[(ret.index >= EVENT_START) & (ret.index <= EVENT_END) & ~ret.index.isin(ev["day"])]
    print(f"announcement-day mean return {ev['r0'].mean():.3f}% (sd {ev['r0'].std():.2f}, up {100*(ev['r0']>0).mean():.0f}% of days) | "
          f"all other days {other.mean():.3f}% (sd {other.std():.2f}, up {100*(other>0).mean():.0f}%)")
    res, ev2 = event_study(ev)
    res.to_csv("out/csv/event_regressions.csv", index=False)
    print(res.to_string(index=False))
    stats = pd.DataFrame([
        {"group": "announce", "n": len(ev), "mean_pct": ev["r0"].mean(), "sd_pct": ev["r0"].std(), "pct_up": 100 * (ev["r0"] > 0).mean()},
        {"group": "other", "n": len(other), "mean_pct": other.mean(), "sd_pct": other.std(), "pct_up": 100 * (other > 0).mean()}])
    stats["t_vs_other"] = [(ev["r0"].mean() - other.mean()) / np.sqrt(ev["r0"].var() / len(ev) + other.var() / len(other)), np.nan]
    stats.to_csv("out/csv/announcement_stats.csv", index=False)
    tab, curves, ci, span, sub = timing(ev, spx, ret)
    tab.to_csv("out/csv/timing_strategies.csv")
    sub.to_csv("out/csv/timing_subperiods.csv", index=False)
    pd.DataFrame([{"strategy": k, "mean_gain_pp": m, "ci_lo": lo, "ci_hi": hi, "p_no_gain": p} for k, (m, lo, hi, p) in ci.items()]
                 ).to_csv("out/csv/timing_bootstrap.csv", index=False)
    # the equity curves are not saved: the buy-and-hold column is the S&P 500 price index rescaled, and the source data is
    # not redistributable. The figure is regenerated by running this script.
    print(f"\nTiming comparison, {span[0].date()} to {span[1].date()} (price-only S&P, cash at effective fed funds rate, {int(COST*1e4)}bp per switch)")
    print(tab.to_string())
    for k, (m, lo, hi, p) in ci.items():
        print(f"{k}: mean log-return gain vs buy-and-hold per release interval {m:+.3f}pp  95% CI [{lo:+.3f}, {hi:+.3f}]  P(<=0)={p:.2f}")
    plots(ev2, curves)
