"""Build the dashboard's JSON data files from the model results in out/csv/.

Run from the repo root, after predict.py:
    python3 dashboard/build_data.py dashboard/data

Writes releases / predictions / period_scores / gains / scorers .json ({rows, meta}).
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, os.getcwd())
from predict import block_bootstrap, to_label  # noqa: E402

warnings.filterwarnings("ignore")
out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)

PERIODS = {
    "all": ("1995-01-01", "2020-12-31"),
    "ex2020": ("1995-01-01", "2019-12-31"),
    "pre2008": ("1995-01-01", "2007-12-31"),
    "zlb": ("2009-01-01", "2019-12-31"),
}
MODELS = ["hold", "last_move", "ridge_state", "ridge_both", "ridge_lean_act", "ridge_g_both", "ridge_g_lean_act",
          "ridge_g_lean_old", "gbm_state", "gbm_g_both", "gbm_g_lean_act"]

pred = pd.read_csv("out/csv/predictions.csv", index_col=0, parse_dates=True)
feat = pd.read_csv("out/csv/features.csv", index_col=0, parse_dates=True)

predict_code = open("predict.py").read()
meta_model = [
    {"label": "Method", "value": "Expanding-window walk-forward: for each Beige Book release from 1995-01, models are fit only on earlier "
                                "releases and predict the target change at the next FOMC meeting (target 21 days after release minus target "
                                "on the release date, upper bound after Dec 2008). Ridge regression, scaled features. Predictions are floored "
                                "so no cut is predicted when the target is at the 0.25% bound."},
    {"label": "Inputs", "value": "Beige Book district text 1970-2020 (Minneapolis Fed archive); FRED FEDFUNDS, DFEDTAR, DFEDTARU. Sentiment scored with VADER, "
                                 "TextBlob, Flair, Loughran-McDonald and hand-written directional word lists."},
    {"label": "predict.py", "value": predict_code, "type": "code", "lang": "python"},
]


def write(name, rows, meta=None):
    with open(os.path.join(out_dir, name), "w") as f:
        json.dump({"rows": rows, "meta": meta or []}, f, separators=(",", ":"))
    print(name, len(rows))


# 1. releases: one row per release 1983-2020
rel = feat.copy()
rel["chg_bp"] = (rel["chg"] * 100).round(1)
rel = rel.reset_index().rename(columns={"index": "date"})
cols = ["date", "effective", "rate_now", "chg_bp", "label", "zlb", "s_dist_z", "dir_act_z", "dir_price_z", "lm_net_z"]
rel["date"] = rel["date"].dt.strftime("%Y-%m-%d")
rel = rel[cols].copy()
rel["label"] = rel["label"].astype(str)
for c in ["effective", "rate_now", "s_dist_z", "dir_act_z", "dir_price_z", "lm_net_z"]:
    rel[c] = rel[c].round(3)
write("releases.json", json.loads(rel.to_json(orient="records")),
      [{"label": "Method", "value": "One row per Beige Book release. Sentiment columns are district averages z-scored against a trailing 40-release window (no look-ahead). "
                                   "chg_bp is the change in the announced target (upper bound after 2008) between the release date and 21 days later."}])

# 2. predictions (bp)
p = pred.copy()
out = pd.DataFrame({"date": p.index.strftime("%Y-%m-%d"), "label": p["label"].astype(str).values,
                    "rate_now": p["rate_now"].round(3).values, "chg_bp": (p["chg"] * 100).round(1).values})
for m in MODELS:
    out[m] = (p[m] * 100).round(2).values
write("predictions.json", json.loads(out.to_json(orient="records")), meta_model)

# 3. period scores
rows = []
for per, (a, b) in PERIODS.items():
    sub = pred.loc[a:b]
    y = sub["chg"].values
    sse0 = np.sum(y ** 2)
    yl = sub["label"].astype(str).values
    for m in MODELS:
        pm = sub[m].values
        rows.append({"period": per, "model": m, "n": int(len(sub)),
                     "mae_bp": round(100 * float(np.mean(np.abs(y - pm))), 2),
                     "r2_vs_hold": round(float(1 - np.sum((y - pm) ** 2) / sse0), 3),
                     "spearman": round(float(spearmanr(pm, y).statistic), 3) if np.std(pm) > 0 else None,
                     "bal_acc": round(float(balanced_accuracy_score(yl, to_label(pm))), 3)})
write("period_scores.json", rows, meta_model)

# 4. bootstrap gains
PAIRS = [("ridge_state", "ridge_both"), ("ridge_state", "ridge_lean_act"), ("ridge_state", "ridge_g_both"),
         ("ridge_state", "ridge_g_lean_act"), ("ridge_both", "ridge_g_both"), ("ridge_lean_act", "ridge_g_lean_act")]
rows = []
for per, (a, b) in PERIODS.items():
    sub = pred.loc[a:b]
    y = sub["chg"]
    for base, alt in PAIRS:
        d = (((y - sub[base]) ** 2) - ((y - sub[alt]) ** 2)).values * 1e4
        m, ci, pv = block_bootstrap(d)
        rows.append({"period": per, "base": base, "alt": alt, "n": int(len(sub)), "gain_bp2": round(float(m), 1),
                     "ci_lo": round(float(ci[0]), 1), "ci_hi": round(float(ci[1]), 1), "p_no_gain": round(float(pv), 3)})
write("gains.json", rows, meta_model + [{"label": "Bootstrap", "value": "Circular block bootstrap, block length 6 releases, 4000 resamples, seed 0. "
                                                                          "gain = mean squared-error reduction of alt vs base, in bp^2 per release. p_no_gain = share of resamples with gain <= 0."}])

# 5. scorer correlations with the next decision / ~6 months ahead
f = feat.copy()
f["fwd6_bp"] = (f["rate_now"].shift(-4) - f["rate_now"]) * 100
f["chg_bp"] = f["chg"] * 100
SC = ["s_dist_z", "s_nat_z", "lm_net_z", "dir_act_z", "dir_price_z"]
rows = []
for per, (a, b) in PERIODS.items():
    sub = f.loc[a:b]
    for s in SC:
        rows.append({"period": per, "scorer": s, "n": int(len(sub)),
                     "rho_next": round(float(spearmanr(sub[s], sub["chg_bp"]).statistic), 3),
                     "rho_6m": round(float(spearmanr(sub[s], sub["fwd6_bp"], nan_policy="omit").statistic), 3)})
write("scorers.json", rows, [{"label": "Method", "value": "Spearman rank correlation between each sentiment scorer (district average, z-scored vs trailing window) at a release "
                                                          "and (rho_next) the target change at the next meeting, (rho_6m) the change in the target over the following four releases."}])


# ---- S&P 500 section (outputs of markets.py) ----
meta_sp = [{"label": "Method", "value": "Scheduled FOMC announcement days, Feb 2000 to Jun 2020 (dates from the Fed's historical meeting pages). Returns are close-to-close log "
                                          "returns of the S&P 500 price index, in percent. Tone and the model's predicted change come from the Beige Book published about two "
                                          "weeks before each meeting. Surprise = actual minus model-predicted change, a proxy only (not market-implied). Regressions use "
                                          "heteroskedasticity-robust (HC3) standard errors on standardised regressors."},
           {"label": "markets.py", "value": open("markets.py").read(), "type": "code", "lang": "python"}]

ev = pd.read_csv("out/csv/event_study.csv", parse_dates=["day"])
ev["day"] = ev["day"].dt.strftime("%Y-%m-%d")
cols = ["day", "s_dist_z", "dir_act_z", "dir_price_z", "pred_bp", "actual_bp", "surprise_bp", "r0", "crisis"]
ev = ev[cols].copy()
for c in cols[1:-1]:
    ev[c] = ev[c].round(3)
write("events.json", json.loads(ev.to_json(orient="records")), meta_sp)

reg = pd.read_csv("out/csv/event_regressions.csv")
reg["sample"] = reg["sample"].map({"2000-2020 (all)": "all", "excl. 2008-09 crisis & 2020": "excl"})
reg = reg.rename(columns={"coef_pct_per_sd": "coef", "robust_se": "se"})
for c in ["coef", "se", "p", "r2", "spearman", "perm_p"]:
    reg[c] = reg[c].round(4)
write("event_regs.json", json.loads(reg.where(reg.notna(), None).to_json(orient="records")), meta_sp)

st = pd.read_csv("out/csv/announcement_stats.csv").round(4)
write("event_stats.json", json.loads(st.where(st.notna(), None).to_json(orient="records")), meta_sp)

sid = {"Buy and hold S&P 500": "bh", "R1: tone rule (invested if tone > 0)": "r1", "R2: model rule (invested unless a cut is predicted)": "r2"}
tm = pd.read_csv("out/csv/timing_subperiods.csv")
tm["strategy"] = tm["strategy"].map(sid)
tm = tm.round(3)
meta_t = [{"label": "Method", "value": "Buy and hold vs two pre-declared rules, release to release, 1995-01 to 2020-05. R1 is invested if the district tone z-score is above zero. "
                                          "R2 is invested unless the model predicts a cut. Otherwise cash at the effective fed funds rate. The position is set from information "
                                          "at each release and entered at the next close; 10bp cost per switch. The S&P 500 is a price index (no dividends). Not investment advice."},
          {"label": "markets.py", "value": open("markets.py").read(), "type": "code", "lang": "python"}]
write("timing.json", json.loads(tm.to_json(orient="records")), meta_t)
bt = pd.read_csv("out/csv/timing_bootstrap.csv")
bt["strategy"] = bt["strategy"].map(sid)
bt = bt.round(3)
write("timing_boot.json", json.loads(bt.to_json(orient="records")),
      meta_t + [{"label": "Bootstrap", "value": "Circular block bootstrap (6 release intervals, 4000 resamples, seed 0) of the per-interval log-return difference, rule minus buy and hold, in percentage points."}])
