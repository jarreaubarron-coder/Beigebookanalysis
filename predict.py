"""Predict the FOMC target-rate decision following each Beige Book, walk-forward.

For every release t from TEST_START on, models are fit ONLY on releases before t and then
predict release t (expanding window, refit each step) -- no look-ahead.

Feature sets:
  state      lagged policy variables only (previous moves, 12m change in the target)
  sentiment  Beige Book sentiment only
  both       state + sentiment
Baselines: "hold" (always predict no change), "last move" (repeat previous change).
"""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import build, ZLB_LEVEL

warnings.filterwarnings("ignore")

TEST_START = "1995-01-01"
MIN_TRAIN = 60
SENT = ["s_nat_z", "s_dist_z", "d_s_nat", "d_s_dist", "s_disp_z"]
STATE = ["prev_chg", "prev_chg2", "rate_chg_12m"]
LMT = ["lm_net_z", "d_lm_net"]                          # Loughran-McDonald net tone
DIR = ["dir_act_z", "dir_price_z", "d_dir_act", "d_dir_price"]  # directional tone, activity vs prices
SETS = {"state": STATE, "sentiment": SENT, "both": STATE + SENT,
        "lm": LMT, "dir": DIR, "new": LMT + DIR, "state_new": STATE + LMT + DIR,
        "all_sent": SENT + LMT + DIR, "state_all": STATE + SENT + LMT + DIR,
        # lean specs added AFTER seeing the results above (exploratory; no first-difference terms)
        "lean_act": STATE + ["dir_act_z", "dir_price_z"],
        "lean_old": STATE + ["s_dist_z", "dir_price_z"],
        # zero-lower-bound aware: regime dummy + sentiment features zeroed while the target is at the bound
        "g_state": STATE + ["zlb"],
        "g_both": STATE + ["zlb"] + [c + "_g" for c in SENT],
        "g_lean_act": STATE + ["zlb", "dir_act_z_g", "dir_price_z_g"],
        "g_lean_old": STATE + ["zlb", "s_dist_z_g", "dir_price_z_g"]}


def make_models():
    return {
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=30.0)),
        "gbm": lambda: GradientBoostingRegressor(n_estimators=80, max_depth=2, learning_rate=0.05,
                                                  subsample=0.8, random_state=0),
    }


def walk_forward(df, start=TEST_START):
    """Return DataFrame of out-of-sample predictions indexed by release date."""
    rows = []
    idx = np.where(df.index >= start)[0]
    for i in idx:
        train, test = df.iloc[:i], df.iloc[[i]]
        if len(train) < MIN_TRAIN:
            continue
        rate_now = test["rate_now"].iloc[0]
        floor = -max(rate_now - ZLB_LEVEL, 0.0)   # cannot cut below the bound: applied to every model
        rec = {"date": df.index[i], "chg": test["chg"].iloc[0], "label": test["label"].iloc[0],
               "rate_now": rate_now, "hold": 0.0, "last_move": max(test["prev_chg"].iloc[0], floor)}
        for set_name, cols in SETS.items():
            for m_name, mk in make_models().items():
                m = mk().fit(train[cols], train["chg"])
                rec[f"{m_name}_{set_name}"] = max(float(m.predict(test[cols])[0]), floor)
            # 3-class logit (cut/hold/hike); expected change = sum p * class mean change
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=500))
            y = train["label"].astype(str)
            if y.nunique() == 3:
                clf.fit(train[cols], y)
                lab = clf.predict(test[cols])[0]
                rec[f"logit_{set_name}"] = "hold" if (lab == "cut" and floor == 0.0) else lab
                p = clf.predict_proba(test[cols])[0]
                rec[f"logit_{set_name}_pcut"] = p[list(clf.classes_).index("cut")]
                rec[f"logit_{set_name}_phike"] = p[list(clf.classes_).index("hike")]
        rows.append(rec)
    return pd.DataFrame(rows).set_index("date")


def to_label(x, band=0.10):
    return pd.cut(pd.Series(x), [-np.inf, -band, band, np.inf], labels=["cut", "hold", "hike"]).astype(str).values


def evaluate(pred):
    y = pred["chg"].values
    sse0 = np.sum(y ** 2)
    rows = []
    for col in [c for c in pred.columns if c in ("hold", "last_move") or c.startswith(("ridge_", "gbm_"))]:
        p = pred[col].values
        rows.append({"model": col,
                     "MAE_bp": 100 * np.mean(np.abs(y - p)),
                     "R2_vs_hold": 1 - np.sum((y - p) ** 2) / sse0,
                     "spearman": spearmanr(p, y).statistic if np.std(p) > 0 else np.nan,
                     "dir_acc": np.mean(to_label(p) == pred["label"].astype(str).values)})
    for col in [c for c in pred.columns if c.startswith("logit_") and not c.endswith(("pcut", "phike"))]:
        yl = pred["label"].astype(str).values
        rows.append({"model": col, "MAE_bp": np.nan, "R2_vs_hold": np.nan,
                     "spearman": np.nan, "dir_acc": np.mean(pred[col].values == yl),
                     "bal_acc": balanced_accuracy_score(yl, pred[col].values)})
    out = pd.DataFrame(rows).set_index("model")
    # balanced accuracy for regressions too (3-class)
    for col in out.index:
        if col in pred.columns and not col.startswith("logit_"):
            out.loc[col, "bal_acc"] = balanced_accuracy_score(pred["label"].astype(str), to_label(pred[col].values))
    return out


def level_error(pred):
    """MAE (bp) of predicting the post-meeting target LEVEL = rate_now + predicted change."""
    lvl = pred["rate_now"] + pred["chg"]
    return {c: 100 * np.mean(np.abs(lvl - (pred["rate_now"] + pred[c]))) for c in
            ["hold", "last_move", "ridge_state", "ridge_sentiment", "ridge_both",
             "gbm_state", "gbm_sentiment", "gbm_both"]}


PERIODS = {"full": (None, None), "ex-2020": (None, "2019-12-31"), "2009-2019": ("2009-01-01", "2019-12-31")}
COMPARE = [("ridge_state", "ridge_both"), ("ridge_state", "ridge_lean_act"), ("ridge_both", "ridge_g_both"),
           ("ridge_lean_act", "ridge_g_lean_act"), ("ridge_state", "ridge_g_state"), ("ridge_state", "ridge_g_both"), ("ridge_state", "ridge_g_lean_act"),
           ("ridge_g_state", "ridge_g_both"), ("ridge_g_state", "ridge_g_lean_act"), ("ridge_g_state", "ridge_g_lean_old")]


def block_bootstrap(sq_diff, B=4000, block=6, seed=0):
    """Circular block bootstrap of the mean squared-error gain. Returns mean, 95% CI, P(gain <= 0)."""
    rng = np.random.default_rng(seed)
    n = len(sq_diff)
    means = []
    for _ in range(B):
        starts = rng.integers(0, n, int(np.ceil(n / block)))
        idx = np.concatenate([(s + np.arange(block)) % n for s in starts])[:n]
        means.append(sq_diff[idx].mean())
    means = np.array(means)
    return sq_diff.mean(), np.percentile(means, [2.5, 97.5]), (means <= 0).mean()


def robustness(pred):
    """R2-vs-hold by regime, and paired bootstrap of squared-error gains (in bp^2 per release)."""
    cols = ["ridge_state", "ridge_both", "ridge_lean_act", "ridge_g_state", "ridge_g_both", "ridge_g_lean_act",
            "ridge_g_lean_old", "gbm_state", "gbm_g_state", "gbm_g_both", "gbm_g_lean_act"]
    for name, (a, b) in PERIODS.items():
        sub = pred.loc[a:b]
        y = sub["chg"]
        r2 = {c: 1 - ((y - sub[c]) ** 2).sum() / (y ** 2).sum() for c in cols}
        print(f"\n[{name}] n={len(sub)}  R2 vs hold: " + "  ".join(f"{c}={v:+.2f}" for c, v in r2.items()))
        for base, alt in COMPARE:
            d = (((y - sub[base]) ** 2) - ((y - sub[alt]) ** 2)).values * 1e4
            m, ci, p = block_bootstrap(d)
            print(f"   {alt} vs {base}: {m:+.1f} bp2/release  95%CI [{ci[0]:+.1f},{ci[1]:+.1f}]  P(<=0)={p:.2f}")


if __name__ == "__main__":
    df = build()
    feats = sorted({c for cols in SETS.values() for c in cols})
    df = df.dropna(subset=feats + ["chg"])
    pred = walk_forward(df)
    pred.to_csv("out/csv/predictions.csv")
    res = evaluate(pred)
    res.to_csv("out/csv/model_scores.csv")
    pd.set_option("display.width", 160)
    print(f"{len(pred)} out-of-sample releases, {pred.index.min().date()} to {pred.index.max().date()}")
    print(pred["label"].value_counts().to_dict())
    print(res.round(3))
    robustness(pred)
