"""Domain-specific Beige Book scoring -> out/csv/lm_scores.csv (one row per release x district).

Two families of scores, computed over the sentence-split text in txt-split/:

1. Loughran-McDonald (LM) finance dictionary via pysentiment2 (published, validated for financial text):
     lm_net  = (pos - neg) / (pos + neg)     net tone
     lm_unc  = uncertainty words per token   (not available in pysentiment2's LM wrapper -> litigious/uncertainty skipped)
     lm_neg_rate, lm_pos_rate  per token

2. Directional tone split by topic (hand-written lists below; NOT a published lexicon):
     Beige Book prose is mostly "X rose / Y declined". A sentence about prices/wages is scored separately
     from one about activity, because policy responds to both, in different ways:
       dir_act   = (up - down) / (up + down) over non-price sentences   (growth direction)
       dir_price = (up - down) / (up + down) over price/wage sentences  (inflation direction)
     Sentences with a negator ("not", "no", "little", "without") before a direction word flip its sign.
"""
import re
import pandas as pd
import pysentiment2 as ps

from tools import gen

PRICE = re.compile(r"\b(prices?|pricing|wages?|inflation\w*|costs?|compensation|salar\w+|raw material\w*|input)\b", re.I)
UP = {"increase", "increased", "increases", "increasing", "rose", "rise", "rises", "rising", "grew", "grow", "grows",
      "growing", "growth", "expand", "expanded", "expanding", "expansion", "improve", "improved", "improving",
      "improvement", "strong", "stronger", "strongest", "strengthen", "strengthened", "strengthening", "robust",
      "higher", "gain", "gains", "gained", "accelerate", "accelerated", "accelerating", "brisk", "healthy", "solid",
      "boom", "booming", "surge", "surged", "tight", "tighter", "tightening", "shortage", "shortages", "upward",
      "firm", "firmer", "advanced", "picked"}
DOWN = {"decrease", "decreased", "decreases", "decreasing", "decline", "declined", "declines", "declining",
        "fell", "fall", "falls", "falling", "fallen", "drop", "dropped", "drops", "dropping", "weak", "weaker",
        "weakened", "weakening", "weakness", "soft", "softer", "softened", "softening", "softness", "slow", "slowed",
        "slower", "slowing", "slowdown", "contract", "contracted", "contraction", "lower", "reduced", "reduce",
        "reduction", "reductions", "deteriorate", "deteriorated", "deteriorating", "sluggish", "downward", "slack",
        "recession", "layoffs", "cutbacks", "curtailed", "depressed", "dampened"}
NEGATORS = {"not", "no", "little", "without", "never", "neither", "nor", "hardly", "barely", "less"}
WORD = re.compile(r"[a-z']+")
LM = ps.LM()


def direction(sentence):
    """(up, down) counts for one sentence with simple negation flipping."""
    toks = WORD.findall(sentence.lower())
    up = down = 0
    for i, w in enumerate(toks):
        if w in UP or w in DOWN:
            neg = any(t in NEGATORS for t in toks[max(0, i - 3):i])
            is_up = (w in UP) != neg
            up += is_up
            down += not is_up
    return up, down


def score_file(path):
    with open(path, "r") as f:
        sentences = [s for s in f.read().split("\n") if s.strip()]
    toks = LM.tokenize(" ".join(sentences))
    sc = LM.get_score(toks)
    pos, neg, n = sc["Positive"], sc["Negative"], max(len(toks), 1)
    res = {"lm_net": (pos - neg) / (pos + neg) if pos + neg else 0.0, "lm_pos_rate": pos / n, "lm_neg_rate": neg / n}
    agg = {"act": [0, 0], "price": [0, 0]}
    for s in sentences:
        u, d = direction(s)
        key = "price" if PRICE.search(s) else "act"
        agg[key][0] += u
        agg[key][1] += d
    for key, (u, d) in agg.items():
        res[f"dir_{key}"] = (u - d) / (u + d) if u + d else 0.0
        res[f"n_{key}"] = u + d
    return res


def score_all():
    rows = []
    for year, month, region in gen(skip=True):
        path = f"txt-split/{year}/{month:02d}/{year}-{month:02d}-{region}.txt"
        rows.append({"year": year, "month": month, "region": region, **score_file(path)})
    df = pd.DataFrame(rows)
    df.to_csv("out/csv/lm_scores.csv", index=False)
    return df


if __name__ == "__main__":
    df = score_all()
    print(len(df), "rows")
    print(df.describe().T[["mean", "std", "min", "max"]].round(3))
    print(df[df.region == "su"].set_index(["year", "month"]).loc[(2008, 12):(2009, 1)])
