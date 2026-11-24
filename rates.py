"""Fed funds rate series: effective rate and (announced) target rate.

Sources (all from FRED, saved in Data/):
  FEDFUNDS.csv   monthly average effective rate, 1954-
  DFEDTAR.csv    daily target, 1982-09-27 to 2008-12-15 (single number)
  DFEDTARU.csv   daily target upper bound, 2008-12-16 -
  DFEDTARL.csv   daily target lower bound, 2008-12-16 -

Caveat: the FOMC only began announcing its target publicly in Feb 1994.
DFEDTAR before that is the series FRED/the Fed reconstructed from records.
Before 1982-09 there is no target at all, so we fall back on the effective rate.
"""
import pandas as pd


def load_effective():
    df = pd.read_csv("Data/FEDFUNDS.csv", parse_dates=["observation_date"])
    return df.rename(columns={"observation_date": "date", "FEDFUNDS": "effective"}).set_index("date")["effective"]


def load_target():
    """Daily announced target (upper bound after 2008). Index: calendar days."""
    old = pd.read_csv("Data/DFEDTAR.csv", parse_dates=["observation_date"], index_col=0)["DFEDTAR"]
    new = pd.read_csv("Data/DFEDTARU.csv", parse_dates=["observation_date"], index_col=0)["DFEDTARU"]
    s = pd.concat([old, new.loc[new.index > old.index.max()]])
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s.asfreq("D").ffill().rename("target")


def target_on(target, dates):
    """Target in effect on each date (NaN before the series starts)."""
    return target.reindex(pd.DatetimeIndex(dates), method="ffill")


if __name__ == "__main__":
    t = load_target()
    print(t.describe())
    print(t.loc["2008-12-10":"2008-12-20"])
