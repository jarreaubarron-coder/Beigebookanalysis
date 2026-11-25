"""Fetch daily S&P 500 (^GSPC) history into Data/SPX_daily.csv (columns: date, open, high, low, close).

This uses Yahoo Finance's unofficial chart endpoint. Yahoo's terms restrict automated use and redistribution of its data,
so this is for personal research only and Data/SPX_daily.csv is git-ignored. If you have a licensed daily S&P 500 series,
write it to the same CSV format instead and skip this script.

    python3 fetch_spx.py
"""
import json
import urllib.request

import pandas as pd

URL = ("https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC?period1=754704000&period2=1609372800&interval=1d")  # 1993-12-01 .. 2020-12-31


def main():
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    d = json.load(urllib.request.urlopen(req, timeout=60))
    r = d["chart"]["result"][0]
    ts = pd.to_datetime(r["timestamp"], unit="s").tz_localize("UTC").tz_convert("America/New_York").normalize().tz_localize(None)
    q = r["indicators"]["quote"][0]
    df = pd.DataFrame({"date": ts, "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"]}).dropna(subset=["close"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df.to_csv("Data/SPX_daily.csv", index=False)
    print(len(df), df["date"].iloc[0], df["date"].iloc[-1])


if __name__ == "__main__":
    main()
