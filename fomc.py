"""FOMC meeting dates from the Fed's historical pages -> Data/FOMC_dates.csv.

The pages (https://www.federalreserve.gov/monetarypolicy/fomchistorical<YYYY>.htm) list one heading per meeting,
e.g. "January 31-February 1 Meeting - 1995", "March 2 (unscheduled) Meeting - 2020". We keep the last day of
each meeting (the decision day) and whether it was a regular scheduled meeting.

Usage (the downloaded pages are untrusted input, so run isolated and pass the folder explicitly):
    python3 -I fomc.py <folder with fomchistorical<YYYY>.htm files>
"""
import html
import os
import re
import sys

import pandas as pd

HEAD = re.compile(r"<h5[^>]*>(.*?)</h5>", re.S)
PARSE = re.compile(r"^(?P<m1>[A-Za-z/]+)\s+(?P<d1>\d{1,2})(?:\s*-\s*(?:(?P<m2>[A-Za-z]+)\s+)?(?P<d2>\d{1,2}))?"
                   r"\s*(?:\((?P<flag>[^)]*)\))?\s*(?:Meeting)?\s*-\s*(?P<y>\d{4})\s*$")


def parse_year(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    text = re.sub(r"<script.*?</script>|<style.*?</style>", "", text, flags=re.S)
    rows, bad = [], []
    for h in HEAD.findall(text):
        h = html.unescape(re.sub(r"<.*?>", "", h)).strip()
        if "Meeting" not in h and "vote" not in h and "call" not in h:
            continue
        m = PARSE.match(h)
        if not m:
            bad.append(h)
            continue
        y, m1, d1 = int(m["y"]), m["m1"], int(m["d1"])
        m2, d2 = m["m2"] or m1, int(m["d2"] or m["d1"])
        if "/" in m1:                       # "April/May 30-1": starts in the first month, ends in the second
            m1, m2 = m1.split("/", 1)
        try:
            start = pd.Timestamp(f"{m1} {d1}, {y}")
            end = pd.Timestamp(f"{m2} {d2}, {y}")
        except ValueError:
            bad.append(h)
            continue
        flag = (m["flag"] or "").strip().lower()
        rows.append({"start": start, "date": end, "flag": flag, "scheduled": flag == "", "heading": h})
    return rows, bad


def main(folder):
    rows, bad = [], []
    for y in range(1994, 2021):
        r, b = parse_year(os.path.join(folder, f"fomchistorical{y}.htm"))
        rows += r
        bad += b
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    # the 2003 page lists both Sept 15 and 16 (the meeting was 16 Sept): of two scheduled meetings < 3 days apart keep the later
    gap = df["date"].shift(-1) - df["date"]
    dup = df["scheduled"] & df["scheduled"].shift(-1, fill_value=False) & (gap <= pd.Timedelta(days=2))
    print("dropped near-duplicate meetings:", df.loc[dup, "date"].dt.date.tolist())
    df = df[~dup].reset_index(drop=True)
    df["year"] = df["date"].dt.year
    os.makedirs("Data", exist_ok=True)
    df[["date", "start", "scheduled", "flag"]].to_csv("Data/FOMC_dates.csv", index=False)
    print(len(df), "meetings;", int(df["scheduled"].sum()), "scheduled")
    print(df.groupby("year")["scheduled"].agg(["size", "sum"]).T.to_string())
    print("unparsed headings:", bad)
    print(df[~df["scheduled"]][["date", "flag"]].to_string())


if __name__ == "__main__":
    main(sys.argv[1])
