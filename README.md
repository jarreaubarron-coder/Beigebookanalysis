# Beige Book

We use several off-the-shelf text sentiment analysis tools to analyze the sentiment of the Fed's Beige Book reports from 1970--2020.

**Live dashboard: <https://jarreaubarron-coder.github.io/Beigebookanalysis/>** (does Beige Book sentiment predict the next fed funds decision?)

The raw text data is scraped from the [Minneapolis Fed](https://www.minneapolisfed.org/region-and-community/regional-economic-indicators/beige-book-archive) by `scrape.py` and stored in `txt`. Do whatever you want with this data (I clearly do not own the copyright).

The reports are scored in `sentiment.py` using three pre-trained models. I am in the process of adding more. So far the models used are

- VADER from `NLTK`
- Pattern analyzer from `TextBlob`
- LSTM text classifier from `flair`

The analysis is done in `analysis.py`, where we get the following graphs:

![GDP Growth Rate Comparison](out/figs/timeseries_suma_drgdp_int.png)

![Sentiment by District](out/figs/timeseries_district_vader.png)

## Predicting the fed funds target

Can the sentiment of a Beige Book predict the FOMC's target-rate decision at the meeting ~2 weeks later?

```
python rates.py      # sanity-check the target series (Data/*.csv from FRED: FEDFUNDS, DFEDTAR, DFEDTARU/L)
python lmscore.py     # LM + directional scores -> out/csv/lm_scores.csv
python features.py   # one row per release -> out/csv/features.csv
python predict.py    # walk-forward models -> out/csv/predictions.csv, out/csv/model_scores.csv
python plots_ffr.py  # out/figs/sentiment_vs_ffr.png, out/figs/model_oos.png
```

Setup: for each release we record the target in effect that day and the target 21 days later (before the next release);
the difference is the outcome. Features are the 3-scorer composite sentiment (district mean, national summary, cross-district
dispersion), z-scored against a *trailing* 5-year window so there is no look-ahead, plus lagged policy moves. Models are refit
at every release using only earlier data (1995-01 to 2020-05, 204 releases).

![Sentiment vs fed funds](out/figs/sentiment_vs_ffr.png)
![Out-of-sample](out/figs/model_oos.png)

Findings (out-of-sample, 1995--2020; full numbers in the next section):

- Sentiment has real signal: it beats "always hold" on squared error, driven by the district-average score.
- Once predictions respect the zero lower bound (below), sentiment adds significantly to lagged policy moves alone.
- The edge comes mostly from the 2001 and 2007--09 easing cycles. The 2020 COVID cuts are missed by every model.
- On absolute error (bp) the models only barely beat "repeat the last move" (best: 8.7bp vs 8.9bp); they are better at spotting big cuts than at calling routine 25bp moves.

### Domain-specific scorers (`lmscore.py`)

The generic scorers were tried against two alternatives, computed per sentence from `txt-split/`
(needs `pip install pysentiment2`, which bundles the Loughran-McDonald finance dictionary):

- `lm_net`: Loughran-McDonald net tone.
- `dir_act` / `dir_price`: up-vs-down direction words ("rose", "declined", ...) with negation flipping, scored separately on
  activity sentences and on price/wage sentences. These word lists are hand-written (see the top of `lmscore.py`), not a published lexicon.

### Zero lower bound

With the target pinned at ~0 (Dec 2008--Dec 2015, and Apr 2020 on) sentiment cannot move policy, and the first version of the model
kept predicting hikes/cuts anyway (sentiment *hurt* in 2009--2019). Two fixes, both using only information available at release time
and applied to every model so the comparison stays fair:

1. **Floor:** a prediction can never be a cut larger than the distance to the bound (no cuts when the target is at 0.25%).
2. **Gating** (`g_*` feature sets): a `zlb` dummy, and the sentiment features are zeroed while the target is at the bound.

Walk-forward 1995--2020, ridge, R² of squared error vs "always hold" (all with the floor):

| features | full | excl. 2020 | 2009--2019 |
|---|---|---|---|
| policy state only | 0.30 | 0.32 | 0.13 |
| + VADER/TextBlob/Flair | 0.38 | 0.42 | 0.03 |
| + directional activity & price tone | 0.42 | 0.47 | 0.04 |
| + VADER/TextBlob/Flair, **gated** | 0.40 | 0.44 | 0.18 |
| + directional activity & price tone, **gated** | 0.43 | 0.48 | 0.15 |

(Before the floor, state-only scored 0.26 / 0.30 / -0.11 and the sentiment models were negative in 2009--2019.)

Squared-error gain over state-only, block bootstrap (`predict.robustness`), bp² per release, 95% CI:

- existing scorers: +39 [+1, +85]; directional tone: +56 [+15, +106] (full sample; same excl. 2020).
- gating on top of the floor: +3 to +7, CI includes 0 (borderline, p ~ 0.03--0.12), mostly in 2009--2019.

Notes:

- LM tone is the weakest single scorer. All activity-type scorers are highly correlated (directional activity vs. the original district average: r = 0.91), so they mostly measure the same business-cycle factor. The price-direction score is the most distinct one.
- The "lean" feature sets and the ZLB treatment were chosen *after* seeing earlier results (forking paths), and many specs were compared without multiple-comparison correction. Treat p-values as exploratory. The clean test is data after 2020-05, which this repo does not yet have.
- Not shown in the table: gradient boosting (`gbm_*` in `out/csv/model_scores.csv`) gives similar results; `gbm_state` is already strong in 2009--2019 (0.30) so the gating matters less for it.

Caveats: the FOMC only announced its target from Feb 1994 (earlier values are the Fed's reconstruction); before 1982-09 there is no
target, so pre-1983 data is only used for the descriptive plot and for warm-up of the rolling normalisation. After Dec 2008 the upper bound
of the target range is used. Only ~100 non-hold decisions exist, so treat all results as suggestive.

## The S&P 500 and FOMC announcement days

Two questions: does Beige Book tone (or our model's predicted decision) predict the S&P 500's return on the FOMC announcement day, and how do simple rules built on them compare with buy-and-hold?

```
python3 fetch_spx.py                     # daily ^GSPC -> Data/SPX_daily.csv (git-ignored; Yahoo's terms restrict redistribution)
python3 -I fomc.py <folder>              # FOMC meeting dates from the Fed's fomchistorical<YYYY>.htm pages -> Data/FOMC_dates.csv
python3 markets.py                       # event study + timing comparison -> out/csv/event_*.csv, timing_strategies.csv, out/figs/
```

Each release is matched to the next scheduled meeting (13--16 days later; 210 of 212 match). The event study covers the 162 scheduled meetings from Feb 2000
(the FOMC issues a statement after every meeting from then on) to Jun 2020. Returns are close-to-close, so they include the morning before the 2pm statement.

![S&P 500 vs rules](out/figs/spx_vs_rules.png)
![Announcement-day returns](out/figs/fomc_day_returns.png)

Findings:

- Announcement days average +0.26% against +0.01% on other days (same sd), the known FOMC announcement-day premium.
- **Tone does not reliably predict the announcement-day return.** Primary test (tone `s_dist_z`): -0.21% per sd, robust p = 0.08, permutation p = 0.03 in 2000--2020, but -0.04% (p = 0.71) once the 2008--09 crisis and 2020 are dropped.
  A placebo on the day a week earlier gives the same sign (-0.12%, -0.18% ex-crisis), so it is rebound-after-bad-news rather than announcement news.
- The model's predicted change (p = 0.28) and the "surprise" (actual minus predicted; p = 0.77) do not predict returns either. The surprise is only a proxy: markets react to futures-implied surprises and to guidance (the +4.6% day on 2011-08-09 had no rate change).
- Of 6 specifications x 2 samples, the only individual p < 0.05 is `dir_price_z` in the full sample (p = 0.018), which does not survive a multiple-testing correction or the crisis exclusion.
- Timing comparison, 1995--2020 (price-only S&P, cash at the effective fed funds rate, 10bp per switch):

| | CAGR | vol | Sharpe | max drawdown | invested |
|---|---|---|---|---|---|
| Buy and hold | 7.7% | 19.2% | 0.35 | -56.8% | 100% |
| R1: invested if tone > 0 | 8.0% | 10.8% | 0.54 | -16.0% | 55% |
| R2: invested unless the model predicts a cut | 9.4% | 12.6% | 0.58 | -22.6% | 63% |

  The rules mostly cut risk. Their return advantage is not statistically distinguishable from zero (block bootstrap 95% intervals include zero), it comes from avoiding the 2000--02 and 2008--09 drawdowns,
  and they lag badly in bull markets (2009--2019: 6.6%/9.7% vs 12.3% for buy-and-hold). The S&P series is price-only (no dividends) so buy-and-hold is understated, and with three major drawdowns in the sample this is thin evidence.
  This is a research exercise, not investment advice.

## Interactive dashboard

Live at <https://jarreaubarron-coder.github.io/Beigebookanalysis/>.

`dashboard/` holds the source of an interactive dashboard that summarises the findings above (period, scorer and model pickers; timeline, scatter, model scores, forecast and bootstrap-gain charts).

```
python3 dashboard/build_data.py dashboard/data   # after predict.py: JSON data files from out/csv/
```

`dashboard/index.html` is the page. It was published as a Claude Artifact of the Data type, which reads the five JSON files as datasets
(`releases`, `predictions`, `period_scores`, `gains`, `scorers`). The sub-period scores, bootstrap intervals and correlations are computed in
`build_data.py`, so the page only displays them. The published copy is a snapshot: re-run the build and re-upload the files to refresh it.

### Standalone version for GitHub Pages

`dashboard/index.html` only runs inside Claude's dashboard runtime. For a version that runs in any browser:

```
python3 dashboard/build_standalone.py   # writes docs/index.html (data embedded, d3 from cdnjs)
```

It wraps the same page in a normal HTML document with a small stand-in for the runtime. To publish it, open the repository's
Settings > Pages, choose "Deploy from a branch", branch `master`, folder `/docs`. The site is then at
`https://<user>.github.io/<repo>/`. Opening `docs/index.html` locally also works (it needs internet for d3).
Rebuild after `predict.py` and `dashboard/build_data.py` whenever the results change.

## Dependencies

Dependencies are listed in `requirements.txt`. Tested + Developed on Python 3.8.

- `scrape.py`
  - `requests`
  - `beautifulsoup4`
- `files.py`
  - `pandas`
- `clean.py`
  - `cleantext`
- `sentiment.py`
  - `pandas`
  - `nltk`
  - `textblob`
  - `flair`
  - `transformers`
- `analysis.py`
  - `numpy`
  - `pandas`
  - `statsmodels`
  - `matplotlib`

## TODO

- [x] Fix parsing errors
  - [x] Bug with `<br>` tag instead of `<br />` (no more breaks)
  - [x] Remove `<strong>` (ignored)
  - [x] `&nbsp;` problem (check if this gets removed)
  - [x] Delete "learn more" `<p>` at the bottom (`grep -RIl "www\." txt/`)
- [x] Find missing/incomplete files
  - [x] Some files are empty
  - [x] Analyze `errors.txt`
- [x] Grab missing files
  - [x] Grab missing `2016-0(4|6)-su` files
  - [x] Grab missing `2015-07-*` files
  - [ ] Try to find missing `1971-01-bo`
- [x] Clean text
  - [x] Replace `&%-+` with text?
  - [ ] Replace numbers with words
  - [x] Check that text is ASCII
- [x] Run sentiment analysis
  - [x] Check out `flair` package
  - [x] `flair` gives values `x<-0.5 | x>0.5` (fixed in analysis)
  - [ ] Check if all text is used or just first `n` words
  - [ ] `transformers` package
  - [ ] Just extract numbers (bigger is better)
- [x] Get exact dates of publication
- [x] Generate histograms
  - [x] Normalize values
  - [x] Check out outlier (`1971-01-bo` missing doc)
- [x] Regress national sentiment on regional sentiments
  - [ ] Do you add a constant here?
  - [ ] See if coefficient sum to 1
  - [ ] Create proxy measure including all regions
- [x] Graph time series
  - [x] Pretty up plots (title+legend)
  - [x] Get GDP data
  - [x] Check stock market data
  - [ ] Think about timing of Beige Book data
  - [x] By region in a grid
  - [ ] Bond yields
- [ ] Time series regression
- [ ] Investigate discrepancies between sentiment scores
  - [ ] In `su` `TextBlob` is high during 1974 recession and higher during 1990s boom
- [x] Add info + pictures to `README.md`
- [ ] GDP Growth by region
  - [ ] Aggregate state data
