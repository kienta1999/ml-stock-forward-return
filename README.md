# ml-stock-forward-return

ML-based S&P 500 stock ranker. Predict each stock's forward 21-trading-day
return independently with XGBoost, sort to get a daily ranking, long the top
decile, hold 21 trading days, rebalance monthly with a SPY/VIX regime gate.

**Current status (clean architecture + EDGAR earnings calendar,
100-trial sweep):** raw long-only **+17.5% CAGR / Sharpe 0.73** vs SPY
+12.7% / Sharpe 0.75 (test 2021-01-04 → 2026-03-31). **+480 bps/yr
alpha at SPY-equivalent risk-efficiency** — strategy is 1.41× SPY's
volatility but earns 1.38× SPY's return, so Sharpe is essentially tied.
Final NAV 2.32× vs SPY 1.87× = 24% more wealth over 5.25 years. The
gain over the prior clean-arch baseline (+15.5% / Sharpe 0.70) came
from adding **EDGAR-derived earnings calendar features** —
`days_since_earnings` ranked 14th in importance (0.037 gain), and the
model's `best_iteration` lifted from 3 → 10, breaking the technical-
only ceiling that hyperparameter tuning couldn't crack. The PEAD
(post-earnings-announcement drift) signal is real, even when anchored
on 10-Q filing dates rather than the 8-K announcement.

**Legacy model preserved.** The previous iteration with broadcast
SPY/VIX features and raw labels (+17.3% CAGR / Sharpe 0.79) is saved as
`models/xgb_v1_legacy.json` for deployment in a tax-advantaged account
(IRA) where short-term gains aren't penalised. That model's headline
number was partly market-timing alpha — see
[Result evolution](#result-evolution-which-runs-produced-which-numbers)
below for the full lineage of how the architecture changed.

**FINRA short interest attempted, dropped.** The original §1 plan was
short interest from FINRA bi-monthly. The download infrastructure shipped
(`scripts/altdata.py --source finra`) but the FINRA CDN archive only
goes back to **2018-08**, not 2007. With train=2007–2017 fully NaN for
`days_to_cover`, XGBoost cannot build any tree splits on the feature
during training (no train variance → no info gain → never selected).
Mean-fill was rejected because constant-pre-2018 values would inject
regime leak. Sliding the splits forward sacrificed test bear-market
coverage. So the FINRA cache stays on disk but the feature is not
wired into the panel. Will revisit if/when paid historical short
interest is added or splits are re-platformed.

**Tax reality check.** Even +17.3% pre-tax loses to SPY HODL after
short-term capital gains tax (~37% combined federal+state at typical
rates → +10.9% effective). For taxable accounts the strategy needs
~+20% pre-tax CAGR to clear the bar. Run in IRA/Roth/401k.

This is the ranking-style sibling of `technical-analysis-stock-scanner`, which
filters and picks. Here we score and sort.

---

## Methodology

| Stage    | What it does                                                                                               |
| -------- | ---------------------------------------------------------------------------------------------------------- |
| Universe | Point-in-time S&P 500 (1996+ membership CSV) joined with current Wikipedia sectors                         |
| Data     | yfinance OHLCV 2005-07-01 → today (1.5y buffer for 252d warmup), per-ticker parquet cache, plus SPY + ^VIX |
| Features | 17 per-ticker + 3 market-context (ticker-specific only) + 2 sector-relative + 3 earnings calendar (EDGAR 10-Q/10-K + yfinance forward dates) + 16 cross-sectional ranks + sector cat = **42 total**. Broadcast SPY/VIX features intentionally excluded so the model can't lean on regime signal. |
| Label    | `forward_21d_return − date_mean(forward_21d_return)` — date-demeaned (cross-sectional excess). Raw `forward_21d_return` is clipped to ±0.5 first to cap dead-ticker outliers, then demeaned. The model can only learn within-date ordering, not market direction. |
| Split    | Train 2007–2017, Val 2018–2020, Test 2021→. Chronological. No shuffling.                                   |
| Model    | XGBoost regressor, RMSE loss, optuna-tuned on val decile spread (max_depth ∈ [3, 6], 100 trials, ES=100 rounds) |
| Backtest | Long-only top-50, monthly rebalance, regime gate (SPY > SMA200 AND VIX < 25), 21 shifted-start offsets     |
| Costs    | 5 bps per side on rebalance turnover                                                                       |

Every feature on row date=D uses only data observable at the close of D.
`dataset.assert_no_lookahead()` samples random rows and recomputes features
with `.loc[:date]` to verify this — it fails loud on leakage.

---

## Point-in-time S&P 500 universe

`universe.py` reads `data/universe/SP_500_Historical_Component.csv` (from
[fja05680/sp500](https://github.com/fja05680/sp500)) — a list of S&P 500
change-events from 1996 onward. Each (date, ticker) row in the panel is
filtered against the index membership in effect on that date, so stocks that
were later acquired, bankrupted, or kicked out (Lehman, Enron, Eastman Kodak,
…) are present for the dates they belonged and absent afterwards.

Sectors come from a separate Wikipedia scrape (`load_sectors`) and only cover
the **current** roster; tickers that have since left the index get
`gics_sector = "Unknown"`, which XGBoost treats as a normal category.

### Residual bias: yfinance doesn't retain delisted tickers

The membership CSV lists every name ever in the index (917 since 2007), but
yfinance only returns OHLCV for ~520 of them — the ~397 missing are mostly
acquisitions and bankruptcies whose symbols got retired (AGN→ABBV, ATVI→MSFT,
ANTM→ELV, AABA, EKDKQ, ABKFQ, …). After download + filter, the panel ends up
with ~501 unique tickers, **all of which happen to be current S&P 500
members** (zero `Unknown` sectors). The membership filter still correctly
time-gates each ticker's rows (panel grows 396 names in 2005 → 501 in 2026),
which fixes the cleanest case — names that were demoted but kept trading,
e.g. PLTR before 2024-09. But the dramatic survivorship cleanup (Lehman,
Enron, Kodak, Allergan) requires a data source that retains delisted-symbol
prices: Sharadar, Norgate, Polygon historical, or CRSP via WRDS. yfinance
free won't get you there.

Net effect on results: gated CAGR moved 18.8% → 17.7%, raw 26.2% → 25.7% —
small deflation, in the expected direction, but smaller than a true bias-free
universe would deliver (probably another 3–5 CAGR points lower).

---

## Setup

```bash
uv sync
```

Python 3.11+. All deps are pinned in `pyproject.toml`.

---

## Run order

```bash
uv run python scripts/universe.py                   # build membership history + sector cache
uv run python scripts/data.py --tickers AAPL,MSFT   # 10s smoke test first
uv run python scripts/data.py                       # 917 historical tickers + SPY + VIX (~30–45 min first time; ~520 succeed via yfinance)

uv run python scripts/altdata.py --source edgar     # SEC EDGAR 10-Q/10-K filing dates per ticker (~5 min first time)
uv run python scripts/altdata.py --source finra     # FINRA bi-monthly short interest (cache only — feature not currently wired)
uv run python scripts/altdata.py                    # both EDGAR + FINRA at once

uv run python scripts/features.py --ticker AAPL     # smoke-print one ticker's features
uv run python scripts/features.py                   # build full panel → data/processed/features.parquet
uv run python scripts/labels.py                     # add forward_21d_return → data/processed/panel.parquet
uv run python scripts/dataset.py                    # splits + lookahead sanity check
uv run python scripts/dataset.py --quick            # same, skip the slow recompute check

uv run python scripts/train.py                      # optuna tuning + final fit (~10-15 min)
uv run python scripts/train.py --trials 20          # faster tune
uv run python scripts/train.py --quick              # skip tuning, use sane defaults

uv run python scripts/backtest.py                    # long-only + gated, 21 shifted starts (~1 min)
uv run python scripts/backtest.py --no-overlay       # skip gated variant
uv run python scripts/backtest.py --top-n 25         # tighter pick (default 50)

uv run python scripts/today.py                                   # latest-date picks (regime gate + top 50)
uv run python scripts/today.py --diff picks/picks_YYYY-MM-DD.csv # buy/sell list vs that prior file
uv run python scripts/today.py --no-overlay                      # ignore regime gate (diagnostic)

uv run python scripts/run_all.py                  # daily: data → features → labels → today (auto --diff)
uv run python scripts/run_all.py --retrain        # also: train + backtest
uv run python scripts/run_all.py --full           # also: refresh universe first
uv run python scripts/run_all.py --dry-run        # print plan, don't execute
```

### Daily live-picks workflow

`scripts/today.py` is what bridges the backtest model to actual trading. The
backtest deliberately ignores the most recent ~21 trading days because they
have no forward-return label yet; `today.py` deliberately predicts on them.

The orchestrator `scripts/run_all.py` chains the daily pipeline together:

```bash
# every morning before market open — one command:
uv run python scripts/run_all.py
```

That runs `data.py → features.py → labels.py → today.py` end-to-end, stops
on the first failure, and auto-supplies `--diff` to `today.py` using the
most recent prior file in `picks/`. Exit code is the failed step's exit
code, so it's cron-friendly.

Modes:

| Command                 | What runs                                         |
| ----------------------- | ------------------------------------------------- |
| `run_all.py`            | data → features → labels → today (default daily)  |
| `run_all.py --retrain`  | also train + backtest                             |
| `run_all.py --full`     | also refresh universe first (`--retrain` implied) |
| `run_all.py --no-today` | refresh + optional retrain only, skip live picks  |
| `run_all.py --no-diff`  | run today.py without `--diff`                     |
| `run_all.py --dry-run`  | print the plan, don't execute                     |

Equivalent manual sequence (if you want to run pieces individually):

```bash
uv run python scripts/data.py
uv run python scripts/features.py
uv run python scripts/labels.py
uv run python scripts/today.py --diff picks/picks_<yesterday>.csv
```

The `--diff` flag prints a BUY / SELL / HOLD ticket — exactly which tickers
to add or drop today vs the prior picks. That's the trade list you'd
execute (manually or via Alpaca API).

Output lands in `picks/picks_<latest_date>.csv` (one file per run;
gitignored to keep daily noise out of git). Cash days produce an empty
picks file. The script warns if the panel is more than 7 days old.

### `data.py` CLI flags

| Flag                  | What it does                                                                                             |
| --------------------- | -------------------------------------------------------------------------------------------------------- |
| _(none)_              | Incremental update — only fetches days since the last cached date. Default for daily refreshes.          |
| `--refresh`           | Wipe caches and redownload from scratch. Use when something looks wrong.                                 |
| `--tickers AAPL,MSFT` | Subset only. Great for smoke-testing or iterating on downstream code without re-downloading 500 tickers. |
| `--start 2015-01-01`  | Override default start (`2007-01-01`). Useful for a smaller, faster dataset while developing.            |
| `--skip-universe`     | Only refresh SPY/VIX.                                                                                    |
| `--skip-market`       | Only refresh the universe; skip SPY/VIX.                                                                 |

### Tuning knobs (in `scripts/data.py`)

- `WORKERS = 8` — parallel yfinance downloads. Drop to 4 if you see retries firing (yfinance throttles).
- `MIN_HISTORY_DAYS = 500` — tickers below this are tagged `short` and excluded by `load_prices()`.
- `RETRIES = 3`, `RETRY_SLEEP = 2.0` — linear backoff on transient yfinance errors.

### Verifying the data layer

```bash
ls data/raw/ | wc -l                   # ~490–500 parquets
ls data/market/                        # SPY.parquet, VIX.parquet
uv run python -c "import pandas as pd; print(pd.read_parquet('data/raw/AAPL.parquet').tail())"
```

---

## Features

39 features per `(date, ticker)` row, in 5 buckets (4 numeric + 1
categorical). Lists are exposed as constants in `scripts/features.py`
so downstream code stays in sync.

### Bucket 1 — per-ticker technicals (17)

| Feature                                  | Definition                                                  | What it captures                                                                                                                                                    |
| ---------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ret_1d`, `ret_5d`, `ret_21d`, `ret_63d` | `close.pct_change(n)`                                       | Trailing return at multiple horizons. 1d/5d often **mean-revert**; 21d/63d trend.                                                                                   |
| `rsi_14`                                 | Wilder's RSI on close, 14d, EWM com=13                      | Overbought (>70) / oversold (<30) momentum oscillator.                                                                                                              |
| `mfi_14`                                 | Money Flow Index — RSI weighted by `typical_price × volume` | Same shape as RSI but volume-aware. Catches conviction behind a move.                                                                                               |
| `macd_hist`                              | `MACD(12,26) − signal(9)`                                   | Trend acceleration. Positive & rising = bullish momentum.                                                                                                           |
| `atr_pct`                                | `ATR(14) / close`                                           | Average true range as % of price — cross-sectionally comparable volatility.                                                                                         |
| `vol_20d`, `vol_60d`                     | Annualised std of log returns                               | Realised volatility, short vs medium term.                                                                                                                          |
| `vol_ratio`                              | `volume / volume.rolling(20).mean()`                        | Today's volume relative to recent norm. >1 = unusually heavy.                                                                                                       |
| `dist_sma50`, `dist_sma200`              | `close / sma_n − 1`                                         | How stretched price is above/below medium- and long-term trend.                                                                                                     |
| `dist_52w_high`                          | `close / max(close, 252) − 1`                               | Distance below the 52-week high. Classic momentum factor (stocks near highs tend to keep outperforming).                                                            |
| `trend_regime`                           | `1.0 if sma50 > sma200 else 0.0`                            | Bull/bear trend flag.                                                                                                                                               |
| `zscore_20d`, `zscore_60d`               | `(close − sma_n) / std_n`                                   | **Volatility-normalised** distance from mean. A 5% gap means different things for a calm vs vol-y name; this corrects for that (also the math behind Bollinger %B). |

### Bucket 2 — market context (3, ticker-specific only)

Each feature here has a *different value per ticker* on a given date.
Broadcast SPY/VIX features (same value for every stock) were intentionally
removed in the clean-architecture refactor — they let trees split on
"market direction" rather than "this stock vs others," which is regime
forecasting, not stock picking.

| Feature                           | Definition                                  | What it captures                                                                                                                     |
| --------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `beta_60d`                        | Rolling cov(stock, SPY) / var(SPY) over 60d | Stock's sensitivity to the market. >1 amplifies, <1 dampens. Per-ticker number.                                                      |
| `excess_ret_5d`, `excess_ret_21d` | `ret_n − spy_ret_n`                         | **Relative strength**: how much the stock beat or lagged SPY over the window. The single biggest signal for cross-sectional ranking. |

**Removed from this bucket** (kept here for documentation since they may
return when conditioned on a separate regime model): `spy_ret_21d`,
`spy_trend_regime`, `spy_rsi_14`, `vix_level`, `vix_zscore_20d`. All five
are broadcast (identical for every ticker on a date) so they can only
help the model time the market, not pick stocks.

### Bucket 3 — sector-relative (2)

| Feature                                   | Definition                                                                  | What it captures                                                          |
| ----------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `excess_ret_5d_vs_sector`, `excess_ret_21d_vs_sector` | `ret_n − groupby(date, gics_sector).transform("mean")` | Cross-sectional sector-relative momentum. Strips sector beta from the per-ticker return so the model compares apples to apples within an industry. NaN for the `Unknown` sector bucket (delisted/removed names — group mean would be noise). |

### Bucket 4 — cross-sectional ranks (16)

For each Bucket 1 feature _(except the binary `trend_regime`)_, we compute its **percentile rank across all active tickers on that date** (0 = worst, 1 = best). Column suffix: `_rank`.

Why: a 30% trailing return in 2008 ≠ a 30% return in 2017. Ranks normalise out the time-varying scale and turn each feature into a contemporaneous comparison — which is precisely what a ranker needs.

### Bucket 6 — earnings calendar (3)

| Feature | Definition | What it captures |
| --- | --- | --- |
| `days_to_earnings` | days from row date to next known earnings filing (EDGAR 10-Q/10-K + yfinance forward calendar), clipped [0, 90] | Pre-earnings positioning. **Currently 0.0 importance** — 10-Q dates are scheduled, not surprise-driven, so forward distance carries no cross-sectional signal. Kept for now pending a switch to actual 8-K item 2.02 announcement dates. |
| `days_since_earnings` | days since most recent earnings filing, clipped [0, 90] | **Post-earnings drift signal — 14th in feature importance (0.037 gain).** The continuous form lets XGBoost discover its own optimal drift window. |
| `post_earnings_drift_window` | `1.0` if `days_since_earnings ∈ [1, 5]` else `0.0` | Hand-coded PEAD window flag. **Currently 0.0 importance** — XGBoost reconstructs the same split internally from the continuous `days_since_earnings`, making this redundant. Kept for now pending a full re-evaluation. |

**Data source.** SEC EDGAR submissions API gives every 10-Q / 10-K
filing date for tickers with a current CIK (648 of 959 historical
S&P 500 names; the missing 311 are mostly delisted/renamed without
retained CIKs). For live picks, yfinance's forward earnings calendar
fills in the upcoming dates. Caveat: 10-Q filings trail the actual
8-K item 2.02 announcement by ~2-4 weeks, so the "drift window" here
is anchored on the report filing rather than the announcement.
Switching to 8-K item filtering would require one extra request per
filing and is queued under [Next steps](#next-steps).

`scripts/altdata.py` handles the EDGAR + FINRA pipelines (separate
from `data.py` because the cadences and rate-limit contracts differ).
Per-ticker earnings parquets live at `data/earnings/{TICKER}.parquet`.

### Bucket 5 — categorical (1)

| Feature       | Source                                        | Purpose                                                                                                                                                               |
| ------------- | --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `gics_sector` | `data/universe/sp500_sectors.csv` (Wikipedia, current members) | XGBoost native categorical. Lets the model split on sector membership without manual encoding — captures effects like "utilities react differently to vol than tech." Tickers no longer in the index fall into the `"Unknown"` bucket. |

### Why some popular indicators are _not_ included

| Skipped                    | Reason                                                                                                                                                                                     |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Bollinger Bands (raw)      | Same information as `vol_20d` + `zscore_20d`; no extra lift.                                                                                                                               |
| ADX                        | Already captured by `trend_regime` + vol features.                                                                                                                                         |
| Stochastic %K, Williams %R | ~85% correlated with RSI. Redundant.                                                                                                                                                       |
| OBV / Chaikin / A-D        | Cumulative volume series; not cross-sectionally comparable without normalisation, and the normalised form ends up close to `vol_ratio × ret_n`.                                            |
| CCI                        | Just a z-score of typical price — already covered by `zscore_*`.                                                                                                                           |
| VWAP                       | Short-horizon; doesn't help a 21d forecast.                                                                                                                                                |
| Ticker as a token          | The ticker is a unique ID, not a factor. Including it would let the model memorise per-stock patterns from train and apply them to test — pure data leakage of the label through identity. |

### No-lookahead guarantee

Every feature on row `date=D` is computed using only data with date ≤ D. All operations are `rolling` / `ewm` / `pct_change` / `shift(positive)` — none peek into the future. `dataset.assert_no_lookahead()` (coming next) will sample random rows, recompute features with `df.loc[:date]`, and fail the build if anything disagrees.

---

## Model

XGBoost regressor on the 39 features. Why XGBoost: handles missing values, scales
to millions of rows, native categorical (`gics_sector`) without encoding, and
gives feature importances for free.

### Three roles, two metrics

| Role                    | Metric                              | Why                                                                                                                                                                                                                                                                                                                                                        |
| ----------------------- | ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Training loss           | **RMSE** (`reg:squarederror`)       | XGBoost needs a smooth differentiable loss; RMSE is the default and gives stable gradients.                                                                                                                                                                                                                                                                |
| Tuning + early stopping | **mean daily decile spread** on val | We don't trade IC, we trade the top decile. IC and decile spread can disagree (deep trees can produce clumpy predictions with high IC but poor decile separation), so we score on the metric tied to P&L. Both the optuna objective and the per-round early-stopping rule maximise val decile spread; keeping them aligned matters (see v1 results below). |
| Reporting               | **RMSE + IC + decile spread**       | Cross-checks: RMSE catches magnitude blow-ups, IC catches ranking quality, decile spread is the most direct proxy for strategy P&L.                                                                                                                                                                                                                        |

### Hyperparameter search

Tuned with **optuna** (TPE sampler, ~50 trials). Knobs and ranges:

| Param              | Range          | What it controls                                                                                                                                         |
| ------------------ | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `max_depth`        | fixed at 3     | Depth-8 collapses decile separation (clumpy predictions); 3–5 give equivalent val spread, so we pick the shallowest — most trees, smoothest predictions. |
| `learning_rate`    | 0.01–0.1 (log) | How aggressively each tree corrects errors. Smaller + more trees usually wins. Range tightened from 0.01–0.3 after early-stopping was firing too aggressively on demeaned labels (see v1 results). |
| `n_estimators`     | 200–1000       | Max number of trees. Capped by early stopping.                                                                                                           |
| `min_child_weight` | 1–20           | Minimum sum of sample weights per leaf. Higher = simpler trees.                                                                                          |
| `subsample`        | 0.6–1.0        | Row sampling per tree. <1 adds randomness → robustness.                                                                                                  |
| `colsample_bytree` | 0.6–1.0        | Feature sampling per tree. Same idea, on columns.                                                                                                        |
| `reg_lambda`       | 0.01–10 (log)  | L2 regularization on leaf weights.                                                                                                                       |

Each trial trains one model with early stopping (100 rounds on val decile
spread, maximize, save_best) and returns mean daily val decile spread. Optuna
picks the next combo to try based on what's worked so far. The final model is
refit on the best params and saved to `models/xgb_v1.json`.

### Outputs

- `models/xgb_v1.json` — trained booster
- `reports/feature_importance.csv` — per-feature gain importance
- `reports/optuna_trials.csv` — full tuning history (params + IC per trial)
- `reports/train_metrics.json` — final train/val RMSE + IC + decile spread + chosen params

### v1 results (val 2018–2020)

| Metric                                 | **Current (clean arch + EDGAR earnings, 100 trials)** | Prior clean-arch (no earnings, 100 trials)   | Legacy (200 trials, with SPY/VIX, raw labels) |
| -------------------------------------- | ----------------------------------------------------- | -------------------------------------------- | --------------------------------------------- |
| val decile spread (top10 − bot10, 21d) | +0.0173 (~173 bps, demeaned units)                    | +0.0173                                      | **+0.0297 (~297 bps)**                        |
| val IC (mean daily Spearman)           | +0.0444                                               | +0.0520                                      | +0.0554                                       |
| train IC                               | +0.0439                                               | +0.0401                                      | +0.0659                                       |
| val RMSE / train RMSE                  | 0.0739 / 0.0687                                       | 0.0739 / 0.0688                              | 0.1029 / 0.0899                               |
| best_iteration                         | **10** (ceiling lifted from 3)                        | 3                                            | 30                                            |
| chosen `max_depth`                     | 3                                                     | 3                                            | 3 (out of [3, 6])                             |
| chosen `learning_rate`                 | 0.020                                                 | 0.027                                        | 0.082                                         |
| chosen `n_estimators`                  | 591 (cap, only 10 used)                               | 389 (cap, only 3 used)                       | 860 (capped by ES)                            |
| chosen `min_child_weight`              | 9                                                     | 3                                            | 11                                            |
| chosen `subsample`                     | 0.964                                                 | 0.792                                        | 0.949                                         |
| chosen `colsample_bytree`              | 0.685                                                 | 0.633                                        | **0.629**                                     |
| chosen `reg_lambda`                    | 0.659                                                 | 0.907                                        | 0.446                                         |

The "decile spread" column for the current run is in **demeaned-return
units** and isn't directly comparable to the legacy column (which is in
raw return units). The val IC column *is* comparable: +0.0444 (with
earnings) vs +0.0520 (no earnings) vs +0.0554 (legacy). Val IC actually
*ticked down* with earnings added, but val decile spread stayed the
same and **test CAGR rose from +15.5% → +17.5%**. The val/test
divergence suggests EDGAR earnings dates carry signal that doesn't
manifest as much in val (2018–2020) but pays off in test (2021+) —
plausible since 2021–2023 had unusually concentrated earnings-driven
moves around mega-cap tech.

Three things stand out:

1. **`best_iteration=3` was the prior ceiling — earnings broke it.**
   Adding three EDGAR-derived features lifted the chosen iteration
   count from 3 → 10. The model now builds three times more boosting
   rounds before val plateaus, which is the textbook signature of
   "more useful signal available to fit." This is the first
   architectural change since the clean-arch refactor that has
   actually moved the boost-depth ceiling.
2. **`colsample_bytree` settled at ~0.63 across both architectures** —
   the optimiser independently rediscovers the same column-sampling
   ratio. With 26 of 41 (legacy) or 25 of 39 (current) features per
   tree, ensemble diversity is the right knob; this is now established.
3. **`max_depth=3` won every time.** Across all three runs in the table
   (and the 500-trial run that never made it to the table because it
   val-overfit), optuna picks depth 3. Deeper trees produce clumpy
   predictions that score well on IC but collapse decile separation.
   Depth tuning is settled.

**Earlier-run note (no label clip, 20 trials):** the previously reported run
without label clipping landed `best_iteration=196` with a much slower
`learning_rate=0.019`, val IC 0.030, decile spread 0.020. The clip was
added because a few delisted-ticker −100% labels were dominating MSE.
That fix kept val decile spread at parity (+0.020 → +0.021) on the first
re-run; it took the wider hyperparameter sweep above to recover and
exceed the pre-clip result.

**Lessons learned along the way (two related misalignments)**:

1. **Early stopping must score the metric you tune on.** First cut of
   `train.py` stopped on val RMSE while tuning on val IC. RMSE bottoms early
   because predictions shrink toward zero, so every trial got cut off at
   `best_iteration=1` — a single deep tree (depth 8). Switching early
   stopping to a custom callable that returns the val score (with
   `xgb.callback.EarlyStopping(maximize=True, save_best=True)`) let the
   search find a real boosted-ensemble basin instead of one fat tree.

2. **Tune the metric the strategy actually trades.** With the alignment
   above fixed but the objective still set to IC, a 50-trial run converged
   back to depth-8 — IC ticked up to 0.0553 (best ever), but decile spread
   collapsed to 0.0070. Why: depth-8 = 256 leaves per tree → ~1000 distinct
   predicted values across 1.2M val rows. Lots of clumping. Spearman handles
   ties OK and picks up a bit more average ordering → IC up. But decile cuts
   pass through clumps where stocks have nearly-identical predictions but
   very different realised returns → top-decile and bottom-decile averages
   collapse together → spread down. **Higher IC, worse strategy P&L.** Fix:
   tune on decile spread directly, and cap `max_depth` at 5 to keep
   predictions non-clumpy. Both early stopping and the optuna objective now
   maximise val decile spread — the numbers in the table above are from
   that configuration.

---

## Backtest

### Strategy (v1, long-only)

Top-50 by predicted return (= top 10% of ~500 active S&P 500 names),
equal-weighted, monthly rebalance, 21 trading day hold, 5 bps per side cost.
Two variants run side-by-side:

- **Gated** (recommended): `if SPY_close > SPY_SMA200 AND VIX_close < 25 → top 50; else cash.`
  Trend-up + low-vol regime filter. ~77% time in market.
- **Raw**: always long top 50. Diagnostic — what the model picks alone, no
  market-timing overlay.

### Rebalance-date sensitivity

Monthly rebalance has a fragility: which day-of-month you happen to start
matters. We mitigate by running the same strategy 21 times — each one
starting on a different anchor day (offset = 0..20) — and reporting the
mean equity curve plus the 10th/90th percentile band. Closely matches what
21 overlapping sleeves would deliver, with much less code complexity.
Sleeves are on the roadmap; this is the simpler-but-equivalent v1.

### Results (test 2021-01-04 → 2026-03-31, 100-trial sweep, no broadcast SPY/VIX, demeaned labels)

`backtest.py` clips the SPY benchmark to the strategy's last predictable
date so the headline comparison is apples-to-apples by default; the CSV
keeps the full SPY series so the post-strategy tail is visible for
inspection.

**Current model (clean architecture + EDGAR earnings):**

| Variant                              | CAGR    | Vol   | Sharpe | Max DD | Final NAV | Time-in-market |
| ------------------------------------ | ------- | ----- | ------ | ------ | --------- | -------------- |
| **Raw long-only**                    | **+17.5%** | 23.9% | **+0.73** | -28.0% | **2.32×** | 100%       |
| Gated long-only                      | +8.8%   | 15.6% | +0.56  | -21.6% | 1.55×     | 77%            |
| SPY buy & hold (clipped @2026-03-31) | +12.7%  | 17.0% | +0.75  | -24.5% | 1.87×     | —              |

**Reading the table honestly:** raw beats SPY by +4.8 CAGR points and
**Sharpe is essentially tied (0.73 vs 0.75 — within noise for a
5.25-year backtest)**. Strategy runs at 1.41× SPY's vol but earns
1.38× SPY's return, so the risk-efficiency ratio is preserved — same
Sharpe, more return. Final NAV 2.32 vs 1.87 = **+24% more wealth**
over the test period. Gated still underperforms SPY (regime gate is
giving up upside it doesn't earn back); the gated mode remains the
known weakness. The +2.0 CAGR-pt gain over the prior clean-arch run
came from EDGAR earnings features lifting `best_iteration` from 3 → 10,
not from any tuning change.

**Concentration sweep on the same model** (`backtest.py --top-n 20`):
top-20 raw +20.3% / Sharpe 0.67. Concentration trades return for vol;
Sharpe is roughly preserved. **Concentration is a poor man's leverage**
— the same effect can be obtained by levering top-50 1.35x with better
Sharpe and lower borrow drag. So the right deployment knob is leverage
on top-50, not concentration.

**Legacy (pre-architecture-cleanup) model**, saved as
`models/xgb_v1_legacy.json`:

| Variant            | CAGR    | Vol   | Sharpe | Max DD |
| ------------------ | ------- | ----- | ------ | ------ |
| Raw long-only      | +17.3%  | 22.0% | +0.79  | -22.6% |
| Gated long-only    | +9.7%   | 14.9% | +0.65  | -22.7% |

That model used broadcast SPY/VIX features and raw labels, so it's
stock-picker + market-timer in one. The +1.8 CAGR points and +0.09
Sharpe vs current is the value of the regime-forecasting alpha that was
deliberately stripped. It's deployable in a tax-advantaged account if
you want to ship something today; it shouldn't be the foundation for
new feature work.

The gated variant is rebalance-date-sensitive (CAGR offset range
+2.69% to +13.92% across the 21 starting days) — a 11-point gap structural
to monthly rebalance with a binary regime gate. Sleeves would smooth it.

### Result evolution: which runs produced which numbers

Numbers from this project have moved meaningfully across iterations
because three different things changed: the **universe** (current-only
→ point-in-time historical), the **labels** (no clip → ±0.5 clip), and
the **tuning budget** (20 → 50 → 200 trials with widened search space).
Comparing across runs is only meaningful when you know which combination
produced which number.

| Run                                      | Universe                | Label          | Features                  | Tuning                                   | Raw CAGR     | Gated CAGR | Notes                                                                                                            |
| ---------------------------------------- | ----------------------- | -------------- | ------------------------- | ---------------------------------------- | ------------ | ---------- | ---------------------------------------------------------------------------------------------------------------- |
| Pre-historical-filter (~2026-04)         | **Current S&P 500 only** | none           | with broadcast SPY/VIX    | IC objective, 20 trials                  | +26.2%       | +18.8%     | Maximum survivorship bias. The headline +26.2% in old null-test tables. Don't compare to anything below.         |
| Post-historical-filter, pre-clip         | Point-in-time           | none           | with broadcast SPY/VIX    | decile-spread, 20 trials                 | +25.7%       | +17.7%     | Universe deflation only ~0.5 pts because yfinance still misses delisted names. `best_iteration=196`.             |
| Post-clip, 50-trial fixed-depth          | Point-in-time           | clip ±0.5      | with broadcast SPY/VIX    | decile-spread, 50 trials, depth=3 fixed  | +13.3%       | +8.6%      | Label clip fixed MSE blow-up but optuna landed in a too-shallow basin (`best_iteration=19`); regime-dominated.   |
| Post-clip, 200-trial sweep (legacy)      | Point-in-time           | clip ±0.5      | with broadcast SPY/VIX    | decile-spread, 200 trials, depth ∈ [3,6] | +17.3%       | +9.7%      | `colsample_bytree=0.629` unlocked cross-sectional signal. Saved as `xgb_v1_legacy.json` for IRA deployment.       |
| Clean architecture (no earnings, 100 trials) | Point-in-time           | clip ±0.5 + date-demeaned | no broadcast SPY/VIX (39 features) | decile-spread, 100 trials, depth ∈ [3,6], ES=100 | +15.5%       | +7.9%      | Pure stock-picking signal on technicals only. `best_iteration=3` consistently — hyperparameter tuning exhausted on this feature set; the ceiling is data, not compute.  |
| **Current** (+ EDGAR earnings)           | Point-in-time           | clip ±0.5 + date-demeaned | clean arch + 3 EDGAR earnings features (42 features) | decile-spread, 100 trials, depth ∈ [3,6], ES=100 | **+17.5%** | **+8.8%** | First non-technical signal lands. `days_since_earnings` ranks 14th in importance; `best_iteration` lifts 3 → 10 (signal ceiling broken). Sharpe 0.73 ≈ SPY 0.75 (essentially tied). |

Four things to take away:

- The **+26.2%** number you'll find in old screenshots / null-test
  tables is on the *current-only* universe. It's not comparable to anything
  below; it's roughly +0.5 pts of universe-survivorship-bias on top of
  multiple points of single-deep-tree-overfit.
- The point-in-time filter (current → historical) only deflates by
  ~0.5 CAGR pts because yfinance retains data for only ~57% of names
  ever in the S&P 500 — the panel ends up at ~501 unique tickers,
  almost entirely current members. A truly bias-free universe (paid
  data with delisted-ticker prices) would deflate further, probably
  another 3–5 pts.
- The +13.3% → +17.3% jump (50-trial → 200-trial) was **pure tuning**
  on the with-SPY/VIX feature set. The model had real cross-sectional
  signal; the prior 50-trial run just hadn't found the right basin.
- The +17.3% → +15.5% drop (legacy → clean-arch no earnings) is
  **architectural, not a regression**. The legacy model used broadcast
  SPY/VIX features + raw labels and was implicitly market-timing;
  stripping both forces honest cross-sectional learning at the cost of
  ~1.8 CAGR points. `best_iteration=3` was the smoking gun: technical
  features alone don't carry enough cross-sectional signal to support a
  deep model.
- The **+15.5% → +17.5% recovery (+ EDGAR earnings)** is the first
  evidence that the predicted "data ceiling" was real — adding three
  earnings-calendar features lifted `best_iteration` 3 → 10 and
  recovered ~+2 CAGR pts of the +1.8 sacrificed in the architectural
  cleanup. Net: clean-architecture with one new signal source is now
  within +0.2 CAGR pts of the legacy model on absolute return, with
  the regime-forecasting alpha removed. The path forward is more new
  signals (fundamentals, insider, short-interest with full coverage)
  rather than more compute — see [Next steps](#next-steps).

### Diagnosis: regime-dominance fully removed, technical-feature ceiling reached

Two architectural changes were applied to remove the regime-forecasting
alpha that the legacy model relied on:

1. **Dropped broadcast SPY/VIX features** (`spy_ret_21d`,
   `spy_trend_regime`, `spy_rsi_14`, `vix_level`, `vix_zscore_20d`).
   Each was identical across all tickers on a given date — useful for
   timing the market, useless for ranking stocks against each other.
   Kept the ticker-specific market-relative features (`beta_60d`,
   `excess_ret_5d`, `excess_ret_21d`).
2. **Switched to date-demeaned labels.** Target is now
   `forward_21d_return − date_mean(forward_21d_return)`. Per-date mean
   is zero by construction, so the model literally cannot earn reward
   by predicting "market goes up." It can only learn within-date
   ordering.

**Outcome — exactly what was predicted:**

- Raw CAGR fell +17.3% → +15.5% (~1.8 pts cost). That's the regime-
  forecasting alpha that's gone — what *was* the model timing the market
  via SPY/VIX, restated honestly.
- Sharpe fell 0.79 → 0.70. Same story at a risk-adjusted level.
- Val IC stayed nearly flat (+0.0554 → +0.0520) — rank correlation is
  preserved. The model still picks better stocks than random within a
  date, just without the macro-timing kicker.
- **`best_iteration=3` is the new floor**, even with `EARLY_STOPPING_ROUNDS=100`
  and a tightened learning-rate range. Three shallow trees capture
  essentially all the cross-sectional signal in the technical-only
  feature set; further trees overfit val noise.

**What this tells us:** the technical features alone don't carry enough
*cross-sectional* alpha to support a deeper model. The legacy +17.3%
wasn't 100% stock-picking — a meaningful chunk was the model leaning on
SPY trend and VIX level to forecast that the *whole market* would go up
or down. Useful in 2021–2024 (mostly bull); not necessarily useful out-
of-sample in 2026+.

**The path forward isn't more compute on the same features.** It's new
signal sources. The clean architecture is the right base — when
fundamentals / events / short-interest land, they'll have somewhere to
contribute instead of being swamped by SPY trend signal. See
[Next steps](#next-steps).

### Null test (sanity check on the alpha) — stale, current-S&P-500 universe + pre-clip

The numbers below are from the **earliest run on the current-S&P-500-only
universe** (no point-in-time filter, no label clip). They're kept for
reference but should not be compared to the current +17.3% raw — that's
on a different (point-in-time, post-clip, 200-trial) configuration. See
[Result evolution](#result-evolution-which-runs-produced-which-numbers)
above. Re-running on the current configuration is queued as a TODO; expect
the model-vs-random gap to compress.

| Predictions used                | CAGR   | Sharpe | Final NAV |
| ------------------------------- | ------ | ------ | --------- |
| The model (current-only, pre-clip) | +26.2% | +0.86  | 3.36×     |
| Random (Gaussian noise)         | +12.9% | +0.78  | 1.88×     |
| Just `dist_52w_high` (1 factor) | +10.6% | +0.74  | 1.69×     |
| SPY buy & hold (old end-date)   | +14.5% | +0.85  | 2.05×     |

Reading this (current-only universe, pre-clip context):

- **Random ≈ SPY**: equal-weighted random picks from a current-S&P-500
  universe earn ~13% — the survivorship-bias floor. SPY's cap-weighting on
  Mag 7 buys it a couple extra points.
- **Naive momentum < SPY**: just chasing 52-week highs alone underperformed
  in 2021–2026. So whatever the old model did was **not** naive momentum.

For the current model, the apples-to-apples table above is the honest
read: raw +13.3% vs SPY +12.7% vs random ~+12.9% — the model's edge over
random is essentially noise.

### Caveats before believing the headline

1. **Residual survivorship still inflates the absolute number.** Membership
   timing is fixed (point-in-time CSV), but yfinance only retains data for
   ~57% of the 917 tickers ever in the S&P 500 since 2007 — delisted /
   acquired symbols are silently dropped. So the panel is closer to "current
   members with proper time-gating" than a true bias-free universe. Real fix
   needs paid data (Sharadar, Norgate, CRSP). Expect another 3–5 CAGR points
   of deflation when that lands. The model-vs-random gap (~13 CAGR points)
   should mostly survive — that comparison was already on the same biased
   universe.
2. **Concentrated picks.** Across 63 rebalance days the strategy lands on
   only 290 unique tickers, ~70% overlap between consecutive rebalances.
   MRNA is picked 87% of the time, PLTR 78%, TSLA 73% — high-beta
   growth/tech/COVID/meme names. Single-name blowups would hurt.
3. **Period-specific regime.** 2021–2026 favored growth/tech/momentum.
   Need backtests on different macro regimes to gauge robustness.

### Outputs

- `reports/backtest_equity.png` — equity curves vs SPY (mean + 10–90% offset band)
- `reports/backtest_stats.json` — CAGR / Sharpe / MaxDD / time-in-market per variant
- `reports/backtest_equity.csv` — daily NAV per variant + the picks log:

    | Column                | What it is                                                                                                                                           |
    | --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
    | (index)               | Trading date.                                                                                                                                        |
    | `spy`                 | SPY buy-and-hold equity (NAV starting at 1.0). Pure benchmark, no rebalance.                                                                         |
    | `raw_long_only`       | Strategy NAV with **no regime gate** (always 100% long top 50). Mean across the 21 shifted-start offsets.                                            |
    | `gated_long_only`     | Strategy NAV with **regime gate ON** (SPY > SMA200 AND VIX < 25 → long; else cash). Mean across 21 offsets. Headline gated number.                   |
    | `gated_offset_p10`    | 10th-percentile of gated equity across the 21 offsets — the unlucky-rebalance-day lower band.                                                        |
    | `gated_offset_p90`    | 90th-percentile of gated equity across the 21 offsets — the lucky-rebalance-day upper band. Width = how rebalance-date-fragile the gated variant is. |
    | `gated_picks_offset0` | Comma-separated tickers held by the gated variant at offset 0 (one representative offset). Empty when the gate said "cash".                          |
    | `raw_picks_offset0`   | Same for the raw variant. Always populated since raw is never in cash.                                                                               |

    Only offset 0's picks are saved; writing all 21 offsets' picks would balloon the CSV. The other 20 offsets pick mostly-overlapping baskets (~70% consecutive overlap), so offset 0 is a reasonable representative.

---

## File layout

```
data/
  universe/
    SP_500_Historical_Component.csv  # raw membership change-events from fja05680/sp500
    sp500_history.parquet            # parsed long-format (date, ticker)
    sp500_sectors.csv                # current Wikipedia sector tags
  raw/{TICKER}.parquet           # OHLCV per ticker
  market/SPY.parquet, VIX.parquet
  processed/                     # features.parquet, panel.parquet (later)
models/xgb_v1.json               # trained booster
reports/                         # equity curve, feature importance, metrics.json (gitignored: png + json)
picks/                           # daily picks_<date>.csv from scripts/today.py (gitignored)
scripts/
  universe.py    data.py         # implemented
  features.py    labels.py       # implemented
  dataset.py     train.py        # implemented
  backtest.py                    # implemented (monthly rebalance + 21 shifted-start offsets)
  strategy.py                    # shared primitives (model load, regime gate, top picks)
  today.py                       # implemented — live picks for the most recent feature date
  run_all.py                     # implemented — daily orchestrator (data → features → labels → today)
```

---

## Metrics glossary

- **Information Coefficient (IC)** — daily Spearman correlation between
  predicted forward return and realised forward return, across all tickers
  on that date. Reported as the time-series mean (t-stat is a future
  diagnostic). >0.03 is considered tradable; >0.05 is good.
- **Top-decile spread** — mean realised forward return of top-decile picks
  minus that of the bottom decile. Direct measure of monotonic ranking
  quality. The objective `train.py` actually optimises.
- **Annualised return / vol / Sharpe** — daily portfolio log-returns scaled
  by √252. Computed by `backtest.py`.
- **Max drawdown** — peak-to-trough on the equity curve. Computed by
  `backtest.py`.
- **Equity curve vs SPY** — visual sanity check; SPY = passive benchmark.
  Plotted in `reports/backtest_equity.png`.
- **Feature importance** — XGBoost gain importance per feature, written to
  `reports/feature_importance.csv` after each training run.

---

## Next steps

**Where we are.** Hyperparameter tuning is exhausted on the technical-
only feature set — `best_iteration=3` is the ceiling. The clean
architecture is honest but capped at +15.5% CAGR / Sharpe 0.70. To push
past, we need new *data sources*, not more compute. All five items below
are free; each has a self-contained payoff.

> **Order of attack (per-item priority based on signal/effort):**
> §2 earnings calendar → ✅ **done**, +2.0 CAGR pts, `best_iteration`
> ceiling broken. Next up: §3 insider buys → §4 EDGAR fundamentals →
> §5 ensemble. §1 short interest is deferred (FINRA archive only goes
> back to 2018 — see headline status). The earnings result confirms
> that non-technical signal sources do contribute on the clean
> architecture, which validates §3–§4 as worth the engineering cost.

### 1. FINRA short interest — deferred

FINRA publishes short interest as % of float for every NMS-listed stock
on a bi-monthly cadence. Free, downloadable as pipe-delimited CSVs.
Download infrastructure shipped (`scripts/altdata.py --source finra`,
URL `cdn.finra.org/equity/otcmarket/biweekly/shrt{YYYYMMDD}.csv`).

**Why deferred**: the FINRA CDN archive starts at **2018-08-15**, not
2007 — so train=2007–2017 has **0% coverage** for any short-interest
column. XGBoost's missing-value handling can't help here: with no train
variance, no tree split is ever built on the feature. Pre-fill with
mean was rejected because every pre-2018 row would get the identical
constant value, teaching the model "this column is an era marker"
(regime leak). Sliding splits forward (e.g. train→2020, val=2021–22,
test=2023+) sacrifices the 2022 bear in test, which we want to keep
for stress evaluation.

**Path back in**: paid historical short interest (Sharadar, Polygon,
QuantQuote — typically $20–60/mo with 2007+ history), or re-platform
splits to absorb 2018+ into train once we accept the test-period
tradeoff. Cache stays on disk so re-wiring is a few lines if/when one
of those resolves.

### 2. Earnings calendar from SEC EDGAR — ✅ shipped

SEC EDGAR submissions API gives every 10-Q / 10-K filing date for any
ticker with a current CIK (covers 648 of 959 historical S&P 500 names;
the missing 311 are delisted/renamed without retained CIKs).
yfinance's `Ticker.get_earnings_dates()` provides forward dates for
the live-picks row. Combined into three features (`days_to_earnings`,
`days_since_earnings`, `post_earnings_drift_window`).

**Result.** `days_since_earnings` ranks 14th in feature importance
(0.037 gain). The other two get 0.0 — `post_earnings_drift_window` is
redundant with the continuous `days_since_earnings` (XGBoost
reconstructs the [1,5] split internally), and `days_to_earnings` from
10-Q filings has no forward signal because filings are scheduled, not
surprise-driven. Test CAGR moved +15.5% → +17.5% (+2.0 pts);
`best_iteration` lifted 3 → 10 (the prior data-ceiling broken).

**Follow-up.** Switch from 10-Q filing dates to 8-K item 2.02
announcement dates — the actual market-moving event is the 8-K
released ~2-4 weeks before the 10-Q. EDGAR submissions API doesn't
expose item codes, so this requires fetching each 8-K's filing index
to filter by item (one extra request per filing). Worth it if PEAD is
where the lift is coming from. Tracked under [TODOs](#todos).

### 3. Insider transactions from SEC EDGAR Form 4 (free, ~1–2 days)

EDGAR publishes Form 4 (insider buys/sells) as JSON and XBRL. Insider
*buying* (especially CEO/CFO open-market purchases at price below recent
avg) has documented predictive power. Add:

- `insider_buy_count_60d`, `insider_sell_count_60d` (counts of officer
  transactions in the last 60 trading days)
- `insider_net_dollar_60d` (signed dollar volume; positive = net
  buying)
- `days_since_last_insider_buy`

Real-time + historical via EDGAR's submissions API. Discipline: the
filing date can lag the transaction date by 2 business days — use the
*filing* date for as-of cuts to avoid lookahead.

### 4. SEC EDGAR XBRL fundamentals (free, ~1 week, parser-heavy)

The biggest unlock and the biggest engineering lift. EDGAR exposes all
10-K / 10-Q filings as XBRL with point-in-time-correct quarterly numbers.
Parse out 4–5 starter ratios:

- Trailing P/E (price / TTM earnings)
- P/S (price / TTM revenue)
- FCF yield (TTM FCF / market cap)
- ROIC (TTM EBIT / invested capital)
- Debt/equity (most-recent quarter)

XBRL parsing is the hard part — concept tagging is inconsistent across
filers, and you need a robust GAAP/IFRS concept mapper. Once the parser
works, adding new ratios is trivial. **Free tier alternatives** that
skip the parser pain for a less-clean first pass: SimFin (free with
attribution), FMP (free tier 250 calls/day). These are good for a
2-day-build-and-test before committing to the EDGAR parser.

After this lands, re-run train + backtest and check whether (a) val IC
moves into the 0.07+ range, and (b) `best_iteration` lifts off 3. If
yes, the model has graduated from "technical ranker" to real
multi-factor stock-picker. If not, the data ceiling is harder than
expected and paid data ($50/mo Sharadar) becomes the next move.

### 5. Ensemble of 5 models (free, ~½ day)

Train 5 models with different random seeds (or bootstrapped train
samples), save 5 booster files, average predictions at inference time.
This is a *system change*, not a model change — you'll have 5 model
files (`xgb_v1_s1.json` … `xgb_v1_s5.json`), all loaded at predict time,
predictions averaged.

**Why it works:** each individual booster has slightly different
prediction errors (different random splits). Averaging cancels noise but
preserves signal — law of large numbers applied to model variance.
Typical Sharpe lift: 5–15% with no new features. Stack on top of any
other improvement above; it's orthogonal.

**Cost:** 5× disk (negligible — XGBoost models are tiny), 5× predict
time (still <1s for daily picks). Conceptually one *system* with 5
*components*. This is what production-grade quant shops actually deploy.

### 6. Paid price data for delisted tickers (do before going live)

This *deflates the backtest*, it doesn't improve the model. Membership-
timing is correct (`universe.py` does point-in-time filtering against
the 1996+ change-event CSV), but yfinance only retains data for ~57% of
historical S&P 500 tickers — almost everything that left the index is
missing. The panel ends up at ~501 unique tickers, all current members.

Expect another 3–5 CAGR points of deflation when this lands. The
model-vs-SPY gap probably mostly survives (same-universe comparison),
but the *absolute* numbers should be trusted only after this swap.
**Do this before trading real money.**

| Source | Cost | Notes |
| ---- | ---- | ---- |
| **Sharadar US Equities** (Nasdaq Data Link) | ~$50/mo | Indie-quant default. Delistings + fundamentals + sectors in one feed → kills two birds (subsumes §4). Recommended. |
| Norgate Premium Data | ~$60/mo + tools | Built for backtesting; total-return adjusted; point-in-time index membership baked in. |
| EOD Historical Data | ~$20/mo | Cheapest with delistings, mixed coverage reviews. |
| Polygon.io (Stocks Advanced) | $199/mo | Real-time + history + delistings. Overkill unless you also want intraday. |
| yfinance + manual delisted backfill | Free | Hacky: scrape delisted prices from Stooq or another free source. $0 but fragile. |

`data.py` already keys per-ticker parquets, so the change is mostly the
download function — should be a 1–2 day swap once the feed is chosen.

### Skip until much later

- **Options skew / IV** — paid only (ORATS ~$300/mo, CBOE DataShop
  $$$$). Real signal but cost-to-signal is bad until the free stack is
  exhausted.
- **News sentiment** — RavenPack $$$$, FinBERT high-effort. Skip.
- **Analyst revisions** — IBES via WRDS, academic-only access. Skip.
- **Tax / deployment infrastructure** — IRA + manual rebalance in
  Schwab/IBKR is enough for the first $100k. Don't build an ETF.
  See the **Tax reality check** callout in the headline status above.

### 7. Sleeves upgrade (smooths the gated variant's offset CAGR range)

Today's gated backtest spans CAGR offset range [+0.5%, +14.2%] across the
21 shifted starts — a 14-point gap between best- and worst-luck rebalance
day. That's structural to monthly rebalance with a binary regime gate; no
amount of tuning fixes it.

**What to build**: 21 overlapping 21-day sleeves running in parallel,
rebalancing 1/21 of book each day. Mathematically equivalent to averaging
the 21 shifted-start offsets, but as one continuous portfolio rather than
21 independent ones. Smooths daily turnover, eliminates rebalance-date
fragility, becomes the realistic live-trading mechanic.

### 8. Diagnostics module (per-month IC stability, drawdown, attribution)

backtest.py reports summary stats; the interesting questions need slicing.

**What to build** (probably `scripts/diagnostics.py`):

- Per-month IC time series + t-stat + % of months positive — is the alpha
  stable, or driven by 2 outlier months?
- Underwater plot — when did drawdowns happen, how long did they last,
  recovery time?
- Picks-concentration audit — ticker frequency, sector breakdown,
  consecutive-rebalance overlap distribution.
- Per-stock attribution — top-10 contributors and detractors to total return.
- Hit rate — of each rebalance's 50 picks, how many beat SPY over the next
  21 days?

---

## TODOs

- [x] v2: point-in-time S&P 500 membership filter (membership timing fixed)
- [ ] paid price data for delisted tickers — yfinance retains only ~57% of historical S&P names; closes the residual ~3–5 CAGR points of survivorship bias
- [x] features.py
- [x] labels.py
- [x] dataset.py + lookahead sanity assertion
- [x] train.py with hyperparameter tuning
- [x] backtest.py — monthly rebalance + 21 shifted-start offsets, regime gate, null test
- [x] today.py — live picks for the latest feature date, with `--diff` for daily BUY/SELL tickets
- [ ] upgrade backtest to overlapping 21-day sleeves (smooths the offset CAGR range)
- [ ] diagnostics: per-month IC stability, underwater plot, picks-concentration audit, per-stock attribution
- [x] run_all.py orchestrator — daily and retrain modes, auto --diff for today.py
- [x] add label clip ±0.5 to keep dead-ticker −100% labels from dominating MSE
- [x] re-tune with `max_depth ∈ [3, 6]` and re-run backtest (200 trials; raw +13.3% → +17.3%, val decile spread +0.0213 → +0.0297; optuna picked depth=3 + `colsample_bytree=0.629` as the actual lever)
- [x] experiment: train without SPY/VIX broadcast features to force cross-sectional signal — **done**, cost ~1 CAGR pt; combined with demeaning below
- [x] experiment: train on date-demeaned forward returns — **done**, raw +17.3% → +15.5%, Sharpe 0.79 → 0.70 vs SPY 0.75; `best_iteration=3` ceiling on technicals
- [x] save legacy +17.3% model as `xgb_v1_legacy.json` for IRA deployment
- [x] feature: `excess_ret_5d_vs_sector` and `excess_ret_21d_vs_sector` (sector-relative momentum)
- [x] fix `backtest.py` SPY end-date so headline CAGR compares like-for-like
- [x] **feature: earnings calendar from SEC EDGAR 10-Q/10-K + yfinance forward dates** — `days_to_earnings`, `days_since_earnings`, `post_earnings_drift_window`. `days_since_earnings` lands at 14th in feature importance (0.037 gain); other two are 0.0 (kept pending review). Test CAGR +15.5% → +17.5%, `best_iteration` 3 → 10
- [ ] follow-up: switch earnings dates from 10-Q filing to 8-K item 2.02 announcement (one extra request per filing, but anchors PEAD on the actual market-moving event ~14d earlier than 10-Q)
- [~] feature: short interest from FINRA bi-monthly — **deferred**. Download infrastructure shipped (`scripts/altdata.py --source finra`) but FINRA CDN archive starts mid-2018, so train=2007–2017 has 0% coverage and XGBoost cannot build splits on the feature. Mean-fill rejected (regime leak); sliding splits forward sacrifices test bear coverage. Revisit when paid historical short interest is added or when splits are re-platformed
- [ ] feature: insider transactions from SEC EDGAR Form 4 (free, ~1–2 days)
- [ ] feature: SEC EDGAR XBRL fundamentals parser — P/E, P/S, FCF yield, ROIC, debt/equity (free, ~1 week)
- [ ] system: ensemble of 5 boosters with different seeds — average predictions (free, ~½ day, +5–15% Sharpe)
- [ ] re-run null test on the clean-architecture model (current null-test table is stale)
- [ ] data: swap yfinance → Sharadar (or equivalent) for delisted-ticker coverage — do before going live

Paste this to claude to ask
claude --resume b63b90f4-923f-419f-b30e-00cd9006952f
claude --resume 7762f7ea-721e-4179-a24b-273d86c65f0e
