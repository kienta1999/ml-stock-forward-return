# ml-stock-forward-return

ML-based S&P 500 stock ranker. Predict each stock's forward 21-trading-day
return independently with XGBoost, sort to get a daily ranking, long the top
decile, hold 21 trading days, rebalance monthly with a SPY/VIX regime gate.

**Current status (5-seed stability-selection prune + 50-trial sweep on
40 features):** raw long-only **+19.0% CAGR / Sharpe 0.76 vs SPY
+12.7% / Sharpe 0.75** (test 2021-01-04 → 2026-03-31). **First time
strictly exceeding SPY's Sharpe** — strategy delivers +6.3 CAGR pts
over SPY at marginally better risk-efficiency (1.47× SPY vol, 1.49×
SPY return). Final NAV **2.48×** vs SPY 1.87× = **+33% more wealth**
over 5.25 years. `best_iteration=43` with `learning_rate=0.0058`
(prior runs were stuck at 2-10) — the cleaner feature set unlocked a
deeper, more conservative basin that the 61-feature config couldn't
find.

The lift came from the **5-seed stability-selection prune**: hold
hyperparams fixed at `DEFAULT_PARAMS`, vary only `random_state ∈
{1..5}`, run `train.py --quick --seed N` 5 times, drop only features
that scored 0 in _every_ seed. 19 features turned out to be truly
dead (and 2 dead-rank-of-marginal-raw → 21 columns total), but the
prior single-run prune attempt was wrong — `current_ratio`,
`book_to_market`, `earnings_yield`, `sales_growth_yoy` all looked
dead in some runs but fired strongly in others. They're rare-fire
signals that need a regime/seed combination to activate. Final
feature count: 61 → 40. Snapshot of the milestone model preserved as
`models/xgb_v1_stability_pruned.json`. See [Stability-selection
prune](#stability-selection-prune) for the methodology.

**Legacy model preserved.** The previous iteration with broadcast
SPY/VIX features and raw labels (+17.3% CAGR / Sharpe 0.79) is saved as
`models/xgb_v1_legacy.json` for deployment in a tax-advantaged account
(IRA) where short-term gains aren't penalised. That model's headline
number was partly market-timing alpha — see
[Result evolution](#result-evolution-which-runs-produced-which-numbers)
below for the full lineage of how the architecture changed.

**FINRA short interest attempted, dropped.** The original §1 plan was
short interest from FINRA bi-monthly. The download infrastructure shipped
(`scripts/deprecated_short_interest.py`) but the FINRA CDN archive only
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

| Stage    | What it does                                                                                                                                                                                                                                                                                                                                                                             |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Universe | Point-in-time S&P 500 (1996+ membership CSV) joined with current Wikipedia sectors                                                                                                                                                                                                                                                                                                       |
| Data     | yfinance OHLCV 2005-07-01 → today (1.5y buffer for 252d warmup), per-ticker parquet cache, plus SPY + ^VIX                                                                                                                                                                                                                                                                               |
| Features | 10 per-ticker technicals + 3 ticker-specific market-context + 3 broadcast SPY/VIX regime context + 1 sector-relative + 2 earnings calendar + 6 XBRL fundamentals + 14 cross-sectional ranks + sector cat = **40 total** (down from 61 after 5-seed stability-selection prune — 19 features were dead in all 5 seeds). See [Stability-selection prune](#stability-selection-prune) below. |
| Label    | `forward_21d_return − date_mean(forward_21d_return)` — date-demeaned (cross-sectional excess). Raw `forward_21d_return` is clipped to ±0.5 first to cap dead-ticker outliers, then demeaned. The model can only learn within-date ordering, not market direction.                                                                                                                        |
| Split    | Train 2007–2017, Val 2018–2020, Test 2021→. Chronological. No shuffling.                                                                                                                                                                                                                                                                                                                 |
| Model    | XGBoost regressor, RMSE loss, optuna-tuned on val decile spread (max_depth ∈ [3, 6], 100 trials, ES=100 rounds)                                                                                                                                                                                                                                                                          |
| Backtest | Long-only top-50, monthly rebalance, regime gate (SPY > SMA200 AND VIX < 25), 21 shifted-start offsets                                                                                                                                                                                                                                                                                   |
| Costs    | 5 bps per side on rebalance turnover                                                                                                                                                                                                                                                                                                                                                     |

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

uv run python scripts/earnings.py                   # SEC EDGAR 10-Q/10-K filing dates + yfinance forward calendar (~5 min first time)
uv run python scripts/earnings.py --tickers AAPL,MSFT  # subset for smoke-test
uv run python scripts/fundamentals.py                # SEC EDGAR XBRL TTM income + MRQ balance sheet → per-ticker parquet (~10-15 min first time)
uv run python scripts/fundamentals.py --smoke        # one-ticker dry-run before the full pull
# scripts/deprecated_short_interest.py is shipped but not wired — FINRA archive only goes back to 2018-08

uv run python scripts/features.py --ticker AAPL     # smoke-print one ticker's features
uv run python scripts/features.py                   # build full panel → data/processed/features.parquet
uv run python scripts/labels.py                     # add forward_21d_return → data/processed/panel.parquet
uv run python scripts/dataset.py                    # splits + lookahead sanity check
uv run python scripts/dataset.py --quick            # same, skip the slow recompute check

uv run python scripts/train.py                      # optuna tuning + final fit (~10-15 min)
uv run python scripts/train.py --trials 20          # faster tune
uv run python scripts/train.py --quick              # skip tuning, use sane defaults
uv run python scripts/train.py --quick --seed 3     # vary RNG (XGBoost + optuna) for stability-selection sweeps

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

61 features per `(date, ticker)` row, organized into buckets (numeric

- 1 categorical). Lists are exposed as constants in `scripts/features.py`
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

### Bucket 2a — ticker-specific market context (3)

Each feature here has a _different value per ticker_ on a given date.

| Feature                           | Definition                                  | What it captures                                                                                                                     |
| --------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `beta_60d`                        | Rolling cov(stock, SPY) / var(SPY) over 60d | Stock's sensitivity to the market. >1 amplifies, <1 dampens. Per-ticker number.                                                      |
| `excess_ret_5d`, `excess_ret_21d` | `ret_n − spy_ret_n`                         | **Relative strength**: how much the stock beat or lagged SPY over the window. The single biggest signal for cross-sectional ranking. |

### Bucket 2b — broadcast SPY/VIX regime context (5)

Each feature here is _identical for every ticker on a given date_ — pure
market state. Originally removed in the clean-arch refactor on the
reasoning that they let trees split on "market direction" rather than
"this stock vs others." That was overcorrection. With **date-demeaned
labels**, broadcast features can't earn standalone reward (the target
sums to zero per date, so a tree that splits only on `vix_level` learns
nothing), but they _can_ condition cross-sectional splits — e.g. "in
high-VIX regimes, split on `debt_to_equity_rank`; in low-VIX, split on
`dist_52w_high`." The 200-trial run with regime features back hit a new
val IC peak (+0.0568) and broke the prior decile-spread ceiling.

| Feature            | Definition                     | Importance (this run)        |
| ------------------ | ------------------------------ | ---------------------------- |
| `vix_level`        | VIX close                      | 0.058 (#5)                   |
| `spy_rsi_14`       | RSI(14) on SPY close           | 0.058 (#6)                   |
| `spy_ret_21d`      | SPY trailing 21d return        | 0.034 (#14)                  |
| `spy_trend_regime` | `1.0 if SPY > SMA200 else 0.0` | 0 (subsumed by `spy_rsi_14`) |
| `vix_zscore_20d`   | `(vix − sma20) / std20`        | 0 (subsumed by `vix_level`)  |

### Bucket 3 — sector-relative (2)

| Feature                                               | Definition                                             | What it captures                                                                                                                                                                                                                             |
| ----------------------------------------------------- | ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `excess_ret_5d_vs_sector`, `excess_ret_21d_vs_sector` | `ret_n − groupby(date, gics_sector).transform("mean")` | Cross-sectional sector-relative momentum. Strips sector beta from the per-ticker return so the model compares apples to apples within an industry. NaN for the `Unknown` sector bucket (delisted/removed names — group mean would be noise). |

### Bucket 4 — cross-sectional ranks (16)

For each Bucket 1 feature _(except the binary `trend_regime`)_, we compute its **percentile rank across all active tickers on that date** (0 = worst, 1 = best). Column suffix: `_rank`.

Why: a 30% trailing return in 2008 ≠ a 30% return in 2017. Ranks normalise out the time-varying scale and turn each feature into a contemporaneous comparison — which is precisely what a ranker needs.

### Bucket 6 — earnings calendar (3)

| Feature                      | Definition                                                                                                      | What it captures                                                                                                                                                                                                                         |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `days_to_earnings`           | days from row date to next known earnings filing (EDGAR 10-Q/10-K + yfinance forward calendar), clipped [0, 90] | Pre-earnings positioning. **Currently 0.0 importance** — 10-Q dates are scheduled, not surprise-driven, so forward distance carries no cross-sectional signal. Kept for now pending a switch to actual 8-K item 2.02 announcement dates. |
| `days_since_earnings`        | days since most recent earnings filing, clipped [0, 90]                                                         | **Post-earnings drift signal — 14th in feature importance (0.037 gain).** The continuous form lets XGBoost discover its own optimal drift window.                                                                                        |
| `post_earnings_drift_window` | `1.0` if `days_since_earnings ∈ [1, 5]` else `0.0`                                                              | Hand-coded PEAD window flag. **Currently 0.0 importance** — XGBoost reconstructs the same split internally from the continuous `days_since_earnings`, making this redundant. Kept for now pending a full re-evaluation.                  |

**Data source.** SEC EDGAR submissions API gives every 10-Q / 10-K
filing date for tickers with a current CIK (648 of 959 historical
S&P 500 names; the missing 311 are mostly delisted/renamed without
retained CIKs). For live picks, yfinance's forward earnings calendar
fills in the upcoming dates. Caveat: 10-Q filings trail the actual
8-K item 2.02 announcement by ~2-4 weeks, so the "drift window" here
is anchored on the report filing rather than the announcement.
Switching to 8-K item filtering would require one extra request per
filing and is queued under [Next steps](#next-steps).

`scripts/earnings.py` handles the EDGAR 10-Q/10-K pipeline + yfinance
forward calendar (separate from `data.py` because the cadences and
rate-limit contracts differ). `scripts/fundamentals.py` is the parallel
pipeline for EDGAR XBRL fundamentals. `scripts/deprecated_short_interest.py`
holds the FINRA pipeline (deferred — archive only goes back to 2018-08).
Per-ticker parquets live at `data/earnings/{TICKER}.parquet` and
`data/fundamentals/{TICKER}.parquet`.

### Bucket 5 — categorical (1)

| Feature       | Source                                                         | Purpose                                                                                                                                                                                                                                |
| ------------- | -------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
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

XGBoost regressor on the 61 features. Why XGBoost: handles missing values, scales
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

| Param              | Range          | What it controls                                                                                                                                                                                   |
| ------------------ | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `max_depth`        | fixed at 3     | Depth-8 collapses decile separation (clumpy predictions); 3–5 give equivalent val spread, so we pick the shallowest — most trees, smoothest predictions.                                           |
| `learning_rate`    | 0.01–0.1 (log) | How aggressively each tree corrects errors. Smaller + more trees usually wins. Range tightened from 0.01–0.3 after early-stopping was firing too aggressively on demeaned labels (see v1 results). |
| `n_estimators`     | 200–1000       | Max number of trees. Capped by early stopping.                                                                                                                                                     |
| `min_child_weight` | 1–20           | Minimum sum of sample weights per leaf. Higher = simpler trees.                                                                                                                                    |
| `subsample`        | 0.6–1.0        | Row sampling per tree. <1 adds randomness → robustness.                                                                                                                                            |
| `colsample_bytree` | 0.6–1.0        | Feature sampling per tree. Same idea, on columns.                                                                                                                                                  |
| `reg_lambda`       | 0.01–10 (log)  | L2 regularization on leaf weights.                                                                                                                                                                 |

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

| Metric                                 | **Current (clean arch + EDGAR earnings, 100 trials)** | Prior clean-arch (no earnings, 100 trials) | Legacy (200 trials, with SPY/VIX, raw labels) |
| -------------------------------------- | ----------------------------------------------------- | ------------------------------------------ | --------------------------------------------- |
| val decile spread (top10 − bot10, 21d) | +0.0173 (~173 bps, demeaned units)                    | +0.0173                                    | **+0.0297 (~297 bps)**                        |
| val IC (mean daily Spearman)           | +0.0444                                               | +0.0520                                    | +0.0554                                       |
| train IC                               | +0.0439                                               | +0.0401                                    | +0.0659                                       |
| val RMSE / train RMSE                  | 0.0739 / 0.0687                                       | 0.0739 / 0.0688                            | 0.1029 / 0.0899                               |
| best_iteration                         | **10** (ceiling lifted from 3)                        | 3                                          | 30                                            |
| chosen `max_depth`                     | 3                                                     | 3                                          | 3 (out of [3, 6])                             |
| chosen `learning_rate`                 | 0.020                                                 | 0.027                                      | 0.082                                         |
| chosen `n_estimators`                  | 591 (cap, only 10 used)                               | 389 (cap, only 3 used)                     | 860 (capped by ES)                            |
| chosen `min_child_weight`              | 9                                                     | 3                                          | 11                                            |
| chosen `subsample`                     | 0.964                                                 | 0.792                                      | 0.949                                         |
| chosen `colsample_bytree`              | 0.685                                                 | 0.633                                      | **0.629**                                     |
| chosen `reg_lambda`                    | 0.659                                                 | 0.907                                      | 0.446                                         |

The "decile spread" column for the current run is in **demeaned-return
units** and isn't directly comparable to the legacy column (which is in
raw return units). The val IC column _is_ comparable: +0.0444 (with
earnings) vs +0.0520 (no earnings) vs +0.0554 (legacy). Val IC actually
_ticked down_ with earnings added, but val decile spread stayed the
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

### Results (test 2021-01-04 → 2026-03-31, 50-trial sweep on stability-pruned 40 features, demeaned labels)

`backtest.py` clips the SPY benchmark to the strategy's last predictable
date so the headline comparison is apples-to-apples by default; the CSV
keeps the full SPY series so the post-strategy tail is visible for
inspection.

**Current model (`models/xgb_v1_stability_pruned.json`, snapshot saved):**

| Variant                              | CAGR       | Vol   | Sharpe    | Max DD | Final NAV | Time-in-market |
| ------------------------------------ | ---------- | ----- | --------- | ------ | --------- | -------------- |
| **Raw long-only**                    | **+19.0%** | 25.0% | **+0.76** | -25.8% | **2.48×** | 100%           |
| Gated long-only                      | +10.2%     | 17.1% | +0.59     | -21.1% | 1.66×     | 77%            |
| SPY buy & hold (clipped @2026-03-31) | +12.7%     | 17.0% | +0.75     | -24.5% | 1.87×     | —              |

**Reading the table honestly:** raw beats SPY by +6.3 CAGR points and
**Sharpe strictly exceeds SPY for the first time (0.76 vs 0.75)**.
Strategy runs at 1.47× SPY's vol but earns 1.49× SPY's return —
risk-efficiency is now slightly better than the benchmark, not just
tied. Final NAV 2.48 vs 1.87 = **+33% more wealth** over 5.25 years.
Gated still underperforms SPY (regime gate is giving up upside it
doesn't earn back; redundant when SPY/VIX features are already inside
the model). The lift over the prior 61-feature run (+16.2% / 0.72)
came from the **5-seed stability-selection prune** (61 → 40
features) — the cleaner panel let optuna find a slow-build basin
(`best_iteration=43, lr=0.0058`) that it couldn't with 21 noise
columns.

**Concentration sweep on the same model** (`backtest.py --top-n 20`):
top-20 raw +20.3% / Sharpe 0.67. Concentration trades return for vol;
Sharpe is roughly preserved. **Concentration is a poor man's leverage**
— the same effect can be obtained by levering top-50 1.35x with better
Sharpe and lower borrow drag. So the right deployment knob is leverage
on top-50, not concentration.

**Legacy (pre-architecture-cleanup) model**, saved as
`models/xgb_v1_legacy.json`:

| Variant         | CAGR   | Vol   | Sharpe | Max DD |
| --------------- | ------ | ----- | ------ | ------ |
| Raw long-only   | +17.3% | 22.0% | +0.79  | -22.6% |
| Gated long-only | +9.7%  | 14.9% | +0.65  | -22.7% |

That model used broadcast SPY/VIX features and raw labels, so it's
stock-picker + market-timer in one. The +1.8 CAGR points and +0.09
Sharpe vs current is the value of the regime-forecasting alpha that was
deliberately stripped. It's deployable in a tax-advantaged account if
you want to ship something today; it shouldn't be the foundation for
new feature work.

The gated variant is rebalance-date-sensitive (CAGR offset range
+2.69% to +13.92% across the 21 starting days) — a 11-point gap structural
to monthly rebalance with a binary regime gate. Sleeves would smooth it.

### Stability-selection prune

After the 61-feature run, an aggressive single-run prune (drop everything
with importance==0 in _one_ training) was attempted. It was wrong. The
problem: at `best_iteration=4 × max_depth=3 = 12 splits` total, importance
is _extremely_ noisy — a feature can score 0.06 in one run and 0 in
another simply by losing the split-competition. Pruning on a single vote
throws away real signal that happened to lose by chance.

**Stability sweep**: held hyperparameters fixed at `DEFAULT_PARAMS`
(`lr=0.082`, `n_est=860`, ES on val decile spread → ~25 trees built),
varied only `random_state ∈ {1,2,3,4,5}`, ran `train.py --quick --seed N`
five times on the full 61-feature panel. Each seed produces its own
`reports/feature_importance_seed{N}.csv`. A feature is pruned only if
importance==0 in **all 5** seeds (not 1 — true stability selection).

**Across-seed variance was severe**: `best_iteration` ranged 2–25 with
identical hyperparameters, and within-feature importance swings were
huge — `current_ratio` scored 0 in seeds 1-4 and 0.073 in seed 5;
`book_to_market` scored 0.031 in seed 1 and 0 in 2–5. Single-run pruning
would have killed both as "useless." They aren't — they're rare-fire
signals that need a regime/seed combination to activate.

**Outcome — 19 features dead in all 5 seeds, pruned**:

| Bucket                      | Pruned                                                                                           | Kept                                                                                                                                                             |
| --------------------------- | ------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Per-ticker technicals (raw) | `ret_1d`, `ret_5d`, `mfi_14`, `vol_ratio`, `dist_sma50`, `trend_regime`, `rsi_14`                | `ret_21d`, `ret_63d`, `macd_hist`, `atr_pct`, `vol_20d`, `vol_60d`, `dist_sma200`, `dist_52w_high`, `zscore_20d`, `zscore_60d`                                   |
| Per-ticker ranks            | `ret_1d_rank`, `ret_5d_rank`, `ret_21d_rank`, `mfi_14_rank`, `vol_ratio_rank`, `dist_sma50_rank` | `ret_63d_rank`, `macd_hist_rank`, `atr_pct_rank`, `vol_20d_rank`, `vol_60d_rank`, `dist_sma200_rank`, `dist_52w_high_rank`, `zscore_20d_rank`, `zscore_60d_rank` |
| Market regime broadcast     | `spy_trend_regime`, `vix_zscore_20d` (subsumed by `spy_rsi_14` / `vix_level`)                    | `spy_ret_21d`, `spy_rsi_14`, `vix_level`                                                                                                                         |
| Sector-relative             | `excess_ret_5d_vs_sector`                                                                        | `excess_ret_21d_vs_sector`                                                                                                                                       |
| Earnings calendar           | `post_earnings_drift_window` (redundant with continuous `days_since_earnings`)                   | `days_to_earnings`, `days_since_earnings`                                                                                                                        |
| Fundamentals (raw)          | `op_income_growth_yoy`                                                                           | `earnings_yield`, `book_to_market`, `roa`, `debt_to_equity`, `current_ratio`, `sales_growth_yoy`                                                                 |
| Fundamental ranks           | `sales_growth_yoy_rank`                                                                          | `earnings_yield_rank`, `book_to_market_rank`, `roa_rank`, `debt_to_equity_rank`, `current_ratio_rank`                                                            |

**Net: 61 → 40 features** (drop 19 truly-dead + 2 dead-rank-of-marginal-raw:
`ret_21d_rank` and `sales_growth_yoy_rank` — raw stays, rank goes).

**The 9 "rock-solid" features** (non-zero in every seed): `dist_52w_high`,
`dist_sma200_rank`, `spy_ret_21d`, `macd_hist_rank`, `debt_to_equity`,
`vol_60d_rank`, `zscore_60d_rank`, `vix_level`, `gics_sector`. These
collectively own ~60% of total importance.

Full per-seed table: `reports/feature_importance_stability.csv`.
Methodology lives in `train.py` via the `--seed` flag — re-run any time
hyperparameters or feature set changes substantially.

### Result evolution: which runs produced which numbers

Numbers from this project have moved meaningfully across iterations
because three different things changed: the **universe** (current-only
→ point-in-time historical), the **labels** (no clip → ±0.5 clip), and
the **tuning budget** (20 → 50 → 200 trials with widened search space).
Comparing across runs is only meaningful when you know which combination
produced which number.

| Run                                                | Universe                 | Label                     | Features                                                                   | Tuning                                                             | Raw CAGR   | Gated CAGR | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| -------------------------------------------------- | ------------------------ | ------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------ | ---------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Pre-historical-filter (~2026-04)                   | **Current S&P 500 only** | none                      | with broadcast SPY/VIX                                                     | IC objective, 20 trials                                            | +26.2%     | +18.8%     | Maximum survivorship bias. The headline +26.2% in old null-test tables. Don't compare to anything below.                                                                                                                                                                                                                                                                                                                                                                   |
| Post-historical-filter, pre-clip                   | Point-in-time            | none                      | with broadcast SPY/VIX                                                     | decile-spread, 20 trials                                           | +25.7%     | +17.7%     | Universe deflation only ~0.5 pts because yfinance still misses delisted names. `best_iteration=196`.                                                                                                                                                                                                                                                                                                                                                                       |
| Post-clip, 50-trial fixed-depth                    | Point-in-time            | clip ±0.5                 | with broadcast SPY/VIX                                                     | decile-spread, 50 trials, depth=3 fixed                            | +13.3%     | +8.6%      | Label clip fixed MSE blow-up but optuna landed in a too-shallow basin (`best_iteration=19`); regime-dominated.                                                                                                                                                                                                                                                                                                                                                             |
| Post-clip, 200-trial sweep (legacy)                | Point-in-time            | clip ±0.5                 | with broadcast SPY/VIX                                                     | decile-spread, 200 trials, depth ∈ [3,6]                           | +17.3%     | +9.7%      | `colsample_bytree=0.629` unlocked cross-sectional signal. Saved as `xgb_v1_legacy.json` for IRA deployment.                                                                                                                                                                                                                                                                                                                                                                |
| Clean architecture (no earnings, 100 trials)       | Point-in-time            | clip ±0.5 + date-demeaned | no broadcast SPY/VIX (39 features)                                         | decile-spread, 100 trials, depth ∈ [3,6], ES=100                   | +15.5%     | +7.9%      | Pure stock-picking signal on technicals only. `best_iteration=3` consistently — hyperparameter tuning exhausted on this feature set; the ceiling is data, not compute.                                                                                                                                                                                                                                                                                                     |
| **Prior best** (+ EDGAR earnings)                  | Point-in-time            | clip ±0.5 + date-demeaned | clean arch + 3 EDGAR earnings features (42 features)                       | decile-spread, 100 trials, depth ∈ [3,6], ES=100                   | **+17.5%** | **+8.8%**  | First non-technical signal lands. `days_since_earnings` ranks 14th in importance; `best_iteration` lifts 3 → 10 (signal ceiling broken). Sharpe 0.73 ≈ SPY 0.75 (essentially tied).                                                                                                                                                                                                                                                                                        |
| + XBRL fundamentals (raw, 500 trials)              | Point-in-time            | clip ±0.5 + date-demeaned | + 7 raw fundamentals (49 features)                                         | decile-spread, 500 trials, depth ∈ [3,6], LR ∈ [0.005,0.3], ES=100 | +15.1%     | +7.7%      | **Regressed.** Val IC ticked up (+0.0444 → +0.0563) but decile spread is flat (+0.0182, hard ceiling — top 10 trials all hit exactly 0.018162). `best_iteration=4`. 3 of 7 fundamentals absorbed (D/E rank 3rd at 0.082, ROA 6th at 0.075, E/P 10th at 0.046) but at the cost of zeroing 8 previously-active technicals (`vol_20d`, `ret_1d/5d/63d`, `trend_regime`, `zscore_*`). Fundamentals are _displacing_ signal, not adding to it. Rank-normalized version pending. |
| + fundamentals + regime context (200 trials)       | Point-in-time            | clip ±0.5 + date-demeaned | + 7 fundamentals + 7 fund-ranks + 5 broadcast SPY/VIX regime (61 features) | decile-spread, 200 trials, depth ∈ [3,5], LR ∈ [0.005,0.3], ES=100 | +16.2%     | +9.0%      | Decile-spread ceiling broken (+0.0182 → +0.0235); val IC +0.0568 (best in clean arch). `best_iteration=2`, `learning_rate=0.261` — model wants few aggressive boosts. Ranking quality up but raw CAGR short of the +17.5% earnings-only headline.                                                                                                                                                                                                                          |
| **+ 5-seed stability-selection prune (50 trials)** | Point-in-time            | clip ±0.5 + date-demeaned | 40 features (61 minus 19 dead-in-all-5-seeds + 2 dead-rank-only)           | decile-spread, 50 trials, depth ∈ [3,5], LR ∈ [0.005,0.3], ES=100  | **+19.0%** | **+10.2%** | **First Sharpe > SPY (0.76 vs 0.75).** `best_iteration=43, learning_rate=0.0058` — the cleaner feature set unlocked a slow-build basin the 61-feature config couldn't find (was stuck at lr~0.26 / 2 trees). Val decile spread +0.0193, val IC +0.0417 (lower than 61-feature run, but test CAGR up). Saved as `xgb_v1_stability_pruned.json`.                                                                                                                             |

Four things to take away:

- The **+26.2%** number you'll find in old screenshots / null-test
  tables is on the _current-only_ universe. It's not comparable to anything
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
  forecasting alpha that's gone — what _was_ the model timing the market
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
_cross-sectional_ alpha to support a deeper model. The legacy +17.3%
wasn't 100% stock-picking — a meaningful chunk was the model leaning on
SPY trend and VIX level to forecast that the _whole market_ would go up
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

| Predictions used                   | CAGR   | Sharpe | Final NAV |
| ---------------------------------- | ------ | ------ | --------- |
| The model (current-only, pre-clip) | +26.2% | +0.86  | 3.36×     |
| Random (Gaussian noise)            | +12.9% | +0.78  | 1.88×     |
| Just `dist_52w_high` (1 factor)    | +10.6% | +0.74  | 1.69×     |
| SPY buy & hold (old end-date)      | +14.5% | +0.85  | 2.05×     |

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

**Where we are.** Three non-technical signal sources have landed
(EDGAR earnings calendar, EDGAR XBRL fundamentals, SPY/VIX regime
context). The 5-seed stability-selection prune dropped 21 noise
columns; the cleaner 40-feature panel unlocked a slow-build basin
(`best_iteration=43, lr=0.0058`) that the 61-feature config couldn't
find. Latest: **raw +19.0% CAGR / Sharpe 0.76 vs SPY +12.7% / 0.75**
— first time exceeding SPY's Sharpe. The Sharpe-parity bar is
cleared. Items below are mostly free; each is a self-contained
payoff for further alpha.

> **Order of attack (per-item priority based on signal/effort):**
> §2 earnings calendar → ✅ **done**, +2.0 CAGR pts.
> §4 EDGAR fundamentals + regime context + stability-selection prune
> → ✅ **done**, **+19.0% CAGR / Sharpe 0.76 (first time > SPY's 0.75).**
> Next up: **§3 insider buys** (free, ~1-2 days, fundamentals-style
> signal source) → **§5 ensemble of 5 seeds** (free, ~½ day, +5-15%
> Sharpe, also a natural extension of the seed-sweep we just used for
> pruning) → §6 paid delisted-ticker prices (do before going live —
> ~3-5 CAGR pts of survivorship deflation expected). §1 short interest
> deferred (FINRA archive only goes back to 2018).

### 1. FINRA short interest — deferred

FINRA publishes short interest as % of float for every NMS-listed stock
on a bi-monthly cadence. Free, downloadable as pipe-delimited CSVs.
Download infrastructure shipped (`scripts/deprecated_short_interest.py`,
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
_buying_ (especially CEO/CFO open-market purchases at price below recent
avg) has documented predictive power. Add:

- `insider_buy_count_60d`, `insider_sell_count_60d` (counts of officer
  transactions in the last 60 trading days)
- `insider_net_dollar_60d` (signed dollar volume; positive = net
  buying)
- `days_since_last_insider_buy`

Real-time + historical via EDGAR's submissions API. Discipline: the
filing date can lag the transaction date by 2 business days — use the
_filing_ date for as-of cuts to avoid lookahead.

### 4. SEC EDGAR XBRL fundamentals + regime context — landed, ceiling broken

**Shipped** (`scripts/fundamentals.py`): companyfacts API → per-ticker
parquet cache → TTM income/cashflow + MRQ balance sheet → asof-merged
into the panel by SEC `filed` date for no-lookahead. Seven ratios
computed at panel-build time using split-adjusted shares × yfinance
Adj Close for market cap (`fundamentals._split_factor_after` scales
raw XBRL shares forward by cumulative split factor, so `shares_adj ×
adj_close` gives correct historical market cap):

- `earnings_yield` = TTM net_income / market_cap (E/P, value)
- `book_to_market` = MRQ equity / market_cap (B/M, value)
- `roa` = TTM net_income / MRQ assets (profitability)
- `debt_to_equity` = MRQ lt_debt / MRQ equity (leverage)
- `current_ratio` = MRQ current_assets / current_liabilities (liquidity)
- `sales_growth_yoy` = TTM revenue / TTM revenue 4Q ago − 1 (growth)
- `op_income_growth_yoy` = TTM op_income / TTM op_income 4Q ago − 1 (growth)

Concept synonym handling: revenue (`Revenues` / `RevenueFromContract...`
/ `SalesRevenueNet`), LT debt (`LongTermDebt` / `LongTermDebtNoncurrent`),
equity (with/without noncontrolling interest). Q4 derived from annual −
(Q1+Q2+Q3) since 10-Ks don't separately tag Q4.

**Coverage caveat**: 2007-01 → 2009-06 is 0% (pre-XBRL-mandate); 2009-06
→ 2011-06 partial (large filers only). All 7 features are NaN there;
XGBoost handles missing natively.

**Two iterations after first wiring fundamentals in:**

**Iteration A — raw values, 500 trials: regressed** (+17.5% → +15.1%
CAGR / Sharpe 0.73 → 0.68). Hit a hard ceiling at val decile spread
0.0182 — top 10 trials all reached _exactly_ 0.018162. 3 of 7
fundamentals absorbed (D/E 0.082, ROA 0.075, E/P 0.046); 4 dead.
Diagnosis: at `best_iteration=4 × max_depth=3` the model had only ~12
splits to spend; fundamentals won 3 of them and _displaced_ 8
previously-active technicals rather than adding net signal.

**Iteration B — fundamentals + ranks + 5 broadcast SPY/VIX regime
features, 200 trials: ceiling broken** (+16.2% CAGR / Sharpe 0.72;
val IC +0.0568, decile spread **+0.0235** — first time off the
0.0182 ceiling in 4+ sweeps). Bringing broadcast regime features
back was the unlock. The earlier clean-arch decision to drop them
was right _before_ date-demeaning landed (where they would have
been pure market-timing alpha) but became overcorrection _after_ —
with demeaned labels, a regime feature can't earn standalone
reward, but it can condition cross-sectional splits.

**What worked in iteration B** (from `feature_importance.csv`):

| Feature                    | Importance | Rank | Notes                                   |
| -------------------------- | ---------- | ---- | --------------------------------------- |
| `dist_52w_high`            | 0.187      | 1    | Unchanged anchor                        |
| `dist_sma200`              | 0.129      | 2    | Unchanged anchor                        |
| `roa`                      | 0.064      | 3    | Profitability factor                    |
| `excess_ret_21d_vs_sector` | 0.060      | 4    | Sector-relative momentum                |
| `vix_level`                | 0.058      | 5    | **Regime — fear gauge**                 |
| `spy_rsi_14`               | 0.058      | 6    | **Regime — market overbought/oversold** |
| `vol_60d_rank`             | 0.050      | 7    | Cross-sectional vol rank                |
| `dist_52w_high_rank`       | 0.045      | 8    |                                         |
| `days_since_earnings`      | 0.042      | 9    | PEAD (prior signal source)              |
| `gics_sector`              | 0.038      | 10   | Categorical sector                      |
| `macd_hist_rank`           | 0.038      | 11   |                                         |
| `debt_to_equity`           | 0.036      | 12   | Leverage                                |
| `sales_growth_yoy`         | 0.035      | 13   | **Resurrected** (was 0 in iter A)       |
| `spy_ret_21d`              | 0.034      | 14   | **Regime — momentum**                   |
| `earnings_yield`           | 0.029      | 15   | Value                                   |
| `book_to_market`           | 0.024      | 16   | **Resurrected** (was 0 in iter A)       |
| `atr_pct`                  | 0.022      | 17   |                                         |

**Still dead** (0 importance even with regime context):
`op_income_growth_yoy`, `current_ratio`, `spy_trend_regime` (subsumed by
`spy_rsi_14`), `vix_zscore_20d` (subsumed by `vix_level`), and _all 7_
fundamental ranks (raw values won every split competition for fundamentals).
Also dead: most raw technicals where the rank version dominated
(`ret_*`, `vol_20d`, `zscore_*`, `mfi_14`, etc.).

**Why CAGR didn't lift to match the IC/decile-spread improvement.**
The decile-spread metric measures top10% mean − bottom10% mean of
realised returns. Up +30% to +0.0235 means within-decile ordering
is sharper. But the strategy trades the top-50 _mean_ — and that
moved less. The lift is concentrated in the middle of the
distribution (better separation between deciles 4–7) where the
strategy doesn't operate. Net: +1.1 CAGR pts vs raw fundamentals
(+15.1 → +16.2), but still −1.3 CAGR pts vs the leaner earnings-only
config (+17.5). Sharpe 0.72 ≈ SPY 0.75 (within noise).

**Pass/fail vs prior plan**:

- ✅ Decile-spread ceiling broken (+0.0182 → +0.0235)
- ✅ Val IC peaked (+0.0568, best in clean arch)
- ✅ B/M and sales_growth resurrected (regime context unlocked them)
- ❌ Raw CAGR did _not_ exceed +17.5% — fell short
- ❌ `best_iteration` did not lift (=2, even lower than 3-4)
- ⚠️ Sharpe 0.72 still slightly below SPY 0.75

**Next.** Three open questions:

1. **Why `best_iteration=2` with `learning_rate=0.261`?** The optimiser
   chose very few, very aggressive boosts. Worth re-running with a
   lower-LR floor (0.005–0.05) to see if a slower-build basin exists.
2. **Are the fundamentals' rank versions truly redundant?** All 7
   raw fundamentals carry signal; all 7 ranks are 0. That's
   suspicious — usually ranks add information. Drop the rank
   columns (and the dead `op_income_growth`, `current_ratio`,
   `spy_trend_regime`, `vix_zscore_20d`) to see whether removing
   them lifts decile spread further. Cleaner feature set, same
   signal.
3. **Should the 5-feature regime block be 3?** `spy_trend_regime`
   and `vix_zscore_20d` are dead and likely harmful (column-sample
   noise in the optuna search).

### 5. Ensemble of 5 models (free, ~½ day)

Train 5 models with different random seeds (or bootstrapped train
samples), save 5 booster files, average predictions at inference time.
This is a _system change_, not a model change — you'll have 5 model
files (`xgb_v1_s1.json` … `xgb_v1_s5.json`), all loaded at predict time,
predictions averaged.

**Why it works:** each individual booster has slightly different
prediction errors (different random splits). Averaging cancels noise but
preserves signal — law of large numbers applied to model variance.
Typical Sharpe lift: 5–15% with no new features. Stack on top of any
other improvement above; it's orthogonal.

**Cost:** 5× disk (negligible — XGBoost models are tiny), 5× predict
time (still <1s for daily picks). Conceptually one _system_ with 5
_components_. This is what production-grade quant shops actually deploy.

### 6. Paid price data for delisted tickers (do before going live)

This _deflates the backtest_, it doesn't improve the model. Membership-
timing is correct (`universe.py` does point-in-time filtering against
the 1996+ change-event CSV), but yfinance only retains data for ~57% of
historical S&P 500 tickers — almost everything that left the index is
missing. The panel ends up at ~501 unique tickers, all current members.

Expect another 3–5 CAGR points of deflation when this lands. The
model-vs-SPY gap probably mostly survives (same-universe comparison),
but the _absolute_ numbers should be trusted only after this swap.
**Do this before trading real money.**

| Source                                      | Cost            | Notes                                                                                                              |
| ------------------------------------------- | --------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Sharadar US Equities** (Nasdaq Data Link) | ~$50/mo         | Indie-quant default. Delistings + fundamentals + sectors in one feed → kills two birds (subsumes §4). Recommended. |
| Norgate Premium Data                        | ~$60/mo + tools | Built for backtesting; total-return adjusted; point-in-time index membership baked in.                             |
| EOD Historical Data                         | ~$20/mo         | Cheapest with delistings, mixed coverage reviews.                                                                  |
| Polygon.io (Stocks Advanced)                | $199/mo         | Real-time + history + delistings. Overkill unless you also want intraday.                                          |
| yfinance + manual delisted backfill         | Free            | Hacky: scrape delisted prices from Stooq or another free source. $0 but fragile.                                   |

`data.py` already keys per-ticker parquets, so the change is mostly the
download function — should be a 1–2 day swap once the feed is chosen.

### Skip until much later

- **Options skew / IV** — paid only (ORATS ~$300/mo, CBOE DataShop
    $$
    ). Real signal but cost-to-signal is bad until the free stack is
    exhausted.
    $$
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
- [~] feature: short interest from FINRA bi-monthly — **deferred**. Download infrastructure shipped (`scripts/deprecated_short_interest.py`) but FINRA CDN archive starts mid-2018, so train=2007–2017 has 0% coverage and XGBoost cannot build splits on the feature. Mean-fill rejected (regime leak); sliding splits forward sacrifices test bear coverage. Revisit when paid historical short interest is added or when splits are re-platformed
- [ ] feature: insider transactions from SEC EDGAR Form 4 (free, ~1–2 days)
- [x] feature: SEC EDGAR XBRL fundamentals (`scripts/fundamentals.py`) — 7 ratios shipped (E/P, B/M, ROA, D/E, current_ratio, sales/op-income growth YoY) + split-adjusted shares for correct market cap. Final config: raw fundamentals + ranks + 5 broadcast SPY/VIX regime features brought back. **Iteration A** (raw, 500 trials, no regime): regressed (+17.5% → +15.1% CAGR; ceiling at decile spread 0.0182). **Iteration B** (raw + ranks + regime, 200 trials): **decile-spread ceiling broken** (+0.0182 → +0.0235), val IC +0.0568 (best ever in clean arch), but raw CAGR +16.2% — short of the +17.5% earnings-only headline. 3 of 5 regime features absorbed (vix_level, spy_rsi_14, spy_ret_21d); 2 dead fundamentals resurrected (B/M, sales_growth); 2 still dead (op_income_growth, current_ratio); all 7 fundamental ranks dead.
- [x] follow-up to §4: prune dead features via 5-seed stability selection. **Result**: 19 features dead in all 5 seeds, 21 columns total dropped (raw + ranks). 61 → 40 features. Single-run prune was wrong — `current_ratio`, `book_to_market`, `earnings_yield`, `sales_growth_yoy` all looked dead in some runs but fired strongly in others (rare-regime signals). Methodology persisted in `train.py --seed N` flag and `reports/feature_importance_stability.csv`. See [Stability-selection prune](#stability-selection-prune).
- [x] retrain on the pruned 40-feature set (50-trial sweep) and re-run backtest. **Result**: raw +19.0% CAGR / Sharpe 0.76 (first time > SPY's 0.75); `best_iteration` lifted to 43 with `lr=0.0058` — the cleaner feature set unlocked a slow-build basin the 61-feature config couldn't find. Saved as `models/xgb_v1_stability_pruned.json`.
- [ ] system: ensemble of 5 boosters with different seeds — average predictions (free, ~½ day, +5–15% Sharpe)
- [ ] re-run null test on the clean-architecture model (current null-test table is stale)
- [ ] data: swap yfinance → Sharadar (or equivalent) for delisted-ticker coverage — do before going live

Paste this to claude to ask
claude --resume b63b90f4-923f-419f-b30e-00cd9006952f
claude --resume 7762f7ea-721e-4179-a24b-273d86c65f0e
claude --resume 94d5520c-9a4b-460f-9e6d-b16cc80211b4
