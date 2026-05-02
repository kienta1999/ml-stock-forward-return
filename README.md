# ml-stock-forward-return

ML-based S&P 500 stock ranker. Predict each stock's forward 21-trading-day
return independently with XGBoost, sort to get a daily ranking, long the top
decile, hold 21 trading days, rebalance monthly with a SPY/VIX regime gate.

**Current status (depth-3 model, label clip ±0.5):** raw long-only roughly
matches SPY apples-to-apples (CAGR +13.3% vs SPY +12.7% over the same end
date 2026-03-31), gated underperforms (+8.6%). Feature importance is
dominated by market-context signals (SPY/VIX), so the model is effectively
acting as a market-direction timer rather than a stock picker — see
[Diagnosis](#diagnosis-current-model-is-a-market-timer-not-a-stock-picker).
Earlier runs without label clipping reported higher CAGRs (+25.7% raw /
+17.7% gated) but those were driven by a single deep tree fit to extreme
delisted-ticker labels — see lessons-learned below.

This is the ranking-style sibling of `technical-analysis-stock-scanner`, which
filters and picks. Here we score and sort.

---

## Methodology

| Stage    | What it does                                                                                               |
| -------- | ---------------------------------------------------------------------------------------------------------- |
| Universe | Point-in-time S&P 500 (1996+ membership CSV) joined with current Wikipedia sectors                         |
| Data     | yfinance OHLCV 2005-07-01 → today (1.5y buffer for 252d warmup), per-ticker parquet cache, plus SPY + ^VIX |
| Features | 17 per-ticker + 8 market-context + 16 cross-sectional ranks + sector cat.                                  |
| Label    | `close[t+21] / close[t] - 1`, clipped to ±0.5 (~0.27% rows clipped) so dead-ticker −100% labels don't dominate MSE |
| Split    | Train 2007–2017, Val 2018–2020, Test 2021→. Chronological. No shuffling.                                   |
| Model    | XGBoost regressor, RMSE loss, optuna-tuned on val decile spread (max_depth fixed at 3)                     |
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

41 features per `(date, ticker)` row, in 4 buckets. Lists are exposed as
constants in `scripts/features.py` so downstream code stays in sync.

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

### Bucket 2 — market context (8)

Same value broadcast across all tickers on a given date — gives the model awareness of the macro regime.

| Feature                           | Definition                                               | What it captures                                                                                                                     |
| --------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `spy_ret_21d`                     | SPY 21-day pct change                                    | Market trend. Bull tape vs correction.                                                                                               |
| `spy_trend_regime`                | SPY 50d MA > 200d MA                                     | Binary bull/bear regime for the index.                                                                                               |
| `spy_rsi_14`                      | RSI(14) of SPY                                           | Market overbought/oversold.                                                                                                          |
| `vix_level`                       | ^VIX close                                               | Implied vol — fear gauge. <15 calm, >25 stressed.                                                                                    |
| `vix_zscore_20d`                  | `(vix − vix.rolling(20).mean()) / vix.rolling(20).std()` | VIX shock relative to the recent baseline.                                                                                           |
| `beta_60d`                        | Rolling cov(stock, SPY) / var(SPY) over 60d              | Stock's sensitivity to the market. >1 amplifies, <1 dampens.                                                                         |
| `excess_ret_5d`, `excess_ret_21d` | `ret_n − spy_ret_n`                                      | **Relative strength**: how much the stock beat or lagged SPY over the window. The single biggest signal for cross-sectional ranking. |

### Bucket 3 — cross-sectional ranks (16)

For each Bucket 1 feature _(except the binary `trend_regime`)_, we compute its **percentile rank across all active tickers on that date** (0 = worst, 1 = best). Column suffix: `_rank`.

Why: a 30% trailing return in 2008 ≠ a 30% return in 2017. Ranks normalise out the time-varying scale and turn each feature into a contemporaneous comparison — which is precisely what a ranker needs.

### Bucket 4 — categorical (1)

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

XGBoost regressor on the 41 features. Why XGBoost: handles missing values, scales
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
| `learning_rate`    | 0.01–0.3 (log) | How aggressively each tree corrects errors. Smaller + more trees usually wins.                                                                           |
| `n_estimators`     | 200–1000       | Max number of trees. Capped by early stopping.                                                                                                           |
| `min_child_weight` | 1–20           | Minimum sum of sample weights per leaf. Higher = simpler trees.                                                                                          |
| `subsample`        | 0.6–1.0        | Row sampling per tree. <1 adds randomness → robustness.                                                                                                  |
| `colsample_bytree` | 0.6–1.0        | Feature sampling per tree. Same idea, on columns.                                                                                                        |
| `reg_lambda`       | 0.01–10 (log)  | L2 regularization on leaf weights.                                                                                                                       |

Each trial trains one model with early stopping (50 rounds on val decile
spread, maximize, save_best) and returns mean daily val decile spread. Optuna
picks the next combo to try based on what's worked so far. The final model is
refit on the best params and saved to `models/xgb_v1.json`.

### Outputs

- `models/xgb_v1.json` — trained booster
- `reports/feature_importance.csv` — per-feature gain importance
- `reports/optuna_trials.csv` — full tuning history (params + IC per trial)
- `reports/train_metrics.json` — final train/val RMSE + IC + decile spread + chosen params

### v1 results (val 2018–2020, 50-trial tune, depth-3, label clip ±0.5)

| Metric                                 | Value                  |
| -------------------------------------- | ---------------------- |
| val decile spread (top10 − bot10, 21d) | +0.0213 (~213 bps)     |
| val IC (mean daily Spearman)           | +0.0524                |
| train IC                               | +0.0955                |
| val RMSE / train RMSE                  | 0.1033 / 0.0871        |
| best_iteration                         | **19**                 |
| chosen `max_depth`                     | 3 (fixed)              |
| chosen `learning_rate`                 | 0.076                  |
| chosen `n_estimators`                  | 938 (capped by ES)     |

`best_iteration=19` is the headline problem — even with the label clip in
place, optuna picked a fast `learning_rate=0.076` and early stopping fires
after 19 trees. 19 depth-3 trees ≈ ~150 leaves total, and as the next
section shows the splits land overwhelmingly on SPY/VIX features → most
tickers fall into the same leaf path → ranking is determined by market
regime, not stock-specific signal.

**Earlier-run note (no label clip, 20 trials):** the previously reported run
without label clipping landed `best_iteration=196` with a much slower
`learning_rate=0.019`, val IC 0.030, decile spread 0.020. The clip was
added because a few delisted-ticker −100% labels were dominating MSE; that
fix worked (RMSE is now well-behaved), but it appears to have removed
whatever signal the slow ensemble was finding in the tails. Net val decile
spread is unchanged (+0.020 → +0.021), but the resulting model is
shallower and more regime-driven.

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

### Results (test 2021-01-04 → 2026-03-31, depth-3 + label clip ±0.5)

#### As reported by `backtest.py` (unequal end dates)

| Variant                    | CAGR   | Vol   | Sharpe | Max DD | Final NAV | Time-in-market | End date   |
| -------------------------- | ------ | ----- | ------ | ------ | --------- | -------------- | ---------- |
| Raw long-only              | +13.2% | 20.3% | +0.65  | -24.6% | 1.91×     | 100%           | 2026-03-31 |
| Gated long-only            | +8.7%  | 12.4% | +0.70  | -18.1% | 1.54×     | 77%            | 2026-03-31 |
| SPY buy & hold (benchmark) | +14.7% | 16.9% | +0.87  | -24.5% | 2.07×     | —              | 2026-05-01 |

`backtest.py` reports SPY's CAGR through the latest market date (2026-05-01)
while the strategy curves stop at the last date with a 21-day-old prediction
(2026-03-31). SPY rallied ~8% in those 21 days (1.87 → 2.07), so the
default comparison is unfair to the strategy.

#### Apples-to-apples (all curves clipped at 2026-03-31)

| Variant                    | NAV @ 2026-03-31 | CAGR (5.22 yr) |
| -------------------------- | ---------------- | -------------- |
| Raw long-only              | 1.911            | **+13.3%**     |
| Gated long-only            | 1.543            | +8.6%          |
| SPY buy & hold (benchmark) | 1.868            | +12.7%         |

Apples-to-apples, **raw long-only narrowly beats SPY (~+0.6 CAGR points)**,
and the gated variant underperforms by ~4 points. The "alpha" is small
enough to be inside the survivorship-bias buffer (next subsection).

The gated variant is rebalance-date-sensitive (CAGR offset range +0.5%
to +14.2% across the 21 starting days) — a 14-point gap structural to
monthly rebalance with a binary regime gate. Sleeves would smooth it.

### Diagnosis: current model is a market-timer, not a stock-picker

Three observations from `reports/feature_importance.csv` and the latest
`scripts/today.py` output point to the same conclusion:

1. **Feature importance is dominated by market-context features.** Top 6
   by gain importance are `spy_trend_regime`, `spy_ret_21d`,
   `vix_zscore_20d`, `vix_level`, `dist_52w_high`, `spy_rsi_14` — five out
   of six are SPY/VIX features that broadcast the **same value to every
   ticker on a given date**. Adding their share gives ~38% of total gain
   on macro signals; only `dist_52w_high` (a per-ticker feature) cracks
   the top 5.
2. **Most tickers get an identical predicted return.** On 2026-03-31,
   only the top ~11 picks have unique scores; tickers ranked 12–50 (and
   beyond) all land at exactly `+0.0194`. With depth-3 × 19 trees and
   most splits on date-broadcast SPY/VIX features, the bulk of the
   universe shares a leaf path → no cross-sectional differentiation. The
   strategy's "top 50" is really "top ~10 picks plus 40 random tickers
   tied at the regime-baseline value."
3. **Removing the regime gate helps, not hurts.** Normally a regime gate
   adds value when the model can pick stocks but can't time the market.
   Here, raw (no-gate) outperforms gated by ~4.7 CAGR points — consistent
   with a model whose ranking is *already* primarily timing the market;
   the gate then double-times the same signal and costs ~23% time-out-of-
   market for no benefit.

What this means in practice: the v1 model's val IC of +0.052 is real, but
it's mostly date-level signal (good days vs. bad days for the whole
universe) rather than cross-sectional signal (which stock will beat which
on a given day). For a top-N strategy that's the wrong kind of signal —
the strategy needs the model to discriminate *within* a day, not across
days. Fixes likely live in (a) shrinking the influence of broadcast
features (drop or down-weight SPY/VIX features in training; or train on
demeaned-by-date labels), (b) deepening trees so per-ticker features can
take more splits (the next experiment is `max_depth ∈ [3, 6]` — already
wired in `train.py`), and (c) better stock-specific features
(fundamentals, earnings, options skew — already on the next-steps list).

### Null test (sanity check on the alpha) — stale, pre-clip

The numbers below are from the earlier run before the label clip + depth-3
fix and are kept for reference. They need to be re-run on the current model
before they can be trusted; expect the model-vs-random gap to compress
significantly given the diagnosis above.

| Predictions used                | CAGR   | Sharpe | Final NAV |
| ------------------------------- | ------ | ------ | --------- |
| The model (pre-clip run)        | +26.2% | +0.86  | 3.36×     |
| Random (Gaussian noise)         | +12.9% | +0.78  | 1.88×     |
| Just `dist_52w_high` (1 factor) | +10.6% | +0.74  | 1.69×     |
| SPY buy & hold (old end-date)   | +14.5% | +0.85  | 2.05×     |

Reading this (pre-clip context):

- **Random ≈ SPY**: equal-weighted random picks from our (current-S&P-500)
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

Roadmap in rough priority order (highest leverage first). Each item has a
self-contained payoff; you can pick them off one at a time.

### 1. Restore stock-specific signal (fix the regime-dominance issue)

This is the new top priority — see [Diagnosis](#diagnosis-current-model-is-a-market-timer-not-a-stock-picker)
for the symptoms. Three concrete experiments, in order:

1. **Tune `max_depth` in [3, 6].** Already wired in `train.py` (the
   currently-saved model was trained with depth fixed at 3). Deeper trees
   give per-ticker features more chance to take splits after the regime
   features burn through the early ones. Watch decile spread, not IC: a
   higher IC with the same regime-dominance pattern wouldn't fix the
   strategy.
2. **Down-weight or drop the SPY/VIX broadcast features.** Train a variant
   without `spy_*` and `vix_*` (and possibly `beta_60d` and the binary
   `trend_regime`) so the model is forced to pick on cross-sectional
   features only. Compare val decile spread and per-day prediction
   variance.
3. **Train on date-demeaned labels.** Replace `y = forward_21d_return` with
   `y = forward_21d_return − date_mean(forward_21d_return)`. This
   explicitly removes the date-level component the model is currently
   chasing and forces it to learn within-date ordering. The original raw
   label can still be reported on, but the loss is computed against the
   demeaned target.

### 2. Paid price data for delisted tickers (closes the residual bias)

Membership-timing is now correct (`universe.py` does point-in-time filtering
against the 1996+ change-event CSV), but yfinance only retains data for
~57% of historical S&P 500 tickers — almost everything that left the index
is missing. The panel ends up at ~501 unique tickers, all current members.
Random-pick CAGR barely budged from the survivorship-biased run (12.9% →
~12.5%) because we never had the dead names in the first place.

**What to build**: swap the data layer's source from yfinance to a feed
that retains delisted symbols — Sharadar US Equities (~$50/mo), Norgate,
Polygon historical, or CRSP via WRDS. `data.py` already keys per-ticker
parquets, so the change is mostly the download function. Expect another
3–5 CAGR points of deflation when this lands; the model-vs-random gap
should mostly survive (it was already same-universe-vs-same-universe).

### 3. Sleeves upgrade (smooths the gated variant's offset CAGR range)

Today's gated backtest spans CAGR offset range [+0.5%, +14.2%] across the
21 shifted starts — a 14-point gap between best- and worst-luck rebalance
day. That's structural to monthly rebalance with a binary regime gate; no
amount of tuning fixes it.

**What to build**: 21 overlapping 21-day sleeves running in parallel,
rebalancing 1/21 of book each day. Mathematically equivalent to averaging
the 21 shifted-start offsets, but as one continuous portfolio rather than
21 independent ones. Smooths daily turnover, eliminates rebalance-date
fragility, becomes the realistic live-trading mechanic.

### 4. Diagnostics module (per-month IC stability, drawdown, attribution)

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

### 5. Better features (the real lever for IC > 0.05)

We've extracted what the current 41 features can give us (val IC ~0.03,
val decile spread ~0.02). The next ~50 bps of IC won't come from
hyperparameter tuning — it'll come from new signals.

Candidates to try:

- **Fundamentals**: trailing P/E, P/S, EV/EBITDA, FCF yield, ROIC. Quarterly
  cadence so leakage discipline is harder; needs careful as-of dating.
- **Earnings/event flags**: days-to-next-earnings, post-earnings drift window.
- **Sector-relative momentum**: `excess_ret_21d` vs sector mean (not just
  SPY).
- **Options skew**: 25-delta put/call IV ratio per ticker — captures
  fear/greed not visible in price alone.
- **Short interest / borrow rate**: contrarian signal, especially for
  high-vol names.

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
- [ ] re-tune with `max_depth ∈ [3, 6]` (already wired in train.py) and re-run backtest
- [ ] experiment: train without SPY/VIX broadcast features to force cross-sectional signal
- [ ] experiment: train on date-demeaned forward returns
- [x] fix `backtest.py` SPY end-date so headline CAGR compares like-for-like
- [ ] re-run null test on the post-clip model (current numbers are stale)

Paste this to claude to ask
claude --resume b63b90f4-923f-419f-b30e-00cd9006952f
claude --resume 7762f7ea-721e-4179-a24b-273d86c65f0e
