# ml-stock-forward-return

ML-based S&P 500 stock ranker. Predict each stock's forward 21-trading-day
return independently with XGBoost, sort to get a daily ranking, long the top
decile, hold 21 trading days, rebalance monthly with a SPY/VIX regime gate.
Backtested on test 2021→ vs SPY buy-and-hold; gated variant +18.8% CAGR
(Sharpe 0.91) vs SPY +14.5% (Sharpe 0.85). Survivorship caveat applies.

This is the ranking-style sibling of `technical-analysis-stock-scanner`, which
filters and picks. Here we score and sort.

---

## Methodology

| Stage    | What it does                                                                                               |
| -------- | ---------------------------------------------------------------------------------------------------------- |
| Universe | Current S&P 500 from Wikipedia (v1)                                                                        |
| Data     | yfinance OHLCV 2005-07-01 → today (1.5y buffer for 252d warmup), per-ticker parquet cache, plus SPY + ^VIX |
| Features | 17 per-ticker + 8 market-context + 16 cross-sectional ranks + sector cat.                                  |
| Label    | `close[t+21] / close[t] - 1`                                                                               |
| Split    | Train 2007–2017, Val 2018–2020, Test 2021→. Chronological. No shuffling.                                   |
| Model    | XGBoost regressor, RMSE loss, optuna-tuned on val decile spread (max_depth ≤ 5)                            |
| Backtest | Long-only top-50, monthly rebalance, regime gate (SPY > SMA200 AND VIX < 25), 21 shifted-start offsets     |
| Costs    | 5 bps per side on rebalance turnover                                                                       |

Every feature on row date=D uses only data observable at the close of D.
`dataset.assert_no_lookahead()` samples random rows and recomputes features
with `.loc[:date]` to verify this — it fails loud on leakage.

---

## ⚠ Survivorship bias (known v1 caveat)

`universe.py` returns the **current** S&P 500 and applies that list backwards
through the entire 2007→today range. Stocks that were in the index but later
got removed (acquired, bankrupted, delisted) are silently absent. The backtest
will look better than reality.

**v2 fix**: replace with point-in-time index membership (a monthly snapshot of
S&P 500 additions/removals).

---

## Setup

```bash
uv sync
```

Python 3.11+. All deps are pinned in `pyproject.toml`.

---

## Run order

```bash
uv run python scripts/universe.py                   # cache S&P 500 ticker list
uv run python scripts/data.py --tickers AAPL,MSFT   # 10s smoke test first
uv run python scripts/data.py                       # full universe + SPY + VIX (~10–20 min first time)

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

# (still stubbed)
# uv run python scripts/evaluate.py
```

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
| `gics_sector` | `data/universe/sp500_members.csv` (Wikipedia) | XGBoost native categorical. Lets the model split on sector membership without manual encoding — captures effects like "utilities react differently to vol than tech." |

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

| Role             | Metric                        | Why                                                                                                                                              |
| ---------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Training loss    | **RMSE** (`reg:squarederror`) | XGBoost needs a smooth differentiable loss; RMSE is the default and gives stable gradients.                                                      |
| Tuning + early stopping | **mean daily decile spread** on val | We don't trade IC, we trade the top decile. IC and decile spread can disagree (deep trees can produce clumpy predictions with high IC but poor decile separation), so we score on the metric tied to P&L. Both the optuna objective and the per-round early-stopping rule maximise val decile spread; keeping them aligned matters (see v1 results below). |
| Reporting        | **RMSE + IC + decile spread** | Cross-checks: RMSE catches magnitude blow-ups, IC catches ranking quality, decile spread is the most direct proxy for strategy P&L.              |

### Hyperparameter search

Tuned with **optuna** (TPE sampler, ~50 trials). Knobs and ranges:

| Param              | Range          | What it controls                                                               |
| ------------------ | -------------- | ------------------------------------------------------------------------------ |
| `max_depth`        | 3–5            | Tree depth. Capped at 5 — depth-8 trees produce clumpy predictions that score well on IC but collapse decile separation (see v1 lessons). |
| `learning_rate`    | 0.01–0.3 (log) | How aggressively each tree corrects errors. Smaller + more trees usually wins. |
| `n_estimators`     | 200–1000       | Max number of trees. Capped by early stopping.                                 |
| `min_child_weight` | 1–20           | Minimum sum of sample weights per leaf. Higher = simpler trees.                |
| `subsample`        | 0.6–1.0        | Row sampling per tree. <1 adds randomness → robustness.                        |
| `colsample_bytree` | 0.6–1.0        | Feature sampling per tree. Same idea, on columns.                              |
| `reg_lambda`       | 0.01–10 (log)  | L2 regularization on leaf weights.                                             |

Each trial trains one model with early stopping (50 rounds on val decile
spread, maximize, save_best) and returns mean daily val decile spread. Optuna
picks the next combo to try based on what's worked so far. The final model is
refit on the best params and saved to `models/xgb_v1.json`.

### Outputs

- `models/xgb_v1.json` — trained booster
- `reports/feature_importance.csv` — per-feature gain importance
- `reports/optuna_trials.csv` — full tuning history (params + IC per trial)
- `reports/train_metrics.json` — final train/val RMSE + IC + decile spread + chosen params

### v1 baseline results (val 2018–2020, 20-trial tune, decile-spread objective)

| Metric                         | Value                  |
| ------------------------------ | ---------------------- |
| val decile spread (top10 − bot10, 21d) | **+0.0203 (~203 bps)** |
| val IC (mean daily Spearman)   | +0.0298                |
| train IC                       | +0.0495                |
| val RMSE / train RMSE          | 0.1059 / 0.0891        |
| best_iteration                 | 196                    |
| chosen `max_depth`             | 3                      |
| chosen `learning_rate`         | 0.019                  |
| chosen `n_estimators`          | 443                    |

203 bps of 21d decile spread annualises (×~12 sleeves) to ~24% on the
long-short spread, ~12% long-only alpha before costs and survivorship
correction. Real economics land after `backtest.py`.

Note that **val IC dropped from 0.052 → 0.030** vs the IC-tuned run, but
**decile spread improved (0.018 → 0.020)** — exactly the disagreement that
motivated the objective switch. The model is no longer chasing rank
correlation it can't trade; it's directly optimising what the strategy
captures. `best_iteration=196` (vs `16` previously) shows a real
boosted-ensemble doing the work — slow `learning_rate=0.019` × 196 shallow
depth-3 trees, no single-tree dominance.

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

### Results (test 2021-01-04 → 2026-03-26)

| Variant                    | CAGR    | Vol    | Sharpe | Max DD  | Final NAV | Time-in-market |
| -------------------------- | ------- | ------ | ------ | ------- | --------- | -------------- |
| **Raw long-only**          | +26.2%  | 30.6%  | +0.86  | -35.5%  | 3.36×     | 100%           |
| **Gated long-only**        | +18.8%  | 20.6%  | +0.91  | -27.4%  | 2.45×     | 77%            |
| SPY buy & hold (benchmark) | +14.5%  | 17.0%  | +0.85  | -24.5%  | 2.05×     | —              |

**Both variants beat SPY in absolute return.** Raw delivers higher CAGR
(and higher drawdown); gated trades CAGR for risk-adjusted performance and
a shallower drawdown.

The gated variant is rebalance-date-sensitive (CAGR offset range ~+8% to
+29% across the 21 starting days). That spread is structural to monthly
rebalance with a binary regime gate — sleeves would smooth it. The
headline "+18.8%" is the mean across the 21 offsets.

### Null test (sanity check on the alpha)

Headline numbers can flatter — the universe is survivorship-biased and the
test period favored growth/tech. To check the alpha isn't just universe +
factor exposure, we replaced the model's predictions with two alternatives
and re-ran the **raw** long-only backtest:

| Predictions used                | CAGR    | Sharpe | Final NAV |
| ------------------------------- | ------- | ------ | --------- |
| **The model**                   | +26.2%  | +0.86  | 3.36×     |
| Random (Gaussian noise)         | +12.9%  | +0.78  | 1.88×     |
| Just `dist_52w_high` (1 factor) | +10.6%  | +0.74  | 1.69×     |
| SPY buy & hold                  | +14.5%  | +0.85  | 2.05×     |

Reading this:

- **Random ≈ SPY**: equal-weighted random picks from our (current-S&P-500)
  universe earn ~13% — the survivorship-bias floor. SPY's cap-weighting on
  Mag 7 buys it a couple extra points.
- **Naive momentum < SPY**: just chasing 52-week highs alone underperformed
  in 2021–2026. So whatever the model does is **not** naive momentum.
- **Model − random ≈ +13% CAGR**: this is the cleanest measure of true
  alpha — the model's ranking adds 13 CAGR points over random ranking on
  the same universe.

### Caveats before believing the headline

1. **Survivorship still inflates the absolute number.** The model-vs-random
   gap (~13 CAGR points) should largely survive a point-in-time universe
   (v2 TODO), but the absolute 26% CAGR almost certainly won't. Realistic
   post-fix expectation: 15–20% CAGR raw, 12–15% gated.
2. **Concentrated picks.** Across 63 rebalance days the strategy lands on
   only 290 unique tickers, ~70% overlap between consecutive rebalances.
   MRNA is picked 87% of the time, PLTR 78%, TSLA 73% — high-beta
   growth/tech/COVID/meme names. Single-name blowups would hurt.
3. **Period-specific regime.** 2021–2026 favored growth/tech/momentum.
   Need backtests on different macro regimes to gauge robustness.

### Outputs

- `reports/backtest_equity.png` — equity curves vs SPY (mean + 10–90% offset band)
- `reports/backtest_stats.json` — CAGR / Sharpe / MaxDD / time-in-market per variant
- `reports/backtest_equity.csv` — daily NAV per variant (gated mean + offset bands, raw mean, SPY)

---

## File layout

```
data/
  universe/sp500_members.csv     # cached ticker list + sector
  raw/{TICKER}.parquet           # OHLCV per ticker
  market/SPY.parquet, VIX.parquet
  processed/                     # features.parquet, panel.parquet (later)
models/xgb_v1.json               # trained booster
reports/                         # equity curve, feature importance, metrics.json
scripts/
  universe.py    data.py         # implemented
  features.py    labels.py       # implemented
  dataset.py     train.py        # implemented
  backtest.py                    # implemented (monthly rebalance + 21 shifted-start offsets)
  evaluate.py                    # stub
  run_all.py                     # stub
```

---

## Metrics (defined for when evaluate.py lands)

- **Information Coefficient (IC)** — daily Spearman correlation between
  predicted forward return and realised forward return, across all tickers
  on that date. Reported as the time-series mean (and t-stat). >0.03 is
  considered tradable; >0.05 is good.
- **Top-decile spread** — mean realised forward return of top-decile picks
  minus that of the bottom decile. Direct measure of monotonic ranking quality.
- **Annualised return / vol / Sharpe** — daily portfolio log-returns
  scaled by √252.
- **Max drawdown** — peak-to-trough on the equity curve.
- **Equity curve vs SPY** — visual sanity check; SPY = passive benchmark.
- **Feature importance** — XGBoost gain importance per feature.

---

## TODOs

- [ ] v2: point-in-time S&P 500 membership (kill survivorship bias) — biggest single thing left to fix
- [x] features.py
- [x] labels.py
- [x] dataset.py + lookahead sanity assertion
- [x] train.py with hyperparameter tuning
- [x] backtest.py — monthly rebalance + 21 shifted-start offsets, regime gate, null test
- [ ] upgrade backtest to overlapping 21-day sleeves (smooths the offset CAGR range)
- [ ] evaluate.py + plots (per-month IC, drawdown plot, picks audit)
- [ ] run_all.py orchestrator

Paste this to claude to ask
claude --resume b63b90f4-923f-419f-b30e-00cd9006952f
