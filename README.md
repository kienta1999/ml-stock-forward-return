# ml-stock-forward-return

ML-based S&P 500 stock ranker. Predict each stock's forward 21-trading-day return
independently with XGBoost, sort to get a daily ranking, long the top decile,
hold ~1 month, rebalance daily via overlapping monthly sleeves.

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
| Model    | XGBoost regressor, RMSE loss, optuna-tuned on val IC                                                       |
| Backtest | Daily predictions → rank → long top 10% equal-weight, 21d hold, 21 sleeves                                 |
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

# (the rest is stubbed — building piece by piece)
# uv run python scripts/backtest.py
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
| Tuning objective | **mean daily IC** on val      | We're a ranker, not a forecaster. RMSE rewards getting magnitudes right; IC rewards getting the _order_ right — which is what the strategy uses. |
| Reporting        | **RMSE + IC + decile spread** | Cross-checks: RMSE catches magnitude blow-ups, IC catches ranking quality, decile spread is the most direct proxy for strategy P&L.              |

### Hyperparameter search

Tuned with **optuna** (TPE sampler, ~50 trials). Knobs and ranges:

| Param              | Range          | What it controls                                                               |
| ------------------ | -------------- | ------------------------------------------------------------------------------ |
| `max_depth`        | 3–8            | Tree depth. Deeper = more interaction terms but more overfitting.              |
| `learning_rate`    | 0.01–0.3 (log) | How aggressively each tree corrects errors. Smaller + more trees usually wins. |
| `n_estimators`     | 200–1000       | Max number of trees. Capped by early stopping.                                 |
| `min_child_weight` | 1–20           | Minimum sum of sample weights per leaf. Higher = simpler trees.                |
| `subsample`        | 0.6–1.0        | Row sampling per tree. <1 adds randomness → robustness.                        |
| `colsample_bytree` | 0.6–1.0        | Feature sampling per tree. Same idea, on columns.                              |
| `reg_lambda`       | 0.01–10 (log)  | L2 regularization on leaf weights.                                             |

Each trial trains one model with early stopping (50 rounds on val RMSE) and
returns mean daily val IC. Optuna picks the next combo to try based on what's
worked so far. The final model is refit on the best params and saved to
`models/xgb_v1.json`.

### Outputs

- `models/xgb_v1.json` — trained booster
- `reports/feature_importance.csv` — per-feature gain importance
- `reports/optuna_trials.csv` — full tuning history (params + IC per trial)
- `reports/train_metrics.json` — final train/val RMSE + IC + decile spread + chosen params

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
  backtest.py    evaluate.py     # stub
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

- [ ] v2: point-in-time S&P 500 membership (kill survivorship bias)
- [x] features.py
- [x] labels.py
- [x] dataset.py + lookahead sanity assertion
- [x] train.py with hyperparameter tuning
- [ ] backtest.py with overlapping 21-day sleeves
- [ ] evaluate.py + plots
- [ ] run_all.py orchestrator

Paste this to claude to ask
claude --resume b63b90f4-923f-419f-b30e-00cd9006952f

```
Loaded 2,396,643 rows; dropped 126,457 early-window NaN → 2,270,186 usable rows.
Trimmed 49,871 pre-2007-01-01 buffer rows → 2,220,315 model-ready rows.

Train: 1,207,798 rows  |  Val: 361,939 rows

Running optuna (50 trials, maximize val IC)...

  0%|          | 0/50 [00:00<?, ?it/s]
Best trial: 0. Best value: 0.0170279:   0%|          | 0/50 [00:08<?, ?it/s]
Best trial: 0. Best value: 0.0170279:   2%|▏         | 1/50 [00:08<07:10,  8.79s/it]
Best trial: 1. Best value: 0.0517669:   2%|▏         | 1/50 [00:17<07:10,  8.79s/it]
Best trial: 1. Best value: 0.0517669:   4%|▍         | 2/50 [00:17<07:10,  8.97s/it]
Best trial: 1. Best value: 0.0517669:   4%|▍         | 2/50 [00:23<07:10,  8.97s/it]
Best trial: 1. Best value: 0.0517669:   6%|▌         | 3/50 [00:23<05:49,  7.45s/it]
Best trial: 1. Best value: 0.0517669:   6%|▌         | 3/50 [00:27<05:49,  7.45s/it]
Best trial: 1. Best value: 0.0517669:   8%|▊         | 4/50 [00:27<04:46,  6.23s/it]
Best trial: 1. Best value: 0.0517669:   8%|▊         | 4/50 [00:33<04:46,  6.23s/it]
Best trial: 1. Best value: 0.0517669:  10%|█         | 5/50 [00:33<04:34,  6.10s/it]
Best trial: 1. Best value: 0.0517669:  10%|█         | 5/50 [00:44<04:34,  6.10s/it]
Best trial: 1. Best value: 0.0517669:  12%|█▏        | 6/50 [00:44<05:42,  7.79s/it]
Best trial: 1. Best value: 0.0517669:  12%|█▏        | 6/50 [00:51<05:42,  7.79s/it]
Best trial: 1. Best value: 0.0517669:  14%|█▍        | 7/50 [00:51<05:16,  7.37s/it]
Best trial: 1. Best value: 0.0517669:  14%|█▍        | 7/50 [01:00<05:16,  7.37s/it]
Best trial: 1. Best value: 0.0517669:  16%|█▌        | 8/50 [01:00<05:30,  7.86s/it]
Best trial: 1. Best value: 0.0517669:  16%|█▌        | 8/50 [01:07<05:30,  7.86s/it]
Best trial: 1. Best value: 0.0517669:  18%|█▊        | 9/50 [01:07<05:08,  7.52s/it]
Best trial: 1. Best value: 0.0517669:  18%|█▊        | 9/50 [01:14<05:08,  7.52s/it]
Best trial: 1. Best value: 0.0517669:  20%|██        | 10/50 [01:14<04:58,  7.46s/it]
Best trial: 10. Best value: 0.0555798:  20%|██        | 10/50 [01:31<04:58,  7.46s/it]
Best trial: 10. Best value: 0.0555798:  22%|██▏       | 11/50 [01:31<06:46, 10.41s/it]
Best trial: 10. Best value: 0.0555798:  22%|██▏       | 11/50 [01:42<06:46, 10.41s/it]
Best trial: 10. Best value: 0.0555798:  24%|██▍       | 12/50 [01:42<06:37, 10.46s/it]
Best trial: 10. Best value: 0.0555798:  24%|██▍       | 12/50 [01:52<06:37, 10.46s/it]
Best trial: 10. Best value: 0.0555798:  26%|██▌       | 13/50 [01:52<06:22, 10.33s/it]
Best trial: 10. Best value: 0.0555798:  26%|██▌       | 13/50 [02:01<06:22, 10.33s/it]
Best trial: 10. Best value: 0.0555798:  28%|██▊       | 14/50 [02:01<05:59,  9.98s/it]
Best trial: 10. Best value: 0.0555798:  28%|██▊       | 14/50 [02:09<05:59,  9.98s/it]
Best trial: 10. Best value: 0.0555798:  30%|███       | 15/50 [02:09<05:30,  9.44s/it]
Best trial: 10. Best value: 0.0555798:  30%|███       | 15/50 [02:19<05:30,  9.44s/it]
Best trial: 10. Best value: 0.0555798:  32%|███▏      | 16/50 [02:19<05:24,  9.54s/it]
Best trial: 10. Best value: 0.0555798:  32%|███▏      | 16/50 [02:27<05:24,  9.54s/it]
Best trial: 10. Best value: 0.0555798:  34%|███▍      | 17/50 [02:27<05:04,  9.23s/it]
Best trial: 10. Best value: 0.0555798:  34%|███▍      | 17/50 [02:37<05:04,  9.23s/it]
Best trial: 10. Best value: 0.0555798:  36%|███▌      | 18/50 [02:37<05:01,  9.43s/it]
Best trial: 10. Best value: 0.0555798:  36%|███▌      | 18/50 [02:48<05:01,  9.43s/it]
Best trial: 10. Best value: 0.0555798:  38%|███▊      | 19/50 [02:48<05:03,  9.80s/it]
Best trial: 10. Best value: 0.0555798:  38%|███▊      | 19/50 [02:58<05:03,  9.80s/it]
Best trial: 10. Best value: 0.0555798:  40%|████      | 20/50 [02:58<04:55,  9.86s/it]
Best trial: 10. Best value: 0.0555798:  40%|████      | 20/50 [03:10<04:55,  9.86s/it]
Best trial: 10. Best value: 0.0555798:  42%|████▏     | 21/50 [03:10<05:05, 10.52s/it]
Best trial: 10. Best value: 0.0555798:  42%|████▏     | 21/50 [03:25<05:05, 10.52s/it]
Best trial: 10. Best value: 0.0555798:  44%|████▍     | 22/50 [03:25<05:35, 12.00s/it]
Best trial: 10. Best value: 0.0555798:  44%|████▍     | 22/50 [03:38<05:35, 12.00s/it]
Best trial: 10. Best value: 0.0555798:  46%|████▌     | 23/50 [03:38<05:33, 12.35s/it]
Best trial: 10. Best value: 0.0555798:  46%|████▌     | 23/50 [03:50<05:33, 12.35s/it]
Best trial: 10. Best value: 0.0555798:  48%|████▊     | 24/50 [03:50<05:14, 12.10s/it]
Best trial: 10. Best value: 0.0555798:  48%|████▊     | 24/50 [04:02<05:14, 12.10s/it]
Best trial: 10. Best value: 0.0555798:  50%|█████     | 25/50 [04:02<05:00, 12.02s/it]
Best trial: 10. Best value: 0.0555798:  50%|█████     | 25/50 [04:12<05:00, 12.02s/it]
Best trial: 10. Best value: 0.0555798:  52%|█████▏    | 26/50 [04:12<04:37, 11.55s/it]
Best trial: 10. Best value: 0.0555798:  52%|█████▏    | 26/50 [04:23<04:37, 11.55s/it]
Best trial: 10. Best value: 0.0555798:  54%|█████▍    | 27/50 [04:23<04:22, 11.43s/it]
Best trial: 10. Best value: 0.0555798:  54%|█████▍    | 27/50 [04:35<04:22, 11.43s/it]
Best trial: 10. Best value: 0.0555798:  56%|█████▌    | 28/50 [04:35<04:11, 11.45s/it]
Best trial: 10. Best value: 0.0555798:  56%|█████▌    | 28/50 [04:46<04:11, 11.45s/it]
Best trial: 10. Best value: 0.0555798:  58%|█████▊    | 29/50 [04:46<03:58, 11.37s/it]
Best trial: 10. Best value: 0.0555798:  58%|█████▊    | 29/50 [04:54<03:58, 11.37s/it]
Best trial: 10. Best value: 0.0555798:  60%|██████    | 30/50 [04:54<03:28, 10.44s/it]
Best trial: 10. Best value: 0.0555798:  60%|██████    | 30/50 [05:03<03:28, 10.44s/it]
Best trial: 10. Best value: 0.0555798:  62%|██████▏   | 31/50 [05:03<03:09,  9.99s/it]
Best trial: 10. Best value: 0.0555798:  62%|██████▏   | 31/50 [05:13<03:09,  9.99s/it]
Best trial: 10. Best value: 0.0555798:  64%|██████▍   | 32/50 [05:13<02:59,  9.95s/it]
Best trial: 10. Best value: 0.0555798:  64%|██████▍   | 32/50 [05:23<02:59,  9.95s/it]
Best trial: 10. Best value: 0.0555798:  66%|██████▌   | 33/50 [05:23<02:46,  9.80s/it]
Best trial: 10. Best value: 0.0555798:  66%|██████▌   | 33/50 [05:31<02:46,  9.80s/it]
Best trial: 10. Best value: 0.0555798:  68%|██████▊   | 34/50 [05:31<02:30,  9.38s/it]
Best trial: 10. Best value: 0.0555798:  68%|██████▊   | 34/50 [05:40<02:30,  9.38s/it]
Best trial: 10. Best value: 0.0555798:  70%|███████   | 35/50 [05:40<02:18,  9.26s/it]
Best trial: 10. Best value: 0.0555798:  70%|███████   | 35/50 [05:46<02:18,  9.26s/it]
Best trial: 10. Best value: 0.0555798:  72%|███████▏  | 36/50 [05:46<01:57,  8.40s/it]
Best trial: 10. Best value: 0.0555798:  72%|███████▏  | 36/50 [05:54<01:57,  8.40s/it]
Best trial: 10. Best value: 0.0555798:  74%|███████▍  | 37/50 [05:54<01:47,  8.30s/it]
Best trial: 10. Best value: 0.0555798:  74%|███████▍  | 37/50 [06:04<01:47,  8.30s/it]
Best trial: 10. Best value: 0.0555798:  76%|███████▌  | 38/50 [06:04<01:43,  8.59s/it]
Best trial: 10. Best value: 0.0555798:  76%|███████▌  | 38/50 [06:13<01:43,  8.59s/it]
Best trial: 10. Best value: 0.0555798:  78%|███████▊  | 39/50 [06:13<01:37,  8.84s/it]
Best trial: 10. Best value: 0.0555798:  78%|███████▊  | 39/50 [06:23<01:37,  8.84s/it]
Best trial: 10. Best value: 0.0555798:  80%|████████  | 40/50 [06:23<01:32,  9.26s/it]
Best trial: 10. Best value: 0.0555798:  80%|████████  | 40/50 [06:30<01:32,  9.26s/it]
Best trial: 10. Best value: 0.0555798:  82%|████████▏ | 41/50 [06:30<01:17,  8.62s/it]
Best trial: 10. Best value: 0.0555798:  82%|████████▏ | 41/50 [06:40<01:17,  8.62s/it]
Best trial: 10. Best value: 0.0555798:  84%|████████▍ | 42/50 [06:40<01:11,  8.95s/it]
Best trial: 10. Best value: 0.0555798:  84%|████████▍ | 42/50 [06:50<01:11,  8.95s/it]
Best trial: 10. Best value: 0.0555798:  86%|████████▌ | 43/50 [06:50<01:03,  9.13s/it]
Best trial: 10. Best value: 0.0555798:  86%|████████▌ | 43/50 [06:59<01:03,  9.13s/it]
Best trial: 10. Best value: 0.0555798:  88%|████████▊ | 44/50 [06:59<00:55,  9.21s/it]
Best trial: 10. Best value: 0.0555798:  88%|████████▊ | 44/50 [07:08<00:55,  9.21s/it]
Best trial: 10. Best value: 0.0555798:  90%|█████████ | 45/50 [07:08<00:44,  8.98s/it]
Best trial: 10. Best value: 0.0555798:  90%|█████████ | 45/50 [07:18<00:44,  8.98s/it]
Best trial: 10. Best value: 0.0555798:  92%|█████████▏| 46/50 [07:18<00:37,  9.36s/it]
Best trial: 10. Best value: 0.0555798:  92%|█████████▏| 46/50 [07:28<00:37,  9.36s/it]
Best trial: 10. Best value: 0.0555798:  94%|█████████▍| 47/50 [07:28<00:28,  9.54s/it]
Best trial: 10. Best value: 0.0555798:  94%|█████████▍| 47/50 [07:36<00:28,  9.54s/it]
Best trial: 10. Best value: 0.0555798:  96%|█████████▌| 48/50 [07:36<00:18,  9.10s/it]
Best trial: 10. Best value: 0.0555798:  96%|█████████▌| 48/50 [07:45<00:18,  9.10s/it]
Best trial: 10. Best value: 0.0555798:  98%|█████████▊| 49/50 [07:45<00:09,  9.12s/it]
Best trial: 10. Best value: 0.0555798:  98%|█████████▊| 49/50 [07:52<00:09,  9.12s/it]
Best trial: 10. Best value: 0.0555798: 100%|██████████| 50/50 [07:52<00:00,  8.61s/it]
Best trial: 10. Best value: 0.0555798: 100%|██████████| 50/50 [07:52<00:00,  9.46s/it]

Best val IC during tuning: +0.0556
Best params:
  max_depth: 8
  learning_rate: 0.09289825888463438
  n_estimators: 976
  min_child_weight: 1
  subsample: 0.8607466203112715
  colsample_bytree: 0.9630659181130071
  reg_lambda: 0.01748488898051613

Fitting final model with best params...

Final metrics:
  train RMSE: 0.0908   train IC: +0.1005
  val   RMSE: 0.1047   val   IC: +0.0556   decile spread: +0.0064
  best iteration: 1

  -> model saved to /home/talekien1710/personal_project/ml-stock-forward-return/models/xgb_v1.json
  -> reports saved to /home/talekien1710/personal_project/ml-stock-forward-return/reports/
```
