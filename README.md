# ml-stock-forward-return

ML-based S&P 500 stock ranker. Predict each stock's forward 21-trading-day return
independently with XGBoost, sort to get a daily ranking, long the top decile,
hold ~1 month, rebalance daily via overlapping monthly sleeves.

This is the ranking-style sibling of `technical-analysis-stock-scanner`, which
filters and picks. Here we score and sort.

---

## Methodology

| Stage    | What it does                                                                 |
|----------|------------------------------------------------------------------------------|
| Universe | Current S&P 500 from Wikipedia (v1)                                          |
| Data     | yfinance OHLCV 2005-07-01 → today (1.5y buffer for 252d warmup), per-ticker parquet cache, plus SPY + ^VIX |
| Features | 17 per-ticker + 8 market-context + 16 cross-sectional ranks + sector cat.    |
| Label    | `close[t+21] / close[t] - 1`                                                 |
| Split    | Train 2007–2017, Val 2018–2020, Test 2021→. Chronological. No shuffling.     |
| Model    | XGBoost regressor, tuned on the val set                                      |
| Backtest | Daily predictions → rank → long top 10% equal-weight, 21d hold, 21 sleeves   |
| Costs    | 5 bps per side on rebalance turnover                                         |

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

# (the rest is stubbed — building piece by piece)
# uv run python scripts/train.py
# uv run python scripts/backtest.py
# uv run python scripts/evaluate.py
```

### `data.py` CLI flags

| Flag | What it does |
|------|--------------|
| *(none)* | Incremental update — only fetches days since the last cached date. Default for daily refreshes. |
| `--refresh` | Wipe caches and redownload from scratch. Use when something looks wrong. |
| `--tickers AAPL,MSFT` | Subset only. Great for smoke-testing or iterating on downstream code without re-downloading 500 tickers. |
| `--start 2015-01-01` | Override default start (`2007-01-01`). Useful for a smaller, faster dataset while developing. |
| `--skip-universe` | Only refresh SPY/VIX. |
| `--skip-market` | Only refresh the universe; skip SPY/VIX. |

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

| Feature | Definition | What it captures |
|---|---|---|
| `ret_1d`, `ret_5d`, `ret_21d`, `ret_63d` | `close.pct_change(n)` | Trailing return at multiple horizons. 1d/5d often **mean-revert**; 21d/63d trend. |
| `rsi_14` | Wilder's RSI on close, 14d, EWM com=13 | Overbought (>70) / oversold (<30) momentum oscillator. |
| `mfi_14` | Money Flow Index — RSI weighted by `typical_price × volume` | Same shape as RSI but volume-aware. Catches conviction behind a move. |
| `macd_hist` | `MACD(12,26) − signal(9)` | Trend acceleration. Positive & rising = bullish momentum. |
| `atr_pct` | `ATR(14) / close` | Average true range as % of price — cross-sectionally comparable volatility. |
| `vol_20d`, `vol_60d` | Annualised std of log returns | Realised volatility, short vs medium term. |
| `vol_ratio` | `volume / volume.rolling(20).mean()` | Today's volume relative to recent norm. >1 = unusually heavy. |
| `dist_sma50`, `dist_sma200` | `close / sma_n − 1` | How stretched price is above/below medium- and long-term trend. |
| `dist_52w_high` | `close / max(close, 252) − 1` | Distance below the 52-week high. Classic momentum factor (stocks near highs tend to keep outperforming). |
| `trend_regime` | `1.0 if sma50 > sma200 else 0.0` | Bull/bear trend flag. |
| `zscore_20d`, `zscore_60d` | `(close − sma_n) / std_n` | **Volatility-normalised** distance from mean. A 5% gap means different things for a calm vs vol-y name; this corrects for that (also the math behind Bollinger %B). |

### Bucket 2 — market context (8)

Same value broadcast across all tickers on a given date — gives the model awareness of the macro regime.

| Feature | Definition | What it captures |
|---|---|---|
| `spy_ret_21d` | SPY 21-day pct change | Market trend. Bull tape vs correction. |
| `spy_trend_regime` | SPY 50d MA > 200d MA | Binary bull/bear regime for the index. |
| `spy_rsi_14` | RSI(14) of SPY | Market overbought/oversold. |
| `vix_level` | ^VIX close | Implied vol — fear gauge. <15 calm, >25 stressed. |
| `vix_zscore_20d` | `(vix − vix.rolling(20).mean()) / vix.rolling(20).std()` | VIX shock relative to the recent baseline. |
| `beta_60d` | Rolling cov(stock, SPY) / var(SPY) over 60d | Stock's sensitivity to the market. >1 amplifies, <1 dampens. |
| `excess_ret_5d`, `excess_ret_21d` | `ret_n − spy_ret_n` | **Relative strength**: how much the stock beat or lagged SPY over the window. The single biggest signal for cross-sectional ranking. |

### Bucket 3 — cross-sectional ranks (16)

For each Bucket 1 feature *(except the binary `trend_regime`)*, we compute its **percentile rank across all active tickers on that date** (0 = worst, 1 = best). Column suffix: `_rank`.

Why: a 30% trailing return in 2008 ≠ a 30% return in 2017. Ranks normalise out the time-varying scale and turn each feature into a contemporaneous comparison — which is precisely what a ranker needs.

### Bucket 4 — categorical (1)

| Feature | Source | Purpose |
|---|---|---|
| `gics_sector` | `data/universe/sp500_members.csv` (Wikipedia) | XGBoost native categorical. Lets the model split on sector membership without manual encoding — captures effects like "utilities react differently to vol than tech." |

### Why some popular indicators are *not* included

| Skipped | Reason |
|---|---|
| Bollinger Bands (raw) | Same information as `vol_20d` + `zscore_20d`; no extra lift. |
| ADX | Already captured by `trend_regime` + vol features. |
| Stochastic %K, Williams %R | ~85% correlated with RSI. Redundant. |
| OBV / Chaikin / A-D | Cumulative volume series; not cross-sectionally comparable without normalisation, and the normalised form ends up close to `vol_ratio × ret_n`. |
| CCI | Just a z-score of typical price — already covered by `zscore_*`. |
| VWAP | Short-horizon; doesn't help a 21d forecast. |
| Ticker as a token | The ticker is a unique ID, not a factor. Including it would let the model memorise per-stock patterns from train and apply them to test — pure data leakage of the label through identity. |

### No-lookahead guarantee

Every feature on row `date=D` is computed using only data with date ≤ D. All operations are `rolling` / `ewm` / `pct_change` / `shift(positive)` — none peek into the future. `dataset.assert_no_lookahead()` (coming next) will sample random rows, recompute features with `df.loc[:date]`, and fail the build if anything disagrees.

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
  dataset.py     train.py        # stub
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
- [ ] train.py with hyperparameter tuning
- [ ] backtest.py with overlapping 21-day sleeves
- [ ] evaluate.py + plots
- [ ] run_all.py orchestrator
