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
| Data     | yfinance OHLCV 2007-01-01 → today, per-ticker parquet cache, plus SPY + ^VIX |
| Features | Per-ticker indicators + market context (SPY/VIX) + cross-sectional ranks     |
| Label    | `close[t+21] / close[t] - 1`                                                 |
| Split    | Train ≤2017, Val 2018–2020, Test 2021→. Chronological. No shuffling.         |
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
uv run python scripts/universe.py            # cache S&P 500 ticker list
uv run python scripts/data.py --tickers AAPL,MSFT   # 10s smoke test first
uv run python scripts/data.py                # full universe + SPY + VIX (~10–20 min first time)

# (the rest is stubbed — building piece by piece)
# uv run python scripts/features.py
# uv run python scripts/labels.py
# uv run python scripts/dataset.py
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
  features.py    labels.py       # stub
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
- [ ] features.py
- [ ] labels.py
- [ ] dataset.py + lookahead sanity assertion
- [ ] train.py with hyperparameter tuning
- [ ] backtest.py with overlapping 21-day sleeves
- [ ] evaluate.py + plots
- [ ] run_all.py orchestrator
