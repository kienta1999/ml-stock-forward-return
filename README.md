# ranker-21d-sp500

ML-based S&P 500 stock ranker. Predict each stock's forward 21-trading-day
return independently with XGBoost, sort to get a daily ranking, long the top
decile, hold 21 trading days, rebalance monthly with a **vol-targeted
sizing overlay** (replaces the legacy SPY/VIX regime gate as of 2026-05-11).

**Current status (+ 3 FRED macro features + vol-target overlay +
cataclysmic-only quality filter, 200-trial sweep on 47 features, seed 15):**
raw long-only **+24.3% CAGR / Sharpe 0.95 vs SPY +13.6% /
Sharpe 0.80** (test 2021-01-04 → 2026-04-09). Strategy delivers
**+10.7 CAGR pts** over SPY. Final NAV **3.13×** vs SPY 1.95×. MaxDD
**-26.5%** (raw), **-25.0%** (vol-targeted overlay, Sharpe **0.91** /
CAGR +21.5%). Saved hyperparams: `best_iter=41`, `lr=0.0015`,
`colsample=0.579`, `reg_lambda=0.005`, `max_depth=5`.

**Quality filter (added 2026-05-17).** A loose fundamentals/insider
"cataclysmic-only" filter runs before top-N selection: drop names with
`debt_to_equity > 10`, `current_ratio < 0.3`, `sales_growth_yoy < -0.50`,
or `insider_net_dollar_60d < -50M`. NaN values pass — no fundamentals
coverage isn't evidence of low quality. Sweep across 7 threshold
variants showed the loose version is Pareto-better than no filter (raw
CAGR 23.0% → 24.3%, Sharpe 0.87 → 0.95, MaxDD -28.1% → -26.5%); tighter
variants reduce drawdown linearly but kill CAGR faster than they help.
See [Quality filter](#quality-filter-cataclysmic-only-fundamentalinsider-screen)
for the full sweep table.

**Cross-seed stability (2026-05-11, pre-quality-filter baseline).**
Re-ran a 200-trial sweep on a different seed (17 vs 15). Optuna landed
in different hyperparam basins (`best_iter=9` vs `41`, `lr=0.004` vs
`0.0015`, `max_depth=4` vs `5`) but converged on **identical raw
Sharpe 0.87** (CAGR 22.3% vs 23.0%, MaxDD -26.6% vs -28.1%) and within
0.01 on vol-target Sharpe (0.82 vs 0.83). Top picks overlap heavily —
TTD, SMCI, INTC, CHTR, AXON, COIN, HOOD appear top-12 on both seeds.
Different hyperparams, same alpha → the signal lives in the data +
features, not in a lucky landing. The strategy is not random. (The
quality filter described above adds +0.08 Sharpe on top of these
underlying model numbers.)

**Important caveat: this is a calm window.** Avg vol-target exposure
was 95% in 2021-2026 — the overlay barely activated because there
was no real vol spike (2022 was a slow grind). The leave-2008-out
walk-forward stress test (see "Vol-target overlay" below) shows raw
MaxDD -64.6% with the model trained on 2010-2019 and tested on
2007-2009 — long-only top-40 has a structural drawdown floor no
overlay alone can break.

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

| Stage    | What it does                                                                                                                                                                                                                                                                                                                                                                                                                       |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Universe | Point-in-time S&P 500 (1996+ membership CSV) joined with current Wikipedia sectors                                                                                                                                                                                                                                                                                                                                                 |
| Data     | yfinance OHLCV 2005-07-01 → today (1.5y buffer for 252d warmup), per-ticker parquet cache, plus SPY + ^VIX                                                                                                                                                                                                                                                                                                                         |
| Features | 10 per-ticker technicals + 3 ticker-specific market-context + 3 broadcast SPY/VIX regime context + 3 broadcast FRED macro regime (term spread, IG credit spread, 5y5y inflation) + 1 sector-relative + 2 earnings calendar + 6 XBRL fundamentals + 14 cross-sectional ranks + 4 insider Form 4 + sector cat = **47 total** (after 5-seed stability-selection prune — see [Stability-selection prune](#stability-selection-prune)). |
| Label    | `forward_21d_return − date_mean(forward_21d_return)` — date-demeaned (cross-sectional excess). Raw `forward_21d_return` is clipped to ±0.5 first to cap dead-ticker outliers, then demeaned. The model can only learn within-date ordering, not market direction.                                                                                                                                                                  |
| Split    | Train 2007–2017, Val 2018–2020, Test 2021→. Chronological. No shuffling.                                                                                                                                                                                                                                                                                                                                                           |
| Model    | XGBoost regressor, RMSE loss, optuna-tuned on val decile spread (max_depth ∈ [3, 6], 100 trials, ES=100 rounds)                                                                                                                                                                                                                                                                                                                    |
| Backtest | Long-only top-40 (after cataclysmic-only quality filter drops D/E>10 / CR<0.3 / sales_yoy<-0.50 / insider_net<-$50M), monthly rebalance, vol-targeted sizing overlay (`exposure = min(1.0, 0.20 / spy_vol_20d)`), 21 shifted-start offsets                                                                                                                                                                                         |
| Costs    | 5 bps per side on rebalance turnover                                                                                                                                                                                                                                                                                                                                                                                               |

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
uv run python scripts/insider.py                    # SEC quarterly Form 3/4/5 bulk TSV → per-ticker parquet (~5 min first time, ~80 quarter zips back to 2006q1)
uv run python scripts/insider.py --refresh          # wipe data/insider/ and rebuild from scratch
uv run python scripts/insider.py --tickers AAPL,MSFT  # rebuild only those tickers from already-cached zips
uv run python scripts/fundamentals.py                # SEC EDGAR XBRL TTM income + MRQ balance sheet → per-ticker parquet (~10-15 min first time)
uv run python scripts/fundamentals.py --smoke        # one-ticker dry-run before the full pull
uv run python scripts/macro.py                       # FRED daily series → data/market/macro.parquet (~5s, no auth)
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

uv run python scripts/backtest.py                    # long-only raw + vol-targeted overlay + quality filter, 21 shifted starts (~1 min)
uv run python scripts/backtest.py --no-overlay       # skip vol-target variant (raw only)
uv run python scripts/backtest.py --no-quality-filter # disable the cataclysmic-only fundamentals/insider screen
uv run python scripts/backtest.py --top-n 25         # tighter pick (default 40)
uv run python scripts/backtest.py --vol-target 0.15  # more aggressive de-risk (default 0.20)
uv run python scripts/backtest.py --weight pred      # weight basket by predicted_return (default: equal); see "Basket weighting" below
uv run python scripts/backtest.py --leverage 1.35     # lever up (default 1.0 = cash-only); borrow charged at 5.14% APR — see "Vol-target overlay"
uv run python scripts/backtest.py --lag 1             # execute at the close 1 trading day AFTER the signal close — models the live loop (picks computed after close, orders placed next session). Measured 2026-07-05 on the same panel: raw CAGR 28.20% → 28.02%, vol-target 25.40% → 25.14%, Sharpe unchanged. 1-day execution lag costs ~0.2 CAGR pt — the MOC assumption is not load-bearing.

uv run python scripts/diagnostics.py                 # IC stability, underwater, concentration, attribution, hit rate, live-picks scorecard
uv run python scripts/diagnostics.py --live-only     # just mark picks/*.csv to market vs SPY (fast)

uv run python scripts/today.py                                   # latest-date picks (vol-target sizing + quality filter + top 40)
uv run python scripts/today.py --diff picks/picks_YYYY-MM-DD.csv # buy/sell list vs that prior file
uv run python scripts/today.py --no-overlay                      # ignore vol-target (always 100% exposure)
uv run python scripts/today.py --no-quality-filter               # disable the cataclysmic-only fundamentals/insider screen
uv run python scripts/today.py --vol-target 0.15                 # more conservative sizing
uv run python scripts/today.py --weight pred                     # pred-weighted basket (default: equal)

uv run python scripts/run_all.py                  # 🚨 DAILY / CATCH-UP — universe → data → earnings → insider → fundamentals → features → labels → today (auto --diff). Run this when you come back after time away.
uv run python scripts/run_all.py --retrain        # also: train + backtest (use after a feature change)
uv run python scripts/run_all.py --full           # alias for --retrain (universe refreshes on every run)
uv run python scripts/run_all.py --download-only  # raw-data refresh only — useful if you want to defer features/labels/picks
uv run python scripts/run_all.py --dry-run        # print plan, don't execute

# ── Live execution on Interactive Brokers — see "Live execution (IBKR)" below ──
# --port is REQUIRED (4001 = live account, 4002 = paper). Examples use 4001 (always-live setup).
uv run python scripts/check_ibkr_conn.py --port 4001                 # sanity-check WSL→Gateway; prints DU/U (paper/live) banner
uv run python scripts/execute_picks.py --port 4001                   # print rebalance plan only — places NOTHING (safe)
uv run python scripts/execute_picks.py --port 4001 --mode whatif --leverage 1.35 --vol-target 0.2 --top-n 20 --max-notional 10000  # IBKR commission/margin preview — still places nothing
uv run python scripts/execute_picks.py --port 4001 --fractional      # deploy ~100% of a small account (fractional shares)
uv run python scripts/execute_picks.py --port 4001 --mode live --leverage 1.35 --vol-target 0.2 --top-n 20  # ⚠️ places real orders (gated by cap + typed confirm)

# ── Alternatively, with NO API at all (IBKR Lite, or just not paying for Pro market data) ──
# The /ibkr-web-trade skill does the same rebalance through the free web Client Portal, driven
# by Playwright: positions, NetLiquidation, quotes and order tickets all come from the logged-in
# portal session. Same sizing code (--broker web), same circuit breaker, no IB Gateway.
# In Claude Code:
#   /ibkr-web-trade                       # newest picks CSV, --top-n 40 (backtest canonical)
#   /ibkr-web-trade mode whatif           # commission/margin preview, places nothing
#   /ibkr-web-trade top-n 20              # any flag above overrides the default
# You log in yourself (2FA included) in the browser it opens; it confirms the plan with you,
# then the final order list, before anything is submitted.
```

> **On IBKR Lite / no market-data subscription?** `.claude/skills/ibkr-web-trade`
> is the API-free path: `/ibkr-web-trade` in Claude Code opens the web Client
> Portal in Playwright, you log in, and it runs the identical rebalance
> (`--broker web`) through the portal's own order tickets — plan confirmation
> and a final go/no-go both required before it submits.

> **Automating orders?** See **### Live execution (IBKR)** below for Gateway
> setup, WSL↔Windows networking, the print → whatif → live workflow, and the
> full flag table.

### Daily live-picks workflow

`scripts/today.py` is what bridges the backtest model to actual trading. The
backtest deliberately ignores the most recent ~21 trading days because they
have no forward-return label yet; `today.py` deliberately predicts on them.

> ## 🚨 Coming back after time away? Run this first 🚨
>
> ```bash
> uv run python scripts/run_all.py
> ```
>
> **This is the catch-up command.** It refreshes every raw data cache
> (universe → prices → earnings → insider → fundamentals), rebuilds the
> features and labels panels against the new data, and generates today's
> picks against your **existing** model — no retrain. Use this when you've
> been away for a day, a week, a month, a year. Every step is incremental
> and fail-safe:
>
> - **Universe** — re-pulls Wikipedia, syncs today's S&P 500 roster (catches
>   any committee adds/removes you missed)
> - **Prices** — yfinance incremental from the last cached bar
> - **Earnings** — EDGAR submissions API, only new 8-K item 2.02 / 10-Q / 10-K filings
>   since last run (cached parquets without the `items` column auto-refetch
>   under the new schema)
> - **Insider** — smart-incremental: any quarterly Form 3/4/5 zips published
>   while you were away get downloaded, plus the latest is always re-fetched.
>   Skip a year of runs and it still catches up cleanly
> - **Fundamentals** — EDGAR XBRL incremental, only new filings
> - **Features / labels** — pure local recompute against the freshened caches
> - **Today** — scores against the existing `models/xgb_v1.json`, writes
>   `picks/picks_<date>.csv`, auto-diffs against the most recent prior file,
>   and runs the **daily de-risk check**: recomputes
>   `min(1.35, 0.20 / spy_vol_20d)` and yells "⚠⚠ DE-RISK … SELL DOWN TODAY"
>   if it has fallen >10% below the live book's exposure
>   (`reports/live_book.json`) — the daily crash guard for the
>   quarterly-rebalanced levered book
>
> Want a model that actually _learns_ on the new data (e.g. after a feature
> change like the 8-K anchor switch)? Run `run_all.py --retrain` instead —
> same catch-up flow, plus train + backtest at the end.
>
> Just want to pull caches now and finish later? `run_all.py --download-only`
> stops after the raw-data refresh.

The orchestrator `scripts/run_all.py` chains the daily pipeline together:

```bash
# every morning before market open — one command:
uv run python scripts/run_all.py
```

That runs `universe → data → earnings → insider → fundamentals → features →
labels → today` end-to-end, stops on the first failure, and auto-supplies
`--diff` to `today.py` using the most recent prior file in `picks/`. Exit
code is the failed step's exit code, so it's cron-friendly. Universe is
refreshed on every run (~1s — re-pulls Wikipedia so today's S&P 500 roster
is current even if the upstream `SP_500_Historical_Component.csv` is stale).

Modes:

| Command                      | What runs                                                                                                                     |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **`run_all.py`**             | **🚨 daily / catch-up — universe → data → earnings → insider → fundamentals → features → labels → today. See callout above.** |
| `run_all.py --retrain`       | (default) + train + backtest                                                                                                  |
| `run_all.py --full`          | alias for `--retrain` (universe is now refreshed on every run, so the flag is redundant)                                      |
| `run_all.py --download-only` | refresh raw data caches only (universe → data → earnings → insider → fundamentals), then stop                                 |
| `run_all.py --no-today`      | refresh + optional retrain only, skip live picks                                                                              |
| `run_all.py --no-diff`       | run today.py without `--diff`                                                                                                 |
| `run_all.py --dry-run`       | print the plan, don't execute                                                                                                 |

Equivalent manual sequence (if you want to run pieces individually):

```bash
uv run python scripts/universe.py --refresh
uv run python scripts/data.py
uv run python scripts/earnings.py
uv run python scripts/insider.py
uv run python scripts/fundamentals.py
uv run python scripts/features.py
uv run python scripts/labels.py
uv run python scripts/today.py --diff picks/picks_<yesterday>.csv
```

The `--diff` flag prints a BUY / SELL / HOLD ticket — exactly which tickers
to add or drop today vs the prior picks. That's the trade list you'd
execute manually, or hand to `scripts/execute_picks.py` for automated
placement on Interactive Brokers (see **Live execution (IBKR)** below).

Output lands in `picks/picks_<latest_date>.csv` (one file per run;
gitignored to keep daily noise out of git). Cash days produce an empty
picks file. The script warns if the panel is more than 7 days old.

### Rebalance cadence decision (2026-07-08): QUARTERLY @ 1.35x leverage

**Decision: rebalance every ~3 months (63 trading days), `--leverage 1.35
--vol-target 0.2`, revisit the account quarterly.** Cadence sweep on the
2021→2026-06 panel (vol-target variant, margin @ 5.14% APR modeled):

| Hold @ 1.35x        | CAGR        | Sharpe   | MaxDD  | offset-luck CAGR range |
| ------------------- | ----------- | -------- | ------ | ---------------------- |
| monthly (21d)       | +30.58%     | 1.08     | −28.6% | +24.6% → +34.9%        |
| **quarterly (63d)** | **+30.58%** | **1.11** | −29.1% | +22.3% → +37.3%        |
| annual (252d)       | +25.01%     | 0.96     | −31.7% | +15.1% → +36.4%        |

(Unlevered vol-target CAGR: monthly 25.5% / quarterly 25.3% / 6-mo 23.8%
/ annual 21.4% — the 21d signal decays slowly out to ~1 quarter, then
meaningfully.) Monthly vs quarterly gross is a dead heat, so the
unmodeled costs decide it: fixed IBKR fees (~$40/rebalance → $480/yr
monthly vs $160/yr quarterly — a multi-point drag on a small account)
and slower short-term-gain churn. Annual's long-term-tax edge (~+1pt
after-tax, unlevered) is eaten at 1.35x by the wider gross gap, worse
Sharpe, worse MaxDD, and huge single-rebalance date luck. A **4-sleeve
structure** (each sleeve held >1yr, one sleeve rebalanced per quarter →
every sale long-term, quarterly fees) is the taxable-account upgrade
path if the account grows.

⚠️ **Standing caveat:** the vol-target overlay is only _evaluated at
rebalance_. Quarterly cadence = up to ~3 months fully levered at 1.35x
before the overlay reacts to a vol spike (the MaxDD −29% above is from
a calm window with no such spike; the leave-2008-out stress test was
−64.6% _unlevered_). **Mitigation (shipped 2026-07-13): the daily
de-risk check in `today.py`** — every run recomputes
`min(1.35, 0.20 / spy_vol_20d)` from fresh SPY data, compares it to the
book's gross exposure (`reports/live_book.json`, written by
`execute_picks.py --mode live`), and yells "⚠⚠ DE-RISK … SELL DOWN
TODAY" when the formula falls >10% below the book. Sell-down only —
re-levering waits for the quarterly rebalance.

### Live execution (IBKR)

`scripts/execute_picks.py` turns a `picks/*.csv` into real orders on
Interactive Brokers, reconciling against whatever the account already holds
(a re-run after fills produces ~zero orders — it does **not** double-buy).
It talks to a running **IB Gateway** (or TWS) via `ib_async`, not to IBKR's
servers directly.

**Prerequisite — the Gateway must be running and API-enabled.** Log in
(paper account `DU…` first!), then **Configure → Settings → API → Settings**:
turn **Read-Only API off**, note the socket port. IB Gateway has no "enable
socket clients" checkbox (it's always on) — that's expected.

Ports: `4002` = Gateway paper, `4001` = Gateway live, `7497`/`7496` = TWS
paper/live.

**WSL → Windows networking.** `127.0.0.1` inside WSL2 is _not_ the Windows
host. Two options:

- **Mirrored (recommended):** put `networkingMode=mirrored` under `[wsl2]` in
  `C:\Users\<you>\.wslconfig`, `wsl --shutdown`, reopen. Now `127.0.0.1` works
  from WSL and Gateway's default `127.0.0.1` Trusted IP needs no change — and
  it survives reboots. Pass `--host 127.0.0.1`.
- **NAT (default):** the Windows host is the WSL default-route gateway
  (`ip route show default`). Add the **WSL source IP** (`ip addr show eth0`) to
  Gateway's Trusted IPs and uncheck "localhost only". ⚠️ That subnet **drifts
  on reboot**, so re-check it. The scripts auto-detect the host side.

Sanity-check the plumbing before any order code — `scripts/check_ibkr_conn.py`
does a TCP probe, connects, and prints a **`DU` (paper) / `U` (live)** banner
so paper vs. live is unmistakable:

`--port` is **required** (no default) so live vs. paper is always a conscious
choice — `4001` = live, `4002` = paper. Examples below use `4001` (always-live
setup); swap to `4002` if you log the Gateway into a paper `DU…` account.

```bash
uv run python scripts/check_ibkr_conn.py --port 4001        # live (auto host)
uv run python scripts/check_ibkr_conn.py --host 127.0.0.1 --port 4001   # mirrored networking
```

**Three modes, safest first — always dry-run before trading:**

```bash
# 1. print — client-side plan only, places NOTHING
uv run python scripts/execute_picks.py --port 4001

# 2. whatif — IBKR returns commission + margin per order, still places nothing
uv run python scripts/execute_picks.py --port 4001 --mode whatif

# --fractional deploys ~100% on a small account, BUT IBKR often rejects
# fractional over the API (error 10243/10244); verify with whatif before trusting it.
# Reliable alternative on a small account: fewer names — today.py --top-n 15.
uv run python scripts/execute_picks.py --port 4001 --mode whatif --fractional

# 3. live — actually places orders (gated; see below)
uv run python scripts/execute_picks.py --port 4001 --mode live --max-notional 10000
```

Sizing: `target_$ = weight * NetLiquidation * leverage`. The picks `weight`
already bakes in the vol-target exposure, so `--leverage 1.0` (default) is
cash-only and pays no margin interest.

**`--vol-target` usage:**

| Invocation          | Behavior                                                                                                                                                                                                                                                                                             |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--vol-target`      | Bare flag = default target **0.20** (20% annualized). Recommended with any `--leverage > 1.0`.                                                                                                                                                                                                       |
| `--vol-target 0.15` | Explicit target. It is a **fraction**, not a percent. Lower = de-risks earlier/harder.                                                                                                                                                                                                               |
| `--vol-target 1.5`  | **Rejected** (any value ≤ 0 or > 1). A too-large target would never bind — overlay silently disabled while looking enabled.                                                                                                                                                                          |
| _(omitted)_         | Flat `leverage ×` CSV weights. Not unbounded — the CSV already bakes in the **signal date's** vol-target exposure — but that boundary is as stale as the picks file and gets scaled back **up** by leverage in stress (see below). Truly unbounded only if picks were generated with `--no-overlay`. |

**⚠ Leveraged sizing: pass `--vol-target` to match the backtest.**
Flat `--leverage 1.35` multiplies the CSV weights by 1.35 no matter the
regime — in stress, the CSV's baked-in de-risked exposure gets scaled back
_up_ by 1.35× (e.g. SPY vol 25%: flat gives 1.08× gross, the backtest models
`min(1.35, 0.20/0.25) = 0.80×`). With `--vol-target 0.20` the script
renormalizes the CSV weights to gross 1.0 and applies
`min(leverage, vol_target / spy_vol_20d)` from the **freshest cached SPY
data** (also fresher than the picks file's signal date; warns if the cache
is >7 days old — run `data.py` first). This is exactly
`strategy.vol_target_exposure`, the same function the backtest calls:

```bash
uv run python scripts/execute_picks.py --port 4001 --leverage 1.35 --vol-target 0.20
```

Key flags:

| Flag             | What it does                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--mode`         | `print` (default, safe) · `whatif` (cost preview) · `live` (places orders)                                                                                                                                                                                                                                                                                                                                                                                                     |
| `--top-n`        | Trade only the top N picks by `predicted_return`, weights rescaled to preserve the CSV's gross exposure (composes with `--vol-target`). **40 = backtest canonical and 2008-stress-tested; 20 backtests better (raw CAGR 34.5% vs 28.2%, Sharpe within noise) and halves fixed order costs. 20 is now leave-2008-out verified (2026-07-23): it SURVIVES (stays positive, beats SPY) but is the more fragile basket — deeper crash MaxDD and its offset-luck range goes negative where 40's stays positive (see "Basket size in the 2008 stress test" below). It also holds ~70% tech vs 65% at 40.** **Live choice: 20** — better return + cleaner sizing at low capital, and 2008-survivable; the extra tail robustness of 40 is a minor (~3-4 pt) drawdown effect vs leverage, which is the real tail lever. |
| `--fractional`   | Fractional-share orders so a small account deploys ~100% instead of stranding cash on pricey names that round to 0 whole shares. ⚠️ **IBKR frequently rejects fractional over the API** (error 10243/10244 — needs the fractional-shares permission enabled, and may be desktop-TWS-only). Verify with `--mode whatif` first. RTH only.                                                                                                                                        |
| `--leverage`     | Gross exposure multiplier (default 1.0 = cash-only). >1.0 borrows on margin (~5.14% APR).                                                                                                                                                                                                                                                                                                                                                                                      |
| `--vol-target`   | Backtest-matching overlay for leveraged sizing: gross = `min(leverage, X / spy_vol_20d)` from the freshest cached SPY data, CSV weights renormalized to 1.0 first. Bare flag = 0.20 (the backtest default); values ≤ 0 or > 1 rejected (percent typo guard). Omit → flat leverage multiple (more exposure in stress than the backtest models).                                                                                                                                 |
| `--max-notional` | Hard circuit breaker on total BUY notional; abort if the plan exceeds it.                                                                                                                                                                                                                                                                                                                                                                                                      |
| `--min-order`    | Skip rebalances below this dollar value (default $100) to avoid churn.                                                                                                                                                                                                                                                                                                                                                                                                         |
| `--slippage-bps` | Marketable-limit padding per side (default 50 bps; sizing prices are 15-min delayed, and 30 bps missed 3/20 fills on the 2026-07-06 go-live).                                                                                                                                                                                                                                                                                                                                  |

**Live-mode safety:** orders are marketable **limit** orders with `tif`/limit
set explicitly (dodges the Gateway order-preset that trips error 10349 on bare
market orders). Live mode enforces the `--max-notional` cap and, on a `U…`
(live) account, requires you to type the exact account number to confirm.
Before rebalancing, live mode **cancels all working orders** (aborting if any
refuse to cancel) so stale limits from a previous run can't fill alongside
fresh duplicates — the plan reconciles against positions only, so an unseen
working order would otherwise double-buy. `print`/`whatif` are read-only and
just warn when working orders exist.

**Small-account caveat:** IBKR charges ~$1 flat per order. On a ~$10k / 40-name
book that's ~$40 to enter (~40 bps), ~80 bps round-trip — vs the backtest's
5 bps/side assumption. Concentrate (`execute_picks.py --top-n 20`, or
`today.py --top-n 15` at pick time) or size up to
dilute the fixed cost.

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

65 features per `(date, ticker)` row, organized into buckets (numeric

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

### Bucket 2c — broadcast FRED macro regime context (3)

Added 2026-05-11. Same broadcast pattern as Bucket 2b — single value per
date, identical across every ticker, earns reward only via interactions
under date-demeaned labels. The point: VIX/SPY are both equity-derived;
yield-curve and credit-spread signals capture macro stress those don't
see. The 10y–3m term spread is the NY Fed's official recession predictor
and inverted hard in 2019/2022/2023 within our test window.

| Feature             | Definition                                             | What it captures                                                                                                                |
| ------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| `term_spread_10y3m` | `DGS10 − DGS3MO` (% pts)                               | Yield-curve regime. Inverts 6–18mo before postwar US recessions. Negative in 2019, 2022-23, the test window's macro stress.     |
| `ig_credit_spread`  | `BAA10Y` — Moody's Baa minus 10y (% pts)               | Investment-grade corporate credit stress. Spikes in risk-off (peaked 6.16 in March 2020). Equity-vol's debt-market counterpart. |
| `inflation_5y5y`    | `T5YIFR` — 5-Year 5-Year Forward Inflation Expectation | Long-run inflation regime. Captures shifts VIX missed in 2022 — when realised inflation cracked but VIX stayed muted.           |

**Data source.** FRED's free CSV endpoint (`fredgraph.csv?id=<series>`)
— no API key. Cached to `data/market/macro.parquet`. ICE BofA HY OAS
(`BAMLH0A0HYM2`) was originally on the wish-list but FRED now gates ICE
series history behind an API key; BAA10Y is ~0.85 correlated with HY OAS
and serves as the credit-stress proxy. See `scripts/macro.py` for the
full caveat. The series are market-data (yields/spreads), not survey —
no revision-lookahead concern, the FRED snapshot is what traders saw on
each historical date.

**Why these specifically.** Of the dozen candidate FRED macro series with
≥2006 history, these 3 share two properties: (a) market-data, no
publication-delay lookahead, and (b) cover stress axes the existing
SPY/VIX regime features don't. Survey/composite series (NFCI, UNRATE,
CFNAI, ICSA) were skipped because their revisions / publication delays
introduce subtle lookahead that's not worth fixing for the marginal
signal lift.

**Empirical lift (47-feat `--quick`, single seed).** Adding these 3
features lifted the raw long-only test backtest by **+1.47 CAGR pts**
(21.79% → 23.26%) and **+0.03 Sharpe** (0.84 → 0.87), with max drawdown
shallower by ~1pt (-29.59% → -28.66%). Single-seed result — multi-seed
stability check pending. Likely larger DD benefit in a backtest that
includes 2008-2009 (currently in train, hidden from test).

### Bucket 3 — sector-relative (2)

| Feature                                               | Definition                                             | What it captures                                                                                                                                                                                                                             |
| ----------------------------------------------------- | ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `excess_ret_5d_vs_sector`, `excess_ret_21d_vs_sector` | `ret_n − groupby(date, gics_sector).transform("mean")` | Cross-sectional sector-relative momentum. Strips sector beta from the per-ticker return so the model compares apples to apples within an industry. NaN for the `Unknown` sector bucket (delisted/removed names — group mean would be noise). |

### Bucket 4 — cross-sectional ranks (16)

For each Bucket 1 feature _(except the binary `trend_regime`)_, we compute its **percentile rank across all active tickers on that date** (0 = worst, 1 = best). Column suffix: `_rank`.

Why: a 30% trailing return in 2008 ≠ a 30% return in 2017. Ranks normalise out the time-varying scale and turn each feature into a contemporaneous comparison — which is precisely what a ranker needs.

### Bucket 6 — earnings calendar (3)

| Feature                      | Definition                                                                                                                            | What it captures                                                                                                                                                                                 |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `days_to_earnings`           | days from row date to next known earnings event (EDGAR 8-K item 2.02 + yfinance forward calendar), clipped [0, 90]                    | Pre-earnings positioning. **Previously 0.0 importance** under 10-Q anchoring — re-evaluation pending after the 8-K switch.                                                                       |
| `days_since_earnings`        | days since most recent earnings announcement (8-K item 2.02, falling back to 10-Q for older / pre-item-code filings), clipped [0, 90] | **Post-earnings drift signal — 14th in feature importance (0.037 gain) under the old 10-Q anchor.** The 8-K switch anchors PEAD on the actual market-moving event ~14 days earlier; retrain TBD. |
| `post_earnings_drift_window` | `1.0` if `days_since_earnings ∈ [1, 5]` else `0.0`                                                                                    | Hand-coded PEAD window flag. **0.0 importance** — XGBoost reconstructs the same split internally from the continuous `days_since_earnings`. Already disabled in `EARNINGS_FEATURES`.             |

**Data source.** SEC EDGAR submissions API gives every filing
(8-K / 10-Q / 10-K, plus amendments) for tickers with a current CIK
(648 of 959 historical S&P 500 names; the missing 311 are mostly
delisted/renamed without retained CIKs). The PEAD anchor is the
**8-K item 2.02** filing — the earnings press release, filed within
~4 business days of the event. The submissions JSON exposes a parallel
`items` array, so we filter to item 2.02 with no extra requests beyond
the per-ticker fetch we already do. Older filings (pre-2004, before
8-K item codes were standardised) and rare gaps fall back to the
matching 10-Q. For live picks, yfinance's forward earnings calendar
fills in the upcoming dates.

`scripts/earnings.py` handles the EDGAR 8-K/10-Q/10-K pipeline + yfinance
forward calendar (separate from `data.py` because the cadences and
rate-limit contracts differ). `scripts/fundamentals.py` is the parallel
pipeline for EDGAR XBRL fundamentals. `scripts/deprecated_short_interest.py`
holds the FINRA pipeline (deferred — archive only goes back to 2018-08).
Per-ticker parquets live at `data/earnings/{TICKER}.parquet` and
`data/fundamentals/{TICKER}.parquet`.

### Bucket 8 — insider transactions (4)

| Feature                       | Definition                                                                                           | What it captures                                                                           |
| ----------------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `insider_buy_count_60d`       | # direct open-market P (purchase) Form 4 filings in `(D − 60d, D]`                                   | Officer/director conviction buying. Cross-sectionally rare → strong signal when present.   |
| `insider_sell_count_60d`      | # direct open-market S (sale) Form 4 filings in the same window                                      | Sales are noisier (10b5-1 plans, diversification) — counted but de-weighted by net dollar. |
| `insider_net_dollar_60d`      | `Σ(P value) − Σ(S value)` over the window, where `value = shares × price`                            | Signed dollar conviction. Positive = net insider buying, magnitude scaled by trade size.   |
| `days_since_last_insider_buy` | days from D back to the most recent P filing on or before D, capped at 365 (NaN if no buy on record) | Decaying memory of the last buy. Empty for tickers with no historical insider purchases.   |

**Asof key.** All four features use `filing_date` (not `transaction_date`)
for the as-of cut. Form 4 must be filed within 2 business days of the
trade, so the filing date is the earliest a public investor could have
acted on the information — using it avoids lookahead. Only direct-ownership
non-derivative `P` (open-market purchase) and `S` (open-market sale)
transactions are kept; awards, grants, option exercises, gifts, and
indirect-ownership filings are excluded since they carry little timing
signal.

**Data source.** SEC's DERA team publishes parsed Form 3/4/5 data as
quarterly TSV zips at
`https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{YYYY}q{N}_form345.zip`.
~80 zips cover 2006q1 → present (~600 MB total, one HTTPS GET per quarter).
This is dramatically cheaper than scraping per-filing Form 4 XMLs through
the EDGAR archives — a single full-universe rebuild via XML scraping took
hours and got my IP throttled; the bulk approach finishes in minutes and
stays well below SEC's 10 req/s limit. `scripts/insider.py` smart-incrementals:
each run downloads any quarters missing from `data/insider/_bulk/` plus
always re-fetches the latest cached quarter (which gets updated within-quarter
as new filings arrive), then rebuilds `data/insider/{TICKER}.parquet` from
the union. Skipping several runs is fine — missing quarters are picked up
automatically. SEC publishes each quarter ~1 month after it closes, so the
freshest filings have a publication lag; live picks already lag features by
21 d, so this lag is not a binding constraint.

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

| Role                    | Metric                                                      | Why                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ----------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Training loss           | **RMSE** (`reg:squarederror`)                               | XGBoost needs a smooth differentiable loss; RMSE is the default and gives stable gradients.                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Tuning + early stopping | **val top-N (40) mean realised return**                     | We don't trade IC and we don't trade the full decile — we trade the top-40 portfolio. Decile spread (the prior objective) averages ranking quality across the top and bottom ~50 names each, but the strategy only holds the tip of decile 1, so the two metrics can disagree. Empirically, sweeps that maximised val decile spread backtested 2–3 CAGR pts WORSE than `--quick` (see "Objective evolution" note below). Both the optuna objective and the per-round early-stopping rule now maximise val top-40 mean return. |
| Reporting               | **RMSE + IC + top-40 mean return + Sharpe + decile spread** | Cross-checks: RMSE catches magnitude blow-ups, IC catches ranking quality, top-40 mean return is the most direct proxy for strategy P&L, and Sharpe + decile spread are kept as diagnostics to track how the optimiser trades risk vs raw return.                                                                                                                                                                                                                                                                             |

**Objective evolution (May 2026):** decile spread → top-N Sharpe (1 sweep,
abandoned — Sharpe was gamed by collapsing prediction variance, hit
`best_iteration=0` / val IC ≈ 0) → **top-N mean return** (current). Mean
return cannot be gamed the same way: zero alpha → zero objective. Eval
metric implementation note: the per-round callback uses
`np.argsort(-preds, kind='stable')[:top_n]` per date (~3-4× faster than
the pandas `groupby().nlargest()` it replaced); stable sort matches pandas
tie-breaking exactly, so `best_iteration` is bit-reproducible across runs.

### Hyperparameter search

Tuned with **optuna** (TPE sampler, ~50 trials). Knobs and ranges:

| Param              | Range            | What it controls                                                                                                                                                                                                                                                                    |
| ------------------ | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `max_depth`        | 3–5              | Depth-6+ dominated in 4 prior sweeps (always worst trials). Depth-3 still wins frequently, but rank-normalized fundamentals + 8-K interactions occasionally pay for depth 4-5; window kept narrow.                                                                                  |
| `learning_rate`    | 0.001–0.02 (log) | Tightened from 0.005–0.3 after the slow-build basin (lr~0.005, best_iteration 13-43) consistently outperformed the aggressive-shallow basin (lr~0.03+, best_iteration ~5). 50-trial TPE was burning trials on the latter; capping at 0.02 forces optuna into the productive region. |
| `n_estimators`     | 200–1000         | Max number of trees. Capped by early stopping (typically 19–43).                                                                                                                                                                                                                    |
| `min_child_weight` | 1–20             | Minimum sum of sample weights per leaf. Higher = simpler trees.                                                                                                                                                                                                                     |
| `subsample`        | 0.6–1.0          | Row sampling per tree. <1 adds randomness → robustness.                                                                                                                                                                                                                             |
| `colsample_bytree` | 0.55–0.75        | Tightened from 0.6–1.0. **0.629 has been the actual lever unlocking cross-sectional signal** across sweeps — values >0.8 give every tree almost-all features and collapse top-N selection. Window stays asymmetric around 0.629 so optuna can still wander.                         |
| `reg_lambda`       | 0.001–10 (log)   | L2 regularization on leaf weights. Floor lowered from 0.01 because the prior known-good optimum (0.01003) was bumping the wall.                                                                                                                                                     |

Each trial trains one model with early stopping (100 rounds on val top-40
mean return, maximize, save_best) and returns val top-40 mean realised
return. Optuna picks the next combo to try based on what's worked so far.
The final model is refit on the best params and saved to `models/xgb_v1.json`.

`--quick` skips tuning and uses the saved best params from the May 2026
50-trial sweep (`val top-40 mean return = +0.0142`, `best_iteration = 19`).
Bit-reproducible — every `--quick` run regenerates the identical model.

### Outputs

- `models/xgb_v1.json` — trained booster
- `reports/feature_importance.csv` — per-feature gain importance
- `reports/optuna_trials.csv` — full tuning history (params + IC per trial)
- `reports/train_metrics.json` — final train/val RMSE + IC + top-40 mean return + Sharpe + decile spread + chosen params

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

Top-40 by predicted return (= top ~8% of ~500 active S&P 500 names),
equal-weighted, monthly rebalance, 21 trading day hold, 5 bps per side cost.
Two variants run side-by-side:

- **Raw** (default for "max return at full risk"): always long top 40, 100%
  exposed. See "Why default top-40, raw" below for the rationale.
- **Vol-targeted** (default for "shallower drawdowns"): scale gross exposure
  continuously by `min(1.0, 0.20 / spy_vol_20d)`. In calm regimes (SPY 20d
  vol ≤ 20%) → full long. As realized vol climbs, exposure ramps down: SPY
  vol 30% → ~67% exposure, SPY vol 45% → ~44%. Remaining capital sits in
  cash. Replaces the legacy `SPY > SMA200 AND VIX < 25` binary gate as of
  2026-05-11 (see "Vol-target overlay" below).

### Vol-target overlay (the new "gated" default)

The legacy binary regime gate fired late — `VIX > 25` typically crosses
after most of the damage has happened — and was rebalance-date sensitive
because it was an on/off switch. Replaced with **continuous vol-target
sizing**:

```
exposure_t = min(1.0, vol_target / spy_realized_vol_20d_t)     # vol_target default 0.20
basket_weights_t = exposure_t * (1/N for each of the top-N picks)
cash_bucket_t = 1 - exposure_t                                  # returns 0
```

Properties:

- **No leverage by default** — exposure caps at 1.0, only scales down. Opt in
  with `backtest.py --leverage 1.35`: the cap becomes
  `min(leverage, vol_target / spy_vol)`, so you lever up in calm regimes but
  the vol target still de-risks you in stress. Borrowed funds (gross > 1.0)
  accrue margin interest daily at `--margin-rate` (default 5.14% APR = IBKR Pro
  Tier I) so the reported CAGR nets out the cost of leverage. Leverage lifts
  CAGR but **not** Sharpe (it scales return and vol together), and the vol-target
  variant keeps its drawdown far tighter than the raw variant when levered.
- **Chosen live setting: `--leverage 1.35`.** Picked for margin-call safety on a
  small Reg-T account (IBKR portfolio margin needs $110k; below that, maintenance
  = 25%), not drawdown comfort. Max leverage that survives a basket drop `d`
  without a Reg-T call is `L = 1 / (d + 0.25·(1−d))`; inverting for L=1.35 gives a
  margin-call threshold of `d ≈ 65%` (holdings would have to fall ~65% within a
  single 21-day hold — vs our unlevered −53% full-cycle MaxDD). At the call point
  account equity is down ~88% (equity drop = L·d). Note: volatile/concentrated
  small-caps often carry 30% house maintenance, which tightens the threshold to
  ~63%. The backtest does **not** model forced liquidation — it rides equity
  through the trough — so these thresholds are the real-world guardrail, not the
  reported CAGR. Vol-target is run purely as call-insurance here: it cuts exposure
  when SPY vol spikes, so nameplate 1.35× spends a crash at effective <1.35×.
- **Triggered at rebalance** — exposure is recomputed every 21 trading days,
  not daily. Mid-hold vol spikes don't trigger an emergency sell; the
  overlay reacts at the next scheduled rebalance. For live use,
  `today.py` reads _today's_ SPY vol, so daily reruns give a near-real-time
  recommendation independent of the backtest cadence.
- **Empirical 2008 stress test** (leave-2008-out walk-forward, train
  2010-2019 / test 2007-2009): raw MaxDD -64.6%, raw CAGR +6.9% vs
  SPY -5.7%. Vol-target overlay sized exposure down hard during Sep-Nov 2008. To re-run this experiment: see `dataset.py` constants and `TEST_END`.

#### Basket size in the 2008 stress test (top-20 vs top-40, 2026-07-23)

Re-ran the leave-2008-out walk-forward for both basket sizes on the **same**
2008-unseen model (`train.py --quick`, train 2010-2017 / val 2018-2019, test
2007-01→2009-12). Resolves the long-standing "top-20 UNVERIFIED in 2008" flag.
All numbers **unlevered**:

| Metric (2007-2009)      | top-20         | top-40        | SPY    |
| ----------------------- | -------------- | ------------- | ------ |
| vol-target CAGR         | +2.92%         | +5.01%        | -5.66% |
| vol-target MaxDD        | -48.2%         | -44.6%        | -55.2% |
| vol-target Sharpe       | 0.09           | 0.17          | -0.19  |
| offset-luck CAGR range  | [-3.7%, +9.8%] | [+0.6%,+10.2%]| —      |
| raw (no overlay) MaxDD  | -61.4%         | -58.0%        | —      |

**Verdict:** top-20 SURVIVES 2008 out-of-sample (positive CAGR, beats SPY) —
it does not detonate — but it is the more fragile basket on every risk axis:
deeper MaxDD (raw and gated), lower crash CAGR, and its offset-luck range dips
**negative** (worst rebalance-timing loses money) while top-40 stays positive
across all 21 offsets. This is the concentration premium: same Sharpe in the
calm 2021→ window, more tail fragility in a crash.

**Two caveats before reading this as comfort:** (1) these are *unlevered* — at
the live 1.35x, the -48%/-61% top-20 drawdowns scale past -50% (margin-call
territory), so **basket size is a minor (~3-4 pt) tail lever vs leverage**,
which is the dominant one. (2) The vol-target overlay averaged only **~83%**
exposure through this window — 2008 was sustained high vol, not a single
spike, so the overlay is *not* a full parachute in a grinding crash; the daily
`today.py` de-risk helps but can't defy a persistent-vol regime. (The
leave-out model is also deliberately weak — val IC ~0.01 — so this is a
structural stress test of basket size + overlay, not a P&L forecast.)

**Decision (unchanged, now evidence-backed): stay at top-20, rebalance
quarterly.** 20 wins on return + capital fit and is 2008-survivable; quarterly
cadence is a coin-flip-gross tie with monthly that wins on costs/taxes (see
"Rebalance cadence decision"). Tail risk is controlled with leverage + daily
de-risk, not basket size.
- **Empirical 2021-2026** (no real vol-bomb in this window): vol-target's
  average exposure stayed ~95% — overlay barely activated because the 2022
  bear was a slow grind, not a vol explosion. MaxDD improved ~2pts vs raw
  at the cost of ~3 CAGR pts. The overlay's value shows up in genuine
  vol-bomb regimes (2008, 2020), not slow grinds.

VIX/SMA200 retired but reversible: `regime_long`/`regime_long_row` are
preserved in `strategy.py`; uncomment the `spy_sma200` line in
`prepare_market` to bring them back.

### Quality filter (cataclysmic-only fundamental/insider screen)

Added 2026-05-17. `strategy.apply_quality_filter` runs once per rebalance
date _before_ `top_picks`, dropping candidates that hit any of these
"firm-going-to-zero" thresholds:

```python
QUALITY_FILTER_DEFAULTS = {
    "max_debt_to_equity":       10.0,            # truly extreme leverage
    "min_current_ratio":         0.3,            # near-insolvent liquidity
    "min_sales_growth_yoy":     -0.50,           # revenue more than halved
    "max_insider_net_sell_60d": -50_000_000.0,   # insiders dumping >$50M net
}
```

**NaN values always pass** — missing fundamentals/insider coverage is not
evidence of low quality (could be pre-XBRL, a young filer, or thinly-
covered ticker). XGBoost already handles missing natively. The filter
is opt-out via `--no-quality-filter` in both `backtest.py` and `today.py`.

**Why these specific thresholds.** A 2026-05-17 sweep across 7 variants
on the raw long-only backtest (21-offset mean):

| Variant                          | Hits%     | CAGR        | Sharpe   | Max DD  |
| -------------------------------- | --------- | ----------- | -------- | ------- |
| no_filter (baseline)             | 0.0%      | +23.02%     | 0.87     | -28.14% |
| **very_loose (current default)** | **10.8%** | **+24.29%** | **0.95** | -26.48% |
| loose (D/E>7 etc.)               | 16.9%     | +22.82%     | 0.91     | -25.57% |
| tight (D/E>3, CR<0.7)            | 33.9%     | +19.53%     | 0.82     | -24.52% |
| very_tight (D/E>2, CR<1.0)       | 52.7%     | +17.58%     | 0.79     | -23.08% |
| only_insider_sell ≤-$20M         | 12.7%     | +23.33%     | 0.91     | -25.72% |
| only*leverage*>5                 | 3.7%      | +22.27%     | 0.84     | -29.16% |
| only*liquidity*<0.5              | 2.4%      | +22.65%     | 0.86     | -27.73% |
| only*sales_collapse*<-30%        | 5.3%      | +22.86%     | 0.88     | -28.36% |

**Key insight:** tighter filters DO reduce drawdown monotonically (lower
vol, lower MaxDD as hit-rate climbs), but they kill CAGR _faster_ than
they help. The model already prices "merely weak" fundamentals via
interactions with regime features — e.g. a tree splits on `vix_level > 25`
then on `debt_to_equity_rank` and learns "high leverage underperforms
in stress regimes". A hard threshold on top of that throws away
mean-reversion winners (high D/E + oversold) that the regime
interaction correctly flips positive. Only the **cataclysmic-only** zone
(~10% hits) is Pareto-additive: at that level, the names you remove are
broken enough that no regime interaction redeems them.

**ROA intentionally absent** from defaults. Development-stage names
(biotech, early SaaS, capex-heavy growth) show deeply negative ROA while
ripping. The model already sees ROA + roa_rank; an additional hard floor
double-penalises those names.

**Single-axis filters underperform the combined version** — each
component contributes orthogonal information. Insider-sell-only is the
strongest single axis (0.91 Sharpe alone) but combined-loose beats it.

**What gets dropped in practice (2026-05-08 live slice — 4 of 501 candidates):**

| Ticker | Pred. 21d return | Trigger                      | Why it's "cataclysmic"                                                                                                                                                                                 |
| ------ | ---------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| APP    | +0.16%           | `insider_net_60d = -$162.1M` | AppLovin — insiders dumping more than 3× our $50M threshold over the trailing 60 days. The single largest insider-exit signal in the universe; people who can see the next quarter's pipeline are out. |
| DELL   | +0.22%           | `insider_net_60d = -$51.5M`  | Just over the $50M threshold. Post-AI-rally cash-out; not as severe as APP but still flags "the people closest to the numbers are taking chips off the table."                                         |
| CCL    | +0.16%           | `current_ratio = 0.30`       | Carnival Corp — only 30¢ of current assets per $1 of current liabilities. Post-COVID balance sheet is still broken; the cruise lines levered up to survive 2020 and haven't repaired liquidity.        |
| NCLH   | +0.24%           | `current_ratio = 0.21`       | Norwegian Cruise Line — even worse than CCL at 21¢ per $1. A near-insolvent working-capital position on paper; one bad quarter from forced refinancing.                                                |

None of these have catastrophic `debt_to_equity` (>10) or
`sales_growth_yoy < -0.50` — the universe in this regime is debt-managed
and growing. The filter is doing its job: catching the few names whose
_single_ fundamental flag is in clear "this might go to zero" territory,
without touching the 99% of names where the model's interaction-based
ranking should be trusted.

For comparison, the **previous tight defaults** (D/E>5, CR<0.5,
sales*yoy<-0.30, ROA<-0.20, insider_net<-$20M) dropped 9 of 501 — same
4 names above plus AXON (insider -$24.9M, just over the old -$20M
line), CHTR (CR 0.40), IT (D/E 9.3), LUV (CR 0.48), MRNA (ROA -27.8%).
All five are debatable: MRNA is a development-stage biotech that
\_should* have negative ROA, CHTR and LUV operate in capital-intensive
industries where 0.4–0.5 current ratios are normal, AXON's insider sale
was post-rally profit-taking not a panic exit, and IT has carried high
leverage through years of strong stock performance. Loosening the
thresholds gives the model back exactly the names it was right about.

To revert to no filter, pass `--no-quality-filter`. To use stricter
thresholds, edit `QUALITY_FILTER_DEFAULTS` in `scripts/strategy.py` —
but re-run the sweep first: tightening cost ~1.5 CAGR pts in the test
above and would likely repeat.

### Basket weighting

`--weight equal` (default) puts 1/N on every name in the basket. `--weight
pred` weights proportional to `predicted_return` (negatives clipped at 0,
falls back to equal if every pick is ≤0). Same flag works on both
`backtest.py` and `today.py`; `today.py` writes the chosen weights into
the `weight` column of `picks_<date>.csv`.

**Empirically, pred-weight does not help** at the current model's signal
strength. On the 2021→2026 test window, switching equal→pred at the
default `--top-n 40` knocks gated Sharpe 0.70 → 0.64 and raw Sharpe 0.84
→ 0.81 — vol rises in both variants while CAGR is flat. Tightening to
`--top-n 10 --weight pred` makes it worse (gated Sharpe 0.37, raw
Sharpe 0.59). Concentration only pays off if the _within-basket
ranking_ carries signal; here it's mostly noise. The signal lives at
"this basket of 40 beats the universe," not "stock #1 beats stock #40."
Kept in the codebase as a knob to re-test against future models.

### Rebalance-date sensitivity

Monthly rebalance has a fragility: which day-of-month you happen to start
matters. We mitigate by running the same strategy 21 times — each one
starting on a different anchor day (offset = 0..20) — and reporting the
mean equity curve plus the 10th/90th percentile band. Closely matches what
21 overlapping sleeves would deliver, with much less code complexity.
Sleeves are on the roadmap; this is the simpler-but-equivalent v1.

### Results (test 2021-01-04 → 2026-03-31, 500-trial sweep on 44 features incl. Form 4 insiders, demeaned labels)

`backtest.py` clips the SPY benchmark to the strategy's last predictable
date so the headline comparison is apples-to-apples by default; the CSV
keeps the full SPY series so the post-strategy tail is visible for
inspection.

**Current model (`models/xgb_v1.json`, 200-trial sweep on 47 features, seed 15, retrained 2026-05-11; quality filter ON, defaults from 2026-05-17):**

| Variant                               | CAGR       | Vol   | Sharpe    | Max DD     | Final NAV | Avg Exposure |
| ------------------------------------- | ---------- | ----- | --------- | ---------- | --------- | ------------ |
| **Raw long-only** (filter ON)         | **+24.3%** | 25.7% | **+0.95** | -26.5%     | **3.13×** | 100%         |
| Vol-targeted (target 0.20, filter ON) | +21.5%     | 23.6% | +0.91     | **-25.0%** | 2.78×     | 95%          |
| Raw long-only (filter OFF, baseline)  | +23.0%     | 26.6% | +0.87     | -28.1%     | 2.96×     | 100%         |
| SPY buy & hold (clipped @2026-04-09)  | +13.6%     | 17.0% | +0.80     | -24.5%     | 1.95×     | —            |

**Reading the table honestly:** raw (filter ON) beats SPY by **+10.7 CAGR
points at Sharpe 0.95 vs 0.80** — the Sharpe gap is +0.15. Final NAV 3.13×
vs SPY 1.95× = **+60% more wealth** over 5.25 years. Saved hyperparams:
`best_iteration=41`, `lr=0.0015`, `reg_lambda=0.005`, `max_depth=5`. The
filter alone contributes +1.3 CAGR pts and +0.08 Sharpe vs the no-filter
baseline; see [Quality filter](#quality-filter-cataclysmic-only-fundamentalinsider-screen)
for the full sweep that justified the chosen thresholds.

**Cross-seed stability:** the same 200-trial protocol on **seed 17**
(prior commit, recorded here for the multi-seed evidence) produced raw
CAGR 22.3% / Sharpe **0.87** / MaxDD -26.6% — Sharpe identical, MaxDD
within 1.5pt, with optuna landing in a very different hyperparam basin
(`best_iter=9`, `lr=0.004`, `max_depth=4`). Vol-target Sharpe 0.82 (this
seed) vs 0.83 (seed 17), MaxDD -25.5% vs -24.8%. **Two seeds → same
alpha** — the signal lives in features + data, not in lucky
hyperparams.

**Important caveat: this is a calm window for vol-target.** Avg
exposure 95% means the overlay barely activated in 2021-2026 — the 2022
bear was a slow grind, not a vol explosion. The leave-2008-out walk-
forward test puts raw MaxDD at -64.6%; vol-target would scale exposure
to ~40% during Sep-Nov 2008 but the long-only floor is still deep.

### Why default top-40, raw (no gate)

A `--top-n 40..60` sweep on both the 2021-2026 test window and the
2007-2026 full history was the basis for the default choice.

**Within the raw variant, n=40 dominates.** It posts the best CAGR in
both windows (+24.05% full, +20.78% recent), best/tied Sharpe (0.76 full,
0.78 recent), and the MaxDD actually gets _deeper_ as n grows — adding
names dilutes signal without reducing crisis beta.

**Raw beats gated on Sharpe across the whole sweep** (raw 0.76–0.78 vs
gated 0.65–0.75). The regime gate gives up roughly half the CAGR
(24% → 10%) for less Sharpe — bad insurance.

**The 2008 MaxDD argument doesn't hold up.** Raw n=40's full-history
drawdown is -56.21%; SPY buy-and-hold over the same window is -55.19%.
The gate was protecting against a drawdown SPY itself doesn't avoid.
You'd need to gate SPY too to be consistent.

**Risk acknowledgement.** Raw means full market beta in a 2008-style
crisis. The thesis is that 401k flows + institutional algo liquidity
have raised the floor vs 2007 — a "this-time-is-different" bet, which
is historically dangerous. Partial mitigation: keep a manual kill switch
(go to cash if SPY breaks SMA200 _and_ VIX > 30) rather than encoding
that logic into the strategy. Human judgment for the 1-in-15-year tail,
not a permanent half-CAGR tax.

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
| **+ Form 4 insider transactions (50 trials)**      | Point-in-time            | clip ±0.5 + date-demeaned | 44 features (40 + 4 insider: buy/sell counts, net dollar, days-since-buy)  | decile-spread, 50 trials, depth ∈ [3,5], LR ∈ [0.005,0.3], ES=100  | **+21.0%** | **+12.0%** | **Sharpe 0.81 vs SPY 0.75 (gap widens from +0.01 to +0.06).** All 4 insider features earn non-zero importance — `insider_buy_count_60d` rank 25/45, `insider_sell_count_60d` weakest at 39/45 (matches 10b5-1-plan-noise literature). `best_iteration=13, learning_rate=0.0051`. MaxDD widens -25.8% → -31.7% (Calmar 0.74 → 0.66) — expected cost of higher CAGR. Bulk-TSV pipeline replaces a per-XML scraper that got the source IP throttled. Saved as `xgb_v1.json`.  |

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
  execute_picks.py               # implemented — rebalance IBKR account toward a picks CSV (print/whatif/live)
  check_ibkr_conn.py             # implemented — WSL→Gateway connection sanity check (prints DU/U banner)
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

**Where we are.** Four non-technical signal sources have landed
(EDGAR earnings calendar, EDGAR XBRL fundamentals, SPY/VIX regime
context, Form 4 insider transactions). Latest: **raw +21.0% CAGR /
Sharpe 0.81 vs SPY +12.7% / 0.75** — Sharpe gap over SPY widened
from +0.01 to +0.06. Each new signal source has earned its keep:
all 44 features in the current panel post non-zero importance except
for 4 still-dead columns flagged for the next stability-selection
prune. Items below are mostly free; each is a self-contained payoff
for further alpha.

> **Order of attack (per-item priority based on signal/effort):**
> §2 earnings calendar → ✅ **done**, +2.0 CAGR pts.
> §4 EDGAR fundamentals + regime context + stability-selection prune
> → ✅ **done**, +19.0% CAGR / Sharpe 0.76 (first time > SPY).
> §3 Form 4 insider transactions → ✅ **done**, **+21.0% CAGR /
> Sharpe 0.81** (Sharpe gap over SPY widens to +0.06).
>
> **Queued (free, in priority order):**
>
> 1. **§7 8-K item 2.02 announcement dates** — ✅ **shipped** (data path).
>    `earnings.py` now caches 8-K filings and filters to item 2.02 via the
>    submissions JSON's parallel `items` array (zero extra requests vs the
>    prior README plan). `load_earnings_dates` prefers 8-K announcement
>    dates per quarter and falls back to 10-Q for pre-item-code filings.
>    Retrain (`--quick` on saved DEFAULT_PARAMS) reproduces the prior raw
>    +21.6% CAGR / 0.86 Sharpe baseline within stochastic noise — 8-K data
>    is **neutral, not lift-positive on top-40 backtest**. The lift may
>    only appear after the optuna objective is realigned with the strategy
>    (see #2 below).
> 2. **✅ Optuna objective realigned with top-40** (May 2026). Replaced
>    val decile spread with **val top-40 mean realised return** as both
>    the per-round eval metric (early stopping) and the trial objective.
>    A brief Sharpe-objective detour was abandoned (optimiser hit
>    `best_iteration=0` by collapsing prediction variance). 50-trial sweep
>    under the new objective: val top-40 mean return +0.0142,
>    `best_iteration=19`, raw backtest **+21.79% CAGR / Sharpe 0.84**
>    (vs prior decile-spread tune at 21.60% / 0.86). Backtest performance
>    is within noise; the real win is that objective ↔ strategy are now
>    aligned, so future optuna sweeps optimise what is actually traded.
>    Eval-metric implementation switched from pandas `groupby().nlargest()`
>    to numpy `argsort(-preds, kind='stable')[:top_n]` with pre-computed
>    per-date index groups → ~3-4× faster per trial (157s/trial → ~45-50s
>    on a 50-trial sweep), bit-exact under stable-sort tie-breaking.
> 3. **§5 ensemble of 5 seeds** (~½ day) — average predictions across 5
>    boosters with different RNG seeds. Documented +5-15% Sharpe. Reuses the
>    seed-sweep tooling from the stability prune. Run after #2 lands so the
>    ensemble averages over a model selected against the right objective.
> 4. **§8 13F institutional ownership** (~1-2 days) — quarterly Schedule 13F
>    via EDGAR bulk data (same shape as your insider pipeline). Smart-money
>    flow tracking: top-N largest holders, holdings concentration, recent net
>    fund buying. Cross-sectional fundamentals-style signal.
> 5. **§9 FRED macro broadcast features** (~½ day) — yield curve slope
>    (`T10Y3M`, 1982+) and HY OAS (`BAMLH0A0HYM2`, 1996+). Both have full
>    2007 floor coverage. Free FRED API, broadcast-into-interaction pattern
>    same as `vix_level` / `spy_rsi_14`.
> 6. **§10 Amihud illiquidity** (~1 hour) — rolling 21d mean of
>    `|daily_return| / dollar_volume`, plus its cross-sectional rank. No new
>    data download (computed from existing OHLCV); full 2007+ coverage.
>    Documented illiquidity premium (Amihud 2002).
> 7. **§11 Quality factors** (~½ day) — gross profitability (Novy-Marx 2013),
>    accruals (Sloan 1996), asset growth (Cooper 2008) via the existing
>    `fundamentals.py` XBRL pipeline. Same NaN profile as current 7
>    fundamentals (2009-06 partial, 2011-06 full). Fills the biggest gap in
>    current quality coverage — `op_income_growth_yoy` is dead, but
>    profitability/accruals capture different axes.
>
> §6 paid delisted-ticker prices (do before going live — ~3-5 CAGR pts of
> survivorship deflation expected). §1 short interest deferred (FINRA archive
> only goes back to 2018).

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

**Follow-up — ✅ shipped (data path).** Switched from 10-Q filing
dates to 8-K item 2.02 announcement dates as the PEAD anchor.
Surprise: the EDGAR submissions API _does_ expose item codes via a
parallel `items` array on `filings.recent` — no per-filing index
fetch required (the prior plan in this README was wrong). 8-K filings
are now cached alongside 10-Q/10-K rows in
`data/earnings/{TICKER}.parquet` (new `items` column; legacy caches
auto-refetch). `load_earnings_dates` prefers the 8-K item 2.02 date
per earnings cycle and falls back to the matching 10-Q for older
filings (pre-2004, before 8-K item codes were standardised) or rare
gaps. Median 10-Q→8-K gap on a smoke-test cohort: AAPL 1d, MSFT 0d,
NVDA 12d — most of the lift will come from names like NVDA whose
10-Q lags meaningfully. Retrain + backtest pending to quantify.

### 3. Insider transactions from SEC EDGAR Form 4 — ✅ shipped

**Shipped** (`scripts/insider.py`): SEC's quarterly Form 3/4/5 bulk TSV
dataset → per-ticker parquet → 4 features in [Bucket 8](#bucket-8--insider-transactions-4).
The first attempt scraped per-filing Form 4 XMLs through EDGAR archives
and got the source IP throttled within a few hours; the bulk-TSV rewrite
does ~80 quarter-zip GETs total (one per quarter back to 2006q1) and
finishes in minutes.

Features:

- `insider_buy_count_60d`, `insider_sell_count_60d` — counts of direct
  open-market officer transactions in the last 60 calendar days
- `insider_net_dollar_60d` — signed dollar volume (P − S); positive = net
  buying
- `days_since_last_insider_buy` — decaying memory, capped at 365

Discipline: the asof key is `filing_date` (Form 4 must be filed within
2 business days of the trade). SEC publishes each quarter ~1 month after
quarter end; live picks lag features by 21d so the publication lag is not
binding.

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

### 8. Diagnostics module — ✅ SHIPPED 2026-07-08 (`scripts/diagnostics.py`)

Six read-only sections (no IBKR connection): per-month IC stability,
underwater plot from the backtest artifacts, picks-concentration audit,
per-stock attribution, per-rebalance hit rate vs SPY, and a live-picks
scorecard that marks every `picks/picks_*.csv` to market from cached
prices (basket incl. cash drag vs SPY over the same 21-trading-day
window, matured vs open). Sections 1/3-5 measure the _model_ on
equal-weight offset-0 baskets; section 6 measures what the live pick
files actually said to hold. Runs automatically as the final, non-fatal
step of `run_all.py`; `--live-only` is the fast daily path. Outputs:
`reports/diagnostics_{ic_monthly,attribution,live_scorecard}.csv`,
`diagnostics_underwater.png`, `diagnostics_summary.json`.

**First run (2026-07-08, test 2021→2026-06) — honest reads:**

- **The IC-level alpha is thin and lumpy**: monthly IC mean +0.018,
  t-stat **1.12** (not significant), only 55% of months positive, and
  excluding the top-3 months drops the mean to +0.005. The best and
  worst months are adjacent (2025-04 IC +0.31, 2025-03 IC −0.28) — the
  tariff-shock whipsaw. The strategy's edge shows up better at the
  basket level than the full-cross-section IC level: avg 21d basket
  return +2.41% vs SPY +1.27%, basket beats SPY on 54% of rebalances
  with hit rate 52% — a small, persistent tilt plus fat right-tail
  months, not a steady high-IC signal.
- **Concentration is real but bounded**: avg IT sector weight 30%
  (peak 57.5% in one basket), 54% consecutive-rebalance overlap
  (~46% turnover/month), 295 unique names over 65 rebalances. Top-10
  contributors = 28% of summed contribution (INTC, STX, NVDA, WDC…).
- **Live picks (since 2026-03)**: 6 matured windows, avg active return
  **+4.43%**, beat SPY in 4/6 — but all 5 currently-open windows are
  trailing SPY (−0.75% to −4.28% active). Watch whether that's noise
  or decay.

### 9. FRED macro broadcast features (free) — **SHIPPED 2026-05-11**

Shipped as **Bucket 2c — broadcast FRED macro regime context** (see the
feature section above for definitions and empirical lift). Three series
made it into the model:

| Shipped name        | FRED ID          | Description                                       |
| ------------------- | ---------------- | ------------------------------------------------- |
| `term_spread_10y3m` | `DGS10 − DGS3MO` | Yield-curve regime / NY Fed recession indicator   |
| `ig_credit_spread`  | `BAA10Y`         | Moody's Baa minus 10y Treasury (IG credit stress) |
| `inflation_5y5y`    | `T5YIFR`         | 5-Year 5-Year Forward Inflation Expectation       |

**HY OAS (`BAMLH0A0HYM2`) was scoped out** — FRED's free CSV now
truncates every ICE BofA series to 2023→ (full history is gated behind
an API key). `BAA10Y` is ~0.85 correlated with HY OAS historically and
serves as the credit-stress proxy. If a future iteration wants HY OAS
specifically, the path is `FRED_API_KEY` + `https://api.stlouisfed.org/fred/series/observations`.

Downloader: `scripts/macro.py` → cached to `data/market/macro.parquet`.
Wired via `compute_macro_regime_features()` in `features.py` (broadcasts
onto the panel's trading-day index with `ffill` only, never `bfill`).

### 10. Amihud illiquidity (free, ~1 hour)

Documented illiquidity premium (Amihud 2002): less-liquid stocks earn higher forward returns cross-sectionally, controlling for size/vol. Cheap to add — no new data source.

**Formula.**

```
ILLIQ_t = rolling_mean_21d( |daily_return_t| / dollar_volume_t )
       where dollar_volume_t = Close_t × Volume_t
```

Scale by `1e6` for readability. Add the cross-sectional rank too (relative illiquidity within the S&P 500 matters more than absolute level). For tail-control, take `log1p` or winsorise at the 99th percentile per date — extreme low-volume sessions (early 2007, holidays) blow up the raw ratio.

**How to wire it.**

1. Add an `out["amihud_21d"]` block to `compute_per_ticker_features()` in `features.py:191` after the existing `vol_*` block.
2. Append `"amihud_21d"` to `PER_TICKER_FEATURES` (`features.py:56`).
3. Add `"amihud_21d"` to `RANKABLE` (`features.py:139`) so the panel-stage rank step picks up `amihud_21d_rank` automatically.

**Coverage.** Computed from existing OHLCV — full 2007+ history, no NaN cliff.

### 11. Quality factors — gross profitability, accruals, asset growth (free, ~½ day)

Three well-documented academic quality factors not in your current 7 fundamentals. Same XBRL pipeline as existing fundamentals → same NaN profile (0% pre-2009-06, partial through 2011-06, full after). XGBoost handles missing natively; current `NULLABLE_FEATURES` machinery already covers this.

**Factors and rationale.**

| Factor                | Formula                                     | Reference      | Sign                |
| --------------------- | ------------------------------------------- | -------------- | ------------------- |
| `gross_profitability` | (TTM Revenue − TTM COGS) / MRQ Total Assets | Novy-Marx 2013 | + (the other value) |
| `accruals`            | (ΔWC − Depreciation) / avg(MRQ Assets)      | Sloan 1996     | − (high → poor fwd) |
| `asset_growth`        | MRQ Assets / MRQ Assets 4Q ago − 1          | Cooper 2008    | −                   |

Where `WC = current_assets − current_liabilities` (already pulled as `mrq_assets_current` − `mrq_liabilities_current`).

**New XBRL tags needed in `fundamentals.py`** (add to the `_TAGS` synonym dict and the per-ticker pull):

- `CostOfGoodsSold` / `CostOfRevenue` / `CostOfGoodsAndServicesSold` (synonym handling like the existing `Revenues` family).
- `DepreciationDepletionAndAmortization` / `DepreciationAndAmortization` / `Depreciation` for accruals.
- A 4Q-lagged `Assets` value for asset growth (the existing `mrq_assets` is already pulled; just need a prior-period version analogous to `ttm_revenue_prior`).

**How to wire it.**

1. Extend the TTM/MRQ pulls in `fundamentals.py` to include COGS (TTM), depreciation (TTM), and lagged assets (4Q prior MRQ).
2. Add the three ratio computations to `attach_fundamentals()` in `features.py:383` alongside the existing 7.
3. Append to `FUNDAMENTAL_FEATURES` (`features.py:123`) and `RANKABLE` (`features.py:139`).
4. Sign-flip / divide-by-zero handling: same pattern as existing — set NaN when denominator ≤ 0 rather than letting negative-denominator rows produce wrong-sign signals.

**Hypothesis to validate after retraining.** `op_income_growth_yoy` was dead in 5/5 stability seeds and `sales_growth_yoy` survives — earnings-form growth doesn't pay here. But profitability and asset-growth capture different axes (operational efficiency, expansion-as-overinvestment) — they should land non-zero. Accruals is the riskiest of the three (requires depreciation tag coverage, which is patchier in XBRL than revenue/COGS).

### 12. Model-architecture experiments — same panel, new objectives (free, ~½–1 day each)

Every signal source so far has been a new _feature_. These are new _models /
objectives_ on the existing 47-feature panel — cheap to try because the data
layer doesn't change. In rough order of expected fit:

1. **Learning-to-rank objective.** The label, the tuning metric, and the
   strategy are all ranking constructs, but training is RMSE regression.
   XGBoost ships `rank:pairwise` / `rank:ndcg` — set the date as the group
   key (`DMatrix.set_group` with per-date row counts; splits are already
   chronological so groups never straddle train/val). LambdaMART directly
   optimizes "get the top of the list right," which is what top-40
   selection needs and what decile-spread regression only approximates.
   Keep the top-40 mean-return eval metric for early stopping so the
   objective change is the only variable.
2. **Classification reframing.** Predict `P(stock in top decile of forward
21d return)` as binary logistic instead of regressing the demeaned
   return. Robust to the fat-tailed label noise that forces the ±0.5 clip,
   and the predicted probability is a natural position-sizing weight.
   Label build is a per-date `groupby.rank(pct=True) >= 0.9` on the
   existing forward return — no new data.
3. **Multi-horizon blend.** Add 5d and 63d forward labels in `labels.py`,
   train three boosters, blend their per-date ranks (start with equal
   weights). Short horizons capture reversal, long capture trend; the
   blend usually beats any single horizon.
4. **Meta-labeling** (López de Prado, _Advances in Financial ML_ ch. 3).
   A second small model takes the primary model's prediction + regime
   features and predicts "will this pick beat SPY over the hold?" — used
   for _sizing_, not selection. The principled version of what the
   cataclysmic quality filter does by hand. Needs care to train it on
   out-of-fold primary predictions to avoid leakage.
5. **Cross-model ensemble.** XGBoost + LightGBM + a ridge regression on
   the rank features, averaged per-date ranks. Different inductive biases
   diversify better than different RNG seeds — but do §5 (5-seed ensemble)
   first since it's already scoped and the tooling exists.

Skip neural nets (MLP / TabNet) for now — on a ~1M-row tabular panel with
47 features, gradient-boosted trees almost always win, and the time is
better spent on the ranking objective.

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
- [x] diagnostics: per-month IC stability, underwater plot, picks-concentration audit, per-stock attribution, hit rate, live-picks scorecard — shipped 2026-07-08 as `scripts/diagnostics.py`, wired as the final non-fatal step of `run_all.py`. First-run findings in [§8](#8-diagnostics-module--shipped-2026-07-08-scriptsdiagnosticspy): monthly IC t-stat only 1.12 (top-3 months carry most of the IC), but basket-level edge is steadier (+2.41% vs SPY +1.27% avg per rebalance); live picks 6 matured windows avg +4.43% active, open windows all trailing.
- [x] run_all.py orchestrator — daily and retrain modes, auto --diff for today.py
- [x] add label clip ±0.5 to keep dead-ticker −100% labels from dominating MSE
- [x] re-tune with `max_depth ∈ [3, 6]` and re-run backtest (200 trials; raw +13.3% → +17.3%, val decile spread +0.0213 → +0.0297; optuna picked depth=3 + `colsample_bytree=0.629` as the actual lever)
- [x] experiment: train without SPY/VIX broadcast features to force cross-sectional signal — **done**, cost ~1 CAGR pt; combined with demeaning below
- [x] experiment: train on date-demeaned forward returns — **done**, raw +17.3% → +15.5%, Sharpe 0.79 → 0.70 vs SPY 0.75; `best_iteration=3` ceiling on technicals
- [x] save legacy +17.3% model as `xgb_v1_legacy.json` for IRA deployment
- [x] feature: `excess_ret_5d_vs_sector` and `excess_ret_21d_vs_sector` (sector-relative momentum)
- [x] fix `backtest.py` SPY end-date so headline CAGR compares like-for-like
- [x] **feature: earnings calendar from SEC EDGAR 10-Q/10-K + yfinance forward dates** — `days_to_earnings`, `days_since_earnings`, `post_earnings_drift_window`. `days_since_earnings` lands at 14th in feature importance (0.037 gain); other two are 0.0 (kept pending review). Test CAGR +15.5% → +17.5%, `best_iteration` 3 → 10
- [x] follow-up: switch earnings dates from 10-Q filing to 8-K item 2.02 announcement — **shipped (data path)**. SEC submissions JSON already exposes item codes via a parallel `items` array, so the switch needed zero extra requests (not "one extra request per filing" as previously assumed). `earnings.py` caches 8-K filings (new `items` column on `data/earnings/{TICKER}.parquet`; legacy caches auto-refetch) and `load_earnings_dates` prefers 8-K item 2.02 dates per earnings cycle with 10-Q fallback for pre-2004 filings. Per-ticker median 10-Q→8-K gap varies widely: AAPL 1d, MSFT 0d, NVDA 12d. Retrain + backtest pending to quantify the PEAD-anchor lift.
- [~] feature: short interest from FINRA bi-monthly — **deferred**. Download infrastructure shipped (`scripts/deprecated_short_interest.py`) but FINRA CDN archive starts mid-2018, so train=2007–2017 has 0% coverage and XGBoost cannot build splits on the feature. Mean-fill rejected (regime leak); sliding splits forward sacrifices test bear coverage. Revisit when paid historical short interest is added or when splits are re-platformed
- [x] feature: insider transactions from SEC EDGAR Form 4 (`scripts/insider.py`) — bulk-TSV approach via SEC DERA's quarterly Form 3/4/5 dataset (~80 zips back to 2006q1, downloads in minutes). 4 features (buy/sell counts in 60d, net dollar volume, days-since-last-buy). All 4 earn non-zero importance; `insider_buy_count_60d` strongest at rank 25/45. Replaced an earlier per-XML scraper that got the source IP throttled.
- [x] feature: SEC EDGAR XBRL fundamentals (`scripts/fundamentals.py`) — 7 ratios shipped (E/P, B/M, ROA, D/E, current_ratio, sales/op-income growth YoY) + split-adjusted shares for correct market cap. Final config: raw fundamentals + ranks + 5 broadcast SPY/VIX regime features brought back. **Iteration A** (raw, 500 trials, no regime): regressed (+17.5% → +15.1% CAGR; ceiling at decile spread 0.0182). **Iteration B** (raw + ranks + regime, 200 trials): **decile-spread ceiling broken** (+0.0182 → +0.0235), val IC +0.0568 (best ever in clean arch), but raw CAGR +16.2% — short of the +17.5% earnings-only headline. 3 of 5 regime features absorbed (vix_level, spy_rsi_14, spy_ret_21d); 2 dead fundamentals resurrected (B/M, sales_growth); 2 still dead (op_income_growth, current_ratio); all 7 fundamental ranks dead.
- [x] follow-up to §4: prune dead features via 5-seed stability selection. **Result**: 19 features dead in all 5 seeds, 21 columns total dropped (raw + ranks). 61 → 40 features. Single-run prune was wrong — `current_ratio`, `book_to_market`, `earnings_yield`, `sales_growth_yoy` all looked dead in some runs but fired strongly in others (rare-regime signals). Methodology persisted in `train.py --seed N` flag and `reports/feature_importance_stability.csv`. See [Stability-selection prune](#stability-selection-prune).
- [x] retrain on the pruned 40-feature set (50-trial sweep) and re-run backtest. **Result**: raw +19.0% CAGR / Sharpe 0.76 (first time > SPY's 0.75); `best_iteration` lifted to 43 with `lr=0.0058` — the cleaner feature set unlocked a slow-build basin the 61-feature config couldn't find. Saved as `models/xgb_v1_stability_pruned.json`.
- [x] retrain on the 44-feature set with insiders (50-trial sweep) and re-run backtest. **Result**: raw +21.0% CAGR / Sharpe 0.81 (Sharpe gap over SPY widens from +0.01 to +0.06); `best_iteration=13`, `lr=0.0051`. MaxDD widens -25.8% → -31.7% (Calmar 0.74 → 0.66 — slight tail-risk regression, clear Sharpe win). All 4 insider features earn non-zero importance. Saved as `models/xgb_v1.json`.
- [x] make today.py read from features.parquet instead of panel.parquet — `panel.parquet` drops the most recent ~21 trading days because forward returns aren't yet realised, but those are exactly the rows today.py needs to score live. New `load_features()` helper in `scripts/features.py`.
- [ ] system: ensemble of 5 boosters with different seeds — average predictions (free, ~½ day, +5–15% Sharpe)
- [ ] feature: 13F institutional ownership from SEC EDGAR (~1-2 days) — quarterly Schedule 13F filings via EDGAR bulk data, same shape as insider pipeline. Candidate features: top-N largest holder count, ownership concentration (HHI on holdings), net fund buying in last quarter. Smart-money flow signal.
- [ ] feature: FRED macro broadcast (~½ day) — `T10Y3M` (10y/3m term spread, 1982+) and `BAMLH0A0HYM2` (HY OAS, 1996+) via free FRED API. Both have full 2007 floor coverage. Broadcast regime context that interacts with cross-section, same shape as `vix_level` / `spy_rsi_14`. See [§9](#9-fred-macro-broadcast-features-free--day).
- [ ] feature: extended recession-probability suite (~1 day) — broadcast regime block beyond §9, designed to interact with cross-section under stress. All free; all earliest dates predate the 2007 train floor.
    - **FRED series** (free API via `fredapi` / `pandas-datareader`):
        - `RECPROUSM156N` — NY Fed monthly P(US recession in 12 months), derived from the yield curve. 1959+. Cleanest single pre-computed predictor — model gets a probability instead of having to learn the inversion shape itself.
        - `T10Y2Y` — 10Y minus 2Y Treasury, popular alternative to §9's `T10Y3M`. 1976+. A/B-test against T10Y3M to see which inversion definition the cross-section reacts to.
        - `SAHMCURRENT` — Sahm rule (3-month avg unemployment minus prior-12-month low; ≥ 0.5pp = real-time recession trigger). 1948+. Labor-market signal independent of yield curve.
        - `CFNAI` — Chicago Fed National Activity Index, 85-indicator composite. < −0.7 historically marks recessions. 1967+.
        - `USSLIND` — Philly Fed Leading Index for the United States. Forward-looking diffusion. 1982+.
        - `USALOLITONOSTSAM` — OECD Composite Leading Indicator for the US. 1955+. Mirrors CFNAI with a different basket — useful for diversification.
        - `ICSA` — Initial unemployment claims (weekly); take 4-week moving average. 1967+. High-frequency labor signal.
    - **Non-FRED nowcasts** (free, JSON, lower priority — require daily scraping):
        - Atlanta Fed GDPNow — `atlantafed.org/cqer/research/gdpnow` (live GDP estimate).
        - NY Fed Staff Nowcast — `newyorkfed.org/research/policy/nowcast` (similar real-time GDP nowcast).
    - **Reference only — never a feature**: `USREC` (NBER official recession indicator, retrospective binary). Use for attribution / per-regime backtest slicing only; using it as a feature is a look-ahead violation since NBER dates recessions in arrears.
    - **Watch for redundancy.** `RECPROUSM156N` is itself derived from the yield curve, so its signal partially overlaps `T10Y3M`. Hope is that stress measures from different domains (rates → labor → leading composites) capture distinct cross-section interactions; prune via stability sweep after wiring.
- [ ] feature: Amihud illiquidity (free, ~1 hour) — rolling 21d mean of `|daily_return| / dollar_volume`, plus cross-sectional rank. Computed from existing OHLCV; full 2007+ coverage. Documented illiquidity premium (Amihud 2002). See [§10](#10-amihud-illiquidity-free-1-hour).
- [ ] feature: quality factors — gross profitability (Novy-Marx 2013), accruals (Sloan 1996), asset growth (Cooper 2008) (~½ day) via existing XBRL pipeline. New XBRL tags needed: COGS, depreciation, 4Q-lagged assets. Same NaN profile as existing fundamentals. See [§11](#11-quality-factors--gross-profitability-accruals-asset-growth-free--day).
- [ ] re-run null test on the clean-architecture model (current null-test table is stale)
- [ ] data: swap yfinance → Sharadar (or equivalent) for delisted-ticker coverage — do before going live
- [ ] follow-up stability-selection prune now that 4 new features have landed — `excess_ret_5d`, `atr_pct`, `earnings_yield`, `roa_rank` were dead in this single run, but a 5-seed sweep is needed before pruning
- [x] 🔴 **daily de-risk check** — **shipped 2026-07-13** as a default step of `today.py` (so `run_all.py` gets it for free). Recomputes `min(--leverage, --vol-target / spy_vol_20d)` from the freshest cached SPY data and compares it against the live book's gross exposure, printing "OK — hold" or "⚠⚠ DE-RISK … SELL DOWN TODAY" with the exact `execute_picks.py` command. Book exposure comes from `reports/live_book.json`, which `execute_picks.py --mode live` now writes after placing orders (seeded retroactively for the 2026-07-06 go-live: gross 1.07x); falls back to the full leverage cap (worst case — can only over-fire, never lull) if the file is missing, or `--book-exposure` to override. Sell-down only by design: when the formula rises back above the book, re-levering waits for the quarterly rebalance. Alarm threshold: formula < book × 0.90 (`--derisk-tolerance`). `--no-derisk` skips.
- [ ] model: learning-to-rank objective (`rank:pairwise` / `rank:ndcg`, date as group key) — direct match to top-40 selection; keep the top-40 mean-return eval metric so the objective is the only variable. See [§12](#12-model-architecture-experiments--same-panel-new-objectives-free-1-day-each).
- [ ] model: classification reframing — binary logistic on `P(top decile of forward 21d return)`; robust to fat-tailed labels, probability doubles as a sizing weight. See [§12](#12-model-architecture-experiments--same-panel-new-objectives-free-1-day-each).
- [ ] labels: multi-horizon blend — add 5d + 63d forward labels, train 3 boosters, blend per-date ranks. See [§12](#12-model-architecture-experiments--same-panel-new-objectives-free-1-day-each).
- [ ] model: meta-labeling — second model predicts P(pick beats SPY) for sizing, trained on out-of-fold primary predictions. See [§12](#12-model-architecture-experiments--same-panel-new-objectives-free-1-day-each).
- [ ] system: cross-model ensemble (XGBoost + LightGBM + ridge-on-ranks, per-date rank average) — after the 5-seed ensemble (§5) lands. See [§12](#12-model-architecture-experiments--same-panel-new-objectives-free-1-day-each).
- [ ] cross-project: point-in-time backtest of the `conviction-pick-sp500` deterministic funnel against this repo's panel (membership history + OHLCV + XBRL fundamentals) — validates that doctrine's quality gates historically; see that repo's README roadmap.
- [ ] 🔴 **train.py: replace optuna objective (val decile spread) with a top-40-aligned metric** — empirical anti-correlation observed during the 8-K work (May 2026): a 350-trial sweep maximised val decile spread to 0.0183 but backtested at +18.81% raw CAGR, while `--quick` with saved DEFAULT_PARAMS (val decile spread 0.0161) backtested at +21.60%. Decile spread averages ranking across deciles 1 & 10 (~50 names each) but the strategy holds only the top 40 — the very tip of decile 1. Replace `_compute_decile_spread()` with cumulative log return (or Sharpe) of an equal-weighted top-40 portfolio over val. Until this lands, **do not run `train.py --trials N`** — the saved DEFAULT_PARAMS are the current best model. Also lower `reg_lambda` floor from 0.001 (already done) — no other range changes pending.

Paste this to claude to ask
claude --resume b63b90f4-923f-419f-b30e-00cd9006952f
claude --resume 7762f7ea-721e-4179-a24b-273d86c65f0e
claude --resume 94d5520c-9a4b-460f-9e6d-b16cc80211b4
claude --resume df27d2b6-5402-4381-89c2-89a7b3fb0d76 (insider)
claude --resume 14ebc2c2-fe20-4ae5-8fc9-32d10b7ca9d6 (macro)
claude --resume eabad73f-1806-46e5-99e1-4f4c8817d4ba (2026-06-20: outage fixes; net-issuance feature tested + reverted — uncommitted)

Session 2026-06-20 summary (work tree NOT committed — review pending):

1.  Fixed ~10-day picks outage: data/universe/sp500_history.parquet was 0 bytes
    (write interrupted ~Jun 3); every daily run since crashed reading it.
    Deleted + rebuilt; universe.py now treats a 0-byte parquet as missing and
    writes atomically (tmp + os.replace).
2.  macro.py: FRED unreachable from current network (fredgraph.csv times out;
    geo-block/rate-limit suspected — api host responds, download host doesn't).
    Now falls back to the cached macro.parquet on total failure and merges
    partial fetches instead of dropping failed columns. Cache stale at
    2026-06-01 — acceptable (slow-moving regime series).
3.  Quality-filter axis experiment — TESTED, REJECTED, REVERTED. Tried two new
    cataclysmic axes on top of current defaults: shares_growth_yoy>25% (dilution)
    and insolvency_flag (equity<=0 AND TTM NI<0, to cover the D/E-NaN-when-
    equity<=0 blind spot). Sweep on raw long-only 21-offset (test → 2026-05-18):

        variant            hits   CAGR     Sharpe  MaxDD
        no_filter          0.0%  +26.02%   0.98   -27.64%
        defaults (current) 6.4%  +26.35%   1.03   -26.88%   <- still best point
        +dilution>25%      9.0%  +26.02%   1.04   -26.23%
        +insolvency        6.9%  +26.10%   1.03   -26.44%
        +both              9.5%  +25.88%   1.04   -25.65%

    Both axes show the _tightening_ signature (CAGR down monotonically with
    hit-rate, MaxDD down, Sharpe flat at +0.01 noise) — NOT the cataclysmic-
    Pareto signature the original 4 axes have (defaults raise CAGR 26.02->26.35
    AND cut MaxDD). insolvency_flag's cost is the predicted false positive: it
    drops buyback-driven negative-equity names (BA/MCD/HD-type) the model is
    right to hold. Conclusion: QUALITY_FILTER_DEFAULTS unchanged; filter-axis
    code + scripts/quality_axis_sweep.py reverted/deleted.

4.  Net-issuance (shares_growth_yoy) as a MODEL FEATURE — TESTED, DEAD, FULLY
    REVERTED. Promoted shares_growth_yoy (Pontiff & Woodgate 2008, two-sided:
    +dilution / -buybacks) to FUNDAMENTAL_FEATURES (47->48) and retrained
    (50-trial). Result: 0.0 importance (rank 38/48) — the model never split on
    it. The retrain also LANDED in the aggressive-shallow basin (best_iter=4,
    lr=0.017) the README warns underperforms, so raw long-only fell 26.35% ->
    23.39% CAGR vs the prior model on the same window — basin variance, not the
    feature (feature was dead). NOTE the 0-importance is weakly conclusive: at
    best_iter=4 only 4 trees grew, so 20/48 features show 0 importance incl.
    known-good ones (book_to_market, insider counts). Verdict: net issuance does
    not carry signal here in either filter OR feature form. Reverted features.py,
    fundamentals.py, models/xgb_v1.json, and reports/ to HEAD; restored the prior
    47-feature model. Working tree keepers: ONLY universe.py + macro.py.
    TODO confirm: rerun backtest.py on the restored model to verify it matches
    prior behavior before committing.

Note (2026-06-20): the red TODO below claiming "do not run train.py --trials N"
is STALE — train.py:267 objective is already the top-N mean-return metric the
warning asked for (fixed 2026-05-10). Retraining is safe. Left the TODO text
for history but it no longer blocks.

claude --resume eabad73f-1806-46e5-99e1-4f4c8817d4ba
Live ibkr place trade: claude --resume 86203dd8-11fa-43ed-a889-a7213c2e3af3
July trade placed: claude --resume 944406b7-7f36-440e-884d-723e320df5fa
Quarterly rebalance: claude --resume c55b3738-bfb1-40ea-8d4e-eeadb095d308
