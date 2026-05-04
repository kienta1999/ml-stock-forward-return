#!/usr/bin/env python3
"""Compute features for the ranker.

Five feature buckets:
  1. Per-ticker technicals  (returns, oscillators, trend, vol, volume, z-scores)
  2. Market context         (beta, excess returns vs SPY — ticker-specific only;
                             broadcast SPY/VIX features are intentionally
                             omitted so the model can't lean on regime signal)
  3. Sector-relative        (ret_n minus the equal-weighted within-sector mean)
  4. Cross-sectional ranks  (per-date percentile of every numeric feature in {1})
  5. Categorical            (gics_sector — XGBoost native categorical)

LOOKAHEAD: every value on row date=D uses only data with date <= D.
All operations are rolling / ewm / pct_change / shift(positive). No row peeks
into the future. dataset.py runs an explicit sanity assertion on this later.

CLI:
    python scripts/features.py --ticker AAPL    # smoke test, print last 5 rows
    python scripts/features.py                  # build full panel → data/processed/features.parquet
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from earnings import load_earnings_dates  # noqa: E402
from data import load_market, load_prices  # noqa: E402
from fundamentals import load_fundamentals  # noqa: E402
from insider import load_insider_transactions  # noqa: E402
from universe import (  # noqa: E402
    UNKNOWN_SECTOR,
    filter_to_members,
    load_history,
    load_sectors,
)

_ROOT = os.path.dirname(_HERE)
PANEL_PATH = os.path.join(_ROOT, "data", "processed", "features.parquet")

# Feature columns by bucket — kept here so labels.py / dataset.py / train.py can import them.
# Pruned 2026-05-03 via 5-seed stability selection (--quick × seeds 1-5).
# A feature is dropped only if importance==0 in ALL 5 seeds. Marginal features
# (alive in just 1-2 seeds) are kept — single-run noise can easily mask real
# signal. See `reports/feature_importance_stability.csv` for the full table.
PER_TICKER_FEATURES: list[str] = [
    # "ret_1d", "ret_5d",  # dead in 5/5
    "ret_21d", "ret_63d",
    # "rsi_14", "mfi_14",  # dead in 5/5
    "macd_hist",
    "atr_pct", "vol_20d", "vol_60d",
    # "vol_ratio", "dist_sma50",  # dead in 5/5
    "dist_sma200", "dist_52w_high",
    # "trend_regime",  # dead in 5/5
    "zscore_20d", "zscore_60d",
]
MARKET_FEATURES: list[str] = [
    "beta_60d", "excess_ret_5d", "excess_ret_21d",
]
# Broadcast market-regime features — same value for every ticker on a given
# date. Removed in the clean-arch refactor because they let the model time
# the market with raw labels. RE-ADDED 2026-05-03: with date-demeaned labels,
# a broadcast feature cannot earn reward as a standalone predictor (target
# sums to zero per date), but CAN earn reward via interactions — e.g. a
# tree splits on `vix_level > 25` then on `debt_to_equity_rank`, encoding
# "in stress regimes high-leverage names underperform". That regime-cross-
# section interaction is what we want and what's been missing.
MARKET_REGIME_FEATURES: list[str] = [
    "spy_ret_21d",
    # "spy_trend_regime",  # dead in 5/5 (subsumed by spy_rsi_14)
    "spy_rsi_14",
    "vix_level",
    # "vix_zscore_20d",  # dead in 5/5 (subsumed by vix_level)
]
# Sector-relative features: ret_n minus the equal-weighted within-(date, sector)
# mean. Computed at panel-assembly time once all tickers are stacked. Excluded
# from the per-ticker lookahead check because they depend on cross-sectional
# state — but they have no lookahead by construction (mean is over
# contemporaneous returns only).
SECTOR_FEATURES: list[str] = [
    # "excess_ret_5d_vs_sector",  # dead in 5/5
    "excess_ret_21d_vs_sector",
]
# Bucket 6 — earnings calendar (per-ticker, computed from EDGAR 10-Q/10-K
# filing dates + yfinance forward calendar for the live row). Caveat: the
# EDGAR filing date trails the actual 8-K item-2.02 announcement by ~2-4
# weeks, so the PEAD window here is anchored on the 10-Q filing rather than
# the announcement. Live `days_to_earnings` blends in yfinance upcoming dates.
EARNINGS_FEATURES: list[str] = [
    "days_to_earnings", "days_since_earnings",
    # "post_earnings_drift_window",  # dead in 5/5, redundant with continuous days_since
]

# Bucket 7 — fundamentals from SEC EDGAR XBRL (TTM income/cashflow + MRQ
# balance sheet, asof-merged into the panel by `filed` date for no-lookahead).
# Coverage caveat: 2007-01 → 2009-06 is 0% (pre-XBRL-mandate); 2009-06 → 2011-06
# is partial (large filers only). NaN for all rows in those windows; XGBoost
# handles missing natively.
#   earnings_yield      = TTM net_income / market_cap          (E/P, value)
#   book_to_market      = MRQ equity / market_cap              (B/M, value)
#   roa                 = TTM net_income / MRQ assets          (profitability)
#   debt_to_equity      = MRQ lt_debt / MRQ equity             (leverage)
#   current_ratio       = MRQ assets_current / liab_current    (liquidity)
#   sales_growth_yoy    = TTM revenue / TTM revenue 4Q ago − 1 (growth)
#   op_income_growth_yoy = TTM op_income / TTM op_income 4Q ago − 1 (growth)
# All 7 fundamentals retained for the regime-interaction test (2026-05-03).
# In the rank-only sweep, B/M / sales_growth_yoy / op_income_growth_yoy
# showed 0 importance in both raw and rank form — but those runs had no
# regime context. Adding the 5 broadcast SPY/VIX features re-opens the
# possibility that growth and B/M carry regime-conditional signal (e.g.
# high-growth names underperforming in high-VIX, B/M working only in
# trend-up regimes). Re-evaluate after the regime-included run.
FUNDAMENTAL_FEATURES: list[str] = [
    "earnings_yield", "book_to_market", "roa",
    "debt_to_equity", "current_ratio",
    "sales_growth_yoy",
    # "op_income_growth_yoy",  # dead in 5/5
]

# Features that get cross-sectional ranks. Skip binary trend_regime.
# All fundamentals ranked for cross-sector comparability (5% E/P =
# cheap-for-tech / median-for-banks; rank normalizes that). Even
# current_ratio — although its rank showed 0 in the regime-less sweep
# while raw won — gets a rank in this run so the regime-interaction
# story can give it a fair chance. Prune later based on importance.
# Explicit list — skip features whose RANK was dead in 5/5 seeds even though
# the raw value stays: ret_21d_rank and sales_growth_yoy_rank are excluded
# despite ret_21d / sales_growth_yoy keeping their raw form.
RANKABLE: list[str] = [
    "ret_63d", "macd_hist", "atr_pct",
    "vol_20d", "vol_60d",
    "dist_sma200", "dist_52w_high",
    "zscore_20d", "zscore_60d",
    "earnings_yield", "book_to_market", "roa",
    "debt_to_equity", "current_ratio",
]
RANK_FEATURES: list[str] = [f"{c}_rank" for c in RANKABLE]
CATEGORICAL_FEATURES: list[str] = ["gics_sector"]

# Bucket 8 — insider transactions from SEC EDGAR Form 4 (asof filing_date).
# Only direct-ownership non-derivative open-market purchases (P) and sales (S).
# CEO/CFO open-market buying has documented cross-sectional predictive power.
# days_since_last_insider_buy is NaN if no historical buy is on record; all
# four features are nullable so tickers with missing EDGAR coverage keep rows.
INSIDER_FEATURES: list[str] = [
    "insider_buy_count_60d",
    "insider_sell_count_60d",
    "insider_net_dollar_60d",
    "days_since_last_insider_buy",
]

# Features allowed to remain NaN (XGBoost handles missing natively). dataset.py
# excludes these from its dropna gate so tickers without EDGAR coverage (mostly
# delisted/renamed names with no current SEC CIK) keep their OHLCV-only rows.
# The *_rank versions of fundamentals inherit NaN from their raw values (since
# pandas .rank() preserves NaN), so they must be nullable too — otherwise the
# dropna gate kills every pre-2009-XBRL row.
# Only include ranks that actually exist (RANKABLE ∩ FUNDAMENTAL_FEATURES).
_FUNDAMENTAL_RANKS: list[str] = [
    f"{c}_rank" for c in FUNDAMENTAL_FEATURES if c in RANKABLE
]
NULLABLE_FEATURES: list[str] = (
    EARNINGS_FEATURES + FUNDAMENTAL_FEATURES + _FUNDAMENTAL_RANKS
    + INSIDER_FEATURES
)

ALL_FEATURES: list[str] = (
    PER_TICKER_FEATURES + MARKET_FEATURES + MARKET_REGIME_FEATURES
    + SECTOR_FEATURES
    + EARNINGS_FEATURES + FUNDAMENTAL_FEATURES + RANK_FEATURES
    + INSIDER_FEATURES
    + CATEGORICAL_FEATURES
)


# ─────────────────────────────────────────────────────────────────────────────
# Bucket 1 — per-ticker technicals
# ─────────────────────────────────────────────────────────────────────────────


def compute_per_ticker_features(prices: pd.DataFrame) -> pd.DataFrame:
    """OHLCV (Open, High, Low, Close, Volume) → DataFrame indexed by Date with
    Bucket 1 features. Early rows will have NaN where the rolling window isn't
    full yet — downstream code drops them.
    """
    h, l, c, v = prices["High"], prices["Low"], prices["Close"], prices["Volume"]
    out = pd.DataFrame(index=prices.index)

    # Returns (trailing).
    for n in (1, 5, 21, 63):
        out[f"ret_{n}d"] = c.pct_change(n)

    # RSI(14), Wilder's smoothing via EWM com=13.
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    out["rsi_14"] = 100 - 100 / (1 + gain / loss)

    # MFI(14) — RSI weighted by money flow (typical_price * volume).
    typ = (h + l + c) / 3
    money_flow = typ * v
    typ_diff = typ.diff()
    pos_flow = money_flow.where(typ_diff > 0, 0.0)
    neg_flow = money_flow.where(typ_diff < 0, 0.0)
    pos_sum = pos_flow.rolling(14).sum()
    neg_sum = neg_flow.rolling(14).sum()
    out["mfi_14"] = 100 - 100 / (1 + pos_sum / neg_sum)

    # MACD histogram (12, 26, 9).
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    out["macd_hist"] = macd_line - macd_signal

    # ATR(14) as % of price — cross-sectionally comparable.
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(com=13, adjust=False).mean()
    out["atr_pct"] = atr / c

    # Realized vol (annualized log-return std).
    log_ret = np.log(c / c.shift())
    out["vol_20d"] = log_ret.rolling(20).std() * np.sqrt(252)
    out["vol_60d"] = log_ret.rolling(60).std() * np.sqrt(252)

    # Volume ratio vs 20d avg.
    out["vol_ratio"] = v / v.rolling(20).mean()

    # Trend / position relative to MAs.
    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()
    sma60 = c.rolling(60).mean()
    sma200 = c.rolling(200).mean()
    out["dist_sma50"] = c / sma50 - 1
    out["dist_sma200"] = c / sma200 - 1
    out["dist_52w_high"] = c / c.rolling(252).max() - 1
    out["trend_regime"] = (sma50 > sma200).astype(float)

    # Z-scores: (close − sma_n) / std_n. A volatility-normalised "distance from mean".
    std20 = c.rolling(20).std()
    std60 = c.rolling(60).std()
    out["zscore_20d"] = (c - sma20) / std20
    out["zscore_60d"] = (c - sma60) / std60

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Bucket 2 — market context
# ─────────────────────────────────────────────────────────────────────────────


def attach_market_context(
    per_ticker: pd.DataFrame,
    prices: pd.DataFrame,
    spy: pd.DataFrame,
) -> pd.DataFrame:
    """Add Bucket 2 features — only the ticker-specific ones (beta + excess
    returns vs SPY). Broadcast market-wide features (SPY trend/RSI, VIX level)
    are deliberately excluded — they're identical across all tickers on a given
    date and dominate split selection without contributing to cross-sectional
    ranking.
    """
    out = per_ticker.copy()

    # Beta(60d) — rolling cov(ticker_ret, spy_ret) / var(spy_ret).
    ret_1d = prices["Close"].pct_change()
    spy_ret_1d = spy["Close"].pct_change().reindex(ret_1d.index)
    out["beta_60d"] = ret_1d.rolling(60).cov(spy_ret_1d) / spy_ret_1d.rolling(60).var()

    # Excess returns (relative strength vs SPY).
    spy_ret_5d = spy["Close"].pct_change(5).reindex(ret_1d.index)
    spy_ret_21d = spy["Close"].pct_change(21).reindex(ret_1d.index)
    out["excess_ret_5d"] = out["ret_5d"] - spy_ret_5d
    out["excess_ret_21d"] = out["ret_21d"] - spy_ret_21d
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-ticker convenience: compute_features = Bucket 1 + Bucket 2
# ─────────────────────────────────────────────────────────────────────────────


def compute_features(
    prices: pd.DataFrame,
    spy: pd.DataFrame,
) -> pd.DataFrame:
    """One ticker's full feature frame (Buckets 1 + 2). Indexed by Date."""
    bucket1 = compute_per_ticker_features(prices)
    out = attach_market_context(bucket1, prices, spy)
    # Carry close for label construction + backtest pricing. Not a model feature.
    out["close"] = prices["Close"]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Bucket 3 — cross-sectional ranks (panel-level)
# ─────────────────────────────────────────────────────────────────────────────


def add_cross_sectional_ranks(
    panel: pd.DataFrame,
    cols: list[str] = RANKABLE,
) -> pd.DataFrame:
    """For each date, compute the percentile rank (0–1) of `cols` across tickers."""
    panel = panel.copy()
    grouped = panel.groupby("date", sort=False)
    for col in cols:
        panel[f"{col}_rank"] = grouped[col].rank(pct=True)
    return panel


# ─────────────────────────────────────────────────────────────────────────────
# Bucket 6 — earnings calendar (per-ticker, attached during build_panel)
# ─────────────────────────────────────────────────────────────────────────────


_PEAD_LO, _PEAD_HI = 1, 5
_EARNINGS_CLIP = 90  # cap days_to / days_since at 90 — beyond that the signal flatlines


def attach_earnings_features(
    feats: pd.DataFrame,
    earnings_dates: list[pd.Timestamp] | pd.DatetimeIndex,
) -> pd.DataFrame:
    """Add EARNINGS_FEATURES to a per-ticker frame indexed by Date.

    `earnings_dates` is the sorted list of all known earnings filing dates for
    the ticker (EDGAR historicals + optional yfinance forward calendar). For
    each row date D:
      days_to_earnings   = next_edate ≥ D minus D            (NaN if none known)
      days_since_earnings = D minus most-recent edate < D    (NaN if none known)
      post_earnings_drift_window = 1 if days_since ∈ [1, 5] else 0
    """
    out = feats.copy()
    edates = pd.DatetimeIndex(earnings_dates).sort_values().unique()
    if len(edates) == 0:
        out["days_to_earnings"] = np.nan
        out["days_since_earnings"] = np.nan
        out["post_earnings_drift_window"] = 0.0
        return out

    dates = pd.DatetimeIndex(out.index)
    edates = pd.DatetimeIndex(edates)

    # `pos_next` = first index in edates where edate >= D.
    pos_next = edates.searchsorted(dates.values, side="left")
    has_next = pos_next < len(edates)
    next_dates = edates[np.minimum(pos_next, len(edates) - 1)]
    days_to = (next_dates - dates).days.to_numpy(dtype="float64")
    days_to[~has_next] = np.nan

    # `pos_prev` = last index where edate < D, i.e. searchsorted(side='left') - 1.
    pos_prev = pos_next - 1
    has_prev = pos_prev >= 0
    prev_dates = edates[np.maximum(pos_prev, 0)]
    days_since = (dates - prev_dates).days.to_numpy(dtype="float64")
    days_since[~has_prev] = np.nan

    out["days_to_earnings"] = np.clip(days_to, 0, _EARNINGS_CLIP)
    out["days_since_earnings"] = np.clip(days_since, 0, _EARNINGS_CLIP)
    out["post_earnings_drift_window"] = (
        ((days_since >= _PEAD_LO) & (days_since <= _PEAD_HI)).astype("float64")
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Bucket 7 — fundamentals (panel-level, asof-merged from SEC XBRL)
# ─────────────────────────────────────────────────────────────────────────────


def attach_fundamentals(
    panel: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    """Asof-merge the fundamentals panel into the price panel by `(ticker,
    asof_date <= date)` and compute the 7 ratio features.

    Lookahead-safe: each row sees only fundamentals that were publicly
    `filed` on or before the row's date. `shares_adj` is on the same
    split-adjusted basis as `close`, so `market_cap = shares_adj × close`.

    Sign-flip / divide-by-zero handling:
      - Growth ratios set to NaN when the prior-period denominator ≤ 0.
        A turnaround (loss → profit) reads as a *positive* signal but the
        percent change would be encoded as a misleading negative; cleaner
        to drop those points than to feed XGBoost the wrong sign.
      - debt_to_equity NaN when equity ≤ 0 (technically-insolvent firms;
        the ratio is meaningless and the few cases would dominate splits).
      - earnings_yield can go negative cleanly (loss-making firms = low
        value); no special handling needed.

    Drops the intermediate TTM/MRQ/share columns from the output — only the
    7 ratio features are retained on the panel.
    """
    if fundamentals is None or fundamentals.empty:
        for c in FUNDAMENTAL_FEATURES:
            panel[c] = np.nan
        return panel

    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    fund = fundamentals.sort_values(["ticker", "asof_date"]).reset_index(drop=True)

    # merge_asof requires identical dtypes on the asof keys. The panel's date
    # comes from parquet round-trip (datetime64[ms]) while the fundamentals
    # asof_date is freshly built in pandas (datetime64[ns]); pin both to ns.
    panel = panel.copy()
    fund = fund.copy()
    panel["date"] = pd.to_datetime(panel["date"]).astype("datetime64[ns]")
    fund["asof_date"] = pd.to_datetime(fund["asof_date"]).astype("datetime64[ns]")

    # merge_asof with `by="ticker"` requires both frames sorted by the asof
    # key globally (not just within each ticker).
    panel_sorted = panel.sort_values("date").reset_index(drop=False).rename(
        columns={"index": "_orig_idx"}
    )
    fund_sorted = fund.sort_values("asof_date").reset_index(drop=True)

    merged = pd.merge_asof(
        panel_sorted,
        fund_sorted,
        left_on="date",
        right_on="asof_date",
        by="ticker",
        direction="backward",
    ).sort_values("_orig_idx").reset_index(drop=True)
    merged = merged.drop(columns=["_orig_idx"])

    close = merged["close"]
    shares_adj = merged["shares_adj"]
    market_cap = shares_adj * close

    ttm_ni = merged["ttm_net_income"]
    ttm_rev = merged["ttm_revenue"]
    ttm_rev_prior = merged["ttm_revenue_prior"]
    ttm_oi = merged["ttm_operating_income"]
    ttm_oi_prior = merged["ttm_operating_income_prior"]
    mrq_assets = merged["mrq_assets"]
    mrq_assets_curr = merged["mrq_assets_current"]
    mrq_liab_curr = merged["mrq_liabilities_current"]
    mrq_equity = merged["mrq_equity"]
    mrq_lt_debt = merged["mrq_lt_debt"]

    merged["earnings_yield"] = ttm_ni / market_cap.where(market_cap > 0)
    merged["book_to_market"] = mrq_equity / market_cap.where(market_cap > 0)
    merged["roa"] = ttm_ni / mrq_assets.where(mrq_assets > 0)
    merged["debt_to_equity"] = mrq_lt_debt / mrq_equity.where(mrq_equity > 0)
    merged["current_ratio"] = mrq_assets_curr / mrq_liab_curr.where(mrq_liab_curr > 0)
    merged["sales_growth_yoy"] = ttm_rev / ttm_rev_prior.where(ttm_rev_prior > 0) - 1
    merged["op_income_growth_yoy"] = ttm_oi / ttm_oi_prior.where(ttm_oi_prior > 0) - 1

    drop_cols = [
        "asof_date", "period_end",
        "ttm_revenue", "ttm_revenue_prior",
        "ttm_net_income",
        "ttm_operating_income", "ttm_operating_income_prior",
        "mrq_assets", "mrq_assets_current", "mrq_liabilities_current",
        "mrq_equity", "mrq_lt_debt",
        "shares", "shares_adj",
    ]
    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Bucket 8 — insider transactions (per-ticker, asof filing_date)
# ─────────────────────────────────────────────────────────────────────────────

_INSIDER_DAYS = 60       # rolling window (calendar days)
_INSIDER_DSB_CAP = 365   # cap days_since_last_insider_buy beyond this


def attach_insider_features(
    feats: pd.DataFrame,
    insider_txns: pd.DataFrame | None,
) -> pd.DataFrame:
    """Add INSIDER_FEATURES to a per-ticker frame indexed by Date.

    `insider_txns` is the slice of load_insider_transactions() for this ticker:
    sorted by filing_date, columns [filing_date, transaction_code, value].

    For each row date D (using filing_date as the asof key):
      insider_buy_count_60d       = # direct open-market P transactions in (D-60d, D]
      insider_sell_count_60d      = # direct open-market S transactions in (D-60d, D]
      insider_net_dollar_60d      = sum(P values) − sum(S values) in (D-60d, D]
      days_since_last_insider_buy = D − most-recent P filing_date ≤ D  (NaN if none)
    """
    out = feats.copy()

    _NAN_DEFAULTS = {
        "insider_buy_count_60d": 0.0,
        "insider_sell_count_60d": 0.0,
        "insider_net_dollar_60d": 0.0,
        "days_since_last_insider_buy": np.nan,
    }

    if insider_txns is None or insider_txns.empty:
        for col, val in _NAN_DEFAULTS.items():
            out[col] = val
        return out

    insider_txns = insider_txns.sort_values("filing_date").reset_index(drop=True)

    # Separate buy and sell arrays for fast searchsorted.
    buy_mask = insider_txns["transaction_code"] == "P"
    sell_mask = insider_txns["transaction_code"] == "S"

    buy_dates = insider_txns.loc[buy_mask, "filing_date"].values.astype("datetime64[ns]")
    buy_values = insider_txns.loc[buy_mask, "value"].fillna(0.0).values.astype(float)
    sell_dates = insider_txns.loc[sell_mask, "filing_date"].values.astype("datetime64[ns]")
    sell_values = insider_txns.loc[sell_mask, "value"].fillna(0.0).values.astype(float)

    n = len(out)
    buy_counts = np.zeros(n, dtype=float)
    sell_counts = np.zeros(n, dtype=float)
    net_dollars = np.zeros(n, dtype=float)
    days_since_buy = np.full(n, np.nan)

    window_td = np.timedelta64(_INSIDER_DAYS, "D")

    for i, d in enumerate(out.index):
        d_ns = np.datetime64(d, "ns")
        window_lo = d_ns - window_td

        # Buys in (window_lo, d_ns]
        lo_b = int(np.searchsorted(buy_dates, window_lo, side="right"))
        hi_b = int(np.searchsorted(buy_dates, d_ns, side="right"))
        buy_counts[i] = hi_b - lo_b
        if hi_b > lo_b:
            net_dollars[i] += buy_values[lo_b:hi_b].sum()

        # Sells in (window_lo, d_ns]
        lo_s = int(np.searchsorted(sell_dates, window_lo, side="right"))
        hi_s = int(np.searchsorted(sell_dates, d_ns, side="right"))
        sell_counts[i] = hi_s - lo_s
        if hi_s > lo_s:
            net_dollars[i] -= sell_values[lo_s:hi_s].sum()

        # Days since most recent buy (all history up to d)
        if hi_b > 0:
            last_buy = buy_dates[hi_b - 1]
            days = float((d_ns - last_buy).astype("timedelta64[D]").astype(int))
            days_since_buy[i] = min(days, _INSIDER_DSB_CAP)

    out["insider_buy_count_60d"] = buy_counts
    out["insider_sell_count_60d"] = sell_counts
    out["insider_net_dollar_60d"] = net_dollars
    out["days_since_last_insider_buy"] = days_since_buy
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Bucket 2b — broadcast market-regime features (panel-level, same value
# per date for every ticker). Earn reward only via interactions with
# cross-sectional features under date-demeaned labels.
# ─────────────────────────────────────────────────────────────────────────────


def compute_market_regime_features(
    spy: pd.DataFrame, vix: pd.DataFrame
) -> pd.DataFrame:
    """Date-indexed DataFrame with 5 broadcast features. Computed once;
    merged onto every ticker's row by date in attach_market_regime.
    """
    spy_close = spy["Close"]
    vix_close = vix["Close"]

    out = pd.DataFrame(index=spy_close.index)
    out["spy_ret_21d"] = spy_close.pct_change(21)
    out["spy_trend_regime"] = (
        spy_close.rolling(50).mean() > spy_close.rolling(200).mean()
    ).astype(float)

    # SPY RSI(14) with Wilder's smoothing (same form as per-ticker rsi_14).
    delta = spy_close.diff()
    gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    out["spy_rsi_14"] = 100 - 100 / (1 + gain / loss)

    out["vix_level"] = vix_close.reindex(spy_close.index)
    vix_aligned = out["vix_level"]
    vix_mean = vix_aligned.rolling(20).mean()
    vix_std = vix_aligned.rolling(20).std()
    out["vix_zscore_20d"] = (vix_aligned - vix_mean) / vix_std
    return out


def attach_market_regime(
    panel: pd.DataFrame, regime: pd.DataFrame
) -> pd.DataFrame:
    """Merge the date-indexed regime frame onto the panel by date.
    Same value broadcast to every ticker on a given date — that's the point.
    """
    panel = panel.copy()
    regime = regime.copy()
    regime.index.name = "date"
    return panel.merge(
        regime[MARKET_REGIME_FEATURES].reset_index(),
        on="date",
        how="left",
    )


def add_sector_relative_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Bucket 3 — `excess_ret_n_vs_sector = ret_n − mean(ret_n)` within
    (date, gics_sector). Equal-weighted; the standard cross-sectional
    sector-momentum form.

    NaN for the UNKNOWN_SECTOR bucket — its members are structurally
    heterogeneous (delisted/removed names with no real sector tag) and the
    group mean would be noise.
    """
    panel = panel.copy()
    grouped = panel.groupby(["date", "gics_sector"], observed=True, sort=False)
    for n in (5, 21):
        col = f"ret_{n}d"
        panel[f"excess_{col}_vs_sector"] = panel[col] - grouped[col].transform("mean")

    is_unknown = panel["gics_sector"] == UNKNOWN_SECTOR
    if is_unknown.any():
        panel.loc[is_unknown, SECTOR_FEATURES] = np.nan
    return panel


# ─────────────────────────────────────────────────────────────────────────────
# Panel assembly — stack tickers, add sector, add ranks
# ─────────────────────────────────────────────────────────────────────────────


def build_panel(
    prices: dict[str, pd.DataFrame] | None = None,
    spy: pd.DataFrame | None = None,
    vix: pd.DataFrame | None = None,
    sectors: pd.DataFrame | None = None,
    history: pd.DataFrame | None = None,
    earnings: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
    insider: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the full long-format feature panel, filtered to point-in-time
    S&P 500 membership.

    Columns (in order):
        date, ticker, gics_sector,
        <PER_TICKER_FEATURES>, <MARKET_FEATURES>, <MARKET_REGIME_FEATURES>,
        <SECTOR_FEATURES>, <EARNINGS_FEATURES>, <FUNDAMENTAL_FEATURES>,
        <RANK_FEATURES>, <INSIDER_FEATURES>, close
    """
    if prices is None:
        prices = load_prices()
    market = None
    if spy is None or vix is None:
        market = load_market()
    if spy is None:
        spy = market["SPY"]
    if vix is None:
        vix = market["VIX"]
    if sectors is None:
        sectors = load_sectors()
    if history is None:
        history = load_history()
    if earnings is None:
        earnings = load_earnings_dates(include_upcoming=True)
    if fundamentals is None:
        fundamentals = load_fundamentals()
    if insider is None:
        insider = load_insider_transactions()

    earnings_by_ticker: dict[str, pd.DatetimeIndex] = {}
    if earnings is not None and not earnings.empty:
        for tkr, grp in earnings.groupby("ticker"):
            earnings_by_ticker[tkr] = pd.DatetimeIndex(grp["report_date"]).sort_values()

    insider_by_ticker: dict[str, pd.DataFrame] = {}
    if insider is not None and not insider.empty:
        for tkr, grp in insider.groupby("ticker"):
            insider_by_ticker[tkr] = grp.reset_index(drop=True)

    print(
        f"Computing features for {len(prices)} tickers "
        f"(earnings: {len(earnings_by_ticker)}, insider: {len(insider_by_ticker)})...",
        flush=True,
    )
    frames: list[pd.DataFrame] = []
    for ticker, df in tqdm(prices.items(), desc="Features"):
        feats = compute_features(df, spy)
        feats = attach_earnings_features(
            feats, earnings_by_ticker.get(ticker, pd.DatetimeIndex([]))
        )
        feats = attach_insider_features(
            feats, insider_by_ticker.get(ticker)
        )
        feats["ticker"] = ticker
        frames.append(feats)

    panel = pd.concat(frames, axis=0)
    panel = panel.reset_index().rename(columns={"Date": "date"})

    # Drop (date, ticker) rows where the ticker wasn't an S&P 500 member on
    # date — this is what de-biases the backtest.
    before = len(panel)
    panel = filter_to_members(panel, history=history)
    print(
        f"Point-in-time filter: {before:,} → {len(panel):,} rows "
        f"(dropped {before - len(panel):,} non-member observations).",
        flush=True,
    )

    # Sector (Bucket 5, categorical). Wikipedia only knows current members;
    # tickers that have since left the index get UNKNOWN_SECTOR rather than NaN
    # so XGBoost's native categorical handling treats them as a real bucket.
    sector_map = sectors.set_index("Ticker")["GICS Sector"]
    panel["gics_sector"] = (
        panel["ticker"].map(sector_map).fillna(UNKNOWN_SECTOR).astype("category")
    )

    print("Computing sector-relative excess returns...", flush=True)
    panel = add_sector_relative_returns(panel)

    print("Attaching broadcast market-regime features (SPY trend / VIX)...", flush=True)
    regime = compute_market_regime_features(spy, vix)
    panel = attach_market_regime(panel, regime)

    print(
        f"Attaching XBRL fundamentals "
        f"({fundamentals['ticker'].nunique() if fundamentals is not None and not fundamentals.empty else 0} tickers)...",
        flush=True,
    )
    panel = attach_fundamentals(panel, fundamentals)

    print("Computing cross-sectional ranks...", flush=True)
    panel = add_cross_sectional_ranks(panel)

    # `ret_1d` is dropped from PER_TICKER_FEATURES (dead in all 5 seeds) but
    # backtest.py still needs it for day-to-day portfolio P&L — keep it in the
    # panel but not in the model's feature set.
    cols_order = (
        ["date", "ticker", "gics_sector"]
        + PER_TICKER_FEATURES + MARKET_FEATURES + MARKET_REGIME_FEATURES
        + SECTOR_FEATURES
        + EARNINGS_FEATURES + FUNDAMENTAL_FEATURES + RANK_FEATURES
        + INSIDER_FEATURES
        + ["ret_1d", "close"]
    )
    panel = panel[cols_order].sort_values(["ticker", "date"]).reset_index(drop=True)
    return panel


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ticker", help="Compute features for a single ticker (smoke test)")
    ap.add_argument("--out", default=PANEL_PATH, help="Output parquet for full panel")
    args = ap.parse_args()

    if args.ticker:
        prices = load_prices([args.ticker])
        if not prices:
            raise SystemExit(f"{args.ticker}: no cached data (or <500 rows). Run scripts/data.py first.")
        spy = load_market()["SPY"]
        feats = compute_features(prices[args.ticker], spy)
        print(f"\n{args.ticker} — last 5 rows ({feats.shape[0]} total, {feats.shape[1]} cols):")
        with pd.option_context("display.width", 220, "display.max_columns", 50):
            print(feats.tail(5))
        print(f"\nNaN counts (top 10):")
        print(feats.isna().sum().sort_values(ascending=False).head(10).to_string())
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    panel = build_panel()
    panel.to_parquet(args.out, index=False)
    print(
        f"\nWrote {args.out}\n"
        f"  rows: {len(panel):,}\n"
        f"  tickers: {panel['ticker'].nunique()}\n"
        f"  date range: {panel['date'].min().date()} → {panel['date'].max().date()}\n"
        f"  feature cols: {len(ALL_FEATURES)}"
    )


if __name__ == "__main__":
    main()
