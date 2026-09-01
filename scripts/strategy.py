#!/usr/bin/env python3
"""Shared strategy primitives used by backtest.py and today.py.

Centralises the pieces both scripts use:
    - Model loading + prediction
    - Market-context frame (SPY close + SMA200, VIX close, SPY 1d return)
    - Regime gate (SPY > SMA200 AND VIX < threshold)
    - Top-N pick selection on a single date
    - Filtering panel rows to those with non-NaN features (live use)

Defaults exported as module constants — both callers expose CLI flags to
override them; this module just owns the canonical values.
"""

import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Import ALL_FEATURES directly from features.py rather than via dataset.py to
# avoid a circular import — dataset.py imports DEFAULT_SEED from this module.
from features import (  # noqa: E402
    ALL_FEATURES as FEATURE_COLS,
    CATEGORICAL_FEATURES,
    NULLABLE_FEATURES,
)

# Default knobs (override at the call site if needed).
TOP_N = 40
HOLD_DAYS = 21
COST_PER_SIDE = 0.0005   # 5 bps

# Legacy VIX/SMA200 binary regime gate. Retired 2026-05-11 in favor of
# vol-target sizing (see DEFAULT_VOL_TARGET below). The functions
# regime_long / regime_long_row are kept for reference and A/B testing —
# to re-enable, uncomment the spy_sma200 line in prepare_market.
VIX_THRESHOLD = 25.0

# Vol-target sizing: scale total gross exposure inversely to SPY's 20d
# realized vol. Reads exposure ∈ [0, 1] (no leverage). Replaces the
# binary VIX/SMA200 gate as the default "gated" overlay because it
# starts pulling exposure down as vol rises rather than waiting for an
# arbitrary VIX threshold cross.
DEFAULT_VOL_TARGET = 0.20
VOL_LOOKBACK = 20

# Annual margin interest rate charged on borrowed funds (gross exposure > 1.0).
# IBKR Lite USD = 6.13% APR (BM + 2.5%, flat across all tiers), confirmed against
# the published rate table 2026-09-01. The live account U26645119 is Lite: fills
# route ZERO/DARK at $0 commission, which Pro's $1.00-minimum never does. Pro
# Tier I would be 5.13% (BM + 1.5%) — 1pp cheaper, but that is not the account
# we trade. Used by the backtest --leverage path and execute_picks.py to cost
# leverage honestly: leverage is not free, and on a small account the borrow
# drag is material.
DEFAULT_MARGIN_RATE = 0.0613

# Live-account gross-exposure cap — the leverage leg of the 2026-07-08
# rebalance-cadence decision (quarterly @ 1.35x, vol-target 0.20).
# today.py's daily de-risk check uses it as the formula ceiling and as
# the worst-case book assumption when reports/live_book.json is absent.
LIVE_LEVERAGE = 1.35

# Live basket sector cap — max share of the basket from any one GICS sector.
# Chosen 2026-09-01 from a sweep at leverage 1.35 / vol-target 0.20 over the
# 2021-2026 test set: 0.40 weakly dominates no cap (CAGR +27.14% vs +26.94%,
# Sharpe 0.96 vs 0.95, MaxDD -28.94% vs -29.19%), while tighter caps cost real
# return (0.20 → +23.59% / 0.89). The differences at 0.40 are inside the
# rebalance-offset noise band, so the case for it is "a free 40% ceiling on any
# one sector", not measured alpha. Wired into run_all.py and the
# ibkr-web-trade skill; every script still defaults to no cap.
LIVE_SECTOR_CAP = 0.40

# De-risk alarm threshold: fire when today's formula exposure drops more
# than this fraction below the book's exposure (relative, not points).
# 10% ignores day-to-day vol wiggle but fires on a real regime shift.
DERISK_TOLERANCE = 0.10

WEIGHT_MODES = ("equal", "pred")
DEFAULT_WEIGHT_MODE = "equal"

# Default random seed used everywhere we have an RNG knob (XGBoost
# `random_state`, optuna TPE, dataset.assert_no_lookahead sampling).
# Stability-selection sweeps override this via --seed; the per-seed
# feature-importance CSV is only written when seed != DEFAULT_SEED so
# the baseline file `feature_importance.csv` stays untouched.
DEFAULT_SEED = 15


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────


def load_model(path: str) -> xgb.XGBRegressor:
    """Load the saved xgb_v1 booster with categorical support enabled."""
    booster = xgb.XGBRegressor(enable_categorical=True)
    booster.load_model(path)
    return booster


def predict(df: pd.DataFrame, booster: xgb.XGBRegressor) -> pd.DataFrame:
    """Score rows in `df` using the booster. Returns a copy with a new
    ``predicted_return`` column.
    """
    out = df.copy()
    out["predicted_return"] = booster.predict(out[FEATURE_COLS])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Market context + regime gate
# ─────────────────────────────────────────────────────────────────────────────


def prepare_market(spy_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """Date-indexed frame with spy_close, vix_close, spy_ret_1d, spy_vol_20d.

    spy_vol_20d (annualized) feeds the vol-target sizing overlay.
    spy_sma200 is preserved as a commented line — it was the second leg of
    the legacy binary regime gate (`SPY > SMA200 AND VIX < 25`). Uncomment
    + restore the `regime_long_row` call site to re-enable that gate.
    """
    # Build on the SPY (equity) trading calendar and align VIX onto it. VIX/CBOE
    # carries rows on equity holidays (e.g. Memorial Day, Juneteenth) that SPY
    # lacks; a naive union would NaN out spy_close on those dates, which then
    # poisons pct_change and the rolling std — freezing spy_vol_20d for ~20 rows
    # after every holiday. Reindexing VIX onto SPY's index keeps spy_close dense.
    market = pd.DataFrame({"spy_close": spy_df["Close"]})
    market.index = pd.to_datetime(market.index)
    market["vix_close"] = vix_df["Close"].reindex(market.index)
    # Legacy VIX/SMA200 regime gate (retired 2026-05-11). Uncomment to revive.
    # market["spy_sma200"] = market["spy_close"].rolling(200).mean()
    market["spy_ret_1d"] = market["spy_close"].pct_change()
    market["spy_vol_20d"] = (
        market["spy_ret_1d"].rolling(VOL_LOOKBACK).std() * np.sqrt(252)
    )
    return market


def regime_long(
    spy_close: float,
    spy_sma200: float,
    vix_close: float,
    vix_threshold: float = VIX_THRESHOLD,
) -> bool:
    """Regime gate: SPY trending up AND VIX not stressed.

    Returns False when warm-up data is missing (NaN SMA200 etc).
    """
    if pd.isna(spy_sma200) or pd.isna(vix_close):
        return False
    return bool(spy_close > spy_sma200 and vix_close < vix_threshold)


def regime_long_row(
    market_row: pd.Series, vix_threshold: float = VIX_THRESHOLD
) -> bool:
    """Convenience wrapper for a row from `prepare_market`.

    NOTE: requires `spy_sma200`, which `prepare_market` no longer computes
    by default (retired in favor of vol-target sizing). Uncomment the
    SMA200 line in `prepare_market` to re-enable.
    """
    return regime_long(
        market_row["spy_close"],
        market_row["spy_sma200"],
        market_row["vix_close"],
        vix_threshold,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Vol-target sizing overlay (default "gated" behavior)
# ─────────────────────────────────────────────────────────────────────────────


def vol_target_exposure(
    spy_vol: float,
    target: float = DEFAULT_VOL_TARGET,
    max_exposure: float = 1.0,
) -> float:
    """Map SPY realized vol → portfolio exposure ∈ [0, max_exposure].

    The overlay scales *down* when volatility rises above the target. The
    `max_exposure` cap (default 1.0 = no leverage) is the ceiling in calm
    regimes: pass >1.0 to lever up when vol is low while still letting the
    vol target pull exposure back below 1.0 in stress. NaN spy_vol (during
    the 20-day warmup) returns `max_exposure` so backtests start fully sized.

    With max_exposure=1.0 (default):
        Calm market (SPY vol ≈ 15%): exposure = 1.0 (full)
        Stressed (SPY vol ≈ 30%):     exposure ≈ 0.67
        Crisis (SPY vol ≈ 45%):       exposure ≈ 0.44
    With max_exposure=1.5 (leverage):
        Calm market (SPY vol ≈ 13%): exposure = 1.5 (levered)
        Stressed (SPY vol ≈ 30%):     exposure ≈ 0.67 (target still binds)
    """
    if pd.isna(spy_vol) or spy_vol <= 0:
        return max_exposure
    return min(max_exposure, float(target) / float(spy_vol))


# ─────────────────────────────────────────────────────────────────────────────
# Picks
# ─────────────────────────────────────────────────────────────────────────────


def top_picks(
    day_panel: pd.DataFrame,
    top_n: int = TOP_N,
    sector_cap: float | None = None,
) -> pd.DataFrame:
    """Top-N rows by predicted_return on a single date's slice.

    `sector_cap` bounds any one GICS sector's share of the basket (0.20 = at
    most 20% of the names). None (default) = unconstrained, the historical
    behaviour: the model is free to put 45% of the basket in one sector, which
    is what it did into the June 2026 semiconductor drawdown.

    Greedy fill down the predicted-return ranking, skipping names whose sector
    is already full. If the cap makes top_n unreachable on a thin day the
    basket comes back short rather than breaching the cap; compute_weights
    renormalises over whatever survives.
    """
    if sector_cap is None or "gics_sector" not in day_panel.columns:
        return day_panel.nlargest(top_n, "predicted_return")

    max_per_sector = max(1, int(sector_cap * top_n))
    ranked = day_panel.sort_values("predicted_return", ascending=False)
    counts: dict[str, int] = {}
    keep: list = []
    for idx, sec in zip(ranked.index, ranked["gics_sector"].fillna("__unknown__")):
        if counts.get(sec, 0) >= max_per_sector:
            continue
        counts[sec] = counts.get(sec, 0) + 1
        keep.append(idx)
        if len(keep) == top_n:
            break
    return ranked.loc[keep]


def compute_weights(
    top: pd.DataFrame, mode: str = DEFAULT_WEIGHT_MODE
) -> dict[str, float]:
    """Map ticker → portfolio weight (sums to 1.0) for the given top picks.

    Modes:
        "equal" — 1/N across the basket.
        "pred"  — proportional to predicted_return, with negatives clipped at
                  zero. Falls back to equal-weight if every prediction in the
                  basket is ≤0 (degenerate case where even the "top" picks are
                  all bearish — e.g. an offset that lands on a stressed day).
    """
    tickers = top["ticker"].tolist()
    n = len(tickers)
    if n == 0:
        return {}
    if mode == "equal":
        w = 1.0 / n
        return {t: w for t in tickers}
    if mode == "pred":
        preds = top["predicted_return"].clip(lower=0.0)
        total = float(preds.sum())
        if total <= 0:
            w = 1.0 / n
            return {t: w for t in tickers}
        return {t: float(p) / total for t, p in zip(tickers, preds)}
    raise ValueError(f"unknown weight mode: {mode!r} (expected one of {WEIGHT_MODES})")


# ─────────────────────────────────────────────────────────────────────────────
# Quality filter — drop "trash" names before top-N selection
# ─────────────────────────────────────────────────────────────────────────────

# Hard-coded fundamentals/insider thresholds. A name failing ANY check is
# dropped from the candidate pool. NaN values pass (the company has no
# fundamentals/insider coverage — XGBoost already handles missing natively).
#
# These are intentionally LOOSE — they catch only the "absolutely
# cataclysmic" names (firm-going-to-zero zone), not just "weak quarter" or
# "expensive". A 2026-05-17 sweep on the 2021→2026 raw long-only backtest:
#
#   variant                 hits%   CAGR     Sharpe   MaxDD
#   no_filter (baseline)     0.0%   +23.02%   0.87    -28.14%
#   current (these defaults) 10.8%  +24.29%   0.95    -26.48%   ← winner
#   loose (D/E>7 etc.)       16.9%  +22.82%   0.91    -25.57%
#   tight (D/E>3 etc.)       33.9%  +19.53%   0.82    -24.52%
#   very_tight (D/E>2)       52.7%  +17.58%   0.79    -23.08%
#
# Lesson: tighter filters DO improve drawdown linearly but kill CAGR faster.
# Only the cataclysmic-only filter is Pareto-better than no filter at all —
# the model has already priced "merely weak" names via interactions, and
# stripping them removes mean-reversion winners.
#
# ROA intentionally absent: development-stage names (biotech, early SaaS,
# capex-heavy growth) show deeply negative ROA while ripping. Model already
# sees ROA + roa_rank.
QUALITY_FILTER_DEFAULTS: dict[str, float] = {
    "max_debt_to_equity": 10.0,                       # truly extreme leverage
    "min_current_ratio": 0.3,                         # near-insolvent liquidity
    "min_sales_growth_yoy": -0.50,                    # revenue more than halved
    "max_insider_net_sell_60d": -50_000_000.0,        # insiders dumping >$50M net
}


def apply_quality_filter(
    day_panel: pd.DataFrame,
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Drop rows whose fundamentals/insider columns flag the name as 'trash'.

    Applied *before* `top_picks` so the basket still has TOP_N survivors. NaN
    values always pass — missing fundamentals are not evidence of low quality
    (could be a young filer, a name with patchy EDGAR coverage, or pre-XBRL).

    `thresholds` is a partial dict — only keys present are enforced, so callers
    can mix-and-match (e.g. only the insider check). Missing columns in the
    panel are also silently skipped, so older panels still load.
    """
    if thresholds is None:
        thresholds = QUALITY_FILTER_DEFAULTS

    df = day_panel
    keep = pd.Series(True, index=df.index)

    checks = (
        ("max_debt_to_equity", "debt_to_equity", lambda s, t: s > t),
        ("min_current_ratio", "current_ratio", lambda s, t: s < t),
        ("min_sales_growth_yoy", "sales_growth_yoy", lambda s, t: s < t),
        ("min_roa", "roa", lambda s, t: s < t),
        ("max_insider_net_sell_60d", "insider_net_dollar_60d", lambda s, t: s < t),
    )
    for key, col, mask_fn in checks:
        if key not in thresholds or col not in df.columns:
            continue
        bad = mask_fn(df[col], thresholds[key]).fillna(False)
        keep &= ~bad

    return df[keep]


def filter_valid_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Keep rows where all *required* numeric features are non-NaN.

    Used by today.py: the most recent ~21 days have features but no label
    (forward_21d_return needs prices 21 days ahead). load_panel(drop_na=True)
    drops them; this keeps them as long as the features themselves are valid.

    NULLABLE_FEATURES (earnings calendar + fundamentals) are excluded from
    the check — they're legitimately NaN for tickers without EDGAR/XBRL
    coverage and XGBoost handles missing natively. Without this exclusion,
    every row with even one missing fundamental gets dropped → ~zero rows
    survive on the live slice.
    """
    required = [
        c for c in FEATURE_COLS
        if c not in CATEGORICAL_FEATURES and c not in NULLABLE_FEATURES
    ]
    return panel.dropna(subset=required)
