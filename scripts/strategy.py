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
VIX_THRESHOLD = 25.0

WEIGHT_MODES = ("equal", "pred")
DEFAULT_WEIGHT_MODE = "equal"

# Default random seed used everywhere we have an RNG knob (XGBoost
# `random_state`, optuna TPE, dataset.assert_no_lookahead sampling).
# Stability-selection sweeps override this via --seed; the per-seed
# feature-importance CSV is only written when seed != DEFAULT_SEED so
# the baseline file `feature_importance.csv` stays untouched.
DEFAULT_SEED = 42


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
    """Date-indexed frame with spy_close, spy_sma200, vix_close, spy_ret_1d."""
    market = pd.DataFrame({
        "spy_close": spy_df["Close"],
        "vix_close": vix_df["Close"],
    })
    market.index = pd.to_datetime(market.index)
    market["spy_sma200"] = market["spy_close"].rolling(200).mean()
    market["spy_ret_1d"] = market["spy_close"].pct_change()
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
    """Convenience wrapper for a row from `prepare_market`."""
    return regime_long(
        market_row["spy_close"],
        market_row["spy_sma200"],
        market_row["vix_close"],
        vix_threshold,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Picks
# ─────────────────────────────────────────────────────────────────────────────


def top_picks(day_panel: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    """Top-N rows by predicted_return on a single date's slice."""
    return day_panel.nlargest(top_n, "predicted_return")


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
