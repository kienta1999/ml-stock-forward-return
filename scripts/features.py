#!/usr/bin/env python3
"""Compute features for the ranker.

Four feature buckets:
  1. Per-ticker technicals  (returns, oscillators, trend, vol, volume, z-scores)
  2. Market context         (SPY trend / RSI, VIX level + zscore, beta, excess returns)
  3. Cross-sectional ranks  (per-date percentile of every numeric feature in {1})
  4. Categorical            (gics_sector — XGBoost native categorical)

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

from data import load_market, load_prices  # noqa: E402
from universe import (  # noqa: E402
    UNKNOWN_SECTOR,
    filter_to_members,
    load_history,
    load_sectors,
)

_ROOT = os.path.dirname(_HERE)
PANEL_PATH = os.path.join(_ROOT, "data", "processed", "features.parquet")

# Feature columns by bucket — kept here so labels.py / dataset.py / train.py can import them.
PER_TICKER_FEATURES: list[str] = [
    "ret_1d", "ret_5d", "ret_21d", "ret_63d",
    "rsi_14", "mfi_14", "macd_hist",
    "atr_pct", "vol_20d", "vol_60d", "vol_ratio",
    "dist_sma50", "dist_sma200", "dist_52w_high", "trend_regime",
    "zscore_20d", "zscore_60d",
]
MARKET_FEATURES: list[str] = [
    "spy_ret_21d", "spy_trend_regime", "spy_rsi_14",
    "vix_level", "vix_zscore_20d",
    "beta_60d", "excess_ret_5d", "excess_ret_21d",
]
# Bucket 1 features that get cross-sectional ranks. Skip binary trend_regime.
RANKABLE: list[str] = [c for c in PER_TICKER_FEATURES if c != "trend_regime"]
RANK_FEATURES: list[str] = [f"{c}_rank" for c in RANKABLE]
CATEGORICAL_FEATURES: list[str] = ["gics_sector"]

ALL_FEATURES: list[str] = (
    PER_TICKER_FEATURES + MARKET_FEATURES + RANK_FEATURES + CATEGORICAL_FEATURES
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


def _spy_market_features(spy: pd.DataFrame) -> pd.DataFrame:
    """SPY-derived features (broadcast — same value for every ticker on a date)."""
    c = spy["Close"]
    sma50 = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss)
    return pd.DataFrame({
        "spy_ret_21d": c.pct_change(21),
        "spy_trend_regime": (sma50 > sma200).astype(float),
        "spy_rsi_14": rsi,
    })


def _vix_features(vix: pd.DataFrame) -> pd.DataFrame:
    c = vix["Close"]
    return pd.DataFrame({
        "vix_level": c,
        "vix_zscore_20d": (c - c.rolling(20).mean()) / c.rolling(20).std(),
    })


def attach_market_context(
    per_ticker: pd.DataFrame,
    prices: pd.DataFrame,
    spy: pd.DataFrame,
    vix: pd.DataFrame,
) -> pd.DataFrame:
    """Add Bucket 2 features. `prices` needed for ticker-specific beta + excess return."""
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

    # Broadcast SPY + VIX market-wide features (date-aligned join).
    market = _spy_market_features(spy).join(_vix_features(vix))
    out = out.join(market.reindex(out.index))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-ticker convenience: compute_features = Bucket 1 + Bucket 2
# ─────────────────────────────────────────────────────────────────────────────


def compute_features(
    prices: pd.DataFrame,
    spy: pd.DataFrame,
    vix: pd.DataFrame,
) -> pd.DataFrame:
    """One ticker's full feature frame (Buckets 1 + 2). Indexed by Date."""
    bucket1 = compute_per_ticker_features(prices)
    out = attach_market_context(bucket1, prices, spy, vix)
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
# Panel assembly — stack tickers, add sector, add ranks
# ─────────────────────────────────────────────────────────────────────────────


def build_panel(
    prices: dict[str, pd.DataFrame] | None = None,
    spy: pd.DataFrame | None = None,
    vix: pd.DataFrame | None = None,
    sectors: pd.DataFrame | None = None,
    history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the full long-format feature panel, filtered to point-in-time
    S&P 500 membership.

    Columns (in order):
        date, ticker, gics_sector,
        <PER_TICKER_FEATURES>, <MARKET_FEATURES>, <RANK_FEATURES>,
        close
    """
    if prices is None:
        prices = load_prices()
    if spy is None or vix is None:
        market = load_market()
        spy = market["SPY"] if spy is None else spy
        vix = market["VIX"] if vix is None else vix
    if sectors is None:
        sectors = load_sectors()
    if history is None:
        history = load_history()

    print(f"Computing features for {len(prices)} tickers...", flush=True)
    frames: list[pd.DataFrame] = []
    for ticker, df in tqdm(prices.items(), desc="Features"):
        feats = compute_features(df, spy, vix)
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

    # Sector (Bucket 4, categorical). Wikipedia only knows current members;
    # tickers that have since left the index get UNKNOWN_SECTOR rather than NaN
    # so XGBoost's native categorical handling treats them as a real bucket.
    sector_map = sectors.set_index("Ticker")["GICS Sector"]
    panel["gics_sector"] = (
        panel["ticker"].map(sector_map).fillna(UNKNOWN_SECTOR).astype("category")
    )

    print("Computing cross-sectional ranks...", flush=True)
    panel = add_cross_sectional_ranks(panel)

    cols_order = (
        ["date", "ticker", "gics_sector"]
        + PER_TICKER_FEATURES + MARKET_FEATURES + RANK_FEATURES
        + ["close"]
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
        market = load_market()
        feats = compute_features(prices[args.ticker], market["SPY"], market["VIX"])
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
