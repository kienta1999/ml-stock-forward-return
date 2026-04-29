#!/usr/bin/env python3
"""Compute features for the ranker.

Three feature groups:
  1. Per-ticker technical indicators (returns 1/5/21/63d, RSI(14), MACD hist,
     ATR%, 20/60d realized vol, vol ratio vs 20d avg, dist from 50/200 MA,
     trend regime flag).
  2. Market context (SPY 50/200 regime, SPY 21d return, VIX level, VIX 20d
     z-score, 60d rolling beta to SPY, ticker excess return vs SPY 1/5/21d).
  3. Cross-sectional ranks of every numeric feature on each date.

CRITICAL: every feature on row date=D must use only data observable at the
close of D — no lookahead. dataset.py runs a sanity assertion on this.

NOT YET IMPLEMENTED — stub.
"""

import pandas as pd


def compute_per_ticker_features(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV in → DataFrame indexed by Date with feature columns."""
    raise NotImplementedError


def attach_market_context(
    feats: pd.DataFrame,
    spy: pd.DataFrame,
    vix: pd.DataFrame,
) -> pd.DataFrame:
    raise NotImplementedError


def cross_sectional_ranks(panel: pd.DataFrame) -> pd.DataFrame:
    """For each date, rank every numeric feature across tickers (0–1)."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit("features.py not yet implemented")
