#!/usr/bin/env python3
"""Backtest the trained ranker on the test set.

Each trading day:
  1. Predict forward_21d_return for every active ticker.
  2. Rank by prediction; long top decile (equal-weight).
  3. Hold 21 trading days.

Implementation: 21 overlapping equal-weight sleeves running in parallel; one
sleeve rotates each day. Portfolio = mean of sleeves → smooth daily curve.
Cost: 5 bps per side on rebalance turnover.

NOT YET IMPLEMENTED — stub.
"""

import pandas as pd

TOP_DECILE = 0.10
HOLD_DAYS = 21
COST_BPS = 5  # per side


def run_backtest(predictions: pd.DataFrame) -> pd.DataFrame:
    """predictions: long panel (date, ticker, pred). Returns daily equity curve."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit("backtest.py not yet implemented")
