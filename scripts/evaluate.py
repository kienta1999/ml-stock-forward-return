#!/usr/bin/env python3
"""Evaluate the ranker.

Metrics:
  * Information Coefficient: daily Spearman(pred, realised forward return),
    plus mean IC and IC t-stat.
  * Top-decile vs bottom-decile mean forward-return spread.
  * Portfolio annualised return, vol, Sharpe, max drawdown.
  * Equity curve plot vs SPY (reports/equity_curve.png).
  * Feature importance plot (reports/feature_importance.png).

NOT YET IMPLEMENTED — stub.
"""

import pandas as pd


def information_coefficient(panel: pd.DataFrame) -> pd.Series:
    raise NotImplementedError


def decile_spread(panel: pd.DataFrame) -> float:
    raise NotImplementedError


def portfolio_stats(equity_curve: pd.Series) -> dict:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit("evaluate.py not yet implemented")
