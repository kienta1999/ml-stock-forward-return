#!/usr/bin/env python3
"""Forward-return labels.

label = close[t+21] / close[t] - 1
Drop rows where label is NaN (the last 21 trading days of each ticker).

NOT YET IMPLEMENTED — stub.
"""

import pandas as pd

FORWARD_DAYS = 21


def forward_return(close: pd.Series, days: int = FORWARD_DAYS) -> pd.Series:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit("labels.py not yet implemented")
