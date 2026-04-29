#!/usr/bin/env python3
"""Assemble the long-format panel and apply chronological splits.

Output: data/processed/panel.parquet with columns
    date, ticker, <features...>, forward_21d_return

Splits (chronological, no shuffling):
    train: 2007-01-01 → 2017-12-31
    val:   2018-01-01 → 2020-12-31
    test:  2021-01-01 → today

Includes a lookahead sanity check: pick random (ticker, date) rows, recompute
each feature using only data <= date, assert equality to the panel value.

NOT YET IMPLEMENTED — stub.
"""

import pandas as pd

TRAIN_END = "2017-12-31"
VAL_START = "2018-01-01"
VAL_END = "2020-12-31"
TEST_START = "2021-01-01"


def build_panel() -> pd.DataFrame:
    raise NotImplementedError


def split(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raise NotImplementedError


def assert_no_lookahead(panel: pd.DataFrame, n_samples: int = 50) -> None:
    """Recompute features for sampled rows and assert match. Raises on leakage."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit("dataset.py not yet implemented")
