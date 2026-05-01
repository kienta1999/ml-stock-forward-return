#!/usr/bin/env python3
"""Load the feature+label panel, apply chronological splits, sanity-check leakage.

Reads:  data/processed/panel.parquet  (output of scripts/labels.py)

Splits (chronological, no shuffling):
    train: 2007-01-01 → 2017-12-31
    val:   2018-01-01 → 2020-12-31
    test:  2021-01-01 → today

Lookahead sanity check: pick random (ticker, date) rows, truncate the raw
price/SPY/VIX history to <= date, recompute per-ticker + market features, and
assert each value matches the panel. Cross-sectional ranks are skipped (they
need the full date cross-section, but they're date-aligned so not a leak risk).

CLI:
    python scripts/dataset.py                  # splits + lookahead check
    python scripts/dataset.py --quick          # skip the (slow) lookahead check
    python scripts/dataset.py --n-samples 200  # more thorough lookahead check
"""

import argparse
import os
import sys
import warnings
from typing import NamedTuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data import load_market, load_prices  # noqa: E402
from features import (  # noqa: E402
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    MARKET_FEATURES,
    PER_TICKER_FEATURES,
    compute_features,
)
from labels import LABEL_COL  # noqa: E402

_ROOT = os.path.dirname(_HERE)
PANEL_PATH = os.path.join(_ROOT, "data", "processed", "panel.parquet")

# data.py downloads from 2005-07-01 so 252d rolling features warm up by 2007-01-01.
# Discard those buffer rows here — they exist only to seed the indicators.
TRAIN_START = "2007-01-01"
TRAIN_END = "2017-12-31"
VAL_START = "2018-01-01"
VAL_END = "2020-12-31"
TEST_START = "2021-01-01"

FEATURE_COLS: list[str] = ALL_FEATURES
TARGET_COL: str = LABEL_COL


class Splits(NamedTuple):
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


# ─────────────────────────────────────────────────────────────────────────────
# Load + split
# ─────────────────────────────────────────────────────────────────────────────


def load_panel(path: str = PANEL_PATH, drop_na: bool = True) -> pd.DataFrame:
    """Read the panel parquet and (by default) drop rows with any NaN
    feature/label — those are the early-window rows where rolling indicators
    haven't warmed up yet.
    """
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Run scripts/labels.py first.")
    panel = pd.read_parquet(path)
    panel["date"] = pd.to_datetime(panel["date"])

    if drop_na:
        before = len(panel)
        # Skip categorical (gics_sector) — it has its own "Unknown" bucket for
        # tickers Wikipedia doesn't tag; pandas dropna would treat the category
        # column inconsistently anyway.
        numeric_required = [c for c in FEATURE_COLS if c not in CATEGORICAL_FEATURES]
        panel = panel.dropna(subset=numeric_required + [TARGET_COL]).reset_index(drop=True)
        print(
            f"Loaded {before:,} rows; dropped {before - len(panel):,} early-window NaN "
            f"→ {len(panel):,} usable rows."
        )

    # Drop the 2005-07-01 → 2006-12-31 buffer rows that existed only to warm up rolling features.
    before = len(panel)
    panel = panel[panel["date"] >= TRAIN_START].reset_index(drop=True)
    if len(panel) < before:
        print(
            f"Trimmed {before - len(panel):,} pre-{TRAIN_START} buffer rows "
            f"→ {len(panel):,} model-ready rows."
        )

    return panel


def split(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological 3-way split. No shuffling."""
    train = panel[panel["date"] <= TRAIN_END]
    val = panel[(panel["date"] >= VAL_START) & (panel["date"] <= VAL_END)]
    test = panel[panel["date"] >= TEST_START]
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def prepare(panel: pd.DataFrame | None = None) -> Splits:
    """End-to-end: load → split → return (X, y) tuples for train/val/test.

    `gics_sector` stays as pandas Categorical for XGBoost native categorical.
    """
    if panel is None:
        panel = load_panel()
    tr, va, te = split(panel)

    def xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        return df[FEATURE_COLS].copy(), df[TARGET_COL].copy()

    Xtr, ytr = xy(tr)
    Xv, yv = xy(va)
    Xte, yte = xy(te)
    return Splits(Xtr, ytr, Xv, yv, Xte, yte)


# ─────────────────────────────────────────────────────────────────────────────
# Lookahead sanity check
# ─────────────────────────────────────────────────────────────────────────────

# Features we can verify by recomputing from truncated raw prices for a single
# ticker. Excludes RANK_FEATURES (need full panel cross-section) and the
# categorical sector (no math involved).
_VERIFIABLE_COLS: list[str] = PER_TICKER_FEATURES + MARKET_FEATURES


def assert_no_lookahead(
    panel: pd.DataFrame,
    n_samples: int = 50,
    seed: int = 42,
    tol: float = 1e-6,
) -> None:
    """Sample random (ticker, date) rows, truncate raw price/SPY/VIX history
    to <= date, recompute each Bucket-1/Bucket-2 feature, and assert it matches
    the panel value. Raises AssertionError on any mismatch.
    """
    market = load_market()
    spy, vix = market["SPY"], market["VIX"]

    # Skip very early dates where features wouldn't have warmed up at all.
    candidates = panel[panel["date"] >= "2010-01-01"]
    if len(candidates) == 0:
        print("No samples >= 2010 to check.")
        return
    sample = candidates.sample(
        n=min(n_samples, len(candidates)), random_state=seed
    ).reset_index(drop=True)

    cache: dict[str, pd.DataFrame] = {}
    mismatches: list[tuple] = []
    n_checked = 0

    for _, row in tqdm(
        sample.iterrows(), total=len(sample), desc="Lookahead check"
    ):
        t, d = row["ticker"], row["date"]

        if t not in cache:
            loaded = load_prices([t])
            if t not in loaded:
                continue
            cache[t] = loaded[t]
        prices = cache[t]

        prices_trunc = prices.loc[:d]
        if len(prices_trunc) == 0 or prices_trunc.index[-1] != pd.Timestamp(d):
            # Date not actually a trading day for this ticker — skip.
            continue

        feats = compute_features(prices_trunc, spy.loc[:d], vix.loc[:d])
        recomputed = feats.iloc[-1]

        for col in _VERIFIABLE_COLS:
            actual = row[col]
            new = recomputed[col]
            if pd.isna(actual) and pd.isna(new):
                continue
            if pd.isna(actual) or pd.isna(new):
                mismatches.append((t, d.date(), col, actual, new))
                continue
            if not np.isclose(actual, new, rtol=tol, atol=tol):
                mismatches.append((t, d.date(), col, actual, new))
        n_checked += 1

    if mismatches:
        print(f"\n{len(mismatches)} lookahead mismatches (showing up to 10):")
        for m in mismatches[:10]:
            print(f"  {m[0]} {m[1]} {m[2]}: panel={m[3]!r:<22} recomputed={m[4]!r}")
        raise AssertionError("Lookahead leak detected — see mismatches above.")

    print(
        f"\n[OK] No lookahead detected: {n_checked} (ticker, date) rows "
        f"× {len(_VERIFIABLE_COLS)} features verified."
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _print_splits(tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame) -> None:
    print("\nSplits:")
    for name, df in [("train", tr), ("val", va), ("test", te)]:
        if len(df) == 0:
            print(f"  {name}: 0 rows")
            continue
        print(
            f"  {name:<5}: {len(df):>10,} rows  "
            f"({df['date'].min().date()} → {df['date'].max().date()})  "
            f"{df['ticker'].nunique()} tickers"
        )

    print(f"\nLabel ({TARGET_COL}) per split:")
    for name, df in [("train", tr), ("val", va), ("test", te)]:
        if len(df) == 0:
            continue
        y = df[TARGET_COL]
        print(
            f"  {name:<5}: mean={y.mean():+.4f}  std={y.std():.4f}  "
            f"pos%={100 * (y > 0).mean():.1f}  median={y.median():+.4f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", default=PANEL_PATH)
    ap.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Lookahead check: number of random rows to recompute (default 50).",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Skip the lookahead sanity check (default: run it).",
    )
    args = ap.parse_args()

    panel = load_panel(args.panel)
    tr, va, te = split(panel)
    _print_splits(tr, va, te)

    if not args.quick:
        print(f"\nRunning lookahead sanity check ({args.n_samples} samples)...")
        assert_no_lookahead(panel, n_samples=args.n_samples)


if __name__ == "__main__":
    main()
