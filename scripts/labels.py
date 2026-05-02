#!/usr/bin/env python3
"""Forward-return labels.

label = close[t+N] / close[t] - 1   (default N = 21 trading days)

Per spec: drop rows where the label is NaN — that's the last N trading days
of each ticker (no future data to compute against).

CLI:
    python scripts/labels.py            # add label to features parquet → data/processed/panel.parquet
"""

import argparse
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_ROOT = os.path.dirname(_HERE)
FEATURES_PATH = os.path.join(_ROOT, "data", "processed", "features.parquet")
PANEL_PATH = os.path.join(_ROOT, "data", "processed", "panel.parquet")

FORWARD_DAYS = 21
LABEL_COL = f"forward_{FORWARD_DAYS}d_return"
CLIP_PCT = 0.5  # cap labels at ±50% so dead-ticker crashes don't dominate MSE


def forward_return(close: pd.Series, days: int = FORWARD_DAYS) -> pd.Series:
    """One ticker's forward return series. Last `days` rows will be NaN."""
    return close.shift(-days) / close - 1


def add_label(
    panel: pd.DataFrame,
    days: int = FORWARD_DAYS,
    drop_na: bool = True,
    clip: float | None = CLIP_PCT,
) -> pd.DataFrame:
    """Add forward-return column to a long panel sorted by (ticker, date).

    Requires `panel` to carry a `close` column (features.build_panel does).
    `clip`: cap labels at ±clip (None disables). Without it a few delisted-ticker
    -100% labels blow up MSE and force best_iteration → 1.
    """
    if "close" not in panel.columns:
        raise ValueError("panel needs a 'close' column — see features.build_panel")

    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    label_col = f"forward_{days}d_return"
    panel[label_col] = panel.groupby("ticker", sort=False)["close"].transform(
        lambda s: s.shift(-days) / s - 1
    )
    if drop_na:
        panel = panel.dropna(subset=[label_col]).reset_index(drop=True)
    if clip is not None:
        n_clipped = ((panel[label_col] < -clip) | (panel[label_col] > clip)).sum()
        panel[label_col] = panel[label_col].clip(-clip, clip)
        print(
            f"Clipped {n_clipped:,} of {len(panel):,} labels to ±{clip} "
            f"({n_clipped / max(len(panel), 1):.2%})."
        )
    return panel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features", default=FEATURES_PATH)
    ap.add_argument("--out", default=PANEL_PATH)
    ap.add_argument("--days", type=int, default=FORWARD_DAYS)
    ap.add_argument(
        "--clip",
        type=float,
        default=CLIP_PCT,
        help=f"Cap labels at ±this. Use --clip 0 to disable (default {CLIP_PCT}).",
    )
    args = ap.parse_args()

    if not os.path.exists(args.features):
        raise SystemExit(f"{args.features} not found. Run scripts/features.py first.")

    panel = pd.read_parquet(args.features)
    n_before = len(panel)
    panel = add_label(panel, days=args.days)
    label_col = f"forward_{args.days}d_return"
    print(
        f"Added {label_col}. Rows: {n_before:,} → {len(panel):,} "
        f"(dropped {n_before - len(panel):,} tail rows with NaN label)."
    )
    print(f"Label stats:\n{panel[label_col].describe().to_string()}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    panel.to_parquet(args.out, index=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
