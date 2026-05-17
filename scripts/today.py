#!/usr/bin/env python3
"""Generate today's portfolio picks from the trained model.

Bridges from backtest research to live trading. Predicts on the most recent
date with valid features (label not required — we don't yet know the
realised forward return), applies the same vol-target sizing overlay as
the backtest, and emits the top-N tickers with deployment weights scaled
by the recommended exposure (remainder = cash bucket).

Daily workflow:
    uv run python scripts/data.py        # incremental refresh
    uv run python scripts/features.py    # rebuild features
    uv run python scripts/today.py       # generate today's picks

This script reads `features.parquet` directly (not `panel.parquet`). The
panel drops the most recent ~21 trading days because forward_21d_return
needs prices 21 days ahead — but those rows still have valid *features*
and are exactly the ones we need to score live. Reading features.parquet
also means `labels.py` does not need to be re-run for today's picks.

CLI:
    uv run python scripts/today.py
    uv run python scripts/today.py --top-n 25
    uv run python scripts/today.py --vol-target 0.15   # more conservative sizing
    uv run python scripts/today.py --no-overlay        # ignore vol-target (always 100%)
    uv run python scripts/today.py --diff picks/picks_2026-04-28.csv

Outputs:
    stdout: SPY 20d vol + recommended exposure + top-N tickers with deploy weights
    picks/picks_<latest_date>.csv: machine-readable picks (weights already scaled)
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import strategy  # noqa: E402
from data import load_market  # noqa: E402
from features import load_features  # noqa: E402

_ROOT = os.path.dirname(_HERE)
MODEL_PATH = os.path.join(_ROOT, "models", "xgb_v1.json")
PICKS_DIR = os.path.join(_ROOT, "picks")

STALE_DAYS_WARN = 7


# ─────────────────────────────────────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────────────────────────────────────


def predict_today(
    features: pd.DataFrame, model_path: str
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Score the latest-date stocks. Returns (today_features, latest_date)."""
    valid = strategy.filter_valid_features(features)
    latest = valid["date"].max()
    today = valid[valid["date"] == latest]
    today = strategy.predict(today, strategy.load_model(model_path))
    return today, latest


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────


def _print_vol_regime(market_row: pd.Series, vol_target: float) -> float:
    """Print vol-target sizing status. Returns recommended exposure ∈ [0, 1]."""
    spy_vol = float(market_row["spy_vol_20d"])
    exposure = strategy.vol_target_exposure(spy_vol, vol_target)
    label = "full long" if exposure >= 0.99 else f"scaled down ({(1 - exposure):.0%} cash)"
    print("\nVol regime (sizing overlay):")
    print(f"  SPY close:           ${float(market_row['spy_close']):.2f}")
    print(f"  VIX close:           {float(market_row['vix_close']):.2f}  (informational only)")
    print(f"  SPY 20d realized vol: {spy_vol:.2%}")
    print(f"  Target vol:          {vol_target:.2%}")
    print(f"  → Recommended exposure: {exposure:.1%}  ({label})")
    return exposure


def _print_picks(picks_df: pd.DataFrame, latest_date: pd.Timestamp, top_n: int) -> None:
    print(f"\n=== TOP {top_n} PICKS ({latest_date.date()}) ===")
    for i, row in picks_df.iterrows():
        print(
            f"  {i + 1:>2}. {row['ticker']:<8}  "
            f"pred={row['predicted_return']:+.4f}  "
            f"weight={row['weight']:.4f}"
        )
    avg_pred = float(picks_df["predicted_return"].mean())
    print(f"\n  Avg predicted 21d return for the basket: {avg_pred:+.2%}")


def _print_diff(picks_df: pd.DataFrame, prev: pd.DataFrame, prev_label: str) -> None:
    prev_set = set(prev["ticker"]) if not prev.empty else set()
    curr_set = set(picks_df["ticker"]) if not picks_df.empty else set()
    sells = sorted(prev_set - curr_set)
    buys = sorted(curr_set - prev_set)
    unchanged = sorted(prev_set & curr_set)
    print(f"\n=== DIFF vs {prev_label} ===")
    print(f"  SELL ({len(sells):>2}): {', '.join(sells) if sells else '(none)'}")
    print(f"  BUY  ({len(buys):>2}): {', '.join(buys) if buys else '(none)'}")
    print(f"  HOLD ({len(unchanged):>2}): {len(unchanged)} tickers unchanged")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--top-n", type=int, default=strategy.TOP_N)
    ap.add_argument(
        "--vol-target",
        type=float,
        default=strategy.DEFAULT_VOL_TARGET,
        help=(
            "Target portfolio vol (annualized) for the sizing overlay "
            f"(default {strategy.DEFAULT_VOL_TARGET}). Exposure = "
            "min(1.0, vol_target / spy_vol_20d); picks weights are scaled "
            "by this exposure and the remainder is implicitly cash."
        ),
    )
    ap.add_argument(
        "--no-overlay",
        action="store_true",
        help="Skip vol-target overlay; always recommend 100% exposure.",
    )
    ap.add_argument(
        "--no-quality-filter",
        action="store_true",
        help=(
            "Disable the cataclysmic-only fundamentals/insider filter applied "
            "before top-N selection. Defaults (strategy.QUALITY_FILTER_DEFAULTS): "
            "drop debt_to_equity>10, current_ratio<0.3, sales_growth_yoy<-0.50, "
            "insider_net_dollar_60d<-50M. NaN values always pass."
        ),
    )
    ap.add_argument(
        "--diff",
        help="Path to a previous picks CSV to compute BUY/SELL/HOLD list against.",
    )
    ap.add_argument(
        "--weight",
        choices=strategy.WEIGHT_MODES,
        default=strategy.DEFAULT_WEIGHT_MODE,
        help=(
            "Basket weighting scheme. 'equal' (default) = 1/N. "
            "'pred' = proportional to predicted_return (negatives clipped, "
            "falls back to equal if all picks are ≤0)."
        ),
    )
    ap.add_argument("--model", default=MODEL_PATH)
    args = ap.parse_args()

    # Load --diff file BEFORE we write the new picks: today's filename is
    # picks_<latest_feature_date>.csv, and on a same-day rerun this can
    # collide with the --diff source (we'd be diffing against ourselves).
    # Reading first preserves the prior state in memory regardless.
    prev_picks: pd.DataFrame | None = None
    if args.diff:
        if os.path.exists(args.diff):
            prev_picks = pd.read_csv(args.diff)
        else:
            print(f"  ⚠ --diff file {args.diff} does not exist; will skip diff")

    print("Loading features and predicting today's slice...")
    features = load_features()
    today, latest_date = predict_today(features, args.model)
    today_str = latest_date.date()

    print(f"  Latest date with valid features: {today_str}")
    print(f"  Stocks scored: {len(today)}")
    days_old = (pd.Timestamp.today().normalize() - latest_date).days
    if days_old > STALE_DAYS_WARN:
        print(
            f"  ⚠ features are {days_old} days old — "
            f"run data.py + features.py to refresh"
        )

    market_data = load_market()
    market = strategy.prepare_market(market_data["SPY"], market_data["VIX"])
    latest_market_row = market.iloc[-1]

    if args.no_overlay:
        exposure = 1.0
        print("\n(--no-overlay): vol-target disabled — picks at 100% exposure.")
    else:
        exposure = _print_vol_regime(latest_market_row, args.vol_target)

    if exposure <= 0:
        # Defensive: vol_target_exposure clamps at [0, 1] and only returns 0
        # when vol_target itself is 0. Treat as full cash.
        print("\n=== STAY IN CASH (exposure = 0) ===")
        picks_df = pd.DataFrame(columns=["ticker", "predicted_return", "weight"])
    else:
        candidates = today
        if not args.no_quality_filter:
            before = len(candidates)
            candidates = strategy.apply_quality_filter(candidates)
            dropped = before - len(candidates)
            print(
                f"\nQuality filter: dropped {dropped} of {before} candidates "
                f"({dropped / before * 100:.1f}%) on fundamentals/insider thresholds."
            )
        top = strategy.top_picks(candidates, args.top_n)[
            ["ticker", "predicted_return"]
        ].copy().reset_index(drop=True)
        pick_weights = strategy.compute_weights(top, args.weight)
        # Scale every basket weight by the recommended exposure; the
        # implicit (1 - exposure) is cash. CSV reflects the actual
        # deployment, not the renormalized basket.
        top["weight"] = top["ticker"].map(
            lambda t: pick_weights.get(t, 0.0) * exposure
        )
        picks_df = top
        if exposure < 0.99:
            print(
                f"  Weights below scaled by exposure {exposure:.2f}; "
                f"basket sums to {exposure:.2%}, cash bucket = {1 - exposure:.2%}."
            )
        _print_picks(picks_df, latest_date, args.top_n)

    os.makedirs(PICKS_DIR, exist_ok=True)
    out_path = os.path.join(PICKS_DIR, f"picks_{today_str}.csv")
    picks_df.to_csv(out_path, index=False)
    print(f"\n  -> picks saved to {out_path}")

    if prev_picks is not None:
        _print_diff(picks_df, prev_picks, os.path.basename(args.diff))


if __name__ == "__main__":
    main()
