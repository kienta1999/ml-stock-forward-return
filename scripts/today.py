#!/usr/bin/env python3
"""Generate today's portfolio picks from the trained model.

Bridges from backtest research to live trading. Predicts on the most recent
date with valid features (label not required — we don't yet know the realised
forward return), applies the same regime gate as the backtest, and emits the
top-N tickers (or "CASH").

Daily workflow:
    uv run python scripts/data.py        # incremental refresh
    uv run python scripts/features.py    # rebuild features
    uv run python scripts/labels.py      # rebuild panel.parquet
    uv run python scripts/today.py       # generate today's picks

The most recent ~21 trading days have features but no label (forward_21d_return
needs prices 21 days ahead, which don't exist yet). The backtest deliberately
ignores those days; this script deliberately predicts on them.

CLI:
    uv run python scripts/today.py
    uv run python scripts/today.py --top-n 25
    uv run python scripts/today.py --no-overlay
    uv run python scripts/today.py --diff picks/picks_2026-04-28.csv

Outputs:
    stdout: regime status + top-N tickers with predicted returns
    picks/picks_<latest_date>.csv: machine-readable picks for downstream automation
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
from dataset import load_panel  # noqa: E402

_ROOT = os.path.dirname(_HERE)
MODEL_PATH = os.path.join(_ROOT, "models", "xgb_v1.json")
PICKS_DIR = os.path.join(_ROOT, "picks")

STALE_DAYS_WARN = 7


# ─────────────────────────────────────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────────────────────────────────────


def predict_today(
    panel: pd.DataFrame, model_path: str
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Score the latest-date stocks. Returns (today_panel, latest_date)."""
    valid = strategy.filter_valid_features(panel)
    latest = valid["date"].max()
    today = valid[valid["date"] == latest]
    today = strategy.predict(today, strategy.load_model(model_path))
    return today, latest


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────


def _print_regime(market_row: pd.Series, vix_threshold: float) -> bool:
    """Print regime status, return go_long verdict."""
    go_long = strategy.regime_long_row(market_row, vix_threshold)
    print("\nRegime status:")
    print(f"  SPY close:     ${float(market_row['spy_close']):.2f}")
    print(f"  SPY SMA200:    ${float(market_row['spy_sma200']):.2f}")
    print(f"  SPY > SMA200:  {bool(market_row['spy_close'] > market_row['spy_sma200'])}")
    print(f"  VIX close:     {float(market_row['vix_close']):.2f}")
    print(f"  VIX < {vix_threshold:.0f}:      {bool(market_row['vix_close'] < vix_threshold)}")
    print(f"  Go long?       {go_long}")
    return go_long


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
    ap.add_argument("--vix-threshold", type=float, default=strategy.VIX_THRESHOLD)
    ap.add_argument(
        "--no-overlay",
        action="store_true",
        help="Skip regime gate; always pick top N.",
    )
    ap.add_argument(
        "--diff",
        help="Path to a previous picks CSV to compute BUY/SELL/HOLD list against.",
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

    print("Loading panel and predicting today's slice...")
    panel = load_panel(drop_na=False)
    today, latest_date = predict_today(panel, args.model)
    today_str = latest_date.date()

    print(f"  Latest date with valid features: {today_str}")
    print(f"  Stocks scored: {len(today)}")
    days_old = (pd.Timestamp.today().normalize() - latest_date).days
    if days_old > STALE_DAYS_WARN:
        print(
            f"  ⚠ panel is {days_old} days old — "
            f"run data.py + features.py + labels.py to refresh"
        )

    market_data = load_market()
    market = strategy.prepare_market(market_data["SPY"], market_data["VIX"])
    latest_market_row = market.iloc[-1]
    go_long = _print_regime(latest_market_row, args.vix_threshold)

    if not args.no_overlay and not go_long:
        print("\n=== STAY IN CASH ===")
        picks_df = pd.DataFrame(columns=["ticker", "predicted_return", "weight"])
    else:
        if args.no_overlay and not go_long:
            print(
                f"\n(--no-overlay): regime would say cash; "
                f"ignoring and picking top {args.top_n}"
            )
        top = strategy.top_picks(today, args.top_n)[
            ["ticker", "predicted_return"]
        ].copy().reset_index(drop=True)
        top["weight"] = 1.0 / args.top_n
        picks_df = top
        _print_picks(picks_df, latest_date, args.top_n)

    os.makedirs(PICKS_DIR, exist_ok=True)
    out_path = os.path.join(PICKS_DIR, f"picks_{today_str}.csv")
    picks_df.to_csv(out_path, index=False)
    print(f"\n  -> picks saved to {out_path}")

    if prev_picks is not None:
        _print_diff(picks_df, prev_picks, os.path.basename(args.diff))


if __name__ == "__main__":
    main()
