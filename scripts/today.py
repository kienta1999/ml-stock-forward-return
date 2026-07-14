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

Every run also performs the daily DE-RISK CHECK for the live book: the
account rebalances quarterly, but the vol-target overlay is only evaluated
at rebalance — a mid-cycle vol spike would leave the book levered while the
formula says to be smaller. The check recomputes
min(leverage, vol_target / spy_vol_20d) from the freshest cached SPY data,
compares it against the book's exposure (reports/live_book.json, written by
execute_picks.py --mode live), and prints "OK — hold" or a loud
"⚠ DE-RISK ... sell down today". Sell-down only: when the formula rises
back above the book, re-levering waits for the quarterly rebalance.

CLI:
    uv run python scripts/today.py
    uv run python scripts/today.py --top-n 25
    uv run python scripts/today.py --vol-target 0.15   # more conservative sizing
    uv run python scripts/today.py --no-overlay        # ignore vol-target (always 100%)
    uv run python scripts/today.py --diff picks/picks_2026-04-28.csv
    uv run python scripts/today.py --book-exposure 1.2 # override live_book.json
    uv run python scripts/today.py --no-derisk         # skip the de-risk check
    uv run python scripts/today.py --mode top-bottom   # market-neutral L/S picks
                                                       # -> picks/top_bottom/ (paper only)

Outputs:
    stdout: SPY 20d vol + recommended exposure + de-risk check + top-N picks
    picks/picks_<latest_date>.csv: machine-readable picks (weights already scaled)
"""

import argparse
import json
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
BOOK_STATE_PATH = os.path.join(_ROOT, "reports", "live_book.json")

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


def _print_derisk_check(
    market_row: pd.Series,
    vol_target: float,
    leverage: float,
    book_override: float | None,
    tolerance: float,
) -> None:
    """Daily de-risk check — off-cycle SELL-DOWN alarm for the live book.

    The book's exposure comes from reports/live_book.json (recorded by
    execute_picks.py --mode live at each rebalance), a --book-exposure
    override, or — worst case, when neither exists — the full leverage
    cap, so a missing state file can only make the check MORE likely to
    fire, never silently lull.
    """
    spy_vol = float(market_row["spy_vol_20d"])
    asof = pd.Timestamp(market_row.name).date()
    formula = strategy.vol_target_exposure(spy_vol, vol_target, max_exposure=leverage)

    if book_override is not None:
        book, src = book_override, "--book-exposure override"
    else:
        book, src = None, ""
        if os.path.exists(BOOK_STATE_PATH):
            try:
                with open(BOOK_STATE_PATH) as fh:
                    state = json.load(fh)
                book = float(state["gross_exposure"])
                src = (f"set {str(state.get('executed_at', '?'))[:10]}, "
                       f"reports/live_book.json")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError):
                print(f"\n  ⚠ could not parse {BOOK_STATE_PATH} — ignoring it")
        if book is None:
            book = leverage
            src = "no reports/live_book.json — assuming full cap (worst case)"

    print("\nDe-risk check (daily crash guard for the quarterly-rebalanced book):")
    print(f"  Book gross exposure: {book:.2f}x  ({src})")
    print(f"  Formula today:       min({leverage:.2f}, {vol_target:.2f} / "
          f"{spy_vol:.3f}) = {formula:.2f}x  (SPY vol as of {asof})")
    if formula < book * (1.0 - tolerance):
        print(f"  → ⚠⚠ DE-RISK: formula says {formula:.2f}x, book is at "
              f"{book:.2f}x — SELL DOWN TODAY:")
        print(f"       uv run python scripts/execute_picks.py --port 4001 "
              f"--leverage {leverage} --vol-target {vol_target} --mode live")
    else:
        print(f"  → OK — hold (formula within {tolerance:.0%} of book; "
              f"re-levering waits for the quarterly rebalance)")


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
        "--mode",
        choices=strategy.PORTFOLIO_MODES,
        default=strategy.DEFAULT_PORTFOLIO_MODE,
        help=(
            "'long-only' (default) = the live strategy. 'top-bottom' = "
            "market-neutral: long top-N (quality-filtered) AND short bottom-N "
            "(unfiltered, equal weight, negative CSV weights) at equal "
            "dollars. Writes to picks/top_bottom/ so the live long-only "
            "execution flow can never ingest a short book by accident. "
            "NOTE: execute_picks.py does not support short weights yet — "
            "top-bottom is backtest/paper only."
        ),
    )
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
        "--leverage",
        type=float,
        default=strategy.LIVE_LEVERAGE,
        help=(
            "Gross-exposure cap used by the de-risk check's formula "
            f"(default {strategy.LIVE_LEVERAGE} — the live-account config). "
            "Does NOT change the picks CSV weights; leverage is applied at "
            "execution by execute_picks.py."
        ),
    )
    ap.add_argument(
        "--book-exposure",
        type=float,
        default=None,
        help=(
            "Current gross exposure of the live book, for the de-risk check. "
            "Default: reports/live_book.json (written by execute_picks.py "
            "--mode live), else the full --leverage cap (worst case)."
        ),
    )
    ap.add_argument(
        "--derisk-tolerance",
        type=float,
        default=strategy.DERISK_TOLERANCE,
        help=(
            "Relative drop of formula-vs-book that triggers the DE-RISK alarm "
            f"(default {strategy.DERISK_TOLERANCE:.2f} = fire when formula < "
            "book × 0.90)."
        ),
    )
    ap.add_argument(
        "--no-derisk",
        action="store_true",
        help="Skip the daily de-risk check.",
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

    # Daily de-risk check — independent of the picks overlay above (that
    # scales the CSV weights; this guards the already-executed live book).
    if not args.no_derisk:
        _print_derisk_check(
            latest_market_row,
            args.vol_target,
            args.leverage,
            args.book_exposure,
            args.derisk_tolerance,
        )

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

        if args.mode == "top-bottom":
            # Short leg: bottom-N from the UNFILTERED scored slice — the
            # quality filter shields the long book only; its cataclysmic
            # rejects are exactly the names worth shorting. Equal weight
            # (no 'pred' analogue for shorts), negative in the CSV.
            bottom = strategy.bottom_picks(today, args.top_n)[
                ["ticker", "predicted_return"]
            ].copy().reset_index(drop=True)
            bottom["weight"] = -exposure / len(bottom)
            print(f"\n=== SHORT {len(bottom)} PICKS ({latest_date.date()}) — "
                  f"bottom of the ranking, unfiltered ===")
            for i, row in bottom.iterrows():
                print(f"  {i + 1:>2}. {row['ticker']:<8}  "
                      f"pred={row['predicted_return']:+.4f}  "
                      f"weight={row['weight']:.4f}")
            print(f"\n  Gross {2 * exposure:.2f}x (long {exposure:.2f} / short "
                  f"{exposure:.2f}), net ≈ 0.")
            print("  ⚠ top-bottom is backtest/paper only — execute_picks.py "
                  "does not support short weights yet.")
            picks_df = pd.concat([top, bottom], ignore_index=True)

    out_dir = PICKS_DIR if args.mode == "long-only" \
        else os.path.join(PICKS_DIR, "top_bottom")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"picks_{today_str}.csv")
    picks_df.to_csv(out_path, index=False)
    print(f"\n  -> picks saved to {out_path}")

    if prev_picks is not None:
        _print_diff(picks_df, prev_picks, os.path.basename(args.diff))


if __name__ == "__main__":
    main()
