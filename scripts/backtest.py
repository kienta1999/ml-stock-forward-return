#!/usr/bin/env python3
"""Backtest the trained ranker on the test set (2021→).

Strategy (v1, long-only):
    Every 21 trading days:
        if SPY_close > SPY_SMA200 and VIX_close < 25:
            portfolio = top 50 stocks by predicted forward return, equal-weight
        else:
            portfolio = cash
    Hold each lot 21 trading days, then rebalance.

Rebalance-date robustness:
    The above is run 21 times, each starting on a different anchor day
    (offset = 0..20). Every offset produces an independent equity curve.
    We report the mean and the 10th/90th percentile band — small spread
    means rebalance-date doesn't matter much, large spread means it does.

Variants reported:
    1. SPY buy-and-hold (benchmark)
    2. Long-only, regime gate ON  (the actual proposed strategy)
    3. Long-only, regime gate OFF (diagnostic — raw stock-picking, no timing)

Costs: 5 bps per side on rebalance turnover.
Execution: trade at close of rebalance day (assumes MOC orders work).

Outputs:
    reports/backtest_equity.png   — equity curves vs SPY
    reports/backtest_stats.json   — CAGR / Sharpe / Max DD / time-in-market
    reports/backtest_equity.csv   — daily NAV per variant (mean across offsets)

CLI:
    uv run python scripts/backtest.py
    uv run python scripts/backtest.py --no-overlay     # gated variant off
    uv run python scripts/backtest.py --top-n 25       # tighter pick
    uv run python scripts/backtest.py --vix-threshold 30
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import strategy  # noqa: E402
from data import load_market  # noqa: E402
from dataset import TEST_START, load_panel  # noqa: E402

_ROOT = os.path.dirname(_HERE)
MODEL_PATH = os.path.join(_ROOT, "models", "xgb_v1.json")
REPORTS_DIR = os.path.join(_ROOT, "reports")

PERIODS_PER_YEAR = 252


# ─────────────────────────────────────────────────────────────────────────────
# Data prep
# ─────────────────────────────────────────────────────────────────────────────


def predict_test(panel: pd.DataFrame, model_path: str) -> pd.DataFrame:
    """Score the test slice with the saved xgb_v1 model."""
    test = panel[panel["date"] >= TEST_START]
    return strategy.predict(test, strategy.load_model(model_path))


# ─────────────────────────────────────────────────────────────────────────────
# Backtest core
# ─────────────────────────────────────────────────────────────────────────────


def run_one_offset(
    by_date: dict[pd.Timestamp, pd.DataFrame],
    market: pd.DataFrame,
    test_dates: list[pd.Timestamp],
    offset: int,
    *,
    regime_gate: bool,
    top_n: int,
    hold_days: int,
    cost_per_side: float,
    vix_threshold: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Single backtest with rebalances on day `offset`, `offset+hold_days`, ....

    Returns (equity, in_market, holdings). `holdings` is a Series of
    comma-separated ticker strings (or "" when in cash) — useful for auditing
    what the strategy actually held on each day.
    """
    weights: dict[str, float] = {}
    equity = 1.0
    equity_curve: dict[pd.Timestamp, float] = {}
    in_market: dict[pd.Timestamp, bool] = {}
    holdings: dict[pd.Timestamp, str] = {}

    for i, date in enumerate(test_dates):
        # 1. Apply today's return to current weights.
        if weights and date in by_date:
            day = by_date[date]
            ret_map = dict(zip(day["ticker"], day["ret_1d"]))
            pf_ret = sum(w * ret_map.get(t, 0.0) for t, w in weights.items())
            equity *= 1.0 + pf_ret

        # 2. Rebalance day?
        is_rebalance = (i >= offset) and ((i - offset) % hold_days == 0)
        if is_rebalance:
            go_long = True
            if regime_gate:
                if date in market.index:
                    go_long = strategy.regime_long_row(market.loc[date], vix_threshold)
                else:
                    go_long = False

            if go_long and date in by_date:
                day = by_date[date]
                top = strategy.top_picks(day, top_n)
                new_weights = {t: 1.0 / top_n for t in top["ticker"]}
            else:
                new_weights = {}

            all_tickers = set(new_weights) | set(weights)
            turnover = sum(
                abs(new_weights.get(t, 0.0) - weights.get(t, 0.0)) for t in all_tickers
            )
            equity *= 1.0 - turnover * cost_per_side
            weights = new_weights

        equity_curve[date] = equity
        in_market[date] = bool(weights)
        holdings[date] = ",".join(sorted(weights.keys())) if weights else ""

    return (
        pd.Series(equity_curve, name="equity"),
        pd.Series(in_market, name="in_market"),
        pd.Series(holdings, name="holdings"),
    )


def run_shifted_starts(
    test_panel: pd.DataFrame,
    market: pd.DataFrame,
    *,
    regime_gate: bool,
    top_n: int,
    hold_days: int,
    cost_per_side: float,
    vix_threshold: float,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Run `hold_days` backtests at different rebalance offsets.

    Returns (equity_df, time_in_market_per_offset, holdings_offset_0).
    equity_df is wide: rows=date, columns=offset. holdings_offset_0 is the
    daily holdings (comma-separated tickers) for offset=0 only — one
    representative pick log, kept tractable to write to CSV.
    """
    test_dates = sorted(test_panel["date"].unique())
    by_date = {
        d: g[["ticker", "ret_1d", "predicted_return"]]
        for d, g in test_panel.groupby("date")
    }

    curves: dict[int, pd.Series] = {}
    tim: dict[int, float] = {}
    holdings_offset_0: pd.Series | None = None
    for offset in range(hold_days):
        eq, inm, hold = run_one_offset(
            by_date, market, test_dates, offset,
            regime_gate=regime_gate, top_n=top_n, hold_days=hold_days,
            cost_per_side=cost_per_side, vix_threshold=vix_threshold,
        )
        curves[offset] = eq
        tim[offset] = float(inm.mean())
        if offset == 0:
            holdings_offset_0 = hold

    return (
        pd.DataFrame(curves).sort_index(),
        pd.Series(tim, name="time_in_market"),
        holdings_offset_0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark + stats
# ─────────────────────────────────────────────────────────────────────────────


def spy_buy_and_hold(market: pd.DataFrame, start: pd.Timestamp) -> pd.Series:
    """SPY equity curve starting at NAV=1.0 from `start`."""
    sub = market.loc[market.index >= start, "spy_ret_1d"].fillna(0.0)
    return (1.0 + sub).cumprod().rename("spy")


def compute_stats(equity: pd.Series) -> dict[str, float]:
    """CAGR, ann vol, Sharpe (rf=0), max drawdown."""
    daily_ret = equity.pct_change().dropna()
    n_years = len(daily_ret) / PERIODS_PER_YEAR if len(daily_ret) > 0 else 1.0
    final = float(equity.iloc[-1])
    cagr = final ** (1.0 / n_years) - 1.0 if final > 0 else -1.0
    ann_vol = float(daily_ret.std() * np.sqrt(PERIODS_PER_YEAR))
    sharpe = cagr / ann_vol if ann_vol > 0 else 0.0

    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = float(drawdown.min())

    return {
        "cagr": float(cagr),
        "ann_vol": ann_vol,
        "sharpe": float(sharpe),
        "max_dd": max_dd,
        "final_nav": final,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────


def plot_equity(
    gated: pd.DataFrame,
    raw: pd.DataFrame,
    spy: pd.Series,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))

    gated_mean = gated.mean(axis=1)
    gated_lo = gated.quantile(0.10, axis=1)
    gated_hi = gated.quantile(0.90, axis=1)
    ax.plot(gated_mean.index, gated_mean.values,
            label="Long-only + regime gate (mean of 21 offsets)",
            color="C0", linewidth=2)
    ax.fill_between(gated_mean.index, gated_lo.values, gated_hi.values,
                    color="C0", alpha=0.18, label="gated 10–90% offset band")

    raw_mean = raw.mean(axis=1)
    ax.plot(raw_mean.index, raw_mean.values,
            label="Long-only no gate (mean of 21 offsets)",
            color="C1", linewidth=1.6, linestyle="--")

    ax.plot(spy.index, spy.values, label="SPY buy & hold",
            color="black", linewidth=1.4, alpha=0.7)

    ax.set_title("Backtest equity curves vs SPY — test period 2021→")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (start = 1.0)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _print_stats(label: str, stats: dict, extra: str = "") -> None:
    print(
        f"  {label:<32}  CAGR={stats['cagr']:+.2%}  "
        f"Vol={stats['ann_vol']:.2%}  "
        f"Sharpe={stats['sharpe']:+.2f}  "
        f"MaxDD={stats['max_dd']:+.2%}  "
        f"FinalNAV={stats['final_nav']:.3f}{extra}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--top-n", type=int, default=strategy.TOP_N)
    ap.add_argument("--hold-days", type=int, default=strategy.HOLD_DAYS)
    ap.add_argument("--cost-bps", type=float, default=strategy.COST_PER_SIDE * 1e4,
                    help="Cost per side in basis points (default 5).")
    ap.add_argument("--vix-threshold", type=float, default=strategy.VIX_THRESHOLD)
    ap.add_argument("--no-overlay", action="store_true",
                    help="Skip the gated variant (only run raw long-only + SPY).")
    ap.add_argument("--model", default=MODEL_PATH)
    args = ap.parse_args()

    cost_per_side = args.cost_bps / 1e4

    print("Loading panel and predicting on test set...")
    panel = load_panel()
    test_panel = predict_test(panel, args.model)
    print(f"  test rows: {len(test_panel):,}  "
          f"({test_panel['date'].min().date()} → {test_panel['date'].max().date()})")

    market_data = load_market()
    market = strategy.prepare_market(market_data["SPY"], market_data["VIX"])

    gated_curves: pd.DataFrame | None = None
    gated_tim: pd.Series | None = None
    gated_holdings: pd.Series | None = None
    if not args.no_overlay:
        print(f"Running long-only WITH regime gate "
              f"(SPY>SMA200 AND VIX<{args.vix_threshold}, {args.hold_days} offsets)...")
        gated_curves, gated_tim, gated_holdings = run_shifted_starts(
            test_panel, market,
            regime_gate=True, top_n=args.top_n, hold_days=args.hold_days,
            cost_per_side=cost_per_side, vix_threshold=args.vix_threshold,
        )

    print(f"Running long-only WITHOUT regime gate ({args.hold_days} offsets)...")
    raw_curves, _, raw_holdings = run_shifted_starts(
        test_panel, market,
        regime_gate=False, top_n=args.top_n, hold_days=args.hold_days,
        cost_per_side=cost_per_side, vix_threshold=args.vix_threshold,
    )

    test_start = pd.Timestamp(test_panel["date"].min())
    spy_eq_full = spy_buy_and_hold(market, test_start)

    # Apples-to-apples: strategy curves stop at the last date with a 21-day-
    # old prediction; clip SPY to the same end date so reported CAGRs compare
    # like-for-like. spy_eq_full is kept for the post-strategy tail in the CSV.
    strategy_end = raw_curves.index.max()
    if gated_curves is not None:
        strategy_end = max(strategy_end, gated_curves.index.max())
    spy_eq = spy_eq_full.loc[:strategy_end]

    stats: dict = {}
    if gated_curves is not None:
        gated_mean = gated_curves.mean(axis=1)
        s = compute_stats(gated_mean)
        s["avg_time_in_market"] = float(gated_tim.mean())
        per_offset_cagrs = [
            compute_stats(gated_curves[c])["cagr"] for c in gated_curves.columns
        ]
        s["offset_cagr_min"] = float(min(per_offset_cagrs))
        s["offset_cagr_max"] = float(max(per_offset_cagrs))
        stats["gated_long_only"] = s

    raw_mean = raw_curves.mean(axis=1)
    stats["raw_long_only"] = compute_stats(raw_mean)
    stats["raw_long_only"]["avg_time_in_market"] = 1.0

    stats["spy_buy_hold"] = compute_stats(spy_eq)
    stats["spy_buy_hold"]["end_date"] = str(strategy_end.date())

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, "backtest_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # CSV keeps the full SPY series (extends past strategy end) so the tail
    # is visible for inspection; only stats + plot use the clipped curve.
    nav_df_cols: dict[str, pd.Series] = {"spy": spy_eq_full, "raw_long_only": raw_mean}
    if gated_curves is not None:
        nav_df_cols["gated_long_only"] = gated_curves.mean(axis=1)
        nav_df_cols["gated_offset_p10"] = gated_curves.quantile(0.10, axis=1)
        nav_df_cols["gated_offset_p90"] = gated_curves.quantile(0.90, axis=1)
    # Picks audit (offset 0 representative). Comma-separated tickers per day,
    # "" when in cash. Lets you eyeball who got picked when without re-running.
    if gated_holdings is not None:
        nav_df_cols["gated_picks_offset0"] = gated_holdings
    nav_df_cols["raw_picks_offset0"] = raw_holdings
    nav_df = pd.DataFrame(nav_df_cols).sort_index()
    nav_df.to_csv(os.path.join(REPORTS_DIR, "backtest_equity.csv"))

    if gated_curves is not None:
        plot_equity(
            gated_curves, raw_curves, spy_eq,
            os.path.join(REPORTS_DIR, "backtest_equity.png"),
        )

    print("\nBacktest summary:")
    if "gated_long_only" in stats:
        s = stats["gated_long_only"]
        extra = (f"  TIM={s['avg_time_in_market']:.0%}  "
                 f"CAGR offset range=[{s['offset_cagr_min']:+.2%}, "
                 f"{s['offset_cagr_max']:+.2%}]")
        _print_stats("gated long-only (regime ON)", s, extra)
    _print_stats("raw long-only (regime OFF)", stats["raw_long_only"])
    _print_stats(
        f"SPY buy & hold (clipped @{strategy_end.date()})",
        stats["spy_buy_hold"],
    )

    print(f"\n  -> equity curve: {REPORTS_DIR}/backtest_equity.png")
    print(f"  -> stats:        {REPORTS_DIR}/backtest_stats.json")
    print(f"  -> daily NAV:    {REPORTS_DIR}/backtest_equity.csv")


if __name__ == "__main__":
    main()
