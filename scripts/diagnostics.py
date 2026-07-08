#!/usr/bin/env python3
"""Post-hoc strategy diagnostics: is the alpha broad and stable, or a few
lucky months of concentrated tech?

Sections (each skippable, all read-only — no IBKR connection needed):

  1. IC stability      — daily Spearman(predicted, realized fwd 21d) on the
                         test slice, aggregated per month. t-stat is computed
                         on the MONTHLY means (daily ICs share overlapping
                         21d forward windows, so a daily t-stat overstates
                         independence by ~sqrt(21)).
  2. Underwater plot   — drawdown curves from reports/backtest_equity.csv
                         (run backtest.py first) + top-5 drawdown episodes.
  3. Concentration     — offset-0 monthly rebalance baskets (quality filter +
                         top-N, mirroring the raw backtest): ticker frequency,
                         average sector weights, consecutive-basket overlap.
  4. Attribution       — per-ticker sum of (weight × realized fwd 21d return)
                         across all baskets: top / bottom 10 names.
  5. Hit rate          — per rebalance, % of picks whose realized 21d return
                         beat SPY's over the same window.
  6. Live scorecard    — every picks/picks_*.csv marked to market from cached
                         prices: basket return (incl. cash drag) vs SPY over
                         the same window, matured (21 trading days) or open.

Sections 1 & 3-5 measure the MODEL (equal-weight, no vol overlay — the
cleanest read on ranking skill). Section 6 measures what the live pick
files actually said to hold, including their baked-in vol-target exposure.
Live returns assume fills at the signal close (README: --lag 1 costs
~0.2 CAGR pt, so this is a close approximation, not a tax record).

Outputs:
    reports/diagnostics_ic_monthly.csv
    reports/diagnostics_underwater.png
    reports/diagnostics_attribution.csv
    reports/diagnostics_live_scorecard.csv
    reports/diagnostics_summary.json
    stdout: consolidated report

CLI:
    uv run python scripts/diagnostics.py               # everything
    uv run python scripts/diagnostics.py --live-only   # section 6 only (fast, daily)
    uv run python scripts/diagnostics.py --top-n 20    # audit a tighter basket
"""

import argparse
import json
import os
import sys
import warnings
from glob import glob

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import strategy  # noqa: E402
from backtest import MODEL_PATH, predict_test  # noqa: E402
from data import load_market, load_prices  # noqa: E402
from dataset import load_panel  # noqa: E402

_ROOT = os.path.dirname(_HERE)
REPORTS_DIR = os.path.join(_ROOT, "reports")
PICKS_DIR = os.path.join(_ROOT, "picks")
EQUITY_CSV = os.path.join(REPORTS_DIR, "backtest_equity.csv")

RAW_LABEL = "forward_21d_return"
HOLD_DAYS = strategy.HOLD_DAYS


def _hr(title: str) -> None:
    print(f"\n{'─' * 70}\n  {title}\n{'─' * 70}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. IC stability
# ─────────────────────────────────────────────────────────────────────────────


def ic_stability(test_panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Daily Spearman IC per date → monthly aggregation + stability stats."""
    daily = (
        test_panel.groupby("date")
        .apply(lambda g: spearmanr(g["predicted_return"], g[RAW_LABEL])[0])
        .rename("ic")
        .dropna()
    )
    monthly = daily.groupby(daily.index.to_period("M")).agg(["mean", "std", "count"])
    monthly.columns = ["ic_mean", "ic_std", "n_days"]

    m = monthly["ic_mean"]
    n_months = len(m)
    t_stat = float(m.mean() / m.std() * np.sqrt(n_months)) if n_months > 1 else np.nan
    stats = {
        "daily_ic_mean": float(daily.mean()),
        "monthly_ic_mean": float(m.mean()),
        "monthly_ic_tstat": t_stat,
        "pct_months_positive": float((m > 0).mean()),
        "n_months": n_months,
        "best_month": {"month": str(m.idxmax()), "ic": float(m.max())},
        "worst_month": {"month": str(m.idxmin()), "ic": float(m.min())},
    }

    _hr("1. IC STABILITY (test slice, monthly)")
    print(f"  Daily IC mean:        {stats['daily_ic_mean']:+.4f}")
    print(f"  Monthly IC mean:      {stats['monthly_ic_mean']:+.4f}  "
          f"(t-stat {t_stat:+.2f} on {n_months} monthly means)")
    print(f"  Months positive:      {stats['pct_months_positive']:.0%}")
    print(f"  Best month:           {stats['best_month']['month']}  "
          f"IC {stats['best_month']['ic']:+.4f}")
    print(f"  Worst month:          {stats['worst_month']['month']}  "
          f"IC {stats['worst_month']['ic']:+.4f}")
    # A concentrated alpha shows up here: high overall IC but <60% of months
    # positive, or the t-stat collapsing when the best 2-3 months are removed.
    trimmed = m.drop(m.nlargest(3).index)
    print(f"  Monthly IC mean excl. top-3 months: {trimmed.mean():+.4f}")
    stats["monthly_ic_mean_excl_top3"] = float(trimmed.mean())
    return monthly, stats


# ─────────────────────────────────────────────────────────────────────────────
# 2. Underwater plot (from backtest artifacts)
# ─────────────────────────────────────────────────────────────────────────────


def _drawdown_episodes(equity: pd.Series, top_k: int = 5) -> pd.DataFrame:
    dd = equity / equity.cummax() - 1.0
    episodes = []
    in_dd, start = False, None
    for date, val in dd.items():
        if val < 0 and not in_dd:
            in_dd, start = True, date
        elif val == 0 and in_dd:
            seg = dd.loc[start:date]
            episodes.append((start, seg.idxmin(), float(seg.min()), date))
            in_dd = False
    if in_dd:
        seg = dd.loc[start:]
        episodes.append((start, seg.idxmin(), float(seg.min()), pd.NaT))
    df = pd.DataFrame(episodes, columns=["start", "trough", "depth", "recovered"])
    return df.nsmallest(top_k, "depth").reset_index(drop=True)


def underwater(no_plot: bool) -> dict:
    if not os.path.exists(EQUITY_CSV):
        _hr("2. UNDERWATER — skipped (reports/backtest_equity.csv missing; run backtest.py)")
        return {}
    eq = pd.read_csv(EQUITY_CSV, index_col=0, parse_dates=True)
    curves = {c: eq[c].dropna() for c in ("raw_long_only", "gated_long_only", "spy")
              if c in eq.columns}

    _hr("2. UNDERWATER (from last backtest run)")
    out: dict = {}
    for name, series in curves.items():
        dd = series / series.cummax() - 1.0
        out[name] = {"max_dd": float(dd.min()), "current_dd": float(dd.iloc[-1])}
        print(f"  {name:<18} max DD {dd.min():+.1%}   current DD {dd.iloc[-1]:+.1%}")

    if "raw_long_only" in curves:
        print("\n  Top drawdown episodes (raw long-only):")
        for _, r in _drawdown_episodes(curves["raw_long_only"]).iterrows():
            rec = r["recovered"].date() if pd.notna(r["recovered"]) else "ONGOING"
            print(f"    {r['depth']:+.1%}  {r['start'].date()} → trough "
                  f"{r['trough'].date()} → recovered {rec}")

    if not no_plot:
        fig, ax = plt.subplots(figsize=(13, 5))
        for name, series in curves.items():
            dd = series / series.cummax() - 1.0
            ax.plot(dd.index, dd.values, label=name, linewidth=1.3)
        ax.set_title("Underwater plot — drawdown from running peak")
        ax.set_ylabel("Drawdown")
        ax.legend(loc="lower left")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = os.path.join(REPORTS_DIR, "diagnostics_underwater.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"\n  -> {path}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3-5. Basket simulation: concentration, attribution, hit rate
# ─────────────────────────────────────────────────────────────────────────────


def simulate_baskets(test_panel: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Offset-0 rebalance baskets mirroring the raw backtest: every HOLD_DAYS
    test dates, quality filter → top-N, equal weight. Returns one row per
    (rebalance_date, ticker) with the realized raw forward 21d return."""
    test_dates = sorted(test_panel["date"].unique())
    keep = ["ticker", "predicted_return", RAW_LABEL, "gics_sector"] + [
        c for c in ("debt_to_equity", "current_ratio", "sales_growth_yoy",
                    "insider_net_dollar_60d") if c in test_panel.columns
    ]
    rows = []
    for d in test_dates[::HOLD_DAYS]:
        day = test_panel.loc[test_panel["date"] == d, keep]
        day = strategy.apply_quality_filter(day)
        top = strategy.top_picks(day, top_n)
        w = 1.0 / len(top) if len(top) else 0.0
        for _, r in top.iterrows():
            rows.append((d, r["ticker"], str(r["gics_sector"]), w,
                         float(r["predicted_return"]), float(r[RAW_LABEL])))
    return pd.DataFrame(
        rows, columns=["date", "ticker", "sector", "weight", "pred", "fwd_ret"]
    )


def concentration(baskets: pd.DataFrame) -> dict:
    _hr("3. PICKS CONCENTRATION (offset-0 monthly baskets, test slice)")
    n_reb = baskets["date"].nunique()
    freq = baskets["ticker"].value_counts()
    print(f"  Rebalances: {n_reb}   unique tickers ever picked: {len(freq)}")
    print(f"  Most-picked (of {n_reb}): "
          + ", ".join(f"{t} {c}x" for t, c in freq.head(10).items()))

    sector_w = (baskets.groupby(["date", "sector"])["weight"].sum()
                .groupby("sector").agg(["mean", "max"])
                .sort_values("mean", ascending=False))
    print("\n  Avg sector weight (max in any basket):")
    for sec, r in sector_w.head(6).iterrows():
        print(f"    {sec:<28} {r['mean']:>5.1%}  (max {r['max']:.1%})")

    dates = sorted(baskets["date"].unique())
    by_date = {d: set(g["ticker"]) for d, g in baskets.groupby("date")}
    overlaps = [
        len(by_date[a] & by_date[b]) / max(len(by_date[b]), 1)
        for a, b in zip(dates, dates[1:])
    ]
    avg_overlap = float(np.mean(overlaps)) if overlaps else np.nan
    print(f"\n  Avg consecutive-rebalance overlap: {avg_overlap:.0%} "
          f"(→ ~{1 - avg_overlap:.0%} turnover per rebalance)")
    top_sector = sector_w.index[0] if len(sector_w) else None
    return {
        "n_rebalances": int(n_reb),
        "unique_tickers": int(len(freq)),
        "avg_consecutive_overlap": avg_overlap,
        "top_sector": top_sector,
        "top_sector_avg_weight": float(sector_w["mean"].iloc[0]) if top_sector else None,
    }


def attribution(baskets: pd.DataFrame) -> pd.DataFrame:
    contrib = (baskets.assign(contrib=baskets["weight"] * baskets["fwd_ret"])
               .groupby("ticker")
               .agg(contrib=("contrib", "sum"), times_picked=("date", "count"))
               .sort_values("contrib", ascending=False))
    _hr("4. PER-STOCK ATTRIBUTION (Σ weight × realized 21d return)")
    total = float(contrib["contrib"].sum())
    print(f"  Total summed contribution: {total:+.2%} "
          f"(top-10 names = {contrib['contrib'].head(10).sum() / total:.0%} of it)"
          if total > 0 else f"  Total summed contribution: {total:+.2%}")
    print("\n  Top 10 contributors:")
    for t, r in contrib.head(10).iterrows():
        print(f"    {t:<8} {r['contrib']:+7.2%}  (picked {int(r['times_picked'])}x)")
    print("  Bottom 10 detractors:")
    for t, r in contrib.tail(10).iloc[::-1].iterrows():
        print(f"    {t:<8} {r['contrib']:+7.2%}  (picked {int(r['times_picked'])}x)")
    return contrib


def hit_rate(baskets: pd.DataFrame, market: pd.DataFrame) -> dict:
    spy_fwd = (market["spy_close"].shift(-HOLD_DAYS) / market["spy_close"] - 1.0)
    per_reb = []
    for d, g in baskets.groupby("date"):
        spy_r = spy_fwd.get(d, np.nan)
        if pd.isna(spy_r):
            continue
        per_reb.append((d, float((g["fwd_ret"] > spy_r).mean()),
                        float(g["fwd_ret"].mean()), float(spy_r)))
    df = pd.DataFrame(per_reb, columns=["date", "hit_rate", "basket_ret", "spy_ret"])
    _hr("5. HIT RATE (per rebalance: % of picks beating SPY over 21d)")
    beat = (df["basket_ret"] > df["spy_ret"]).mean()
    print(f"  Avg hit rate:                    {df['hit_rate'].mean():.0%}")
    print(f"  Rebalances where basket > SPY:   {beat:.0%}  ({len(df)} rebalances)")
    print(f"  Avg basket 21d return vs SPY:    {df['basket_ret'].mean():+.2%} "
          f"vs {df['spy_ret'].mean():+.2%}")
    worst = df.nsmallest(3, "hit_rate")
    print("  Worst rebalances: "
          + ", ".join(f"{r['date'].date()} ({r['hit_rate']:.0%})"
                      for _, r in worst.iterrows()))
    return {
        "avg_hit_rate": float(df["hit_rate"].mean()),
        "pct_rebalances_beat_spy": float(beat),
        "n_rebalances": int(len(df)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Live picks scorecard (no IBKR — model-tracked, marked from cached prices)
# ─────────────────────────────────────────────────────────────────────────────


def live_scorecard(market: pd.DataFrame) -> pd.DataFrame | None:
    files = sorted(glob(os.path.join(PICKS_DIR, "picks_*.csv")))
    if not files:
        _hr("6. LIVE PICKS SCORECARD — skipped (no picks/*.csv)")
        return None

    picks = {os.path.basename(f)[6:16]: pd.read_csv(f) for f in files}
    tickers = sorted({t for df in picks.values() if len(df)
                      for t in df["ticker"]})
    prices = load_prices(tickers, min_history=1)
    closes = pd.DataFrame({t: df["Close"] for t, df in prices.items()})
    closes.index = pd.to_datetime(closes.index)
    spy = market["spy_close"]
    cal = spy.index  # trading calendar

    rows = []
    for date_str, df in picks.items():
        d = pd.Timestamp(date_str)
        if d not in cal:
            continue
        i = cal.get_loc(d)
        j = min(i + HOLD_DAYS, len(cal) - 1)
        end = cal[j]
        matured = (j - i) >= HOLD_DAYS
        spy_ret = float(spy.loc[end] / spy.loc[d] - 1.0)
        if len(df) == 0:  # cash day
            basket_ret, exposure, n_marked = 0.0, 0.0, 0
        else:
            exposure = float(df["weight"].sum())
            ret_sum, n_marked = 0.0, 0
            for _, r in df.iterrows():
                t = r["ticker"]
                if t not in closes.columns:
                    continue
                s = closes[t].dropna()
                s0, s1 = s.asof(d), s.asof(end)
                if pd.isna(s0) or pd.isna(s1):
                    continue
                ret_sum += float(r["weight"]) * (float(s1) / float(s0) - 1.0)
                n_marked += 1
            basket_ret = ret_sum  # cash bucket (1 - exposure) returns 0
        rows.append((d.date(), len(df), n_marked, exposure,
                     "matured" if matured else f"open ({j - i}d)",
                     basket_ret, spy_ret, basket_ret - spy_ret))

    sc = pd.DataFrame(rows, columns=[
        "pick_date", "n_picks", "n_marked", "exposure", "status",
        "basket_ret", "spy_ret", "active_ret",
    ])
    _hr("6. LIVE PICKS SCORECARD (fills assumed at signal close; 21-trading-day windows)")
    print(f"  {'date':<12}{'picks':>6}{'expo':>7}{'status':>12}"
          f"{'basket':>9}{'SPY':>9}{'active':>9}")
    for _, r in sc.iterrows():
        print(f"  {str(r['pick_date']):<12}{r['n_picks']:>6}{r['exposure']:>7.0%}"
              f"{r['status']:>12}{r['basket_ret']:>+9.2%}{r['spy_ret']:>+9.2%}"
              f"{r['active_ret']:>+9.2%}")
    mat = sc[sc["status"] == "matured"]
    if len(mat):
        print(f"\n  Matured windows: {len(mat)}   avg active return "
              f"{mat['active_ret'].mean():+.2%}   "
              f"beat SPY in {(mat['active_ret'] > 0).mean():.0%}")
    return sc


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--top-n", type=int, default=strategy.TOP_N)
    ap.add_argument("--live-only", action="store_true",
                    help="Only the live picks scorecard (fast; skips panel load).")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--model", default=MODEL_PATH)
    args = ap.parse_args()

    os.makedirs(REPORTS_DIR, exist_ok=True)
    market_data = load_market()
    market = strategy.prepare_market(market_data["SPY"], market_data["VIX"])
    summary: dict = {"generated": str(pd.Timestamp.now()), "top_n": args.top_n}

    if not args.live_only:
        print("Loading panel and predicting on test slice...")
        panel = load_panel()
        test_panel = predict_test(panel, args.model)
        print(f"  test rows: {len(test_panel):,}  "
              f"({test_panel['date'].min().date()} → {test_panel['date'].max().date()})")

        monthly, ic_stats = ic_stability(test_panel)
        monthly.to_csv(os.path.join(REPORTS_DIR, "diagnostics_ic_monthly.csv"))
        summary["ic"] = ic_stats

        summary["underwater"] = underwater(args.no_plot)

        baskets = simulate_baskets(test_panel, args.top_n)
        summary["concentration"] = concentration(baskets)
        contrib = attribution(baskets)
        contrib.to_csv(os.path.join(REPORTS_DIR, "diagnostics_attribution.csv"))
        summary["hit_rate"] = hit_rate(baskets, market)

    sc = live_scorecard(market)
    if sc is not None:
        sc.to_csv(os.path.join(REPORTS_DIR, "diagnostics_live_scorecard.csv"),
                  index=False)
        mat = sc[sc["status"] == "matured"]
        summary["live"] = {
            "n_windows": int(len(sc)),
            "n_matured": int(len(mat)),
            "avg_active_ret_matured": float(mat["active_ret"].mean()) if len(mat) else None,
        }

    with open(os.path.join(REPORTS_DIR, "diagnostics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  -> summary: {REPORTS_DIR}/diagnostics_summary.json")


if __name__ == "__main__":
    main()
