#!/usr/bin/env python3
"""Point-in-time S&P 500 membership.

Sources
-------
1. **Membership history** — `data/universe/SP_500_Historical_Component.csv`
   from github.com/fja05680/sp500. One row per change-event date with the full
   members list as of that date. Going back to 1996.

2. **Current sectors** — Wikipedia scrape, only available for present-day
   members. Tickers that have since left the index get `gics_sector = Unknown`.

Caches
------
    data/universe/sp500_history.parquet   long format (date, ticker)
    data/universe/sp500_sectors.csv       Ticker, GICS Sector, ...

Public API
----------
    members_on(date)                tickers in the index on/just-before `date`
    all_historical_tickers(since=)  every ticker ever in the index
    filter_to_members(panel, ...)   drop rows where ticker wasn't in index on date
    load_sectors()                  Wikipedia sector tags (current members only)
    load_universe()                 back-compat: today's members + sectors
    get_tickers()                   back-compat: ticker list for today

CLI
---
    python scripts/universe.py             # build/refresh both caches
    python scripts/universe.py --refresh   # re-scrape Wikipedia sectors
"""

import io
import os
import sys
import time
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_UDIR = os.path.join(_ROOT, "data", "universe")

HISTORY_RAW = os.path.join(_UDIR, "SP_500_Historical_Component.csv")
HISTORY_FILE = os.path.join(_UDIR, "sp500_history.parquet")
SECTORS_FILE = os.path.join(_UDIR, "sp500_sectors.csv")

SECTORS_MAX_AGE_DAYS = 7
WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
UNKNOWN_SECTOR = "Unknown"


def _normalize_ticker(t: str) -> str:
    # yfinance uses '-' for class shares (BRK.B → BRK-B, RDS.A → RDS-A).
    return t.strip().replace(".", "-")


# ─────────────────────────────────────────────────────────────────────────────
# Membership history
# ─────────────────────────────────────────────────────────────────────────────


def _build_history_from_raw() -> pd.DataFrame:
    """Parse the change-event CSV into a long (date, ticker) DataFrame."""
    if not os.path.exists(HISTORY_RAW):
        raise SystemExit(
            f"{HISTORY_RAW} not found. Download from "
            f"github.com/fja05680/sp500 and place it there."
        )
    raw = pd.read_csv(HISTORY_RAW)
    raw["date"] = pd.to_datetime(raw["date"])
    rows: list[tuple[pd.Timestamp, str]] = []
    for d, tickers in zip(raw["date"], raw["tickers"]):
        for t in tickers.split(","):
            rows.append((d, _normalize_ticker(t)))
    return (
        pd.DataFrame(rows, columns=["date", "ticker"])
        .drop_duplicates()
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )


def load_history(rebuild: bool = False) -> pd.DataFrame:
    """Long-format (date, ticker) snapshots. Rebuilt from CSV when stale."""
    os.makedirs(_UDIR, exist_ok=True)
    if (
        not rebuild
        and os.path.exists(HISTORY_FILE)
        and os.path.getmtime(HISTORY_FILE) >= os.path.getmtime(HISTORY_RAW)
    ):
        return pd.read_parquet(HISTORY_FILE)

    df = _build_history_from_raw()
    df.to_parquet(HISTORY_FILE)
    print(
        f"Built membership history: {df['date'].nunique()} change-event dates, "
        f"{df['ticker'].nunique()} unique tickers ({df['date'].min().date()} → "
        f"{df['date'].max().date()}).",
        flush=True,
    )
    return df


def members_on(date, history: pd.DataFrame | None = None) -> list[str]:
    """Tickers in the S&P 500 on `date` (uses most-recent snapshot ≤ date)."""
    if history is None:
        history = load_history()
    snap_dates = np.sort(history["date"].unique())
    target = np.datetime64(pd.Timestamp(date), "ns")
    pos = snap_dates.searchsorted(target, side="right") - 1
    if pos < 0:
        return []
    snap_in_effect = snap_dates[pos]
    return history.loc[history["date"] == snap_in_effect, "ticker"].tolist()


def all_historical_tickers(
    since: str | pd.Timestamp | None = None,
    history: pd.DataFrame | None = None,
) -> list[str]:
    """Every ticker ever in the S&P 500. With `since`, only those present in
    at least one snapshot on/after that date."""
    if history is None:
        history = load_history()
    if since is not None:
        history = history[history["date"] >= pd.Timestamp(since)]
    return sorted(history["ticker"].unique())


def filter_to_members(
    panel: pd.DataFrame,
    history: pd.DataFrame | None = None,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """Keep only rows where `ticker` was an S&P 500 member on `date`.

    A ticker is a member at row date `d` iff it appears in the most recent
    membership snapshot whose date ≤ `d`. Handles add → remove → re-add
    correctly (re-adding doesn't backdate membership during the gap).
    """
    if history is None:
        history = load_history()

    snap_dates = pd.DatetimeIndex(np.sort(history["date"].unique()))
    snap_pos = {d: i for i, d in enumerate(snap_dates)}

    # For each ticker → sorted list of snapshot indices where it was present.
    ticker_idx: dict[str, list[int]] = (
        history.assign(_si=history["date"].map(snap_pos))
        .groupby("ticker")["_si"]
        .apply(lambda s: sorted(s.tolist()))
        .to_dict()
    )

    # Precompute end-of-effect timestamps: snap[i]'s membership applies until
    # snap[i+1] - 1 day (or +∞ for the last snapshot).
    snap_end_after = list(snap_dates[1:] - pd.Timedelta(days=1)) + [
        pd.Timestamp("2999-12-31")
    ]

    # Compress consecutive present-snapshots into (start_ts, end_ts) intervals.
    intervals: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for ticker, idxs in ticker_idx.items():
        spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        a = b = idxs[0]
        for k in idxs[1:]:
            if k == b + 1:
                b = k
            else:
                spans.append((snap_dates[a], snap_end_after[b]))
                a = b = k
        spans.append((snap_dates[a], snap_end_after[b]))
        intervals[ticker] = spans

    keep = np.zeros(len(panel), dtype=bool)
    for ticker, group in panel.groupby(ticker_col, sort=False):
        if ticker not in intervals:
            continue
        gd = group[date_col].to_numpy()
        m = np.zeros(len(group), dtype=bool)
        for start, end in intervals[ticker]:
            m |= (gd >= start.to_datetime64()) & (gd <= end.to_datetime64())
        keep[group.index.to_numpy()] = m

    return panel[keep].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sector tags (Wikipedia, current members only)
# ─────────────────────────────────────────────────────────────────────────────


def _sectors_cache_is_fresh() -> bool:
    if not os.path.exists(SECTORS_FILE):
        return False
    age_days = (time.time() - os.path.getmtime(SECTORS_FILE)) / 86400
    return age_days < SECTORS_MAX_AGE_DAYS


def _fetch_sectors_from_wikipedia() -> pd.DataFrame:
    print("Fetching sector tags from Wikipedia...", flush=True)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ml-stock-ranker/1.0)"}
    html = requests.get(WIKIPEDIA_URL, headers=headers, timeout=15).text
    df = pd.read_html(io.StringIO(html))[0]
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    keep = ["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]
    df = df[keep].rename(columns={"Symbol": "Ticker"})
    df["as_of"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return df.reset_index(drop=True)


def load_sectors(force_refresh: bool = False) -> pd.DataFrame:
    """Sector tags for current S&P 500 members (Wikipedia)."""
    os.makedirs(_UDIR, exist_ok=True)
    if not force_refresh and _sectors_cache_is_fresh():
        return pd.read_csv(SECTORS_FILE)
    df = _fetch_sectors_from_wikipedia()
    df.to_csv(SECTORS_FILE, index=False)
    print(f"Sectors cached to {SECTORS_FILE} ({len(df)} tickers).", flush=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Back-compat helpers (so `from universe import load_universe / get_tickers`
# keeps working for callers that wanted the current snapshot).
# ─────────────────────────────────────────────────────────────────────────────


def load_universe(force_refresh: bool = False) -> pd.DataFrame:
    """Today's members joined to Wikipedia sectors. Equivalent to v1 output."""
    sectors = load_sectors(force_refresh=force_refresh)
    today_members = set(members_on(pd.Timestamp.today()))
    return sectors[sectors["Ticker"].isin(today_members)].reset_index(drop=True)


def get_tickers(force_refresh: bool = False) -> list[str]:
    """Today's S&P 500 ticker list."""
    return members_on(pd.Timestamp.today())


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _summarize_history(history: pd.DataFrame) -> None:
    snap_dates = np.sort(history["date"].unique())
    today_members = members_on(pd.Timestamp.today(), history=history)
    all_tickers = history["ticker"].unique()
    since_2007 = all_historical_tickers(since="2007-01-01", history=history)
    print(
        f"\nHistory: {len(snap_dates)} change-event dates, "
        f"{len(all_tickers)} unique tickers, "
        f"{len(since_2007)} since 2007-01-01."
    )
    print(f"Latest snapshot: {pd.Timestamp(snap_dates[-1]).date()} — "
          f"{len(today_members)} members.")


def main() -> None:
    force_sectors = "--refresh" in sys.argv
    rebuild_history = "--rebuild-history" in sys.argv

    history = load_history(rebuild=rebuild_history)
    _summarize_history(history)

    sectors = load_sectors(force_refresh=force_sectors)
    print(f"\nSector tags: {len(sectors)} tickers (current Wikipedia roster).")
    print(sectors["GICS Sector"].value_counts().to_string())


if __name__ == "__main__":
    main()
