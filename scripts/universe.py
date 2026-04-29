#!/usr/bin/env python3
"""Fetch and cache the current S&P 500 ticker list from Wikipedia.

Output: data/universe/sp500_members.csv with columns:
    Ticker, Security, GICS Sector, GICS Sub-Industry, as_of

Cache is refreshed every 7 days. Run with --refresh to force.

------------------------------------------------------------------------------
SURVIVORSHIP BIAS — KNOWN ISSUE (v1)
------------------------------------------------------------------------------
This module returns the *current* S&P 500 constituents and applies that list
historically. Stocks that were removed from the index (acquired, bankrupted,
delisted) are missing from the backtest universe, which biases performance
upward. To fix in v2, replace this with point-in-time membership snapshots
(e.g. a monthly history of S&P 500 additions/removals).
TODO(v2): point-in-time membership.
------------------------------------------------------------------------------
"""

import io
import os
import sys
import time
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

import pandas as pd
import requests

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_FILE = os.path.join(_ROOT, "data", "universe", "sp500_members.csv")
CACHE_MAX_AGE_DAYS = 7
WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def _cache_is_fresh() -> bool:
    if not os.path.exists(CACHE_FILE):
        return False
    age_days = (time.time() - os.path.getmtime(CACHE_FILE)) / 86400
    return age_days < CACHE_MAX_AGE_DAYS


def _fetch_from_wikipedia() -> pd.DataFrame:
    print("Fetching S&P 500 constituent list from Wikipedia...", flush=True)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ml-stock-ranker/1.0)"}
    html = requests.get(WIKIPEDIA_URL, headers=headers, timeout=15).text
    df = pd.read_html(io.StringIO(html))[0]

    # Yahoo uses '-' for class shares (e.g. BRK.B -> BRK-B), match scanner convention.
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)

    keep = ["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]
    df = df[keep].rename(columns={"Symbol": "Ticker"})
    df["as_of"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return df.reset_index(drop=True)


def load_universe(force_refresh: bool = False) -> pd.DataFrame:
    """Return current S&P 500 members, using cache when fresh."""
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)

    if not force_refresh and _cache_is_fresh():
        df = pd.read_csv(CACHE_FILE)
        age_days = (time.time() - os.path.getmtime(CACHE_FILE)) / 86400
        print(
            f"Loaded universe from cache (age: {age_days:.1f} days, {len(df)} tickers). "
            f"Refresh in {CACHE_MAX_AGE_DAYS - age_days:.1f} days or run with --refresh.",
            flush=True,
        )
        return df

    df = _fetch_from_wikipedia()
    df.to_csv(CACHE_FILE, index=False)
    print(f"Universe cached to {CACHE_FILE} ({len(df)} tickers).", flush=True)
    return df


def get_tickers(force_refresh: bool = False) -> list[str]:
    """Convenience: just the ticker list."""
    return load_universe(force_refresh=force_refresh)["Ticker"].tolist()


if __name__ == "__main__":
    force = "--refresh" in sys.argv
    df = load_universe(force_refresh=force)
    print(f"\n{len(df)} tickers. First 10:")
    print(df.head(10).to_string(index=False))
    print("\nSector breakdown:")
    print(df["GICS Sector"].value_counts().to_string())
