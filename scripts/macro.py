#!/usr/bin/env python3
"""Download FRED macro series for recession/crash regime features.

Series fetched (all daily, ≥2006 history, free, no auth required):
    DGS10           — 10y Treasury constant-maturity yield (%)
    DGS3MO          — 3-month Treasury constant-maturity yield (%)
    BAA10Y          — Moody's Baa-10y Treasury spread (IG credit stress, %)
    T5YIFR          — 5-Year, 5-Year Forward Inflation Expectation (%)

All are market-data series (yields/spreads, not survey output), so the FRED
snapshot matches what market participants saw on each historical date —
no revision lookahead. CSV blobs are tiny (~50KB each); we refetch in full
on every run rather than maintaining an incremental tail.

ICE BofA HY OAS (BAMLH0A0HYM2) was on the original wish-list but FRED's
free CSV endpoint now truncates every ICE BofA series to 2023-05→ (the
full history is gated behind an API key). BAA10Y is ~0.85 correlated with
HY OAS historically, so we lean on it as the credit-stress proxy until/
unless we wire up the FRED API.

Cache: data/market/macro.parquet (wide DataFrame, columns = series names,
date-indexed). features.py forward-fills across the panel's trading-day
index before computing the regime features (FRED occasionally skips a
business day for federal holidays even when SPY trades).

CLI:
    uv run python scripts/macro.py
"""

import argparse
import io
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import requests

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_ROOT = os.path.dirname(_HERE)
MARKET_DIR = os.path.join(_ROOT, "data", "market")
MACRO_PATH = os.path.join(MARKET_DIR, "macro.parquet")

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

MACRO_SERIES: list[str] = [
    "DGS10",
    "DGS3MO",
    "BAA10Y",
    "T5YIFR",
]

REQUEST_TIMEOUT = 30
RETRIES = 3
RETRY_SLEEP = 2.0


def _fetch_one(series: str) -> pd.Series | None:
    """Download one FRED series. Returns a date-indexed Series (NaN rows dropped)."""
    url = FRED_CSV_URL.format(series=series)
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.get(url, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            # FRED's column names vary across snapshots — 'DATE' (legacy) vs
            # 'observation_date' (current). Be robust to either.
            date_col = next(
                c for c in df.columns if c.lower() in ("observation_date", "date")
            )
            value_col = next(c for c in df.columns if c != date_col)
            # FRED encodes missing as "." — to_numeric(errors="coerce") drops them.
            s = pd.to_numeric(df[value_col], errors="coerce")
            s.index = pd.to_datetime(df[date_col])
            s.index.name = "date"
            s.name = series
            return s.dropna()
        except Exception as e:
            if attempt == RETRIES:
                print(
                    f"  [{series}] failed after {RETRIES} attempts: {e}", flush=True
                )
                return None
            time.sleep(RETRY_SLEEP * attempt)
    return None


def fetch_macro(series_list: list[str] = MACRO_SERIES) -> pd.DataFrame:
    """Download every FRED series, concat into a wide date-indexed parquet."""
    os.makedirs(MARKET_DIR, exist_ok=True)
    cols: dict[str, pd.Series] = {}
    for s in series_list:
        sr = _fetch_one(s)
        if sr is None or sr.empty:
            print(f"  {s}: download FAILED.", flush=True)
            continue
        cols[s] = sr
        print(
            f"  {s}: {len(sr):,} rows "
            f"({sr.index.min().date()} → {sr.index.max().date()})",
            flush=True,
        )

    if not cols:
        raise SystemExit("No macro series downloaded; check connectivity.")

    df = pd.concat(cols, axis=1).sort_index()
    df.to_parquet(MACRO_PATH)
    print(
        f"\n  → {MACRO_PATH} ({len(df):,} rows × {len(df.columns)} cols)",
        flush=True,
    )
    return df


def load_macro(path: str = MACRO_PATH) -> pd.DataFrame:
    """Load the cached macro parquet. Raises if absent."""
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Run scripts/macro.py first.")
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # Kept for parity with data.py's CLI surface even though the default
    # already refetches everything from scratch.
    ap.add_argument(
        "--refresh",
        action="store_true",
        help="(default) Refetch every FRED series. CSVs are tiny so we always do this.",
    )
    ap.parse_args()
    print(f"Fetching {len(MACRO_SERIES)} FRED series → {MACRO_PATH} ...")
    fetch_macro()


if __name__ == "__main__":
    main()
