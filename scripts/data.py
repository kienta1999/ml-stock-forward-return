#!/usr/bin/env python3
"""Download and cache OHLCV history for the S&P 500 universe + market series.

Per-ticker parquets:   data/raw/{TICKER}.parquet
Market series:         data/market/SPY.parquet, data/market/VIX.parquet

Behaviour:
  * Default range: 2007-01-01 → today.
  * Incremental: if a parquet exists, only the tail since last cached date is
    downloaded and appended.
  * --refresh wipes a ticker's cache and redownloads from scratch.
  * Tickers with fewer than MIN_HISTORY_DAYS rows are flagged but still cached
    (downstream modules filter them out via load_prices()).

CLI:
    python scripts/data.py                      # universe + SPY + VIX, incremental
    python scripts/data.py --refresh            # full redownload of everything
    python scripts/data.py --tickers AAPL,MSFT  # subset
    python scripts/data.py --start 2010-01-01   # custom start
    python scripts/data.py --skip-universe      # only refresh SPY/VIX
"""

import argparse
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

warnings.filterwarnings("ignore")

import pandas as pd
import yfinance as yf
from tqdm import tqdm

# Make sibling scripts importable when run from anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from universe import get_tickers  # noqa: E402

_ROOT = os.path.dirname(_HERE)
RAW_DIR = os.path.join(_ROOT, "data", "raw")
MARKET_DIR = os.path.join(_ROOT, "data", "market")

DEFAULT_START = "2007-01-01"
MIN_HISTORY_DAYS = 500
WORKERS = 8
RETRIES = 3
RETRY_SLEEP = 2.0

# yfinance ticker mapping for symbols that aren't valid as-is.
SPY_TICKER = "SPY"
VIX_TICKER = "^VIX"


# ─────────────────────────────────────────────────────────────────────────────
# Low-level download
# ─────────────────────────────────────────────────────────────────────────────


def _download_one(
    ticker: str,
    start: str,
    end: str | None = None,
) -> pd.DataFrame | None:
    """Download a single ticker with retry. Returns None on hard failure."""
    for attempt in range(1, RETRIES + 1):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if df is None or df.empty:
                return None
            # yfinance can return MultiIndex columns for single tickers (newer versions).
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "Date"
            return df
        except Exception as e:
            if attempt == RETRIES:
                print(f"  [{ticker}] failed after {RETRIES} retries: {e}", flush=True)
                return None
            time.sleep(RETRY_SLEEP * attempt)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Per-ticker cache (incremental)
# ─────────────────────────────────────────────────────────────────────────────


def _cache_path(ticker: str) -> str:
    return os.path.join(RAW_DIR, f"{ticker}.parquet")


def _load_cached(ticker: str) -> pd.DataFrame | None:
    p = _cache_path(ticker)
    if not os.path.exists(p):
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None


def _save_cached(ticker: str, df: pd.DataFrame) -> None:
    df.to_parquet(_cache_path(ticker))


def fetch_ticker(
    ticker: str,
    start: str = DEFAULT_START,
    end: str | None = None,
    refresh: bool = False,
) -> tuple[str, int, str]:
    """Fetch a ticker's history into cache. Returns (ticker, n_rows, status).

    status ∈ {"new", "updated", "fresh", "failed", "short"}
    """
    end = end or date.today().isoformat()
    cached = None if refresh else _load_cached(ticker)

    if cached is not None and len(cached) > 0:
        last_cached = cached.index.max().date()
        # Already up to date (last cached date is today or yesterday on a weekend).
        if last_cached >= date.today() - timedelta(days=1):
            status = "fresh" if len(cached) >= MIN_HISTORY_DAYS else "short"
            return ticker, len(cached), status

        tail_start = (last_cached + timedelta(days=1)).isoformat()
        new = _download_one(ticker, start=tail_start, end=end)
        if new is None or new.empty:
            status = "fresh" if len(cached) >= MIN_HISTORY_DAYS else "short"
            return ticker, len(cached), status

        merged = pd.concat([cached, new])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        _save_cached(ticker, merged)
        status = "updated" if len(merged) >= MIN_HISTORY_DAYS else "short"
        return ticker, len(merged), status

    # No cache (or --refresh): full download.
    df = _download_one(ticker, start=start, end=end)
    if df is None or df.empty:
        return ticker, 0, "failed"
    _save_cached(ticker, df)
    status = "new" if len(df) >= MIN_HISTORY_DAYS else "short"
    return ticker, len(df), status


# ─────────────────────────────────────────────────────────────────────────────
# Universe-level fetch (parallel)
# ─────────────────────────────────────────────────────────────────────────────


def fetch_universe(
    tickers: list[str],
    start: str = DEFAULT_START,
    refresh: bool = False,
) -> pd.DataFrame:
    """Download/refresh all tickers in parallel. Returns a status DataFrame."""
    os.makedirs(RAW_DIR, exist_ok=True)
    results: list[tuple[str, int, str]] = []

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(fetch_ticker, t, start=start, refresh=refresh): t
            for t in tickers
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Tickers"):
            results.append(fut.result())

    summary = pd.DataFrame(results, columns=["Ticker", "Rows", "Status"])
    summary = summary.sort_values(["Status", "Ticker"]).reset_index(drop=True)

    counts = summary["Status"].value_counts().to_dict()
    print(
        f"\nDone. new={counts.get('new', 0)}  updated={counts.get('updated', 0)}  "
        f"fresh={counts.get('fresh', 0)}  short={counts.get('short', 0)}  "
        f"failed={counts.get('failed', 0)}",
        flush=True,
    )

    short = summary[summary["Status"] == "short"]
    if len(short):
        print(
            f"\n{len(short)} tickers have <{MIN_HISTORY_DAYS} rows and will be "
            f"excluded by load_prices():\n  "
            + ", ".join(short["Ticker"].tolist()),
            flush=True,
        )

    failed = summary[summary["Status"] == "failed"]
    if len(failed):
        print(
            f"\n{len(failed)} tickers failed to download:\n  "
            + ", ".join(failed["Ticker"].tolist()),
            flush=True,
        )
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Market series (SPY, VIX)
# ─────────────────────────────────────────────────────────────────────────────


def fetch_market(
    start: str = DEFAULT_START,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Download SPY and ^VIX into data/market/. Returns {name: df}."""
    os.makedirs(MARKET_DIR, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}

    for label, ticker in [("SPY", SPY_TICKER), ("VIX", VIX_TICKER)]:
        path = os.path.join(MARKET_DIR, f"{label}.parquet")
        if not refresh and os.path.exists(path):
            cached = pd.read_parquet(path)
            last_cached = cached.index.max().date()
            if last_cached >= date.today() - timedelta(days=1):
                print(f"{label}: cache fresh ({len(cached)} rows).", flush=True)
                out[label] = cached
                continue
            tail_start = (last_cached + timedelta(days=1)).isoformat()
            new = _download_one(ticker, start=tail_start)
            if new is not None and not new.empty:
                cached = pd.concat([cached, new])
                cached = cached[~cached.index.duplicated(keep="last")].sort_index()
            cached.to_parquet(path)
            print(f"{label}: updated ({len(cached)} rows).", flush=True)
            out[label] = cached
            continue

        df = _download_one(ticker, start=start)
        if df is None or df.empty:
            print(f"{label}: download FAILED.", flush=True)
            continue
        df.to_parquet(path)
        print(f"{label}: downloaded ({len(df)} rows).", flush=True)
        out[label] = df

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public loader (used by features.py / dataset.py later)
# ─────────────────────────────────────────────────────────────────────────────


def load_prices(
    tickers: list[str] | None = None,
    min_history: int = MIN_HISTORY_DAYS,
) -> dict[str, pd.DataFrame]:
    """Load cached OHLCV for the given tickers (or all on disk).

    Returns dict {ticker: DataFrame[Open,High,Low,Close,Volume]}. Tickers with
    fewer than `min_history` rows are excluded.
    """
    if tickers is None:
        tickers = [
            f.removesuffix(".parquet")
            for f in os.listdir(RAW_DIR)
            if f.endswith(".parquet")
        ]
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        df = _load_cached(t)
        if df is None or len(df) < min_history:
            continue
        out[t] = df
    return out


def load_market() -> dict[str, pd.DataFrame]:
    """Load cached SPY and VIX series."""
    out: dict[str, pd.DataFrame] = {}
    for label in ("SPY", "VIX"):
        p = os.path.join(MARKET_DIR, f"{label}.parquet")
        if os.path.exists(p):
            out[label] = pd.read_parquet(p)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--refresh", action="store_true", help="Wipe caches and redownload")
    ap.add_argument("--tickers", help="Comma-separated subset (default: full universe)")
    ap.add_argument("--skip-universe", action="store_true")
    ap.add_argument("--skip-market", action="store_true")
    args = ap.parse_args()

    if not args.skip_universe:
        if args.tickers:
            tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        else:
            tickers = get_tickers()
        print(
            f"Fetching {len(tickers)} tickers from {args.start} "
            f"({WORKERS} workers, refresh={args.refresh})...",
            flush=True,
        )
        fetch_universe(tickers, start=args.start, refresh=args.refresh)

    if not args.skip_market:
        print("\nFetching market series (SPY, VIX)...", flush=True)
        fetch_market(start=args.start, refresh=args.refresh)


if __name__ == "__main__":
    main()
