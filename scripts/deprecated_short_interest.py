#!/usr/bin/env python3
"""DEPRECATED — FINRA bi-monthly short interest pipeline.

Not currently wired into features.py. Kept on disk because the download
infrastructure works and the cache exists; the feature was deferred because
the FINRA CDN archive only resolves from ~2018-08 onward, leaving train=2007–2017
with 0% coverage and XGBoost unable to build any tree splits on the column.

See README §1 (Next steps) for the path back in: paid historical short
interest (Sharadar/Polygon/QuantQuote) or a re-platformed train/val/test
split that absorbs 2018+ into train.

Cache:
    data/short_interest/finra_{YYYYMMDD}.parquet   one file per release

Public loader (currently unused by features.py):
    load_short_interest()       → DataFrame[ticker, publish_date, days_to_cover, ...]

CLI:
    python scripts/deprecated_short_interest.py                       # incremental
    python scripts/deprecated_short_interest.py --refresh             # redownload
    python scripts/deprecated_short_interest.py --finra-start 2007-01 # history floor
"""

import argparse
import io
import os
import sys
import time
import warnings
from datetime import date

warnings.filterwarnings("ignore")

import pandas as pd
import requests
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_ROOT = os.path.dirname(_HERE)
SHORT_DIR = os.path.join(_ROOT, "data", "short_interest")

# FINRA short interest publication endpoint. Bi-monthly settlement dates: the
# 15th and last business day of each month, with weekend/holiday shifts. We
# enumerate a generous window per month and accept whatever 200s.
# File is pipe-delimited despite the .csv extension.
# Archive on FINRA's CDN only goes back to 2014-01.
FINRA_URL = "https://cdn.finra.org/equity/otcmarket/biweekly/shrt{ymd}.csv"
FINRA_PUBLISH_LAG_BDAYS = 8  # FINRA publishes ~8 business days after settlement
FINRA_HEADERS = {
    "User-Agent": "ml-stock-forward-return research bot",
    "Accept": "text/csv,text/plain,*/*",
}
# FINRA's CDN archive starts mid-2018. The page advertises "back to 2014" but
# direct CDN URLs only resolve from ~Aug 2018 onward; earlier dates 403.
FINRA_DEFAULT_START = "2018-08"

RETRIES = 3
RETRY_SLEEP = 2.0


# ─────────────────────────────────────────────────────────────────────────────
# FINRA — bi-monthly short interest files
# ─────────────────────────────────────────────────────────────────────────────


# Candidate days within each month. FINRA settlement dates are the 15th and
# last business day, but holidays shift them. Trying every plausible day costs
# us ~14 HTTP HEADs per month vs hand-maintaining a calendar.
_FINRA_DAY_CANDIDATES = [13, 14, 15, 16, 17, 18, 26, 27, 28, 29, 30, 31, 1]


def _finra_path(yyyymmdd: str) -> str:
    return os.path.join(SHORT_DIR, f"finra_{yyyymmdd}.parquet")


def _try_finra_download(yyyymmdd: str) -> pd.DataFrame | None:
    """Fetch+parse one settlement-date file. Returns None for missing files.

    FINRA's CDN (CloudFront) returns 403 for keys that don't exist (no
    ListBucket permission), so we treat 403 the same as 404 — settlement files
    don't exist for non-trading days, and the archive only goes back to ~2018-08.
    Only retry on real network/5xx errors.
    """
    url = FINRA_URL.format(ymd=yyyymmdd)
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.get(url, headers=FINRA_HEADERS, timeout=30)
            if r.status_code in (403, 404):
                return None  # file not present at this date — silent skip
            r.raise_for_status()
            text = r.text
            # FINRA files are pipe-delimited with a header row.
            df = pd.read_csv(
                io.StringIO(text),
                sep="|",
                dtype=str,
                on_bad_lines="skip",
                engine="python",
            )
            return df
        except requests.exceptions.RequestException as e:
            # Real network error — retry with backoff.
            if attempt == RETRIES:
                print(f"  [{yyyymmdd}] network error after {RETRIES} retries: {e}", flush=True)
                return None
            time.sleep(RETRY_SLEEP * attempt)
        except Exception as e:
            print(f"  [{yyyymmdd}] parse error: {e}", flush=True)
            return None
    return None


def _normalize_finra_columns(df: pd.DataFrame) -> pd.DataFrame:
    """FINRA column names use camelCase since the 2014+ format. Map them to a
    consistent snake_case schema: settlement_date, ticker, current_short,
    avg_daily_volume, days_to_cover.
    """
    rename = {}
    for c in df.columns:
        cl = c.strip().lower().replace(" ", "").replace("_", "")
        if cl in ("settlementdate",):
            rename[c] = "settlement_date"
        elif cl in ("symbolcode", "symbol"):
            # Prefer "Symbol" if present; "symbolCode" is the canonical 2014+ name.
            if cl == "symbol" or "ticker" not in rename.values():
                rename[c] = "ticker"
        elif cl in ("currentshortpositionquantity", "currentshortinterest"):
            rename[c] = "current_short"
        elif cl in ("averagedailyvolumequantity", "avgdailyvolume"):
            rename[c] = "avg_daily_volume"
        elif cl in ("daystocoverquantity", "daystocover"):
            rename[c] = "days_to_cover"
    df = df.rename(columns=rename)

    keep = [c for c in ["settlement_date", "ticker", "current_short",
                        "avg_daily_volume", "days_to_cover"] if c in df.columns]
    df = df[keep].copy()

    if "settlement_date" in df.columns:
        df["settlement_date"] = pd.to_datetime(df["settlement_date"], errors="coerce")
    for c in ("current_short", "avg_daily_volume", "days_to_cover"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
        # Match yfinance share-class convention.
        df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)

    df = df.dropna(subset=["settlement_date", "ticker"]).reset_index(drop=True)
    return df


def _candidate_settlement_dates(start_ym: str, end_ym: str) -> list[str]:
    """Generate YYYYMMDD strings worth trying between start_ym and end_ym."""
    start = pd.Timestamp(start_ym + "-01")
    end = pd.Timestamp(end_ym + "-01") + pd.offsets.MonthEnd(1)
    cur = start
    dates: list[str] = []
    while cur <= end:
        y, m = cur.year, cur.month
        for d in _FINRA_DAY_CANDIDATES:
            try:
                ts = pd.Timestamp(year=y, month=m, day=d)
            except ValueError:
                continue
            if start <= ts <= end:
                dates.append(ts.strftime("%Y%m%d"))
        cur = (cur + pd.offsets.MonthBegin(1))
    return sorted(set(dates))


def fetch_finra_history(
    start_ym: str = FINRA_DEFAULT_START,
    end_ym: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Download every available bi-monthly short-interest file in [start, end].

    Skips dates already cached unless --refresh. Returns status summary.
    """
    os.makedirs(SHORT_DIR, exist_ok=True)
    if end_ym is None:
        end_ym = date.today().strftime("%Y-%m")

    candidates = _candidate_settlement_dates(start_ym, end_ym)
    results: list[tuple[str, int, str]] = []

    for ymd in tqdm(candidates, desc="FINRA"):
        path = _finra_path(ymd)
        if not refresh and os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                results.append((ymd, len(df), "fresh"))
                continue
            except Exception:
                pass

        raw = _try_finra_download(ymd)
        if raw is None:
            results.append((ymd, 0, "missing"))
            continue

        df = _normalize_finra_columns(raw)
        if df.empty:
            results.append((ymd, 0, "empty"))
            continue
        df.to_parquet(path)
        results.append((ymd, len(df), "new"))

    summary = pd.DataFrame(results, columns=["Date", "Rows", "Status"])
    counts = summary["Status"].value_counts().to_dict()
    print(
        f"\nFINRA done. new={counts.get('new', 0)}  fresh={counts.get('fresh', 0)}  "
        f"missing={counts.get('missing', 0)}  empty={counts.get('empty', 0)}",
        flush=True,
    )
    return summary


def load_short_interest(min_year: int | None = 2007) -> pd.DataFrame:
    """Concatenate every cached FINRA file into a long panel.

    Returns DataFrame[ticker, settlement_date, publish_date, current_short,
    avg_daily_volume, days_to_cover]. publish_date = settlement_date + 8 BDays
    so downstream features can asof-merge without lookahead.
    """
    if not os.path.isdir(SHORT_DIR):
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for fname in sorted(os.listdir(SHORT_DIR)):
        if not (fname.startswith("finra_") and fname.endswith(".parquet")):
            continue
        try:
            df = pd.read_parquet(os.path.join(SHORT_DIR, fname))
        except Exception:
            continue
        if df.empty:
            continue
        if min_year is not None and df["settlement_date"].dt.year.min() < min_year:
            df = df[df["settlement_date"].dt.year >= min_year]
        if df.empty:
            continue
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames, ignore_index=True)
    bday = pd.tseries.offsets.BusinessDay(FINRA_PUBLISH_LAG_BDAYS)
    panel["publish_date"] = panel["settlement_date"] + bday
    return (
        panel.sort_values(["ticker", "publish_date"])
        .drop_duplicates(subset=["ticker", "settlement_date"], keep="last")
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refresh", action="store_true", help="Wipe caches and redownload.")
    ap.add_argument("--finra-start", default=FINRA_DEFAULT_START,
                    help=f"FINRA history floor as YYYY-MM (default: {FINRA_DEFAULT_START}).")
    args = ap.parse_args()

    print(f"\nFINRA: pulling bi-monthly short interest from {args.finra_start}...", flush=True)
    fetch_finra_history(start_ym=args.finra_start, refresh=args.refresh)


if __name__ == "__main__":
    main()
