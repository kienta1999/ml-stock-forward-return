#!/usr/bin/env python3
"""Download alternative data: SEC EDGAR earnings filings + FINRA short interest.

Two pipelines, one entry point. Kept separate from data.py because the cadences,
rate-limits, and failure modes are all different (yfinance is daily; EDGAR is
event-driven with a 10 req/sec courtesy cap; FINRA is bi-monthly static files).

Caches:
    data/earnings/{TICKER}.parquet      report_date, form, accession
    data/earnings/_cik_map.json         ticker → SEC CIK (10-digit zero-padded)
    data/earnings/_upcoming.parquet     yfinance forward calendar (ticker, date)
    data/short_interest/finra_{YYYYMMDD}.parquet   one file per release

Public loaders (used by features.py):
    load_earnings_dates()       → DataFrame[ticker, report_date, source]
    load_short_interest()       → DataFrame[ticker, publish_date, days_to_cover, ...]

CLI:
    python scripts/altdata.py                       # both, incremental
    python scripts/altdata.py --source edgar        # only EDGAR
    python scripts/altdata.py --source finra        # only FINRA
    python scripts/altdata.py --refresh             # wipe caches, redownload
    python scripts/altdata.py --tickers AAPL,MSFT   # subset (EDGAR only)
    python scripts/altdata.py --finra-start 2007-01 # FINRA history floor
"""

import argparse
import io
import json
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta

warnings.filterwarnings("ignore")

import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from universe import all_historical_tickers  # noqa: E402

_ROOT = os.path.dirname(_HERE)
EARNINGS_DIR = os.path.join(_ROOT, "data", "earnings")
SHORT_DIR = os.path.join(_ROOT, "data", "short_interest")
CIK_MAP_PATH = os.path.join(EARNINGS_DIR, "_cik_map.json")
UPCOMING_PATH = os.path.join(EARNINGS_DIR, "_upcoming.parquet")

# SEC asks for a descriptive User-Agent with contact info. They throttle
# anonymous traffic and 403 abusive clients. Keep below 10 req/sec.
SEC_HEADERS = {
    "User-Agent": "ml-stock-forward-return research bot",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_RATE_LIMIT_QPS = 8  # below the 10/s ceiling
SEC_WORKERS = 4

# Forms that mark a quarterly/annual earnings report. Filed ~30-60d after the
# period end and ~2-4 weeks after the actual 8-K announcement, but they're a
# stable, deterministic record of "an earnings event happened on this date".
# Pulling 8-K item 2.02 specifically would require fetching each filing's index
# (one request per 8-K) which would balloon the request budget.
EARNINGS_FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A"}

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
# EDGAR — ticker → CIK map
# ─────────────────────────────────────────────────────────────────────────────


def _normalize_for_sec(ticker: str) -> str:
    """SEC's company_tickers.json uses '.' for share classes (BRK.B), but our
    universe stores them as '-' (yfinance style). Convert when looking up."""
    return ticker.replace("-", ".").upper()


def _fetch_cik_map() -> dict[str, str]:
    """Pull the SEC's full ticker→CIK map and normalize to our ticker format."""
    headers = {**SEC_HEADERS, "Host": "www.sec.gov"}
    r = requests.get(SEC_TICKERS_URL, headers=headers, timeout=30)
    r.raise_for_status()
    raw = r.json()
    out: dict[str, str] = {}
    for entry in raw.values():
        sec_ticker = str(entry["ticker"]).upper()
        # Reverse: SEC uses '.' for classes; we use '-'.
        our_ticker = sec_ticker.replace(".", "-")
        cik = str(entry["cik_str"]).zfill(10)
        # If the same CIK has multiple tickers, keep the first.
        out.setdefault(our_ticker, cik)
    return out


def load_cik_map(refresh: bool = False) -> dict[str, str]:
    os.makedirs(EARNINGS_DIR, exist_ok=True)
    if not refresh and os.path.exists(CIK_MAP_PATH):
        with open(CIK_MAP_PATH) as f:
            return json.load(f)
    print("Fetching SEC ticker→CIK map...", flush=True)
    cik_map = _fetch_cik_map()
    with open(CIK_MAP_PATH, "w") as f:
        json.dump(cik_map, f)
    print(f"  cached {len(cik_map):,} ticker→CIK mappings to {CIK_MAP_PATH}", flush=True)
    return cik_map


# ─────────────────────────────────────────────────────────────────────────────
# EDGAR — per-ticker submissions JSON → earnings filing dates
# ─────────────────────────────────────────────────────────────────────────────


_LAST_SEC_REQUEST = [0.0]  # mutable guard for cross-thread courtesy throttle


def _sec_throttle() -> None:
    """Cheap shared-state rate limiter. Keeps us under ~10 req/sec across threads."""
    interval = 1.0 / SEC_RATE_LIMIT_QPS
    now = time.monotonic()
    wait = interval - (now - _LAST_SEC_REQUEST[0])
    if wait > 0:
        time.sleep(wait)
    _LAST_SEC_REQUEST[0] = time.monotonic()


def _sec_get(url: str) -> requests.Response | None:
    """GET with retry and politeness throttle."""
    for attempt in range(1, RETRIES + 1):
        try:
            _sec_throttle()
            r = requests.get(url, headers=SEC_HEADERS, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt == RETRIES:
                print(f"  [{url}] failed after {RETRIES} retries: {e}", flush=True)
                return None
            time.sleep(RETRY_SLEEP * attempt)
    return None


def _earnings_dates_from_submission(payload: dict) -> list[tuple[str, str, str]]:
    """Extract (filingDate, form, accessionNumber) for earnings forms only."""
    out: list[tuple[str, str, str]] = []
    recent = payload.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accs = recent.get("accessionNumber", [])
    for f, d, a in zip(forms, dates, accs):
        if f in EARNINGS_FORMS:
            out.append((d, f, a))
    return out


def _fetch_extra_submission_files(payload: dict) -> list[dict]:
    """SEC paginates filings older than ~1000 entries into separate JSON files
    listed under filings.files[].name. Pull each one."""
    extras: list[dict] = []
    files = payload.get("filings", {}).get("files", [])
    for f in files:
        name = f.get("name")
        if not name:
            continue
        url = f"https://data.sec.gov/submissions/{name}"
        r = _sec_get(url)
        if r is None:
            continue
        try:
            extras.append(r.json())
        except Exception:
            continue
    return extras


def _earnings_path(ticker: str) -> str:
    return os.path.join(EARNINGS_DIR, f"{ticker}.parquet")


def fetch_earnings_for_ticker(
    ticker: str,
    cik_map: dict[str, str],
    refresh: bool = False,
) -> tuple[str, int, str]:
    """Fetch all 10-Q/10-K filing dates for a ticker. Returns (ticker, n, status).

    status ∈ {"new", "fresh", "missing_cik", "failed"}
    """
    path = _earnings_path(ticker)
    if not refresh and os.path.exists(path):
        try:
            cached = pd.read_parquet(path)
            return ticker, len(cached), "fresh"
        except Exception:
            pass

    cik = cik_map.get(ticker)
    if cik is None:
        return ticker, 0, "missing_cik"

    r = _sec_get(SEC_SUBMISSIONS_URL.format(cik=cik))
    if r is None:
        return ticker, 0, "failed"
    payload = r.json()

    rows = _earnings_dates_from_submission(payload)
    # SEC paginated files (filings.files[].name) are flat objects with the same
    # parallel-array shape as filings.recent — no wrapping under .filings.recent.
    for extra in _fetch_extra_submission_files(payload):
        flat = extra if "form" in extra else extra.get("filings", {}).get("recent", {})
        forms = flat.get("form", [])
        dates = flat.get("filingDate", [])
        accs = flat.get("accessionNumber", [])
        for f, d, a in zip(forms, dates, accs):
            if f in EARNINGS_FORMS:
                rows.append((d, f, a))

    if not rows:
        # No earnings filings on record — still cache an empty frame so we
        # don't re-pull every run.
        df = pd.DataFrame(columns=["report_date", "form", "accession"])
    else:
        df = pd.DataFrame(rows, columns=["report_date", "form", "accession"])
        df["report_date"] = pd.to_datetime(df["report_date"])
        df = df.drop_duplicates(subset=["accession"]).sort_values("report_date").reset_index(drop=True)

    df.to_parquet(path)
    return ticker, len(df), "new"


def fetch_earnings_universe(
    tickers: list[str],
    refresh: bool = False,
) -> pd.DataFrame:
    """Pull EDGAR submissions for every ticker. Returns status summary."""
    os.makedirs(EARNINGS_DIR, exist_ok=True)
    cik_map = load_cik_map(refresh=refresh)
    results: list[tuple[str, int, str]] = []

    with ThreadPoolExecutor(max_workers=SEC_WORKERS) as pool:
        futures = {
            pool.submit(fetch_earnings_for_ticker, t, cik_map, refresh): t
            for t in tickers
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="EDGAR"):
            results.append(fut.result())

    summary = pd.DataFrame(results, columns=["Ticker", "Rows", "Status"])
    counts = summary["Status"].value_counts().to_dict()
    print(
        f"\nEDGAR done. new={counts.get('new', 0)}  fresh={counts.get('fresh', 0)}  "
        f"missing_cik={counts.get('missing_cik', 0)}  failed={counts.get('failed', 0)}",
        flush=True,
    )
    missing = summary[summary["Status"] == "missing_cik"]
    if len(missing):
        print(
            f"  {len(missing)} tickers had no SEC CIK (mostly historical/delisted): "
            + ", ".join(missing["Ticker"].head(20).tolist())
            + (" ..." if len(missing) > 20 else ""),
            flush=True,
        )
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# yfinance — forward earnings calendar (for live picks only)
# ─────────────────────────────────────────────────────────────────────────────


def fetch_upcoming_earnings(tickers: list[str]) -> pd.DataFrame:
    """yfinance.Ticker(t).get_earnings_dates() returns past + upcoming events.
    We only keep dates ≥ today — the past ones come from EDGAR.
    """
    today = pd.Timestamp.today().normalize()
    rows: list[tuple[str, pd.Timestamp]] = []
    failed: list[str] = []

    for t in tqdm(tickers, desc="yfinance upcoming"):
        try:
            df = yf.Ticker(t).get_earnings_dates(limit=8)
            if df is None or df.empty:
                continue
            idx = pd.to_datetime(df.index).tz_localize(None)
            for d in idx:
                if pd.Timestamp(d).normalize() >= today:
                    rows.append((t, pd.Timestamp(d).normalize()))
        except Exception:
            failed.append(t)

    out = (
        pd.DataFrame(rows, columns=["ticker", "report_date"])
        .drop_duplicates()
        .sort_values(["ticker", "report_date"])
        .reset_index(drop=True)
    )
    os.makedirs(EARNINGS_DIR, exist_ok=True)
    out.to_parquet(UPCOMING_PATH, index=False)
    print(
        f"\nUpcoming earnings: {len(out)} dates across "
        f"{out['ticker'].nunique()} tickers ({len(failed)} yfinance failures)",
        flush=True,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# EDGAR — public loader
# ─────────────────────────────────────────────────────────────────────────────


def load_earnings_dates(include_upcoming: bool = True) -> pd.DataFrame:
    """All cached earnings dates (historical EDGAR + optional yfinance upcoming).

    Returns long-format DataFrame[ticker, report_date, source].
    """
    if not os.path.isdir(EARNINGS_DIR):
        return pd.DataFrame(columns=["ticker", "report_date", "source"])

    frames: list[pd.DataFrame] = []
    for fname in os.listdir(EARNINGS_DIR):
        if not fname.endswith(".parquet") or fname.startswith("_"):
            continue
        ticker = fname.removesuffix(".parquet")
        try:
            df = pd.read_parquet(os.path.join(EARNINGS_DIR, fname))
        except Exception:
            continue
        if df.empty:
            continue
        df = df[["report_date"]].copy()
        df["ticker"] = ticker
        df["source"] = "edgar"
        frames.append(df)

    hist = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["ticker", "report_date", "source"]
    )

    if include_upcoming and os.path.exists(UPCOMING_PATH):
        try:
            up = pd.read_parquet(UPCOMING_PATH)
            up["source"] = "yfinance"
            hist = pd.concat([hist, up[["ticker", "report_date", "source"]]], ignore_index=True)
        except Exception:
            pass

    if hist.empty:
        return hist
    hist["report_date"] = pd.to_datetime(hist["report_date"])
    return (
        hist.drop_duplicates(subset=["ticker", "report_date"])
        .sort_values(["ticker", "report_date"])
        .reset_index(drop=True)
    )


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
    ap.add_argument(
        "--source",
        choices=["edgar", "finra", "both"],
        default="both",
        help="Which pipeline to run (default: both).",
    )
    ap.add_argument("--refresh", action="store_true", help="Wipe caches and redownload.")
    ap.add_argument("--tickers", help="Comma-separated subset (EDGAR only).")
    ap.add_argument("--skip-upcoming", action="store_true",
                    help="Skip yfinance forward-earnings pull (EDGAR only).")
    ap.add_argument("--finra-start", default=FINRA_DEFAULT_START,
                    help=f"FINRA history floor as YYYY-MM (default: {FINRA_DEFAULT_START}).")
    args = ap.parse_args()

    if args.source in ("edgar", "both"):
        if args.tickers:
            tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        else:
            tickers = all_historical_tickers(since="2005-07-01")
        print(f"\nEDGAR: fetching submissions for {len(tickers)} tickers...", flush=True)
        fetch_earnings_universe(tickers, refresh=args.refresh)

        if not args.skip_upcoming:
            print("\nyfinance: pulling forward earnings calendar...", flush=True)
            # Forward calendar only meaningful for current S&P 500 members
            # (delisted tickers don't report future earnings).
            from universe import get_tickers
            fetch_upcoming_earnings(get_tickers())

    if args.source in ("finra", "both"):
        print(f"\nFINRA: pulling bi-monthly short interest from {args.finra_start}...", flush=True)
        fetch_finra_history(start_ym=args.finra_start, refresh=args.refresh)


if __name__ == "__main__":
    main()
