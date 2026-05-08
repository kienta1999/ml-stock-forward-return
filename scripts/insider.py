#!/usr/bin/env python3
"""Download insider transactions from SEC's quarterly Form 3/4/5 bulk dataset.

The SEC's DERA team publishes parsed Form 3/4/5 data as quarterly TSV zips at
https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/.
Each ~5–10 MB zip contains SUBMISSION.tsv (filing metadata + ticker) and
NONDERIV_TRANS.tsv (one row per non-derivative transaction). One HTTPS GET per
quarter beats parsing every Form 4 XML individually — full S&P 500 history is
~80 quarters total, downloads in minutes, and stays well below SEC's 10 req/s
limit.

Discipline: filing_date (not transaction_date) is the asof key — Form 4 must
be filed within 2 business days of the transaction, so using the filing date
gives clean point-in-time data. Features computed from this data cannot look
ahead.

Only direct-ownership non-derivative P (open-market purchase) and S (open-
market sale) transactions are kept. Awards, grants, option exercises, gifts,
and indirect-ownership filings carry little open-market timing signal.

SEC publishes each quarter's zip ~1 month after the quarter closes. Live picks
already lag features by 21d, so accept the publication lag — no per-ticker
XML scraping needed for recent days.

Caches:
    data/insider/_bulk/{YYYY}q{N}_form345.zip   raw quarterly zips
    data/insider/{TICKER}.parquet               filing_date, transaction_date,
                                                transaction_code, shares,
                                                price_per_share, value

Incremental: each run downloads any quarters missing from the bulk cache, plus
always re-downloads the latest available quarter (it gets updated within-quarter
as new filings arrive). Per-ticker parquets are then rebuilt from the union of
zips. If you skip several runs, missing quarters are picked up automatically.

Public loader (used by features.py):
    load_insider_transactions()  → DataFrame[ticker, filing_date,
                                             transaction_code, shares,
                                             price_per_share, value]

CLI:
    uv run python scripts/insider.py                       # incremental
    uv run python scripts/insider.py --refresh             # wipe + redownload
    uv run python scripts/insider.py --tickers AAPL,MSFT   # rebuild subset
                                                           # (uses cached zips)
"""

import argparse
import os
import shutil
import sys
import time
import warnings
import zipfile

warnings.filterwarnings("ignore")

import pandas as pd
import requests
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from earnings import _sec_throttle  # noqa: E402
from universe import all_historical_tickers  # noqa: E402

_ROOT = os.path.dirname(_HERE)
INSIDER_DIR = os.path.join(_ROOT, "data", "insider")
BULK_DIR = os.path.join(INSIDER_DIR, "_bulk")

BULK_BASE_URL = (
    "https://www.sec.gov/files/structureddata/data/"
    "insider-transactions-data-sets/{q}_form345.zip"
)
BULK_HEADERS = {
    "User-Agent": "ml-stock-forward-return research bot talekien1710@gmail.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov",
}

# SEC began publishing the Form 345 data sets in 2006q1.
START_YEAR = 2006
START_Q = 1

RETRIES = 3
RETRY_SLEEP = 2.0

# Form 4 (and amendments) only — Form 3 is initial-ownership, Form 5 is
# year-end cleanup; neither carries the open-market timing signal.
FORM4_TYPES = {"4", "4/A"}


# ─────────────────────────────────────────────────────────────────────────────
# Quarter enumeration + download
# ─────────────────────────────────────────────────────────────────────────────


def _current_quarter() -> tuple[int, int]:
    today = pd.Timestamp.now(tz="UTC").normalize()
    return today.year, (today.month - 1) // 3 + 1


def _iter_quarters(start: tuple[int, int], end: tuple[int, int]):
    y, q = start
    ey, eq = end
    while (y, q) <= (ey, eq):
        yield y, q
        q += 1
        if q > 4:
            q = 1
            y += 1


def _quarter_label(year: int, q: int) -> str:
    return f"{year}q{q}"


def _quarter_zip_path(year: int, q: int) -> str:
    return os.path.join(BULK_DIR, f"{_quarter_label(year, q)}_form345.zip")


def _download_quarter(year: int, q: int) -> bool:
    """Download one quarter's zip into BULK_DIR. Returns True on success.

    404 means the quarter hasn't been published yet (most recent quarter has
    a ~1-month lag) — caller treats this as a soft skip, not an error.
    """
    url = BULK_BASE_URL.format(q=_quarter_label(year, q))
    dest = _quarter_zip_path(year, q)
    os.makedirs(BULK_DIR, exist_ok=True)
    for attempt in range(1, RETRIES + 1):
        try:
            _sec_throttle()
            r = requests.get(url, headers=BULK_HEADERS, timeout=120)
            if r.status_code == 404:
                return False
            r.raise_for_status()
            tmp = dest + ".tmp"
            with open(tmp, "wb") as f:
                f.write(r.content)
            os.replace(tmp, dest)
            return True
        except Exception:
            if attempt == RETRIES:
                return False
            time.sleep(RETRY_SLEEP * attempt)
    return False


def download_bulk_quarters(refresh_latest: bool = True) -> list[tuple[int, int]]:
    """Ensure all available quarterly zips are cached.

    Logic:
      - For every quarter from 2006q1 → current, download if not on disk.
      - If `refresh_latest`, always re-download the most recently-cached quarter
        (and one quarter ahead in case it has just been published) — the SEC
        keeps amending the most recent quarter's zip as new filings arrive.

    Returns the list of (year, q) tuples that are now cached.
    """
    os.makedirs(BULK_DIR, exist_ok=True)
    cy, cq = _current_quarter()
    targets = list(_iter_quarters((START_YEAR, START_Q), (cy, cq)))

    cached: list[tuple[int, int]] = []
    missing: list[tuple[int, int]] = []
    for y, q in targets:
        if os.path.exists(_quarter_zip_path(y, q)):
            cached.append((y, q))
        else:
            missing.append((y, q))

    # Always refresh the latest currently-cached quarter (it's still being
    # filled until the next quarterly release). This catches new filings
    # added within a quarter's publication window.
    refresh: list[tuple[int, int]] = []
    if refresh_latest and cached:
        refresh.append(cached[-1])

    to_fetch = missing + [r for r in refresh if r not in missing]
    if not to_fetch:
        print(
            f"Insider bulk: {len(cached)} quarters cached, nothing to download.",
            flush=True,
        )
        return cached

    print(
        f"Insider bulk: {len(cached)} cached, {len(missing)} missing, "
        f"{len(refresh)} to refresh. Downloading {len(to_fetch)} zip(s)...",
        flush=True,
    )

    ok = 0
    skipped = 0
    for y, q in tqdm(to_fetch, desc="bulk zips"):
        if _download_quarter(y, q):
            ok += 1
        else:
            skipped += 1

    final_cached = [
        (y, q) for y, q in targets if os.path.exists(_quarter_zip_path(y, q))
    ]
    print(
        f"Insider bulk: downloaded={ok}  not-yet-published={skipped}  "
        f"total_cached={len(final_cached)}",
        flush=True,
    )
    return final_cached


# ─────────────────────────────────────────────────────────────────────────────
# Quarterly TSV → tidy DataFrame
# ─────────────────────────────────────────────────────────────────────────────


_DATE_FMT = "%d-%b-%Y"  # SEC TSVs use "31-OCT-2025"


def _read_tsv_from_zip(zf: zipfile.ZipFile, name: str, usecols: list[str]) -> pd.DataFrame:
    with zf.open(name) as fh:
        return pd.read_csv(
            fh,
            sep="\t",
            usecols=usecols,
            dtype=str,
            keep_default_na=False,
            na_values=[""],
            on_bad_lines="skip",
            low_memory=False,
        )


def parse_quarter_zip(zip_path: str) -> pd.DataFrame:
    """Parse one quarterly zip → long-format trans frame.

    Returns DataFrame with columns:
        ticker, filing_date, transaction_date, transaction_code,
        shares, price_per_share, value
    Already filtered to Form 4 / 4/A, transaction_code in {P, S}, and
    direct ownership.
    """
    with zipfile.ZipFile(zip_path) as zf:
        sub = _read_tsv_from_zip(
            zf,
            "SUBMISSION.tsv",
            usecols=[
                "ACCESSION_NUMBER",
                "FILING_DATE",
                "DOCUMENT_TYPE",
                "ISSUERTRADINGSYMBOL",
            ],
        )
        trans = _read_tsv_from_zip(
            zf,
            "NONDERIV_TRANS.tsv",
            usecols=[
                "ACCESSION_NUMBER",
                "TRANS_DATE",
                "TRANS_CODE",
                "TRANS_SHARES",
                "TRANS_PRICEPERSHARE",
                "DIRECT_INDIRECT_OWNERSHIP",
            ],
        )

    # Filter SUBMISSION to Form 4 + 4/A with a non-empty ticker.
    sub = sub[sub["DOCUMENT_TYPE"].isin(FORM4_TYPES)]
    sub = sub.dropna(subset=["ISSUERTRADINGSYMBOL"])
    sub["ticker"] = sub["ISSUERTRADINGSYMBOL"].str.strip().str.upper()
    # Drop placeholder/non-ticker values used by some filers (private companies,
    # OTC issuers, etc.) so they don't end up as fake parquet files.
    _BAD = {"", "NONE", "N/A", "NA", "NULL"}
    sub = sub[~sub["ticker"].isin(_BAD)]
    sub["filing_date"] = pd.to_datetime(
        sub["FILING_DATE"], format=_DATE_FMT, errors="coerce"
    )
    sub = sub.dropna(subset=["filing_date"])

    # Filter NONDERIV_TRANS to direct-ownership P/S.
    trans = trans[trans["TRANS_CODE"].isin(("P", "S"))]
    trans = trans[trans["DIRECT_INDIRECT_OWNERSHIP"] == "D"]

    merged = trans.merge(
        sub[["ACCESSION_NUMBER", "ticker", "filing_date"]],
        on="ACCESSION_NUMBER",
        how="inner",
    )

    if merged.empty:
        return pd.DataFrame(
            columns=[
                "ticker", "filing_date", "transaction_date",
                "transaction_code", "shares", "price_per_share", "value",
            ]
        )

    merged["transaction_date"] = pd.to_datetime(
        merged["TRANS_DATE"], format=_DATE_FMT, errors="coerce"
    )
    shares = pd.to_numeric(merged["TRANS_SHARES"], errors="coerce").fillna(0.0)
    price = pd.to_numeric(merged["TRANS_PRICEPERSHARE"], errors="coerce").fillna(0.0)

    out = pd.DataFrame({
        "ticker": merged["ticker"].values,
        "filing_date": merged["filing_date"].values,
        "transaction_date": merged["transaction_date"].values,
        "transaction_code": merged["TRANS_CODE"].values,
        "shares": shares.values,
        "price_per_share": price.values,
        "value": (shares * price).values,
    })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-ticker parquet rebuild
# ─────────────────────────────────────────────────────────────────────────────


def _insider_path(ticker: str) -> str:
    return os.path.join(INSIDER_DIR, f"{ticker}.parquet")


_PERTICKER_COLS = [
    "filing_date", "transaction_date", "transaction_code",
    "shares", "price_per_share", "value",
]


def rebuild_per_ticker(
    quarters: list[tuple[int, int]],
    tickers: list[str] | None = None,
) -> dict[str, int]:
    """Read every cached quarterly zip, group by ticker, write per-ticker parquet.

    If `tickers` is provided, only those tickers' files are (re)written;
    other tickers' existing parquets are left untouched.

    Returns {ticker: row_count} for tickers we actually wrote.
    """
    os.makedirs(INSIDER_DIR, exist_ok=True)

    keep_set = {t.upper() for t in tickers} if tickers else None

    frames: list[pd.DataFrame] = []
    for y, q in tqdm(quarters, desc="parse zips"):
        path = _quarter_zip_path(y, q)
        if not os.path.exists(path):
            continue
        df = parse_quarter_zip(path)
        if df.empty:
            continue
        if keep_set is not None:
            df = df[df["ticker"].isin(keep_set)]
            if df.empty:
                continue
        frames.append(df)

    if not frames:
        print("Insider rebuild: no transactions parsed.", flush=True)
        return {}

    full = pd.concat(frames, ignore_index=True)
    # Stable ordering + dedup on the natural key.
    full = (
        full.sort_values(["ticker", "filing_date"])
        .drop_duplicates(
            subset=["ticker", "filing_date", "transaction_date",
                    "transaction_code", "shares", "price_per_share"]
        )
        .reset_index(drop=True)
    )

    counts: dict[str, int] = {}
    for tkr, grp in tqdm(full.groupby("ticker", sort=False), desc="write parquet"):
        out = (
            grp[_PERTICKER_COLS]
            .sort_values("filing_date")
            .reset_index(drop=True)
        )
        out.to_parquet(_insider_path(tkr))
        counts[tkr] = len(out)

    print(
        f"Insider rebuild: wrote {len(counts)} ticker parquets, "
        f"{sum(counts.values())} total transactions.",
        flush=True,
    )
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# Public loader
# ─────────────────────────────────────────────────────────────────────────────


def load_insider_transactions() -> pd.DataFrame:
    """Load all cached insider transactions.

    Returns long-format DataFrame[ticker, filing_date, transaction_code,
    shares, price_per_share, value]. Only P and S codes are stored, so callers
    can distinguish buys (P) from sales (S) directly.
    """
    cols = ["ticker", "filing_date", "transaction_code",
            "shares", "price_per_share", "value"]
    if not os.path.isdir(INSIDER_DIR):
        return pd.DataFrame(columns=cols)

    frames: list[pd.DataFrame] = []
    for fname in os.listdir(INSIDER_DIR):
        if not fname.endswith(".parquet"):
            continue
        ticker = fname.removesuffix(".parquet")
        try:
            df = pd.read_parquet(os.path.join(INSIDER_DIR, fname))
        except Exception:
            continue
        if df.empty:
            continue
        df = df.copy()
        df["ticker"] = ticker
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=cols)

    out = pd.concat(frames, ignore_index=True)
    out["filing_date"] = pd.to_datetime(out["filing_date"])
    return out.sort_values(["ticker", "filing_date"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--refresh", action="store_true",
        help="Wipe data/insider/ entirely and redownload all quarterly zips.",
    )
    ap.add_argument(
        "--tickers",
        help="Comma-separated subset; only rewrite parquets for these tickers "
             "using already-cached zips. No download is skipped.",
    )
    args = ap.parse_args()

    if args.refresh:
        if os.path.isdir(INSIDER_DIR):
            shutil.rmtree(INSIDER_DIR)
        print(f"Wiped {INSIDER_DIR}", flush=True)

    quarters = download_bulk_quarters(refresh_latest=True)

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = all_historical_tickers(since="2005-07-01")

    rebuild_per_ticker(quarters, tickers=tickers)


if __name__ == "__main__":
    main()
