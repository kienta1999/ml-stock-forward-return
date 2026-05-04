#!/usr/bin/env python3
"""Download insider transaction data from SEC EDGAR Form 4.

Uses the same submissions API as earnings.py. For each ticker, fetches the
Form 4 / 4/A filing list from the CIK submissions JSON, then downloads each
filing XML to extract transaction type (P = open-market purchase, S = open-
market sale), shares, and price.

Discipline: filing_date (not transaction_date) is the asof key — Form 4 must
be filed within 2 business days of the transaction, so using the filing date
gives clean point-in-time data. Features computed from this data cannot look
ahead.

Only direct-ownership non-derivative P and S transactions are kept. Grants,
awards, option exercises, and indirect-ownership filings are excluded since
they carry little open-market timing signal.

Caches:
    data/insider/{TICKER}.parquet    filing_date, transaction_date,
                                     transaction_code, shares, price_per_share,
                                     value (shares × price)

Incremental: on each run the submissions JSON is re-fetched per ticker to
detect new filings; only their XMLs are downloaded. The first run for a ticker
fetches its full history (slow; plan for ~2-4 hours for the full S&P 500
universe). Subsequent daily runs add only filings since the last cached date.

Public loader (used by features.py):
    load_insider_transactions()  → DataFrame[ticker, filing_date,
                                             transaction_code, value]

CLI:
    uv run python scripts/insider.py                        # incremental update
    uv run python scripts/insider.py --refresh              # wipe caches, redownload
    uv run python scripts/insider.py --tickers AAPL,MSFT   # subset
"""

import argparse
import os
import sys
import time
import warnings
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

import pandas as pd
import requests
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from earnings import (  # noqa: E402
    SEC_HEADERS,
    SEC_RATE_LIMIT_QPS,
    SEC_SUBMISSIONS_URL,
    SEC_WORKERS,
    _LAST_SEC_REQUEST,
    _fetch_extra_submission_files,
    _sec_throttle,
    load_cik_map,
)
from universe import all_historical_tickers  # noqa: E402

_ROOT = os.path.dirname(_HERE)
INSIDER_DIR = os.path.join(_ROOT, "data", "insider")

INSIDER_FORMS = {"4", "4/A"}

RETRIES = 3
RETRY_SLEEP = 2.0

SEC_ARCHIVE_HEADERS = {
    "User-Agent": "ml-stock-forward-return research bot",
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov",
}


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────


def _sec_get_archive(url: str) -> requests.Response | None:
    """GET an EDGAR archives URL with retry + rate limiting."""
    for attempt in range(1, RETRIES + 1):
        try:
            _sec_throttle()
            r = requests.get(url, headers=SEC_ARCHIVE_HEADERS, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt == RETRIES:
                return None
            time.sleep(RETRY_SLEEP * attempt)
    return None


def _sec_get_submissions(url: str) -> requests.Response | None:
    """GET an EDGAR data.sec.gov URL with retry + rate limiting."""
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
                return None
            time.sleep(RETRY_SLEEP * attempt)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Submissions JSON → Form 4 filing list
# ─────────────────────────────────────────────────────────────────────────────


def _form4_list_from_recent(recent: dict) -> list[tuple[str, str, str]]:
    """Extract (filing_date, accession, primary_doc) for Form 4 / 4/A."""
    out: list[tuple[str, str, str]] = []
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accs = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])
    for f, d, a, doc in zip(forms, dates, accs, docs):
        if f in INSIDER_FORMS and doc:
            out.append((d, a, doc))
    return out


def _form4_list_from_submission(payload: dict) -> list[tuple[str, str, str]]:
    """All Form 4 entries from the main submissions JSON (recent + paginated)."""
    out = _form4_list_from_recent(payload.get("filings", {}).get("recent", {}))
    for extra in _fetch_extra_submission_files(payload):
        flat = extra if "form" in extra else extra.get("filings", {}).get("recent", {})
        out.extend(_form4_list_from_recent(flat))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Form 4 XML fetch + parse
# ─────────────────────────────────────────────────────────────────────────────


def _form4_url(cik_padded: str, accession: str, doc: str) -> str:
    # EDGAR stores Form 4 filings under the ISSUER's CIK directory.
    # primaryDocument often has an XSL viewer prefix (e.g. "xslF345X06/form4.xml");
    # strip any leading path component so we address the raw XML at the root.
    cik_int = str(int(cik_padded))
    accession_nodash = accession.replace("-", "")
    filename = doc.split("/")[-1] if "/" in doc else doc
    return f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{filename}"


def _fetch_form4_xml(cik_padded: str, accession: str, primary_doc: str) -> str | None:
    """Fetch the raw Form 4 XML text for one filing.

    primaryDocument is often "xslF345X06/form4.xml" — the XSL prefix is a
    subdirectory for the HTML viewer; the actual ownershipDocument XML lives
    at the root of the accession directory. _form4_url strips the prefix.
    If the response doesn't contain ownershipDocument (e.g. older .htm
    variant), also tries replacing .htm with .xml.
    """
    r = _sec_get_archive(_form4_url(cik_padded, accession, primary_doc))
    if r is not None:
        text = r.text
        if "<ownershipDocument" in text:
            return text
        # Older filers may use .htm as primary; try .xml sibling
        filename = primary_doc.split("/")[-1]
        if filename.lower().endswith(".htm"):
            xml_doc = filename[:-4] + ".xml"
            r2 = _sec_get_archive(_form4_url(cik_padded, accession, xml_doc))
            if r2 is not None and "<ownershipDocument" in r2.text:
                return r2.text
    return None


def _get_text(elem: ET.Element, path: str) -> str:
    """Navigate to elem.find(path); return <value> child text or direct text."""
    node = elem.find(path)
    if node is None:
        return ""
    val = node.find("value")
    text = (val.text if val is not None else node.text) or ""
    return text.strip()


def _parse_form4_xml(xml_text: str) -> list[dict]:
    """Form 4 XML → list of {transaction_date, transaction_code, shares,
    price_per_share, value} for direct non-derivative P/S transactions only."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    rows: list[dict] = []
    for txn in root.findall(".//nonDerivativeTransaction"):
        code = _get_text(txn, "transactionCoding/transactionCode")
        if code not in ("P", "S"):
            continue
        direct = _get_text(txn, "ownershipNature/directOrIndirectOwnership")
        if direct != "D":
            continue

        date_str = _get_text(txn, "transactionDate")
        shares_str = _get_text(txn, "transactionAmounts/transactionShares")
        price_str = _get_text(txn, "transactionAmounts/transactionPricePerShare")

        try:
            txn_date = pd.Timestamp(date_str) if date_str else pd.NaT
            shares = float(shares_str) if shares_str else 0.0
            price = float(price_str) if price_str else 0.0
        except (ValueError, TypeError):
            continue

        rows.append(
            {
                "transaction_date": txn_date,
                "transaction_code": code,
                "shares": shares,
                "price_per_share": price,
                "value": shares * price,
            }
        )
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Per-ticker fetch + cache
# ─────────────────────────────────────────────────────────────────────────────


def _insider_path(ticker: str) -> str:
    return os.path.join(INSIDER_DIR, f"{ticker}.parquet")


def fetch_insider_for_ticker(
    ticker: str,
    cik_map: dict[str, str],
    refresh: bool = False,
) -> tuple[str, int, str]:
    """Fetch and cache Form 4 insider transactions for a ticker.

    Returns (ticker, total_cached_rows, status) where status ∈
    {"new", "fresh", "missing_cik", "failed"}.

    Incremental: loads existing cache, finds the latest filing_date, and only
    fetches Form 4 XMLs filed after that date. The submissions JSON is always
    re-fetched (1 lightweight request) to detect new filings.
    """
    path = _insider_path(ticker)

    # Load existing cache
    existing_df: pd.DataFrame | None = None
    cutoff_date: pd.Timestamp | None = None
    if not refresh and os.path.exists(path):
        try:
            existing_df = pd.read_parquet(path)
            if not existing_df.empty:
                cutoff_date = pd.Timestamp(existing_df["filing_date"].max())
        except Exception:
            existing_df = None

    cik = cik_map.get(ticker)
    if cik is None:
        if existing_df is not None:
            return ticker, len(existing_df), "fresh"
        return ticker, 0, "missing_cik"

    # Fetch submissions JSON to get the Form 4 filing list.
    r = _sec_get_submissions(SEC_SUBMISSIONS_URL.format(cik=cik))
    if r is None:
        if existing_df is not None:
            return ticker, len(existing_df), "fresh"
        return ticker, 0, "failed"

    try:
        payload = r.json()
    except Exception:
        if existing_df is not None:
            return ticker, len(existing_df), "fresh"
        return ticker, 0, "failed"

    all_filings = _form4_list_from_submission(payload)

    # Filter to filings after the latest cached date (incremental).
    if cutoff_date is not None:
        new_filings = [
            (d, a, doc)
            for d, a, doc in all_filings
            if pd.Timestamp(d) > cutoff_date
        ]
    else:
        new_filings = all_filings

    if not new_filings and existing_df is not None:
        return ticker, len(existing_df), "fresh"

    # Fetch and parse Form 4 XMLs for new filings.
    new_rows: list[dict] = []
    for filing_date_str, accession, primary_doc in new_filings:
        xml_text = _fetch_form4_xml(cik, accession, primary_doc)
        if xml_text is None:
            continue
        txns = _parse_form4_xml(xml_text)
        for txn in txns:
            txn["filing_date"] = pd.Timestamp(filing_date_str)
        new_rows.extend(txns)

    # Build/merge and save.
    _COLS = ["filing_date", "transaction_date", "transaction_code",
             "shares", "price_per_share", "value"]
    if new_rows:
        new_df = pd.DataFrame(new_rows, columns=_COLS)
        new_df["filing_date"] = pd.to_datetime(new_df["filing_date"])
        new_df["transaction_date"] = pd.to_datetime(new_df["transaction_date"])
        if existing_df is not None and not existing_df.empty:
            df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            df = new_df
        df = (
            df.drop_duplicates(subset=["filing_date", "transaction_date",
                                       "transaction_code", "shares"])
            .sort_values("filing_date")
            .reset_index(drop=True)
        )
        status = "new"
    elif existing_df is not None:
        df = existing_df
        status = "fresh"
    else:
        df = pd.DataFrame(columns=_COLS)
        status = "new"

    df.to_parquet(path)
    return ticker, len(df), status


def fetch_insider_universe(
    tickers: list[str],
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch insider transactions for every ticker. Returns status summary."""
    os.makedirs(INSIDER_DIR, exist_ok=True)
    cik_map = load_cik_map(refresh=False)  # reuse existing CIK map
    results: list[tuple[str, int, str]] = []

    with ThreadPoolExecutor(max_workers=SEC_WORKERS) as pool:
        futures = {
            pool.submit(fetch_insider_for_ticker, t, cik_map, refresh): t
            for t in tickers
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Form 4"):
            results.append(fut.result())

    summary = pd.DataFrame(results, columns=["Ticker", "Rows", "Status"])
    counts = summary["Status"].value_counts().to_dict()
    print(
        f"\nForm 4 done. new={counts.get('new', 0)}  fresh={counts.get('fresh', 0)}  "
        f"missing_cik={counts.get('missing_cik', 0)}  failed={counts.get('failed', 0)}",
        flush=True,
    )
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Public loader
# ─────────────────────────────────────────────────────────────────────────────


def load_insider_transactions() -> pd.DataFrame:
    """Load all cached insider transactions.

    Returns long-format DataFrame[ticker, filing_date, transaction_code,
    shares, price_per_share, value]. Only P and S transaction codes are ever
    stored, so callers can distinguish buys (P) from sales (S) directly.
    """
    if not os.path.isdir(INSIDER_DIR):
        return pd.DataFrame(
            columns=["ticker", "filing_date", "transaction_code",
                     "shares", "price_per_share", "value"]
        )

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
        return pd.DataFrame(
            columns=["ticker", "filing_date", "transaction_code",
                     "shares", "price_per_share", "value"]
        )

    out = pd.concat(frames, ignore_index=True)
    out["filing_date"] = pd.to_datetime(out["filing_date"])
    return out.sort_values(["ticker", "filing_date"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refresh", action="store_true",
                    help="Wipe caches and redownload all Form 4 history. "
                         "WARNING: initial full-universe download takes 2-4 hours.")
    ap.add_argument("--tickers", help="Comma-separated subset (for testing).")
    args = ap.parse_args()

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = all_historical_tickers(since="2005-07-01")

    if args.refresh:
        import shutil
        if os.path.isdir(INSIDER_DIR):
            shutil.rmtree(INSIDER_DIR)
        print(f"Wiped {INSIDER_DIR}", flush=True)

    print(
        f"\nForm 4: fetching insider transactions for {len(tickers)} tickers "
        f"(incremental — only new filings since last run)...",
        flush=True,
    )
    fetch_insider_universe(tickers, refresh=args.refresh)


if __name__ == "__main__":
    main()
