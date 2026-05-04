#!/usr/bin/env python3
"""Download SEC EDGAR XBRL fundamentals and compute TTM/MRQ values per ticker.

Pulls from data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json — the same
endpoint that powers SEC's interactive financial-data dashboards. Each
ticker gets one JSON response containing every us-gaap concept ever
tagged on its 10-Qs / 10-Ks, point-in-time correct.

Caches:
    data/fundamentals/{TICKER}.parquet
        concept, period_end, period_start, filed, value, form, fp, fy, accession

Public loader (used by features.py):
    load_fundamentals()  → DataFrame[ticker, period_end, asof_date,
                                     ttm_revenue, ttm_net_income, ttm_op_income,
                                     ttm_revenue_prior, ttm_op_income_prior,
                                     mrq_assets, mrq_assets_current,
                                     mrq_liabilities_current, mrq_equity,
                                     mrq_lt_debt, shares]

asof_date = max(filed) of all rows contributing to a panel row → the first
date that entire row's contents became publicly observable. features.py
should asof-merge using asof_date, NOT period_end.

CLI:
    python scripts/fundamentals.py                       # incremental, all S&P names
    python scripts/fundamentals.py --refresh             # wipe caches, redownload
    python scripts/fundamentals.py --tickers AAPL,MSFT   # subset

Notes on coverage:
- XBRL was phased in 2009-06 (large filers >$5B float), 2010-06 (other large
  accelerated), 2011-06 (everyone). Pre-2009 has 0% coverage; pre-2011 has
  partial coverage. Train period 2007–2017 will have NaN fundamentals for
  ~2007-01 → 2009-06 (no large filers tagged yet).
- Concept synonyms: revenue and LT debt are tagged inconsistently across
  filers and across time. We coalesce to canonical names (see CONCEPT_MAP).
"""

import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from earnings import (  # noqa: E402  — share the cached CIK map from earnings.py
    SEC_HEADERS,
    SEC_RATE_LIMIT_QPS,
    SEC_WORKERS,
    _LAST_SEC_REQUEST,  # cross-module shared throttle (intended)
    _sec_throttle,
    load_cik_map,
)
from universe import all_historical_tickers  # noqa: E402

_ROOT = os.path.dirname(_HERE)
FUNDAMENTALS_DIR = os.path.join(_ROOT, "data", "fundamentals")
SPLITS_PATH = os.path.join(FUNDAMENTALS_DIR, "_splits.parquet")

SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

RETRIES = 3
RETRY_SLEEP = 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Concept registry — XBRL tags we care about, with synonym fallbacks
# ─────────────────────────────────────────────────────────────────────────────
#
# Each canonical name maps to:
#   - tags: list of us-gaap concept names to read (multiple → coalesce)
#   - unit: which unit-key inside .units{} to pick
#   - kind: 'flow' (income/cashflow, has period_start + period_end, sum-aggregable)
#           or 'snapshot' (balance sheet, period_end only, point-in-time)
#
# Why multiple tags per canonical:
#   revenue  — pre-2018 filers used `Revenues`; post-2018 ASC 606 standardised
#              filers onto `RevenueFromContractWithCustomerExcludingAssessedTax`.
#              Many companies' history straddles both. Read both, dedup later.
#   lt_debt  — `LongTermDebt` is the original tag; `LongTermDebtNoncurrent`
#              became more common from ~2013. Same story.
#   equity   — `StockholdersEquity` is the standard; the `…IncludingPortion…`
#              variant exists for filers with material noncontrolling interest.
#
CONCEPT_MAP: dict[str, dict] = {
    # ── Income statement (flow) ─────────────────────────────────────────────
    "revenue": {
        "tags": [
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
        ],
        "unit": "USD",
        "kind": "flow",
    },
    "net_income": {
        "tags": ["NetIncomeLoss"],
        "unit": "USD",
        "kind": "flow",
    },
    "operating_income": {
        "tags": ["OperatingIncomeLoss"],
        "unit": "USD",
        "kind": "flow",
    },
    # ── Balance sheet (snapshot) ────────────────────────────────────────────
    "assets": {
        "tags": ["Assets"],
        "unit": "USD",
        "kind": "snapshot",
    },
    "assets_current": {
        "tags": ["AssetsCurrent"],
        "unit": "USD",
        "kind": "snapshot",
    },
    "liabilities_current": {
        "tags": ["LiabilitiesCurrent"],
        "unit": "USD",
        "kind": "snapshot",
    },
    "equity": {
        "tags": [
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ],
        "unit": "USD",
        "kind": "snapshot",
    },
    "lt_debt": {
        "tags": ["LongTermDebt", "LongTermDebtNoncurrent"],
        "unit": "USD",
        "kind": "snapshot",
    },
    # ── Shares (snapshot, in shares not USD) ────────────────────────────────
    "shares": {
        "tags": ["CommonStockSharesOutstanding"],
        "unit": "shares",
        "kind": "snapshot",
    },
}

FLOW_CONCEPTS = [k for k, v in CONCEPT_MAP.items() if v["kind"] == "flow"]
SNAPSHOT_CONCEPTS = [k for k, v in CONCEPT_MAP.items() if v["kind"] == "snapshot"]

# Period-span tolerances for flow concepts (in days). Fiscal calendars vary
# (52/53-week filers like Apple, Costco; non-calendar FY ends like Walmart's
# late January) so use ranges, not exact equality.
QUARTERLY_DAYS = (80, 100)
ANNUAL_DAYS = (350, 380)


# ─────────────────────────────────────────────────────────────────────────────
# Per-ticker fetcher
# ─────────────────────────────────────────────────────────────────────────────


def _sec_get_companyfacts(cik: str) -> dict | None:
    """GET /api/xbrl/companyfacts/CIK{cik}.json with retry + throttle."""
    url = SEC_COMPANYFACTS_URL.format(cik=cik)
    for attempt in range(1, RETRIES + 1):
        try:
            _sec_throttle()  # shared with earnings.py to keep us under 10 qps
            r = requests.get(url, headers=SEC_HEADERS, timeout=60)
            if r.status_code == 404:
                return None  # no XBRL filings on record (delisted/foreign/etc.)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == RETRIES:
                print(f"  [{cik}] companyfacts failed after {RETRIES} retries: {e}", flush=True)
                return None
            time.sleep(RETRY_SLEEP * attempt)
    return None


def _extract_concept_rows(payload: dict, canonical: str, spec: dict) -> list[dict]:
    """Pull every value for a canonical concept out of the companyfacts payload.

    Walks every us-gaap tag in spec['tags'], reads units[spec['unit']], and
    flattens to a list of rows tagged with the canonical name. Dedup happens
    in _dedup_concept_rows.
    """
    gaap = payload.get("facts", {}).get("us-gaap", {})
    out: list[dict] = []
    for tag in spec["tags"]:
        if tag not in gaap:
            continue
        units = gaap[tag].get("units", {})
        vals = units.get(spec["unit"], [])
        for v in vals:
            row = {
                "concept": canonical,
                "tag": tag,  # kept for debugging; dropped before cache write
                "period_end": v.get("end"),
                "period_start": v.get("start"),  # None for snapshots
                "filed": v.get("filed"),
                "value": v.get("val"),
                "form": v.get("form"),
                "fp": v.get("fp"),
                "fy": v.get("fy"),
                "accession": v.get("accn"),
            }
            out.append(row)
    return out


def _dedup_concept_rows(rows: list[dict]) -> list[dict]:
    """Within a canonical concept, the same (period_end, period_start) can
    appear many times — every later 10-Q/10-K re-reports prior periods as
    comparatives, often under different `fy` labels (XBRL's `fy` is the
    *filing's* fiscal year, not the *value's*). For as-of/no-lookahead
    correctness we want the **first time** each period+amount became publicly
    observable, so dedup keys on (period_end, period_start) only and keeps
    the earliest `filed`.

    This also handles concept-synonym overlaps (e.g. a 2018 Q4 revenue tagged
    under both `Revenues` and `RevenueFromContractWithCustomerExcludingAssessedTax`
    during the ASC 606 transition) since we collapsed all variants to the
    same canonical name in _extract_concept_rows.
    """
    if not rows:
        return rows
    df = pd.DataFrame(rows)
    df["filed_dt"] = pd.to_datetime(df["filed"], errors="coerce")
    df = (
        df.sort_values("filed_dt")
        .drop_duplicates(
            subset=["concept", "period_end", "period_start"],
            keep="first",
        )
        .drop(columns=["filed_dt", "tag"])
    )
    return df.to_dict("records")


def _fundamentals_path(ticker: str) -> str:
    return os.path.join(FUNDAMENTALS_DIR, f"{ticker}.parquet")


def fetch_fundamentals_for_ticker(
    ticker: str,
    cik_map: dict[str, str],
    refresh: bool = False,
) -> tuple[str, int, str]:
    """Pull companyfacts for one ticker, extract concepts in CONCEPT_MAP, cache.

    status ∈ {"new", "fresh", "missing_cik", "no_facts", "failed"}
    """
    path = _fundamentals_path(ticker)
    if not refresh and os.path.exists(path):
        try:
            cached = pd.read_parquet(path)
            return ticker, len(cached), "fresh"
        except Exception:
            pass

    cik = cik_map.get(ticker)
    if cik is None:
        return ticker, 0, "missing_cik"

    payload = _sec_get_companyfacts(cik)
    if payload is None:
        return ticker, 0, "failed"

    rows: list[dict] = []
    for canonical, spec in CONCEPT_MAP.items():
        rows.extend(_extract_concept_rows(payload, canonical, spec))
    rows = _dedup_concept_rows(rows)

    if not rows:
        df = pd.DataFrame(columns=[
            "concept", "period_end", "period_start", "filed",
            "value", "form", "fp", "fy", "accession",
        ])
        df.to_parquet(path)
        return ticker, 0, "no_facts"

    df = pd.DataFrame(rows)
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    df["period_start"] = pd.to_datetime(df["period_start"], errors="coerce")
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["period_end", "filed", "value"]).reset_index(drop=True)
    df = df.sort_values(["concept", "period_end", "filed"]).reset_index(drop=True)
    df.to_parquet(path)
    return ticker, len(df), "new"


def fetch_fundamentals_universe(
    tickers: list[str],
    refresh: bool = False,
) -> pd.DataFrame:
    """Pull companyfacts for every ticker in the universe."""
    os.makedirs(FUNDAMENTALS_DIR, exist_ok=True)
    cik_map = load_cik_map(refresh=False)  # reuse the EDGAR cache; no need to refetch
    results: list[tuple[str, int, str]] = []

    with ThreadPoolExecutor(max_workers=SEC_WORKERS) as pool:
        futures = {
            pool.submit(fetch_fundamentals_for_ticker, t, cik_map, refresh): t
            for t in tickers
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="XBRL"):
            results.append(fut.result())

    summary = pd.DataFrame(results, columns=["Ticker", "Rows", "Status"])
    counts = summary["Status"].value_counts().to_dict()
    print(
        f"\nXBRL done. new={counts.get('new', 0)}  fresh={counts.get('fresh', 0)}  "
        f"no_facts={counts.get('no_facts', 0)}  "
        f"missing_cik={counts.get('missing_cik', 0)}  failed={counts.get('failed', 0)}",
        flush=True,
    )
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Split history — needed because cached yfinance Close is split-adjusted,
# but XBRL shares are raw point-in-time counts. Multiplying naively would
# underprice market cap pre-split (e.g. AAPL 2010: ~6B raw shares × ~$10
# adj_close = $60B "market cap" vs the real ~$300B). Fix: scale raw XBRL
# shares forward to today's basis using cumulative split factors.
# ─────────────────────────────────────────────────────────────────────────────


def fetch_splits(tickers: list[str], refresh: bool = False) -> pd.DataFrame:
    """Cache yfinance split history for each ticker. One row per (ticker, split).

    Returns long DataFrame[ticker, date, ratio]. Tickers with no splits get
    no rows (treated as cumulative_factor = 1.0 in the loader).
    """
    os.makedirs(FUNDAMENTALS_DIR, exist_ok=True)
    if not refresh and os.path.exists(SPLITS_PATH):
        try:
            return pd.read_parquet(SPLITS_PATH)
        except Exception:
            pass

    rows: list[dict] = []
    for t in tqdm(tickers, desc="splits"):
        try:
            s = yf.Ticker(t).splits
        except Exception:
            continue
        if s is None or len(s) == 0:
            continue
        for d, r in s.items():
            try:
                rows.append({
                    "ticker": t,
                    "date": pd.Timestamp(d).tz_localize(None) if pd.Timestamp(d).tz else pd.Timestamp(d),
                    "ratio": float(r),
                })
            except Exception:
                continue

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["ticker", "date", "ratio"])
    if not df.empty:
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df.to_parquet(SPLITS_PATH, index=False)
    print(f"  cached {len(df)} splits across {df['ticker'].nunique() if len(df) else 0} tickers", flush=True)
    return df


def load_splits() -> pd.DataFrame:
    if not os.path.exists(SPLITS_PATH):
        return pd.DataFrame(columns=["ticker", "date", "ratio"])
    try:
        return pd.read_parquet(SPLITS_PATH)
    except Exception:
        return pd.DataFrame(columns=["ticker", "date", "ratio"])


def _split_factor_after(ticker_splits: pd.DataFrame, asof: pd.Timestamp) -> float:
    """Cumulative product of splits whose date is strictly after `asof`.

    A row reported on `asof` reflects the share count at filing; any later
    split inflates that count by `ratio`. Multiplying the raw shares by this
    factor gives the count expressed in today's split-adjusted basis (which
    matches yfinance's Adj Close).
    """
    if ticker_splits.empty:
        return 1.0
    after = ticker_splits[ticker_splits["date"] > asof]["ratio"]
    return float(after.prod()) if not after.empty else 1.0


# ─────────────────────────────────────────────────────────────────────────────
# TTM aggregation
# ─────────────────────────────────────────────────────────────────────────────


def _classify_period_span(period_start: pd.Timestamp, period_end: pd.Timestamp) -> str:
    """Tag a flow-concept row as quarterly, annual, ytd, or unknown."""
    if pd.isna(period_start):
        return "unknown"
    days = (period_end - period_start).days + 1
    if QUARTERLY_DAYS[0] <= days <= QUARTERLY_DAYS[1]:
        return "quarterly"
    if ANNUAL_DAYS[0] <= days <= ANNUAL_DAYS[1]:
        return "annual"
    # 6-month and 9-month spans show up too (some filers report YTD-only).
    if 170 <= days <= 200:
        return "ytd_h1"
    if 260 <= days <= 290:
        return "ytd_q3"
    return "unknown"


def _build_quarterly_series(
    flow_rows: pd.DataFrame,
    canonical: str,
) -> pd.DataFrame:
    """Reduce a flow concept's mixed quarterly/YTD/annual rows to one row per
    fiscal quarter. Returns DataFrame[period_end, period_start, value, filed].

    Strategy:
    1. Take native quarterly rows directly (period span 80–100 days).
    2. For annual rows, derive Q4 = annual − (Q1 + Q2 + Q3) if all three Q's
       fall inside the annual's [period_start, period_end). The annual row's
       `filed` date is when Q4 became publicly inferable.

    We don't rely on `fy`/`fp` labels — they're inconsistent across
    re-filings — and instead group by date ranges from period_start/end.
    """
    empty = pd.DataFrame(columns=["period_end", "period_start", "value", "filed"])
    if flow_rows.empty:
        return empty

    df = flow_rows[flow_rows["concept"] == canonical].copy()
    if df.empty:
        return empty
    df["span"] = [_classify_period_span(s, e) for s, e in zip(df["period_start"], df["period_end"])]

    # 1. Native quarterly rows
    q = df[df["span"] == "quarterly"][
        ["period_end", "period_start", "value", "filed"]
    ].copy()
    q = (
        q.sort_values("filed")
        .drop_duplicates(subset=["period_end"], keep="first")
        .reset_index(drop=True)
    )

    # 2. Q4 derivation: for each annual, find the three Q's that sit inside
    # its fiscal year and synthesize the missing Q4.
    annuals = df[df["span"] == "annual"][
        ["period_end", "period_start", "value", "filed"]
    ].copy()
    if not annuals.empty:
        annuals = (
            annuals.sort_values("filed")
            .drop_duplicates(subset=["period_end"], keep="first")
            .reset_index(drop=True)
        )
        derived: list[dict] = []
        for _, ann in annuals.iterrows():
            ann_start, ann_end = ann["period_start"], ann["period_end"]
            # Skip if a native Q4 already lands exactly at this annual's period_end
            if (q["period_end"] == ann_end).any():
                continue
            # Find Q-rows whose period falls inside [ann_start, ann_end)
            inside = q[
                (q["period_start"] >= ann_start - pd.Timedelta(days=2))
                & (q["period_end"] < ann_end)
            ]
            if len(inside) != 3:
                continue
            q4_val = ann["value"] - inside["value"].sum()
            q4_filed = max(ann["filed"], inside["filed"].max())
            q4_start = inside["period_end"].max() + pd.Timedelta(days=1)
            derived.append({
                "period_end": ann_end,
                "period_start": q4_start,
                "value": q4_val,
                "filed": q4_filed,
            })
        if derived:
            q = pd.concat([q, pd.DataFrame(derived)], ignore_index=True)

    return q.sort_values("period_end").reset_index(drop=True)


def _ttm_from_quarterlies(quarterlies: pd.DataFrame) -> pd.DataFrame:
    """For each quarter-end, sum the trailing 4 quarterlies (incl. self).

    Returns DataFrame[period_end, ttm_value, ttm_asof_date, ttm_value_prior]
    where:
      - ttm_value       = sum of the 4 quarters ending at this row
      - ttm_asof_date   = max(filed) across those 4 quarters (when all 4
                          became publicly observable)
      - ttm_value_prior = ttm_value at period_end ≈ 365 days earlier (±30d
                          tolerance to handle 52/53-week fiscal calendars).
                          Used for YoY growth ratios in features.py.
    """
    cols = ["period_end", "ttm_value", "ttm_asof_date", "ttm_value_prior"]
    if quarterlies.empty:
        return pd.DataFrame(columns=cols)

    df = quarterlies.sort_values("period_end").reset_index(drop=True).copy()
    n = len(df)
    ttm_value: list = [pd.NA] * n
    ttm_filed: list = [pd.NaT] * n

    for i in range(n):
        if i < 3:
            continue
        window = df.iloc[i - 3 : i + 1]
        span_days = (window.iloc[-1]["period_end"] - window.iloc[0]["period_end"]).days
        # 4 quarters should span ~9 months between earliest and latest
        # period_end (3 quarter-gaps × ~91 days). Tolerate 240–320.
        if span_days < 240 or span_days > 320:
            continue
        ttm_value[i] = window["value"].sum()
        ttm_filed[i] = window["filed"].max()

    df["ttm_value"] = ttm_value
    df["ttm_asof_date"] = ttm_filed

    # YoY prior — date-tolerant lookup, not shift(4).
    pe = df["period_end"].to_numpy()
    ttm_arr = df["ttm_value"].tolist()
    prior_vals: list = []
    for i in range(n):
        target = pe[i] - pd.Timedelta(days=365)
        best = None
        best_delta = None
        for j in range(i):
            delta = abs((pe[j] - target).days)
            if delta <= 30 and pd.notna(ttm_arr[j]):
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best = ttm_arr[j]
        prior_vals.append(best if best is not None else pd.NA)
    df["ttm_value_prior"] = prior_vals

    return df[cols]


# ─────────────────────────────────────────────────────────────────────────────
# Per-ticker assembly
# ─────────────────────────────────────────────────────────────────────────────


def _assemble_ticker(
    ticker: str,
    raw: pd.DataFrame,
    splits: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Given one ticker's cached raw facts, build the per-quarter assembled
    panel: TTM flows + MRQ snapshots + shares, one row per fiscal quarter
    where TTM is computable.

    If `splits` is provided, also computes `shares_adj` = raw_shares × cumulative
    forward split factor — this is the count consistent with yfinance Adj Close
    for market-cap calculations downstream.
    """
    if raw.empty:
        return pd.DataFrame()

    flow_df = raw[raw["concept"].isin(FLOW_CONCEPTS)].copy()
    snap_df = raw[raw["concept"].isin(SNAPSHOT_CONCEPTS)].copy()

    # 1. Build TTM series for each flow concept independently.
    ttm_frames: dict[str, pd.DataFrame] = {}
    for c in ["revenue", "net_income", "operating_income"]:
        q = _build_quarterly_series(flow_df, c)
        ttm = _ttm_from_quarterlies(q)
        ttm_frames[c] = ttm.rename(columns={
            "ttm_value": f"ttm_{c}",
            "ttm_asof_date": f"ttm_{c}_asof",
            "ttm_value_prior": f"ttm_{c}_prior",
        })

    # 2. Spine = union of every flow concept's period_ends. Concepts with
    # spotty quarterly tagging (e.g. financials that only tag annual revenue
    # post-2014) drop to NaN where they aren't covered, but other concepts
    # carry the row. Without this, JPM-style filers produce no rows after
    # they stopped tagging quarterly revenue.
    period_frames = [f[["period_end"]] for f in ttm_frames.values() if not f.empty]
    if not period_frames:
        return pd.DataFrame()
    spine = (
        pd.concat(period_frames, ignore_index=True)
        .drop_duplicates()
        .sort_values("period_end")
        .reset_index(drop=True)
    )

    out = spine
    for c in ["revenue", "net_income", "operating_income"]:
        out = out.merge(ttm_frames[c], on=["period_end"], how="left")

    # 3. Snapshot concepts: take the most-recent value per concept where
    # period_end ≤ each row's period_end. Each snapshot also brings its own
    # filed date so we can compute the panel-row asof correctly.
    for c in SNAPSHOT_CONCEPTS:
        sdf = snap_df[snap_df["concept"] == c].copy()
        if sdf.empty:
            out[f"mrq_{c}"] = pd.NA
            out[f"mrq_{c}_asof"] = pd.NaT
            continue
        sdf = sdf.sort_values("period_end")[["period_end", "value", "filed"]]
        sdf = sdf.rename(columns={
            "period_end": "snap_end",
            "value": f"mrq_{c}",
            "filed": f"mrq_{c}_asof",
        })
        out_sorted = out.sort_values("period_end")
        out_sorted = pd.merge_asof(
            out_sorted,
            sdf,
            left_on="period_end",
            right_on="snap_end",
            direction="backward",
        ).drop(columns=["snap_end"])
        out = out_sorted.sort_values(["period_end"]).reset_index(drop=True)

    # 4. Compute panel row asof_date = max of all contributing filed dates.
    # Cast each _asof column to datetime64 first — empty-snapshot columns
    # land as object dtype and mixing dtypes breaks the row-wise max.
    asof_cols = [c for c in out.columns if c.endswith("_asof")]
    for c in asof_cols:
        out[c] = pd.to_datetime(out[c], errors="coerce")
    out["asof_date"] = out[asof_cols].max(axis=1)

    # 5. Drop rows where ALL flow TTMs are missing (no usable signal at all).
    flow_ttm_cols = ["ttm_revenue", "ttm_net_income", "ttm_operating_income"]
    flow_ttm_cols = [c for c in flow_ttm_cols if c in out.columns]
    out = out.dropna(subset=flow_ttm_cols, how="all").reset_index(drop=True)

    # 6. Rename mrq_shares → shares (it's a quantity, not a ratio input)
    out = out.rename(columns={"mrq_shares": "shares"})

    # 7. Split-adjust shares so they're on the same basis as yfinance Adj Close.
    if splits is not None and "shares" in out.columns:
        ticker_splits = splits[splits["ticker"] == ticker] if "ticker" in splits.columns else splits
        factors = [
            _split_factor_after(ticker_splits, pd.Timestamp(d))
            for d in out["asof_date"]
        ]
        out["shares_adj"] = out["shares"] * pd.Series(factors, index=out.index, dtype="float64")
    else:
        out["shares_adj"] = out["shares"]

    out["ticker"] = ticker
    keep_cols = [
        "ticker", "period_end", "asof_date",
        "ttm_revenue", "ttm_revenue_prior",
        "ttm_net_income",
        "ttm_operating_income", "ttm_operating_income_prior",
        "mrq_assets", "mrq_assets_current",
        "mrq_liabilities_current",
        "mrq_equity", "mrq_lt_debt",
        "shares", "shares_adj",
    ]
    keep_cols = [c for c in keep_cols if c in out.columns]
    return out[keep_cols].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Public loader
# ─────────────────────────────────────────────────────────────────────────────


def load_fundamentals(tickers: list[str] | None = None) -> pd.DataFrame:
    """Concatenate every cached ticker into the long fundamentals panel.

    Returns DataFrame[ticker, period_end, asof_date,
                      ttm_revenue, ttm_revenue_prior,
                      ttm_net_income,
                      ttm_operating_income, ttm_operating_income_prior,
                      mrq_assets, mrq_assets_current, mrq_liabilities_current,
                      mrq_equity, mrq_lt_debt, shares]

    features.py should asof-merge on (ticker, asof_date) with direction='backward'.
    """
    if not os.path.isdir(FUNDAMENTALS_DIR):
        return pd.DataFrame()

    if tickers is None:
        tickers = [
            f.removesuffix(".parquet")
            for f in os.listdir(FUNDAMENTALS_DIR)
            if f.endswith(".parquet") and not f.startswith("_")
        ]

    splits_df = load_splits()
    splits_by_ticker: dict[str, pd.DataFrame] = {}
    if not splits_df.empty:
        for t, grp in splits_df.groupby("ticker"):
            splits_by_ticker[t] = grp.reset_index(drop=True)

    frames: list[pd.DataFrame] = []
    for t in tickers:
        path = _fundamentals_path(t)
        if not os.path.exists(path):
            continue
        try:
            raw = pd.read_parquet(path)
        except Exception:
            continue
        if raw.empty:
            continue
        assembled = _assemble_ticker(t, raw, splits_by_ticker.get(t))
        if not assembled.empty:
            frames.append(assembled)

    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(["ticker", "period_end"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refresh", action="store_true", help="Wipe caches and redownload.")
    ap.add_argument("--tickers", help="Comma-separated subset.")
    ap.add_argument("--smoke", action="store_true",
                    help="After fetch, print the assembled panel for one ticker.")
    args = ap.parse_args()

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = all_historical_tickers(since="2005-07-01")

    print(f"\nXBRL: fetching companyfacts for {len(tickers)} tickers...", flush=True)
    fetch_fundamentals_universe(tickers, refresh=args.refresh)

    print(f"\nyfinance: fetching split history for {len(tickers)} tickers...", flush=True)
    fetch_splits(tickers, refresh=args.refresh)

    if args.smoke:
        smoke_t = tickers[0]
        print(f"\nAssembled panel for {smoke_t}:", flush=True)
        panel = load_fundamentals([smoke_t])
        if panel.empty:
            print("  (empty)", flush=True)
        else:
            print(panel.tail(8).to_string(), flush=True)
            print(f"\n  shape: {panel.shape}", flush=True)
            print(f"  asof_date range: {panel['asof_date'].min()} → {panel['asof_date'].max()}", flush=True)


if __name__ == "__main__":
    main()
