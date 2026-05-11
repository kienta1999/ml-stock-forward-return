#!/usr/bin/env python3
"""Orchestrate the daily (or full) pipeline.

Universe (S&P 500 membership) is refreshed on every run — the upstream
github.com/fja05680/sp500 CSV is hand-maintained, so we re-pull Wikipedia
on each invocation to keep the synthetic "today" snapshot in
sp500_history.parquet aligned with the live roster. ~1s extra per run.

Modes:
    (default)         universe → data → earnings/insider/fundamentals/macro → features → labels → today
    --retrain         (default) + train + backtest
    --full            alias for --retrain (retained for back-compat)
    --download-only   universe → data → earnings → insider → fundamentals → macro (refresh caches only)

Other knobs:
    --no-diff       Skip --diff for today.py (don't auto-pick a prior file)
    --no-today      Stop before today.py (just refresh + maybe retrain)
    --dry-run       Print the plan, don't execute

For `--diff`, run_all auto-selects the most recent file in `picks/` whose
date is strictly before today's calendar date. If no prior file exists, it
runs today.py without --diff. Pass `--no-diff` to opt out.

Stops on the first failure. Exit code is the failed step's exit code, so
this script is cron-friendly.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from glob import glob

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
PICKS_DIR = os.path.join(_ROOT, "picks")

PICKS_DATE_RE = re.compile(r"picks_(\d{4}-\d{2}-\d{2})\.csv$")


# ─────────────────────────────────────────────────────────────────────────────
# Pick prior diff file
# ─────────────────────────────────────────────────────────────────────────────


def _pre_today_picks_file() -> str | None:
    """Most recent picks_<date>.csv whose date is strictly before today.

    Filtering by date (not by mtime) means rerunning later in the same
    calendar day still picks yesterday's file as the diff target rather
    than the file we're about to overwrite.
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    candidates: list[tuple[str, str]] = []
    for path in glob(os.path.join(PICKS_DIR, "picks_*.csv")):
        m = PICKS_DATE_RE.search(os.path.basename(path))
        if m and m.group(1) < today_str:
            candidates.append((m.group(1), path))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


# ─────────────────────────────────────────────────────────────────────────────
# Step runner
# ─────────────────────────────────────────────────────────────────────────────


def _run(label: str, cmd: list[str], dry_run: bool) -> int:
    print(f"\n{'═' * 70}")
    print(f"  {label}")
    print(f"  $ {' '.join(cmd)}")
    print("═" * 70)
    if dry_run:
        print("  (dry-run: not executing)")
        return 0
    t0 = time.time()
    result = subprocess.run(cmd, cwd=_ROOT)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"  → {status} in {elapsed:.1f}s")
    return result.returncode


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--retrain",
        action="store_true",
        help="Also run train.py + backtest.py.",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="Alias for --retrain (universe is now refreshed on every run).",
    )
    ap.add_argument(
        "--no-diff",
        action="store_true",
        help="Don't auto-supply --diff to today.py.",
    )
    ap.add_argument(
        "--no-today",
        action="store_true",
        help="Stop before today.py (refresh + optional retrain only).",
    )
    ap.add_argument(
        "--download-only",
        action="store_true",
        help="Stop after refreshing raw data caches (data, earnings, insider, "
             "fundamentals). Skip features, labels, train, and today.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan, don't execute.",
    )
    args = ap.parse_args()

    if args.full:
        args.retrain = True

    py = ["uv", "run", "python"]
    steps: list[tuple[str, list[str]]] = []

    # Universe always runs first: re-pulls Wikipedia (~1s) so that the
    # synthetic "today" snapshot in sp500_history.parquet is always today's
    # roster, not last week's. This is what keeps live picks correct after
    # S&P committee adds/removes a name without us having re-pulled the
    # github.com/fja05680/sp500 CSV.
    steps.append(("Refresh S&P 500 universe", py + ["scripts/universe.py", "--refresh"]))

    steps.append(("Incremental data refresh (yfinance)", py + ["scripts/data.py"]))
    steps.append(("Refresh EDGAR earnings calendar", py + ["scripts/earnings.py"]))
    steps.append(("Refresh EDGAR insider transactions (Form 4)", py + ["scripts/insider.py"]))
    steps.append(("Refresh EDGAR XBRL fundamentals", py + ["scripts/fundamentals.py"]))
    steps.append(("Refresh FRED macro series", py + ["scripts/macro.py"]))

    if args.download_only:
        # Skip everything after the raw-data refresh: no features, labels,
        # train, or today. `--full` still adds universe refresh above.
        return _execute(steps, args.dry_run)

    steps.append(("Rebuild features panel", py + ["scripts/features.py"]))
    steps.append(("Rebuild label panel", py + ["scripts/labels.py"]))

    if args.retrain:
        steps.append(("Train + tune model (~10–30 min)", py + ["scripts/train.py"]))
        steps.append(("Backtest", py + ["scripts/backtest.py"]))

    if not args.no_today:
        today_cmd = py + ["scripts/today.py"]
        if not args.no_diff:
            prev = _pre_today_picks_file()
            if prev:
                today_cmd += ["--diff", os.path.relpath(prev, _ROOT)]
        steps.append(("Generate today's picks", today_cmd))

    return _execute(steps, args.dry_run)


def _execute(steps: list[tuple[str, list[str]]], dry_run: bool) -> int:
    print(f"\nrun_all.py — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{len(steps)} step(s) planned:")
    for label, cmd in steps:
        print(f"  - {label}")

    overall_t0 = time.time()
    for label, cmd in steps:
        code = _run(label, cmd, dry_run)
        if code != 0:
            print(f"\n!!! Pipeline failed at: {label}")
            return code

    elapsed = time.time() - overall_t0
    print(f"\n{'═' * 70}")
    print(f"  Pipeline completed in {elapsed:.1f}s ({len(steps)} steps)")
    print("═" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
