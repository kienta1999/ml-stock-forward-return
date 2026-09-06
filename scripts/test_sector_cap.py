#!/usr/bin/env python3
"""Self-check for strategy.top_picks(sector_cap=...).

    uv run python scripts/test_sector_cap.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strategy  # noqa: E402
import execute_picks  # noqa: E402


def _panel(sectors: list[str]) -> pd.DataFrame:
    """One date's slice, predictions descending so rank == row order."""
    n = len(sectors)
    return pd.DataFrame({
        "ticker": [f"T{i}" for i in range(n)],
        "gics_sector": sectors,
        "predicted_return": [1.0 - i / n for i in range(n)],
    })


def main() -> None:
    # 1. cap=None is the old behaviour: pure top-N by prediction.
    day = _panel(["Tech"] * 10)
    assert list(strategy.top_picks(day, 4)["ticker"]) == ["T0", "T1", "T2", "T3"]

    # 2. The cap binds. 20% of 10 = 2 names per sector, so an all-Tech day
    #    comes back short rather than breaching.
    got = strategy.top_picks(day, 10, sector_cap=0.20)
    assert len(got) == 2, len(got)
    assert list(got["ticker"]) == ["T0", "T1"]

    # 3. It skips down the ranking, it does not just truncate: Tech fills its
    #    2 slots from the top, then Health's best names get picked up.
    day = _panel(["Tech", "Tech", "Tech", "Tech", "Health", "Health"])
    got = strategy.top_picks(day, 4, sector_cap=0.50)  # 2 per sector
    assert list(got["ticker"]) == ["T0", "T1", "T4", "T5"], list(got["ticker"])

    # 4. A cap the basket never reaches changes nothing.
    day = _panel(["Tech", "Health", "Fin", "Energy"])
    assert (list(strategy.top_picks(day, 4, sector_cap=0.50)["ticker"])
            == list(strategy.top_picks(day, 4)["ticker"]))

    # 5. NaN sectors are bucketed together, not dropped.
    day = _panel(["Tech", "Tech", "Tech"])
    day.loc[[1, 2], "gics_sector"] = None
    got = strategy.top_picks(day, 3, sector_cap=0.34)  # 1 per sector
    assert list(got["ticker"]) == ["T0", "T1"], list(got["ticker"])

    # 6. Cap is a share of top_n, so it scales with basket size.
    day = _panel(["Tech"] * 50)
    assert len(strategy.top_picks(day, 40, sector_cap=0.25)) == 10

    # 7. A missing gics_sector column falls back to unconstrained rather than
    #    raising — diagnostics.py slices don't always carry it.
    day = _panel(["Tech"] * 5).drop(columns=["gics_sector"])
    assert len(strategy.top_picks(day, 3, sector_cap=0.20)) == 3

    # ── execute_picks._apply_sector_cap: same rule, applied to a picks CSV ──
    picks = pd.DataFrame({
        "ticker": [f"T{i}" for i in range(10)],
        "predicted_return": [1.0 - i / 10 for i in range(10)],
        "gics_sector": ["Tech"] * 6 + ["Health"] * 4,
        "weight": [0.05] * 10,          # gross 0.5 (a vol-targeted basket)
    })

    # 8. Cap binds: 40% of 10 = 4 per sector, so 2 Tech names are dropped.
    out = execute_picks._apply_sector_cap(picks, 0.40, quiet=True)
    assert list(out["ticker"]) == ["T0", "T1", "T2", "T3", "T6", "T7", "T8", "T9"]

    # 9. Gross exposure is preserved, not shrunk — the survivors carry the
    #    same dollars, which is the whole point of rescaling.
    assert abs(out["weight"].sum() - picks["weight"].sum()) < 1e-12

    # 10. A non-binding cap returns the frame untouched.
    same = execute_picks._apply_sector_cap(picks, 0.90, quiet=True)
    assert same["weight"].sum() == picks["weight"].sum()
    assert list(same["ticker"]) == list(picks["ticker"])

    # 11. No gics_sector column → falls back to sp500_sectors.csv. Use real
    #     S&P tickers so the lookup resolves.
    real = pd.DataFrame({
        "ticker": ["NVDA", "AMD", "INTC", "JPM"],
        "predicted_return": [0.4, 0.3, 0.2, 0.1],
        "weight": [0.25] * 4,
    })
    out = execute_picks._apply_sector_cap(real, 0.50, quiet=True)
    assert len(out) == 3, list(out["ticker"])   # 2 IT max, JPM survives
    assert "JPM" in set(out["ticker"])
    assert abs(out["weight"].sum() - 1.0) < 1e-12

    # 12. Ordering regression: the cap must bound what is actually TRADED.
    #     A 40-name CSV already capped at 16/sector is a no-op at 0.40, so
    #     applying the cap before --top-n 20 would leave the traded basket
    #     uncapped. After the slice, 0.40 of 20 = 8 per sector.
    csv40 = pd.DataFrame({
        "ticker": [f"T{i}" for i in range(40)],
        "predicted_return": [1.0 - i / 40 for i in range(40)],
        # top 16 by rank are Tech: exactly at the cap for n=40, so a
        # pre-slice cap changes nothing.
        "gics_sector": ["Tech"] * 16 + ["Health"] * 12 + ["Fin"] * 12,
        "weight": [0.025] * 40,
    })
    assert len(execute_picks._apply_sector_cap(csv40, 0.40, quiet=True)) == 40

    traded = csv40.head(20).copy()          # what --top-n 20 leaves
    assert (traded["gics_sector"] == "Tech").sum() == 16   # 80% — uncapped
    out = execute_picks._apply_sector_cap(traded, 0.40, quiet=True)
    assert (out["gics_sector"] == "Tech").sum() == 8, out["gics_sector"].tolist()
    assert abs(out["weight"].sum() - traded["weight"].sum()) < 1e-12

    print("all sector-cap checks passed")


if __name__ == "__main__":
    main()
