#!/usr/bin/env python3
"""Self-check for the --broker web path: reconciliation, ordering, staleness.

    uv run python scripts/test_execute_picks_web.py

Money path, so it is worth one assert-based check: the plan must diff against
held shares (not re-buy them), emit sells before buys, price limits on the
right side of the market, and refuse a stale positions file in live mode.
"""

import json
import os
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import execute_picks as ep


def _fixture(tmp: str, *, age_min: float = 0.0) -> tuple:
    picks = os.path.join(tmp, "picks_2099-01-01.csv")
    pd.DataFrame({"ticker": ["AAA", "CCC"],
                  "predicted_return": [0.02, 0.01],
                  "weight": [0.5, 0.5]}).to_csv(picks, index=False)
    state = os.path.join(tmp, "state.json")
    captured = pd.Timestamp.now() - pd.Timedelta(minutes=age_min)
    with open(state, "w") as fh:
        json.dump(dict(
            account="U00000000", equity=10_000.0,
            captured_at=captured.isoformat(timespec="seconds"),
            positions={"AAA": 5, "BBB": 10},
            prices={"AAA": 100.0, "BBB": 50.0, "CCC": 25.0},
            conids={"AAA": "1", "BBB": "2", "CCC": "3"},
        ), fh)
    return picks, state


def _run(picks: str, state: str, out: str, *extra: str) -> None:
    ep.WEB_ORDERS_PATH = out
    argv = sys.argv
    sys.argv = ["execute_picks.py", "--broker", "web", "--mode", "live",
                "--picks", picks, "--account-state", state,
                "--min-order", "1", *extra]
    try:
        ep.main()
    finally:
        sys.argv = argv


def demo() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "orders.json")
        picks, state = _fixture(tmp)
        _run(picks, state, out)
        payload = json.load(open(out))
        by = {o["ticker"]: o for o in payload["orders"]}

        # 5 of AAA already held against a 50-share target -> buy the 45 missing,
        # not the whole 50. BBB is held but unpicked -> full exit.
        assert by["AAA"] == {**by["AAA"], "action": "BUY", "qty": 45}, by["AAA"]
        assert by["CCC"]["action"] == "BUY" and by["CCC"]["qty"] == 200, by["CCC"]
        assert by["BBB"]["action"] == "SELL" and by["BBB"]["qty"] == 10, by["BBB"]

        # Sells first: the UI places one ticket at a time, so cash is freed
        # before it is spent.
        actions = [o["action"] for o in payload["orders"]]
        assert actions == sorted(actions, key=lambda a: a != "SELL"), actions

        # Marketable limits: buys above the reference price, sells below.
        assert by["AAA"]["limit"] > by["AAA"]["ref_price"], by["AAA"]
        assert by["BBB"]["limit"] < by["BBB"]["ref_price"], by["BBB"]
        assert all(o["conid"] and o["coid"] for o in payload["orders"])

        # Re-running the same plan reuses nothing stateful, but a second live
        # run must still be blocked once the position read has gone stale.
        picks, stale = _fixture(tmp, age_min=ep.ACCOUNT_STATE_MAX_AGE_MIN + 5)
        try:
            _run(picks, stale, out)
        except SystemExit as e:
            assert "ABORT" in str(e), e
        else:
            raise AssertionError("stale account state was accepted in live mode")

    print("ok — reconciliation, sell-first ordering, limit sides, staleness gate")


if __name__ == "__main__":
    demo()
