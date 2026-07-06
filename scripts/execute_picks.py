#!/usr/bin/env python3
"""Rebalance an IBKR account toward a picks CSV — with a safe, staged path to live.

Reads a picks file (default: the latest picks/picks_*.csv), pulls current
positions + NetLiquidation from IB Gateway, diffs target vs. held, and turns
the difference into marketable limit orders. Reconciliation means a re-run
after fills produces (near-)zero orders — it will NOT double-buy.

Three modes, safest first — always dry-run before you trade:
    --mode print   client-side only. Prints the rebalance plan (BUY/SELL/HOLD,
                   share deltas, notional). Places NOTHING, and with no orders
                   sent it does not even need order permissions.
    --mode whatif  sends each order to IBKR flagged what-if. IBKR returns
                   commission + margin impact; nothing is placed or queued.
                   The honest total-cost preview.
    --mode live    actually places orders. First cancels ALL working orders on
                   the connection (stale limits from a previous run would
                   otherwise fill alongside fresh duplicates), then rebalances
                   from the clean book. Gated behind: a hard --max-notional
                   circuit breaker, a LIVE-account warning, and (on a U-prefix
                   account) a typed confirmation of the exact account number.

Sizing:
    target_$[ticker] = weight * NetLiquidation * leverage
    Picks `weight` already bakes in the vol-target exposure (sums to <= 1.0),
    so with --leverage 1.0 (default) the book is cash-only and pays no margin
    interest. --leverage > 1.0 borrows and is flagged loudly (IBKR Pro ~5.14%).

    --vol-target X makes leveraged sizing match the backtest overlay: gross
    exposure becomes min(leverage, X / spy_vol_20d) using the freshest cached
    SPY data, instead of a flat leverage multiple on the CSV weights. Without
    it, --leverage 1.35 in a stressed regime scales the CSV's baked-in
    de-risked exposure back UP by 1.35x — more risk than the backtest models
    (which caps at vol_target/spy_vol, not leverage * vol_target/spy_vol).
    The CSV weights are renormalized to gross 1.0 first, so the overlay is
    applied exactly once, from today's vol rather than the signal date's.

Networking (WSL -> Windows Gateway): host auto-detected from the WSL default
route unless --host is given. --port is REQUIRED, no default (4002 = paper,
4001 = live) so the live/paper choice is always explicit. See
scripts/check_ibkr_conn.py first.

CLI:
    uv run python scripts/execute_picks.py --port 4001                 # print plan (safe)
    uv run python scripts/execute_picks.py --port 4001 --mode whatif   # cost preview
    uv run python scripts/execute_picks.py --picks picks/picks_2026-06-30.csv --port 4001
    uv run python scripts/execute_picks.py --port 4001 --mode live --max-notional 10000
"""

import argparse
import glob
import math
import os
import subprocess
import sys

import pandas as pd


_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(_HERE)
PICKS_DIR = os.path.join(_ROOT, "picks")

VOL_STALE_DAYS_WARN = 7


# ─────────────────────────────────────────────────────────────────────────────
# Setup helpers
# ─────────────────────────────────────────────────────────────────────────────


def resolve_windows_host() -> str:
    """Windows host IP as seen from WSL2 NAT = the default-route gateway."""
    try:
        out = subprocess.check_output(
            "ip route show default | awk '{print $3}'", shell=True, text=True
        ).strip()
        return out or "127.0.0.1"
    except Exception:
        return "127.0.0.1"


def latest_picks_file() -> str:
    """Most recent picks/picks_*.csv by filename (dates sort lexically)."""
    files = sorted(glob.glob(os.path.join(PICKS_DIR, "picks_*.csv")))
    if not files:
        sys.exit(f"No picks_*.csv found in {PICKS_DIR}")
    return files[-1]


def current_vol_target_exposure(
    vol_target: float, leverage: float
) -> tuple[float, float, pd.Timestamp]:
    """Backtest-matching gross exposure from the freshest cached SPY data.

    Returns (exposure, spy_vol_20d, asof_date) where
    exposure = min(leverage, vol_target / spy_vol_20d) — the same
    strategy.vol_target_exposure the backtest applies at each rebalance.
    Reads the local SPY/VIX parquet cache (no IBKR needed); run
    scripts/data.py first if the cache is stale.
    """
    import strategy
    from data import load_market

    market_data = load_market()
    market = strategy.prepare_market(market_data["SPY"], market_data["VIX"])
    vol = market["spy_vol_20d"].dropna()
    if vol.empty:
        sys.exit("No spy_vol_20d in the SPY cache — run scripts/data.py first.")
    spy_vol = float(vol.iloc[-1])
    exposure = strategy.vol_target_exposure(spy_vol, vol_target, max_exposure=leverage)
    return exposure, spy_vol, pd.Timestamp(vol.index[-1])


# ─────────────────────────────────────────────────────────────────────────────
# Pricing
# ─────────────────────────────────────────────────────────────────────────────


def _snapshot_prices(ib, contracts: dict) -> dict:
    """One reqMktData pass. marketPrice(), else last close (works after hours)."""
    tickers = {t: ib.reqMktData(c, "", False, False) for t, c in contracts.items()}
    ib.sleep(2.5)
    out = {}
    for t, tk in tickers.items():
        px = tk.marketPrice()
        if px is None or px != px or px <= 0:          # NaN / non-positive
            px = tk.close if (tk.close and tk.close == tk.close) else None
        out[t] = px
    return out


def get_prices(ib, contracts: dict) -> dict:
    """ticker -> price for sizing. Tries live data, falls back to delayed-frozen.

    Any name with no price after the first pass is retried under delayed-frozen
    market data (type 4), so sizing still works with no live subscription or a
    closed market.
    """
    # Request delayed data (type 3) up front — no live L1 subscription needed,
    # and 15-min-delayed prices are fine for sizing. Stragglers retry under
    # delayed-frozen (type 4) for names with no active delayed tick (closed
    # market / thin book).
    ib.reqMarketDataType(3)
    prices = _snapshot_prices(ib, contracts)
    missing = {t: contracts[t] for t, p in prices.items() if not p}
    if missing:
        ib.reqMarketDataType(4)
        for t, px in _snapshot_prices(ib, missing).items():
            if px:
                prices[t] = px
    return prices


# ─────────────────────────────────────────────────────────────────────────────
# Rebalance plan
# ─────────────────────────────────────────────────────────────────────────────


def build_plan(
    picks: pd.DataFrame,
    equity: float,
    prices: dict,
    current_shares: dict,
    *,
    leverage: float,
    min_order: float,
    fractional: bool,
) -> pd.DataFrame:
    """Diff target vs. held. One row per ticker in (picks ∪ current holdings).

    target_$ = weight * equity * leverage; target shares = target_$ / price,
    floored to whole shares (default) or kept to 4dp when `fractional` (IBKR
    fractional trading, so a small account deploys ~100% instead of stranding
    cash on pricey names). Held names not in picks get target 0 (full exit).
    Rows whose |delta*price| < min_order are HOLD (skipped) to avoid churn.
    """
    target_dollars = {
        row.ticker: float(row.weight) * equity * leverage
        for row in picks.itertuples()
    }
    universe = set(target_dollars) | set(current_shares)
    eps = 1e-4 if fractional else 1.0   # smallest tradable delta

    rows = []
    for t in sorted(universe):
        px = prices.get(t)
        held = float(current_shares.get(t, 0.0))
        if not px:
            rows.append(dict(ticker=t, price=None, held=held, target=held,
                             delta=0.0, notional=0.0, action="SKIP (no price)"))
            continue
        raw = target_dollars.get(t, 0.0) / px
        target = round(raw, 4) if fractional else float(math.floor(raw))
        delta = round(target - held, 4)
        notional = abs(delta) * px
        if abs(delta) < eps or notional < min_order:
            action = "HOLD"
            delta = 0.0
        elif held == 0:
            action = "BUY (new)"
        elif target == 0:
            action = "SELL (exit)"
        else:
            action = "BUY (add)" if delta > 0 else "SELL (trim)"
        rows.append(dict(ticker=t, price=px, held=held, target=target,
                         delta=delta, notional=abs(delta) * px, action=action))
    return pd.DataFrame(rows)


def cancel_pending_orders(ib) -> None:
    """Cancel every working order visible on the connection, wait for confirms.

    Run in live mode BEFORE positions are read, so the rebalance plan starts
    from a clean book: a stale limit from a previous run can neither fill
    mid-run (double-buying a name the plan already counts) nor sit alongside
    a fresh duplicate order. Uses reqAllOpenOrders so orders from a previous
    session under the same clientId are picked up too.
    """
    ib.reqAllOpenOrders()
    ib.sleep(1.5)
    working = [t for t in ib.openTrades() if t.isActive()]
    if not working:
        print("Pending orders: none — book is clean.")
        return
    print(f"Pending orders: cancelling {len(working)} working order(s) first:")
    for t in working:
        o = t.order
        print(f"  {t.contract.symbol:<6} {o.action} {o.totalQuantity:.4g} "
              f"@ {o.lmtPrice} ({t.orderStatus.status})")
        ib.cancelOrder(o)
    # Wait for cancel confirms (up to ~10s) so positions reflect final fills.
    for _ in range(20):
        ib.sleep(0.5)
        if not any(t.isActive() for t in ib.openTrades()):
            break
    stuck = [t for t in ib.openTrades() if t.isActive()]
    if stuck:
        names = ", ".join(t.contract.symbol for t in stuck)
        sys.exit(f"🛑 ABORT: could not cancel working order(s): {names}. "
                 f"Cancel them in Gateway/TWS, then re-run.")
    print("  all cancelled.")


def make_order(action_buy: bool, qty: float, price: float, slippage_bps: float,
               account: str):
    """Marketable LIMIT order with fields set explicitly.

    Explicit tif/limit dodges the Gateway order-preset that otherwise trips
    error 10349 on bare market orders. Buys bid up / sells offer down by
    `slippage_bps` for a high fill probability without going full market.
    `qty` may be fractional (whole shares are passed as int). Fractional and
    RTH-only: fractional shares don't trade outside regular hours.
    """
    from ib_async import LimitOrder

    slip = slippage_bps / 1e4
    limit = price * (1 + slip) if action_buy else price * (1 - slip)
    limit = round(limit, 2)
    q = int(qty) if float(qty).is_integer() else round(float(qty), 4)
    order = LimitOrder("BUY" if action_buy else "SELL", q, limit)
    order.tif = "DAY"
    order.outsideRth = False
    order.account = account
    return order


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────


def _q(x: float) -> str:
    """Whole numbers as ints, fractions to 3dp — keeps the plan table tidy."""
    return f"{int(x):d}" if float(x).is_integer() else f"{x:.3f}"


def print_plan(plan: pd.DataFrame) -> None:
    print(f"\n{'TICKER':<8}{'ACTION':<13}{'HELD':>9}{'TARGET':>9}"
          f"{'DELTA':>9}{'PRICE':>10}{'NOTIONAL':>12}")
    print("-" * 70)
    for r in plan.itertuples():
        px = f"{r.price:,.2f}" if r.price else "   —"
        print(f"{r.ticker:<8}{r.action:<13}{_q(r.held):>9}{_q(r.target):>9}"
              f"{_q(r.delta):>9}{px:>10}{r.notional:>12,.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--picks", default=None,
                    help="Picks CSV. Default: latest picks/picks_*.csv.")
    ap.add_argument("--top-n", type=int, default=None,
                    help="Trade only the top N picks by predicted_return, "
                         "weights rescaled to preserve the CSV's gross exposure. "
                         "Default: every name in the CSV. 40 = backtest canonical; "
                         "20 backtests better and halves fixed order costs but is "
                         "UNVERIFIED in the leave-2008-out stress test.")
    ap.add_argument("--host", default=None,
                    help="Gateway host. Default: auto-detect WSL->Windows gateway.")
    ap.add_argument("--port", type=int, required=True,
                    help="API port — REQUIRED, no default (4002 = Gateway paper, "
                         "4001 = live) so live vs. paper is always a conscious choice.")
    ap.add_argument("--client-id", type=int, default=20)
    ap.add_argument("--mode", choices=("print", "whatif", "live"), default="print",
                    help="print = plan only (safe); whatif = cost preview; "
                         "live = actually place orders.")
    ap.add_argument("--leverage", type=float, default=1.0,
                    help="Gross exposure multiplier (default 1.0 = cash-only, "
                         "no margin). >1.0 borrows at IBKR Pro ~5.14%% APR.")
    ap.add_argument("--vol-target", type=float, default=None,
                    nargs="?", const=0.20,
                    help="Backtest-matching overlay: gross exposure = "
                         "min(leverage, VOL_TARGET / spy_vol_20d) from the "
                         "freshest cached SPY data (0.20 = backtest default). "
                         "CSV weights are renormalized to gross 1.0 first. "
                         "Without this flag, --leverage multiplies the CSV "
                         "weights flat — MORE exposure in stressed regimes "
                         "than the backtest models.")
    ap.add_argument("--slippage-bps", type=float, default=50.0,
                    help="Marketable-limit padding per side (default 50 bps). "
                         "Sizing prices are 15-min delayed, so a thin buffer "
                         "misses fills when the market drifts (2026-07-06: 3/20 "
                         "orders missed at 30 bps).")
    ap.add_argument("--fractional", action="store_true",
                    help="Fractional-share orders so a small account deploys ~100%% "
                         "instead of stranding cash on pricey names. NOTE: IBKR often "
                         "REJECTS fractional over the API (error 10243/10244) — it "
                         "needs the fractional-shares permission enabled and may be "
                         "desktop-TWS-only. Verify with --mode whatif first. RTH only.")
    ap.add_argument("--min-order", type=float, default=100.0,
                    help="Skip rebalances below this dollar value (default $100).")
    ap.add_argument("--max-notional", type=float, default=None,
                    help="Hard circuit breaker on total BUY notional. Default: "
                         "equity * leverage * 1.05. Abort if the plan exceeds it.")
    args = ap.parse_args()

    if args.vol_target is not None and not 0.0 < args.vol_target <= 1.0:
        sys.exit(f"--vol-target {args.vol_target} looks wrong — it is an annualized "
                 f"vol FRACTION (0.20 = 20%). A value > 1 would never bind and "
                 f"silently disable the overlay. Did you mean "
                 f"{args.vol_target / 100:.2f}? Bare --vol-target defaults to 0.20.")

    picks_path = args.picks or latest_picks_file()
    picks = pd.read_csv(picks_path)
    picks = picks[picks["weight"] > 0]
    if args.top_n is not None:
        if args.top_n < 1:
            sys.exit(f"--top-n {args.top_n} must be >= 1.")
        if args.top_n < len(picks):
            gross = float(picks["weight"].sum())
            sort_col = ("predicted_return"
                        if "predicted_return" in picks.columns else None)
            if sort_col:
                picks = picks.sort_values(sort_col, ascending=False)
            picks = picks.head(args.top_n).copy()
            picks["weight"] *= gross / float(picks["weight"].sum())
            print(f"--top-n {args.top_n}: kept top {args.top_n} by "
                  f"{sort_col or 'file order'}, weights rescaled to preserve "
                  f"gross {gross:.3f}")
    print(f"Picks: {picks_path}  ({len(picks)} names, "
          f"weights sum to {picks['weight'].sum():.3f})")
    if args.leverage != 1.0:
        print(f"⚠  leverage {args.leverage:.2f}x — this BORROWS on margin "
              f"(~5.14% APR). Cash-only is --leverage 1.0.")

    # Sizing: build_plan multiplies CSV weights by `sizing_leverage`;
    # `gross_target` (× equity) is the intended gross book, used for the
    # default --max-notional cap.
    sizing_leverage = args.leverage
    gross_target = args.leverage
    if args.vol_target is not None:
        weight_sum = float(picks["weight"].sum())
        if weight_sum <= 0:
            print("Vol-target overlay: picks carry no long weight — nothing to scale.")
        else:
            exposure, spy_vol, asof = current_vol_target_exposure(
                args.vol_target, args.leverage
            )
            age_days = (pd.Timestamp.now().normalize() - asof.normalize()).days
            stale = (f"  ⚠ {age_days}d old — run scripts/data.py first!"
                     if age_days > VOL_STALE_DAYS_WARN else "")
            print(f"Vol-target overlay: spy_vol_20d={spy_vol:.1%} "
                  f"(as of {asof.date()}){stale}")
            print(f"  gross = min(leverage {args.leverage:.2f}, "
                  f"{args.vol_target:.2f}/{spy_vol:.3f}) = {exposure:.2f}x  "
                  f"(CSV weights sum {weight_sum:.3f} renormalized to 1.0 first)")
            sizing_leverage = exposure / weight_sum
            gross_target = exposure

    host = args.host or resolve_windows_host()
    print(f"Connecting to {host}:{args.port} (clientId={args.client_id}) ...")

    try:
        from ib_async import IB, Stock
    except ImportError:
        sys.exit("ib_async not installed. Run:  uv add ib_async")

    ib = IB()
    try:
        ib.connect(host, args.port, clientId=args.client_id, timeout=15)
    except Exception as e:
        sys.exit(f"Connection failed: {e}\nRun scripts/check_ibkr_conn.py to debug.")

    try:
        acct = ib.managedAccounts()[0]
        is_live = not acct.startswith("DU")
        equity = float([v for v in ib.accountValues(acct)
                        if v.tag == "NetLiquidation" and v.currency == "USD"][0].value)
        print(f"Account: {acct}  ({'LIVE — REAL MONEY' if is_live else 'paper'})  "
              f"NetLiquidation=${equity:,.2f}")

        # Live mode starts from a clean book: cancel working orders BEFORE
        # reading positions, so a stale limit can't fill mid-run or duplicate
        # a fresh order. print/whatif are read-only — warn instead.
        if args.mode == "live":
            cancel_pending_orders(ib)
        else:
            ib.reqAllOpenOrders()
            ib.sleep(1.5)
            pending = [t for t in ib.openTrades() if t.isActive()]
            if pending:
                names = ", ".join(t.contract.symbol for t in pending)
                print(f"⚠  {len(pending)} working order(s) pending ({names}) — "
                      f"the plan below ignores them. --mode live cancels them "
                      f"before rebalancing.")

        # Qualify contracts; drop any that don't resolve (delisted/renamed).
        contracts = {}
        for t in picks["ticker"]:
            c = Stock(t, "SMART", "USD")
            contracts[t] = c
        qualified = ib.qualifyContracts(*contracts.values())
        good = {c.symbol for c in qualified}
        dropped = [t for t in contracts if t not in good]
        if dropped:
            print(f"⚠  could not qualify (skipping): {', '.join(dropped)}")
        contracts = {t: c for t, c in contracts.items() if t in good}

        # Current positions (STK/USD only).
        current_shares = {}
        for pos in ib.positions(acct):
            if pos.contract.secType == "STK" and pos.contract.currency == "USD":
                current_shares[pos.contract.symbol] = pos.position
        # Need contracts for held names not in picks so we can price + sell them.
        for sym in current_shares:
            if sym not in contracts:
                c = Stock(sym, "SMART", "USD")
                if ib.qualifyContracts(c):
                    contracts[sym] = c

        print(f"Fetching prices for {len(contracts)} names ...")
        prices = get_prices(ib, contracts)

        if args.fractional:
            print("Fractional mode: orders may be fractional shares (RTH only; "
                  "account must have fractional trading enabled).")
        plan = build_plan(
            picks[picks["ticker"].isin(contracts)], equity, prices, current_shares,
            leverage=sizing_leverage, min_order=args.min_order,
            fractional=args.fractional,
        )
        print_plan(plan)

        trades = plan[plan["delta"] != 0].copy()
        buy_notional = trades[trades["delta"] > 0]["notional"].sum()
        sell_notional = trades[trades["delta"] < 0]["notional"].sum()
        n_buy = int((trades["delta"] > 0).sum())
        n_sell = int((trades["delta"] < 0).sum())
        print(f"\nPlan: {n_buy} buys (${buy_notional:,.0f}), "
              f"{n_sell} sells (${sell_notional:,.0f}), "
              f"{int((plan['action'] == 'HOLD').sum())} holds.")

        # Circuit breaker.
        cap = args.max_notional if args.max_notional is not None \
            else equity * gross_target * 1.05
        if buy_notional > cap:
            sys.exit(f"\n🛑 ABORT: buy notional ${buy_notional:,.0f} exceeds "
                     f"cap ${cap:,.0f}. Raise --max-notional if intended.")

        if trades.empty:
            print("\nNothing to do — book already matches picks.")
            return

        if args.mode == "print":
            print("\n[print mode] No orders sent. Re-run with --mode whatif for "
                  "IBKR's commission/margin preview, or --mode live to trade.")
            return

        # Build the orders once; used by both whatif and live.
        order_specs = []
        for r in trades.itertuples():
            px = prices[r.ticker]
            order_specs.append((r.ticker, contracts[r.ticker],
                                make_order(r.delta > 0, abs(r.delta), px,
                                           args.slippage_bps, acct)))

        if args.mode == "whatif":
            print("\n[whatif mode] IBKR preview — NOTHING will be placed:")
            total_comm = 0.0
            n_ok = 0
            for tkr, contract, order in order_specs:
                st = ib.whatIfOrder(contract, order)
                comm = getattr(st, "commission", None)
                if comm is not None and comm == comm:
                    total_comm += float(comm)
                    n_ok += 1
                    print(f"  {tkr:<6} {order.action:<4} {order.totalQuantity:>8.4g} "
                          f"@ {order.lmtPrice:<9.2f} comm≈{float(comm):.2f} "
                          f"initMargin→{getattr(st, 'initMarginAfter', '?')}")
                else:
                    print(f"  {tkr:<6} {order.action:<4} {order.totalQuantity:>8.4g} "
                          f"@ {order.lmtPrice:<9.2f} (no preview returned)")
            n_fail = len(order_specs) - n_ok
            if n_ok:
                print(f"\n  Est. total commission ({n_ok} previews): "
                      f"${total_comm:,.2f} ({total_comm / equity * 1e4:.1f} bps of equity)")
            if n_fail:
                print(f"  ⚠ {n_fail}/{len(order_specs)} order(s) returned NO preview — "
                      f"rejected by IBKR (commonly fractional-via-API: error 10243/10244, "
                      f"needs fractional permission or the desktop TWS). The $ above "
                      f"EXCLUDES them, so it is not the full basket cost.")
            print("  Open orders queued:", len(ib.openTrades()), "(should be 0)")
            return

        # ── live mode ────────────────────────────────────────────────────────
        print(f"\n[LIVE mode] About to place {len(order_specs)} orders on "
              f"{'LIVE ' if is_live else 'paper '}account {acct}.")
        print(f"  Gross buy ${buy_notional:,.0f}, sell ${sell_notional:,.0f}.")
        if is_live:
            resp = input(f"  Type the account number ({acct}) to CONFIRM live "
                         f"orders, or anything else to abort: ").strip()
            if resp != acct:
                sys.exit("Aborted — confirmation did not match.")
        else:
            resp = input("  Type YES to place paper orders: ").strip()
            if resp != "YES":
                sys.exit("Aborted.")

        placed = []
        for tkr, contract, order in order_specs:
            placed.append((tkr, ib.placeOrder(contract, order)))
        ib.sleep(3)
        print("\nOrder status:")
        for tkr, trade in placed:
            print(f"  {tkr:<6} {trade.order.action:<4} "
                  f"{trade.order.totalQuantity:>8.4g} -> {trade.orderStatus.status}"
                  f" (filled {trade.orderStatus.filled:.4g})")
        print("\nMonitor/adjust remaining working orders in Gateway. Re-run this "
              "script later to reconcile any partial fills.")
    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
