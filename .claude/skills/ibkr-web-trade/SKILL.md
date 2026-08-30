---
name: ibkr-web-trade
description: Rebalance the IBKR account toward the latest picks CSV through the free web Client Portal instead of the paid IBKR Pro API — Playwright drives the logged-in portal session for positions, quotes and order placement. Use when the user says /ibkr-web-trade, or asks to rebalance / place the picks / trade the picks without IB Gateway.
---

# IBKR web trade (no API)

Does exactly what this used to do:

```
uv run python scripts/execute_picks.py --port 4001 --mode live --leverage 1.35 --vol-target 0.2 --top-n 20
```

…but with no IB Gateway and no IBKR Pro market-data subscription. The account
state, the quotes and the order tickets all come from the logged-in Client
Portal session in Playwright. Sizing, reconciliation, the vol-target overlay
and the notional circuit breaker are the **same** `scripts/execute_picks.py`
code, run as `--broker web`.

**Default = real execution** of that command against account `U26645119`
(LIVE, real money) using the newest `picks/picks_*.csv`. Any flag the user
names — `--mode whatif`, `--top-n 40`, a specific picks file, a different
`--leverage` — overrides that default and is passed straight through.

## Non-negotiables

1. **Never submit an order without the user's explicit go-ahead in chat.**
   Two gates, both required: the plan, then the final confirmation.
2. The user types their own username/password. Never ask for them, never store
   them, never type into the login form.
3. In live mode, cancel working orders **before** reading positions. A stale
   limit that fills after the read makes the plan re-buy what is already held.
4. Never re-run the placement step "to be sure". Read the order book back
   instead (step 8).
5. If any step returns an error, stop and report it. Do not improvise around a
   failed order.

## Steps

Work from the repo root:
`/home/talekien1710/personal_project/investment_strategy/ranker-21d-sp500`.

### 1. Settle the parameters

Defaults, unless the user said otherwise:

| | default |
|---|---|
| picks | newest `picks/picks_*.csv` (`ls -t picks/picks_*.csv \| head -1`) |
| mode | `live` |
| leverage | `1.35` |
| vol-target | `0.2` |
| top-n | `20` |
| account | `U26645119` |

Call the flags `$FLAGS` below:
`--broker web --mode <mode> --leverage <lev> --vol-target <vt> --top-n <n> [--picks <file>]`

### 2. Open the portal and check the session

`browser_navigate` to
`https://portal.interactivebrokers.com/portal/?loginType=1&action=ACCT_MGMT_MAIN#/portfolio`

Then `browser_evaluate`:

```js
async () => (await fetch('/portal.proxy/v1/portal/iserver/auth/status', {credentials:'include'})).json()
```

If `authenticated` is not `true`, tell the user to log in in the open browser
window (they handle 2FA) and wait for them to say they are done. Re-check
before continuing.

### 3. Cancel working orders — live mode only

Read `cancel_orders.js`, replace `__ACCOUNT__`, run it with `browser_evaluate`.
If `still_working` comes back non-empty, **stop**: report the names and ask the
user to clear them in the portal.

### 4. Which tickers

```bash
uv run python scripts/execute_picks.py $FLAGS --dump-tickers | tail -1
```

### 5. Capture the account

Read `capture_state.js`, replace `__ACCOUNT__` and `__TICKERS__` (a JS array
literal built from step 4), and run it with `browser_evaluate` using
`filename: "web_account.json"` so the JSON never lands in context. Playwright
writes it to the repo root — move it into place:

```bash
mv web_account.json reports/web_account.json
python3 -c "import json;d=json.load(open('reports/web_account.json'));print(d['account'],d['equity'],len(d['positions']),'missing:',d['missing'],'working:',d['working_orders'])"
```

Report `missing` (names with no quote — they get skipped) and any
`working_orders` to the user. Note `equity` = NetLiquidation.

### 6. Build the plan and show it

```bash
uv run python scripts/execute_picks.py $FLAGS
```

This places nothing. It prints the plan table and, for `whatif`/`live`, writes
`reports/web_orders.json` (sells first, marketable DAY limits 50 bps through
the reference price, one `coid` per order).

Show the user the plan table, the buy/sell totals, and the picks file it came
from — then **ask for confirmation to continue**. Stop here if the mode is
`print`.

If the script aborts on the price sanity check (a quote far from the cached
close = a non-US listing resolved), stop and report which ticker.

### 7. Cost preview, then place

Read `place_orders.js`. Substitute `__ACCOUNT__`, `__ORDERS__` (the `orders`
array from `reports/web_orders.json`, verbatim) and `__WHATIF__`.

**7a — `__WHATIF__ = true`.** IBKR returns commission and margin impact per
order and places nothing. Summarise total commission and the equity/margin
after. If the mode is `whatif`, **stop here** — done.

**7b — final confirmation.** Show the user: account `U26645119` (LIVE, real
money), order count, gross buy and sell notional, estimated commission. Ask
plainly whether to submit. Only a clear yes proceeds.

**7c — `__WHATIF__ = false`.** Run it again. Each order is submitted and its
confirmation prompts auto-replied. Report per-ticker: `order_id` / status, and
anything that came back with an `error`.

### 8. Record the book

```bash
uv run python scripts/execute_picks.py --broker web --record-book
```

Stamps `reports/live_book.json` with the intended gross exposure —
`scripts/today.py`'s daily de-risk check reads it. Live mode only.

### 9. Report

Working vs. filled orders (the `book` list returned by 7c), anything rejected,
anything skipped for a missing quote. Market closed → limits sit as DAY orders
and expire unfilled if never touched; say so, and note that re-running the
skill reconciles whatever did not fill.

## Notes

- `reports/web_account.json` goes stale fast; `--mode live` refuses it past 60
  minutes. Re-run step 5 rather than editing the timestamp.
- The portal quotes `mktPrice` for held names free of charge and mark/last for
  the rest; Yahoo only fills gaps the portal left empty.
- Re-running the whole skill after partial fills is safe and is the intended
  way to finish a rebalance — it re-reads positions and re-diffs.
- Account `U27177562` also exists on this login. It is **not** the strategy
  account. Never trade it without the user asking by number.
