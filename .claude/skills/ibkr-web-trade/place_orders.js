// Submit the orders from reports/web_orders.json through the Client Portal's
// own order endpoint — the same one the portal's order ticket posts to.
// WHATIF=true previews commission/margin and places NOTHING.
// Placeholders to substitute before eval: __ACCOUNT__, __WHATIF__, __ORDERS__
//   __ORDERS__ -> the "orders" array from reports/web_orders.json, verbatim
async () => {
  const B = '/portal.proxy/v1/portal', A = '__ACCOUNT__';
  const WHATIF = __WHATIF__;
  const ORDERS = __ORDERS__;

  const auth = await (await fetch(`${B}/iserver/auth/status`, {credentials: 'include'})).json();
  if (!auth.authenticated) return {error: 'not authenticated — log in first'};

  const post = async (u, body) => {
    const r = await fetch(u, {method: 'POST', credentials: 'include',
                              headers: {'Content-Type': 'application/json'},
                              body: JSON.stringify(body)});
    return {status: r.status, json: await r.json().catch(() => null)};
  };

  const out = [];
  for (const o of ORDERS) {
    if (!o.conid) { out.push({ticker: o.ticker, error: 'no conid'}); continue; }
    const order = {
      acctId: A, conid: Number(o.conid), secType: `${o.conid}:STK`,
      orderType: 'LMT', side: o.action, quantity: o.qty, price: o.limit,
      tif: 'DAY', outsideRTH: false, referrer: 'QuickTrade',
    };
    if (!WHATIF) order.cOID = o.coid;          // duplicate cOID = IBKR rejects a re-run

    const url = `${B}/iserver/account/${A}/orders` + (WHATIF ? '/whatif' : '');
    let res = await post(url, {orders: [order]});
    // Live submits come back as a chain of confirmation prompts (price cap,
    // order-value warnings) that each need an explicit reply before the order
    // reaches the market. Bounded so a prompt loop can't spin.
    let guard = 0;
    while (!WHATIF && Array.isArray(res.json) && res.json[0]?.id
           && res.json[0]?.message && guard++ < 6) {
      res = await post(`${B}/iserver/reply/${res.json[0].id}`, {confirmed: true});
    }
    out.push({ticker: o.ticker, action: o.action, qty: o.qty, limit: o.limit,
              status: res.status, resp: res.json});
    await new Promise(r => setTimeout(r, 400));
  }
  if (WHATIF) return out;

  await new Promise(r => setTimeout(r, 3000));
  const live = await (await fetch(`${B}/iserver/account/orders`, {credentials: 'include'})).json();
  return {submitted: out, book: (live.orders || [])
    .filter(o => o.acct === A)
    .map(o => `${o.ticker} ${o.side} ${o.totalSize ?? o.remainingQuantity} `
            + `@${o.price ?? ''} ${o.status} filled=${o.filledQuantity ?? 0}`)};
}
