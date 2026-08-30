// Read the account into reports/web_account.json: NetLiquidation, held shares,
// live quotes and conids for every ticker the plan may touch. This is what
// replaces the IB Gateway API connection in scripts/execute_picks.py.
// Placeholders to substitute before eval: __ACCOUNT__, __TICKERS__
//   __TICKERS__ -> a JS array literal, e.g. ['ON','INTC',...]
async () => {
  const B = '/portal.proxy/v1/portal', A = '__ACCOUNT__';
  const TICKERS = __TICKERS__;
  // A bare symbol matches listings worldwide (STX is Seagate on NASDAQ but also
  // a EUREX index and an ASX miner) — insist on the US line.
  const US = /\((NASDAQ|NYSE|ARCA|AMEX|BATS|IEX|PINK|VALUE)\)/i;
  const num = v => {
    if (v == null) return null;                       // quotes arrive as "C148.70"
    const m = String(v).match(/-?\d[\d,]*\.?\d*/);
    return m ? parseFloat(m[0].replace(/,/g, '')) : null;
  };
  const gj = async u => (await fetch(u, {credentials: 'include'})).json();

  const auth = await gj(`${B}/iserver/auth/status`);
  if (!auth.authenticated) return {error: 'not authenticated — log in first'};

  const sum = await gj(`${B}/portfolio/${A}/summary`);
  const equity = sum.netliquidation.amount;

  const positions = {}, prices = {}, conids = {}, resolved = {};
  let page = 0, batch;
  do {
    batch = await gj(`${B}/portfolio/${A}/positions/${page}`);
    for (const p of batch) {
      if (p.assetClass !== 'STK' || p.currency !== 'USD' || !p.position) continue;
      const t = p.contractDesc.toUpperCase();
      positions[t] = (positions[t] || 0) + p.position;
      conids[t] = String(p.conid);
      resolved[t] = 'held';
      if (p.mktPrice > 0) prices[t] = p.mktPrice;     // held names quote for free
    }
    page++;
  } while (batch.length === 30 && page < 10);

  for (const t of TICKERS) {
    if (conids[t]) continue;
    const j = await gj(`${B}/iserver/secdef/search?symbol=${t}&name=false&secType=STK`);
    const hits = (j || []).filter(x => x.symbol === t);
    const hit = hits.find(x => US.test(x.companyHeader || '')) || hits[0];
    if (hit) { conids[t] = String(hit.conid); resolved[t] = hit.companyHeader; }
  }

  const need = TICKERS.filter(t => conids[t] && !prices[t]);
  if (need.length) {
    const u = `${B}/iserver/marketdata/snapshot?conids=${need.map(t => conids[t]).join(',')}`
            + `&fields=31,84,86,7635`;
    await fetch(u, {credentials: 'include'});         // first call only primes the feed
    await new Promise(r => setTimeout(r, 3500));
    const byId = {};
    for (const m of await gj(u)) byId[String(m.conid)] = m;
    for (const t of need) {
      const m = byId[conids[t]];
      if (!m) continue;
      const bid = num(m['84']), ask = num(m['86']);
      prices[t] = num(m['7635']) ?? num(m['31'])      // mark, else last,
                  ?? (bid && ask ? (bid + ask) / 2 : null);   // else mid
    }
  }

  const orders = await gj(`${B}/iserver/account/orders`);
  return {
    account: A, equity, captured_at: new Date().toISOString(),
    positions, prices, conids, resolved,
    missing: TICKERS.filter(t => !prices[t]),
    working_orders: (orders.orders || [])
      .filter(o => o.acct === A && /Submitted|Pending/.test(o.status || ''))
      .map(o => `${o.ticker} ${o.side} ${o.remainingQuantity} ${o.status}`),
  };
}
