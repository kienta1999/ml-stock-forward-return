// Post-trade readout: what the account actually looks like once the fills land.
// Pairs with reports/live_book.json (pre-trade equity + target gross) to produce
// the before/after summary in step 9.
// Placeholders to substitute before eval: __ACCOUNT__
async () => {
  const B = '/portal.proxy/v1/portal', A = '__ACCOUNT__';
  const gj = async u => (await fetch(u, {credentials: 'include'})).json();

  const s = await gj(`${B}/portfolio/${A}/summary`);
  const pnl = await gj(`${B}/iserver/account/pnl/partitioned`);
  const day = Object.values(pnl?.upnl || {})[0] || {};

  let page = 0, batch, gross = 0, held = 0;
  do {
    batch = await gj(`${B}/portfolio/${A}/positions/${page}`);
    for (const p of batch) {
      if (p.assetClass !== 'STK' || !p.position) continue;
      gross += Math.abs(p.position) * (p.mktPrice || 0); held++;
    }
    page++;
  } while (batch.length === 30 && page < 10);

  const orders = await gj(`${B}/iserver/account/orders`);
  const cash = s.totalcashvalue?.amount ?? 0;
  return {
    netliq: s.netliquidation?.amount,
    gross: Math.round(gross * 100) / 100,
    positions: held,
    cash,                                   // negative = margin loan
    loan: cash < 0 ? -cash : 0,
    maint_margin: s.maintmarginreq?.amount,
    excess_liquidity: s.excessliquidity?.amount,
    day_pnl: day.dpl ?? null,
    unfilled: (orders.orders || [])
      .filter(o => o.acct === A && o.status !== 'Filled' && o.status !== 'Cancelled')
      .map(o => `${o.ticker} ${o.side} ${o.status} rem=${o.remainingQuantity}`),
  };
}
