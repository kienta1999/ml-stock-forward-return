// Cancel every working order on the account, so the rebalance starts from a
// clean book. Must run BEFORE capture_state.js: a stale limit that fills after
// positions are read makes the plan double-buy a name already held.
// Placeholders to substitute before eval: __ACCOUNT__
async () => {
  const B = '/portal.proxy/v1/portal', A = '__ACCOUNT__';
  const gj = async u => (await fetch(u, {credentials: 'include'})).json();

  const auth = await gj(`${B}/iserver/auth/status`);
  if (!auth.authenticated) return {error: 'not authenticated — log in first'};

  const live = new Set(['PreSubmitted', 'Submitted', 'PendingSubmit',
                        'PendingCancel', 'PreSubmitted - Modified']);
  const j = await gj(`${B}/iserver/account/orders`);
  const working = (j.orders || []).filter(
    o => o.acct === A && live.has(o.status) && (o.remainingQuantity ?? 1) > 0);

  const cancelled = [];
  for (const o of working) {
    const r = await fetch(`${B}/iserver/account/${A}/order/${o.orderId}`,
                          {method: 'DELETE', credentials: 'include'});
    cancelled.push({ticker: o.ticker, side: o.side, qty: o.remainingQuantity,
                    orderId: o.orderId, status: r.status,
                    resp: (await r.text()).slice(0, 200)});
    await new Promise(res => setTimeout(res, 300));
  }
  await new Promise(res => setTimeout(res, 2000));
  const after = await gj(`${B}/iserver/account/orders`);
  const still = (after.orders || []).filter(o => o.acct === A && live.has(o.status));
  return {found: working.length, cancelled,
          still_working: still.map(o => `${o.ticker} ${o.side} ${o.status}`)};
}
