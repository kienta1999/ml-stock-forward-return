#!/usr/bin/env python3
"""Sanity-check the WSL -> Windows IB Gateway connection before any order code runs.

This does NOT place orders. It only:
    1. Resolves the Windows host IP (WSL2 NAT gateway) unless --host is given.
    2. Opens a raw TCP socket to the API port to isolate networking from the API.
    3. Connects via ib_async and prints the account number + a big PAPER/LIVE
       banner so you can eyeball the `DU` (paper) vs `U` (live) prefix.
    4. Prints NetLiquidation / buying power so weight->share sizing has real inputs.

Ports:  4002 = IB Gateway paper   4001 = IB Gateway live
        7497 = TWS paper          7496 = TWS live

Networking:
  * Mirrored WSL networking (Win11 22H2+): Gateway is reachable at 127.0.0.1,
    pass `--host 127.0.0.1`.
  * NAT WSL networking (default): the Windows host is the default-route gateway
    (e.g. 172.24.96.1). This script auto-detects it; that IP can change on
    reboot, which is why we resolve it at runtime rather than hardcoding.

CLI:
    python scripts/check_ibkr_conn.py                  # auto host, port 4002, paper
    python scripts/check_ibkr_conn.py --host 127.0.0.1 # mirrored networking
    python scripts/check_ibkr_conn.py --port 4001      # live gateway (be careful)
    python scripts/check_ibkr_conn.py --client-id 7    # avoid a clientId collision
"""

import argparse
import socket
import subprocess
import sys


def resolve_windows_host() -> str:
    """Windows host IP as seen from WSL2 NAT = the default-route gateway."""
    try:
        out = subprocess.check_output(
            "ip route show default | awk '{print $3}'", shell=True, text=True
        ).strip()
        return out or "127.0.0.1"
    except Exception:
        return "127.0.0.1"


def socket_precheck(host: str, port: int, timeout: float = 4.0) -> bool:
    """Raw TCP probe so we can tell 'network/firewall' apart from 'API' problems."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError as e:
        print(f"  ✗ TCP connect to {host}:{port} failed: {e}")
        return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=None,
                   help="Gateway host. Default: auto-detect WSL->Windows gateway.")
    p.add_argument("--port", type=int, default=4002,
                   help="API socket port (default 4002 = Gateway paper).")
    p.add_argument("--client-id", type=int, default=1,
                   help="Unique clientId; bump if you hit a collision.")
    args = p.parse_args()

    host = args.host or resolve_windows_host()
    print(f"Target: {host}:{args.port}  (clientId={args.client_id})")

    print("\n[1/3] TCP socket precheck ...")
    if not socket_precheck(host, args.port):
        print("\n  Networking/firewall issue — the API never got a chance to answer.")
        print("  Check: Gateway is running & logged in, Windows Firewall allows the")
        print("  port inbound, and (NAT mode) the WSL IP is in Gateway Trusted IPs.")
        return 1
    print("  ✓ Port is open and accepting connections.")

    print("\n[2/3] Importing ib_async ...")
    try:
        from ib_async import IB
    except ImportError:
        print("  ✗ ib_async not installed. Run:  uv add ib_async")
        return 1
    print("  ✓ ib_async imported.")

    print("\n[3/3] Connecting to the API ...")
    ib = IB()
    try:
        ib.connect(host, args.port, clientId=args.client_id, timeout=10)
    except Exception as e:
        print(f"  ✗ API handshake failed: {e}")
        print("  Port was open but the API rejected/ignored us. Usual causes:")
        print("   - 'Enable ActiveX and Socket Clients' is off in Gateway API settings")
        print("   - this clientId is already connected (try --client-id 99)")
        print("   - the source IP isn't in Gateway's Trusted IPs")
        return 1

    try:
        accounts = ib.managedAccounts()
        acct = accounts[0] if accounts else "(unknown)"
        is_paper = acct.startswith("DU")
        banner = "PAPER (safe)" if is_paper else "*** LIVE — REAL MONEY ***"
        print("  ✓ Connected.\n")
        print("=" * 52)
        print(f"  Account : {acct}")
        print(f"  Mode    : {banner}")
        print(f"  Server  : {ib.client.serverVersion()}")
        print("=" * 52)

        vals = {v.tag: v.value for v in ib.accountValues(acct)
                if v.currency in ("USD", "BASE")}
        for tag in ("NetLiquidation", "AvailableFunds", "BuyingPower", "TotalCashValue"):
            if tag in vals:
                print(f"  {tag:<16}: {float(vals[tag]):>15,.2f}")

        positions = ib.positions(acct)
        print(f"\n  Open positions: {len(positions)}")
        for pos in positions[:10]:
            print(f"    {pos.contract.symbol:<6} {pos.position:>10.0f}"
                  f" @ avg {pos.avgCost:.2f}")

        if not is_paper:
            print("\n  ⚠  You are pointed at a LIVE account. For testing the pipeline,"
                  "\n     switch Gateway to Paper Trading and use --port 4002.")
    finally:
        ib.disconnect()

    print("\nDone — plumbing is good. Safe to build order code against this.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
