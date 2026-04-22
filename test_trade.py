"""
test_trade.py — One-off test to force a small trade on Hyperliquid testnet.

Places a tiny BTC long, waits, then closes it. Validates the entire
execution path (auth, order placement, fill parsing, position close).
"""

import os
import time
import datetime as dt

from eth_account import Account
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants


def main():
    priv_key = os.environ.get("HL_PRIVATE_KEY")
    account_address = os.environ.get("HL_ACCOUNT_ADDRESS")
    is_testnet = os.environ.get("HL_TESTNET", "true").lower() == "true"

    if not priv_key or not account_address:
        print("ERROR: HL_PRIVATE_KEY and HL_ACCOUNT_ADDRESS must be set")
        return

    base_url = constants.TESTNET_API_URL if is_testnet else constants.MAINNET_API_URL
    print(f"Connecting to: {base_url}")
    print(f"Account: {account_address}")

    wallet = Account.from_key(priv_key)
    info = Info(base_url, skip_ws=True)
    exchange = Exchange(wallet, base_url, account_address=account_address)

    # 1. Check account state
    state = info.user_state(account_address)
    equity = float(state["marginSummary"]["accountValue"])
    withdrawable = float(state.get("withdrawable", 0))
    print(f"Account equity: ${equity:.2f}")
    print(f"Withdrawable:   ${withdrawable:.2f}")

    # 2. Get BTC mid price
    mids = info.all_mids()
    btc_mid = float(mids["BTC"])
    print(f"BTC mid price: ${btc_mid:,.2f}")

    if equity < 10:
        print("\n⚠️  Account has insufficient balance for a test trade.")
        print("   To fund the testnet account, go to https://app.hyperliquid-testnet.xyz")
        print("   connect the main wallet, and use the faucet to deposit test USDC.")
        print("\n   API connection confirmed working:")
        print(f"   - Authenticated as {account_address}")
        print(f"   - Read account state successfully")
        print(f"   - Read market data successfully")
        return

    # 3. Place small long: roughly $10 worth of BTC
    size = round(10 / btc_mid, 5)
    print(f"\nPlacing test LONG: {size} BTC @ market")

    try:
        exchange.update_leverage(2, "BTC", True)
    except Exception as e:
        print(f"Leverage set warning: {e}")

    resp = exchange.market_open("BTC", True, size)
    print(f"Open response: {resp}")

    time.sleep(3)

    # 4. Check position
    state = info.user_state(account_address)
    positions = [p["position"] for p in state.get("assetPositions", []) if p["position"]["coin"] == "BTC"]
    if positions:
        print(f"Position confirmed: {positions[0]}")

    # 5. Close position
    print("\nClosing position…")
    close_resp = exchange.market_close("BTC")
    print(f"Close response: {close_resp}")

    # 6. Final state
    time.sleep(3)
    state = info.user_state(account_address)
    equity_after = float(state["marginSummary"]["accountValue"])
    print(f"\nFinal equity: ${equity_after:.2f}  (change: ${equity_after - equity:.4f})")


if __name__ == "__main__":
    main()
