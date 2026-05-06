"""
intraday_executor.py — Hourly trade executor using the 1h 2-pole oscillator.

Reuses the Hyperliquid client code from hyperliquid_executor but runs on
intraday candles from Hyperliquid's own candle endpoint. Completely
independent state and capital from the daily bot.

Required environment variables (shared with daily bot):
    HL_PRIVATE_KEY / HL_ACCOUNT_ADDRESS / HL_TESTNET
    GIST_TOKEN / INTRADAY_GIST_ID    (separate from daily trading Gist)
    INTRADAY_CAPITAL                  (capital pool dedicated to intraday)
    INTRADAY_MAX_POSITIONS            (defaults to 2)
    INTRADAY_DD_PCT                   (daily drawdown cutoff, e.g. 5)
    INTRADAY_KILL_SWITCH              ("OFF" halts intraday only)
    GMAIL_USER / GMAIL_APP_PASSWORD / NOTIFY_EMAILS
    TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID
"""

import json
import os
import sys
import datetime as dt

import requests

from intraday_data_loader import fetch_all_intraday, HL_SYMBOL_MAP
from intraday_strategy import generate_intraday_signals, classify_intraday_signal
from hyperliquid_executor import (
    ASSETS,
    get_client,
    get_account_equity,
    get_open_positions,
    get_mid_price,
    get_size_decimals,
    round_size,
    _parse_response,
    _send_email,
    _send_telegram,
)
from backtester import get_asset_profile


STATE_FILENAME = "intraday_state.json"
POSITION_SIZE_PCT = 0.01


# ── State persistence (separate Gist from daily bot) ────────────────────────

def load_state() -> dict:
    gist_token = os.environ.get("GIST_TOKEN")
    gist_id = os.environ.get("INTRADAY_GIST_ID")
    if not gist_token or not gist_id:
        return {}
    resp = requests.get(
        f"https://api.github.com/gists/{gist_id}",
        headers={"Authorization": f"token {gist_token}"},
        timeout=15,
    )
    if resp.status_code != 200:
        return {}
    files = resp.json().get("files", {})
    if STATE_FILENAME not in files:
        return {}
    try:
        return json.loads(files[STATE_FILENAME]["content"])
    except Exception:
        return {}


def save_state(state: dict):
    gist_token = os.environ.get("GIST_TOKEN")
    gist_id = os.environ.get("INTRADAY_GIST_ID")
    if not gist_token or not gist_id:
        return
    requests.patch(
        f"https://api.github.com/gists/{gist_id}",
        headers={"Authorization": f"token {gist_token}"},
        json={"files": {STATE_FILENAME: {"content": json.dumps(state, indent=2)}}},
        timeout=15,
    )


# ── Signal computation ─────────────────────────────────────────────────────

def compute_intraday_signals() -> dict:
    """Fetch 1h candles and compute signals per asset."""
    all_data = fetch_all_intraday(list(ASSETS.keys()), interval="1h", lookback_hours=1000)
    current = {}

    for ticker in ASSETS:
        try:
            df = all_data.get(ticker)
            if df is None or df.empty or len(df) < 50:
                continue

            profile = get_asset_profile(ticker)
            sig = generate_intraday_signals(df, allow_short=profile["allow_short"])

            last = int(sig["Signal"].iloc[-1])
            prev = int(sig["Signal"].iloc[-2]) if len(sig) >= 2 else last
            action = classify_intraday_signal(last, prev)
            price = float(df["Close"].iloc[-1])
            osc = float(sig["TwoPole_Osc"].iloc[-1]) if "TwoPole_Osc" in sig.columns else 0.0

            current[ticker] = {
                "signal": last,
                "action": action,
                "price": price,
                "osc": osc,
            }
        except Exception as e:
            print(f"Error on {ticker}: {e}")

    return current


# ── Trade decisions ─────────────────────────────────────────────────────────

def decide_trades(signals: dict, open_positions: dict, max_positions: int) -> list[dict]:
    """Decide trades given new signals vs current HL positions."""
    trades = []

    # Close out positions that should exit
    for ticker, info in signals.items():
        hl_coin = HL_SYMBOL_MAP[ticker]
        pos = open_positions.get(hl_coin)
        if pos is None:
            continue
        is_long = pos["size"] > 0
        is_short = pos["size"] < 0

        action = info["action"]
        if (action == "sell_exit" and is_long) or \
           (action == "cover_short" and is_short) or \
           (action == "buy" and is_short) or \
           (action == "enter_short" and is_long):
            trades.append({
                "ticker": ticker, "hl_coin": hl_coin,
                "action": "close",
                "side": "long" if is_long else "short",
                "reason": f"{action} signal",
            })

    closes = {t["hl_coin"] for t in trades if t["action"] == "close"}
    remaining = {c: p for c, p in open_positions.items() if c not in closes}
    slots = max_positions - len(remaining)

    # Open new positions, prioritized by oscillator magnitude
    candidates = []
    for ticker, info in signals.items():
        hl_coin = HL_SYMBOL_MAP[ticker]
        if hl_coin in remaining:
            continue
        action = info["action"]
        # Open on fresh entry (buy/enter_short) OR sync when strategy
        # says we should be holding but we have no position.
        if action in ("buy", "hold_long"):
            reason = "buy signal" if action == "buy" else "sync to hold_long"
            candidates.append({
                "ticker": ticker, "hl_coin": hl_coin,
                "action": "open_long", "side": "long",
                "reason": reason,
                "priority": abs(info["osc"]),
            })
        elif action in ("enter_short", "hold_short"):
            reason = "enter_short signal" if action == "enter_short" else "sync to hold_short"
            candidates.append({
                "ticker": ticker, "hl_coin": hl_coin,
                "action": "open_short", "side": "short",
                "reason": reason,
                "priority": abs(info["osc"]),
            })

    candidates.sort(key=lambda c: c["priority"], reverse=True)
    trades.extend(candidates[:slots])
    return trades


def execute_trade(info, exchange, trade: dict, capital: float, leverage: float) -> dict:
    coin = trade["hl_coin"]
    if trade["action"] == "close":
        resp = exchange.market_close(coin)
        return _parse_response(trade, resp, info, coin)

    mid = get_mid_price(info, coin)
    notional = capital * POSITION_SIZE_PCT * leverage
    raw_size = notional / mid
    sz_decimals = get_size_decimals(info, coin)
    size = round_size(raw_size, sz_decimals)
    if size <= 0:
        return {**trade, "status": "skipped", "reason": "Size rounded to zero"}

    try:
        exchange.update_leverage(int(leverage), coin, True)
    except Exception as e:
        print(f"Leverage warning for {coin}: {e}")

    is_buy = trade["action"] == "open_long"
    resp = exchange.market_open(coin, is_buy, size)
    return _parse_response(trade, resp, info, coin)


# ── Guardrails ──────────────────────────────────────────────────────────────

def kill_switch_off() -> bool:
    return os.environ.get("INTRADAY_KILL_SWITCH", "ON").upper() == "OFF"


def check_daily_drawdown(state: dict, equity: float, threshold: float) -> tuple[bool, dict]:
    today = dt.date.today().isoformat()
    key = f"day_start_{today}"
    start = state.get(key)
    update = {}
    if start is None:
        update[key] = equity
        return False, update
    dd = (equity - start) / start * 100 if start > 0 else 0
    if dd <= -threshold:
        update["halted_today"] = today
        update["halt_reason"] = f"Intraday DD {dd:.2f}% exceeded {-threshold}%"
        return True, update
    return False, update


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print(f"Intraday executor started at {dt.datetime.now(dt.UTC).isoformat()}")

    if kill_switch_off():
        print("Intraday KILL_SWITCH is OFF — halting")
        sys.exit(0)

    try:
        info, exchange, address = get_client()
    except Exception as e:
        print(f"Client init failed: {e}")
        _send_email([], f"INTRADAY CLIENT INIT FAILED: {e}")
        sys.exit(1)

    state = load_state()
    equity = get_account_equity(info, address)
    threshold = float(os.environ.get("INTRADAY_DD_PCT", "5"))
    halted, state_update = check_daily_drawdown(state, equity, threshold)
    state.update(state_update)

    if halted:
        msg = f"Intraday halted: {state_update.get('halt_reason')}"
        print(msg)
        _send_email([], msg)
        save_state(state)
        sys.exit(0)

    today = dt.date.today().isoformat()
    if state.get("halted_today") == today:
        print(f"Already halted today: {state.get('halt_reason')}")
        sys.exit(0)

    signals = compute_intraday_signals()
    open_positions = get_open_positions(info, address)
    capital = float(os.environ.get("INTRADAY_CAPITAL", "5000"))
    max_positions = int(os.environ.get("INTRADAY_MAX_POSITIONS", "2"))

    # Filter out assets not listed on this Hyperliquid environment
    available = set(info.all_mids().keys())
    signals = {t: s for t, s in signals.items() if HL_SYMBOL_MAP[t] in available}
    skipped = [t for t in ASSETS if t not in signals]
    if skipped:
        print(f"Skipping unavailable assets on this env: {skipped}")

    # Ownership tracking: only manage positions this bot opened
    owned_coins = set(state.get("owned_coins", []))

    # Reconcile: drop owned coins that no longer have a position on the exchange
    stale_owned = owned_coins - set(open_positions.keys())
    if stale_owned:
        print(f"Dropping stale owned coins (no position on exchange): {stale_owned}")
        owned_coins -= stale_owned

    managed_positions = {c: p for c, p in open_positions.items() if c in owned_coins}

    trades = decide_trades(signals, managed_positions, max_positions)
    print(f"Decided on {len(trades)} intraday trade(s) (own {len(owned_coins)} position(s))")

    results = []
    for trade in trades:
        leverage = 2.0  # Fixed 2x for intraday; more conservative than daily
        result = execute_trade(info, exchange, trade, capital, leverage)
        results.append(result)
        print(f"  {result['ticker']} {result['action']}: {result.get('status')}")

        if result.get("status") == "filled":
            coin = result["hl_coin"]
            if result["action"] == "close":
                owned_coins.discard(coin)
            else:
                owned_coins.add(coin)

    history = state.get("history", [])
    for r in results:
        history.append({
            "timestamp": dt.datetime.now(dt.UTC).isoformat(),
            **{k: v for k, v in r.items() if k != "raw"},
        })
    state["history"] = history[-500:]
    state["last_equity"] = equity
    state["last_run"] = dt.datetime.now(dt.UTC).isoformat()
    state["owned_coins"] = sorted(owned_coins)
    latest = get_open_positions(info, address)
    state["open_positions"] = {c: p for c, p in latest.items() if c in owned_coins}
    state["last_signals"] = signals
    save_state(state)

    summary = f"{len(results)} intraday trade(s) | Equity: ${equity:,.2f}"
    if results:
        _send_email(results, summary)
        _send_telegram(results, summary)
    print("Done")


if __name__ == "__main__":
    main()
