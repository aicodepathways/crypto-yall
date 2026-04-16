"""
hyperliquid_executor.py — Live trading executor for Hyperliquid DEX.

Translates signals from signal_utils into actual trades on Hyperliquid.
Runs daily via GitHub Actions, enforces risk guardrails, and sends
execution notifications through notifier.py.

Required environment variables:
    HL_PRIVATE_KEY      – API wallet private key (trading-only, no withdraw)
    HL_ACCOUNT_ADDRESS  – Main wallet address (0x…) that owns the funds
    HL_TESTNET          – "true" to use testnet, else mainnet
    SEGREGATED_CAPITAL  – USDC allocated to bot (e.g. "10000")
    DAILY_DD_PCT        – Max daily drawdown % before auto-pause (e.g. "5")
    MAX_POSITIONS       – Max concurrent open positions (4 aggressive)
    KILL_SWITCH         – "OFF" to halt all trading, else trades enabled
    GIST_TOKEN / GIST_ID – State persistence (same as notifier)
    GMAIL_USER / GMAIL_APP_PASSWORD / NOTIFY_EMAILS – email alerts
    TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID – telegram alerts
"""

import json
import os
import sys
import datetime as dt
from decimal import Decimal, ROUND_DOWN

import requests
from eth_account import Account

from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

from data_loader import fetch_data
from indicators import compute_all
from hmm_engine import causal_hmm_regimes
from strategy import generate_signals
from backtester import get_asset_profile
from signal_utils import classify_signal


# ── Config ───────────────────────────────────────────────────────────────────

# Map yfinance tickers → Hyperliquid symbols
HL_TICKER_MAP = {
    "BTC-USD": "BTC",
    "ETH-USD": "ETH",
    "SOL-USD": "SOL",
    "AVAX-USD": "AVAX",
    "LINK-USD": "LINK",
    "SUI20947-USD": "SUI",
    "XRP-USD": "XRP",
}

ASSETS = {
    "BTC-USD": "Bitcoin (BTC)",
    "ETH-USD": "Ethereum (ETH)",
    "SOL-USD": "Solana (SOL)",
    "AVAX-USD": "Avalanche (AVAX)",
    "LINK-USD": "Chainlink (LINK)",
    "SUI20947-USD": "Sui (SUI)",
    "XRP-USD": "XRP",
}

STATE_FILENAME = "trading_state.json"
POSITION_SIZE_PCT = 0.01  # 1% of segregated capital per trade


# ── State Persistence (GitHub Gist) ─────────────────────────────────────────

def load_trading_state() -> dict:
    gist_token = os.environ.get("GIST_TOKEN")
    gist_id = os.environ.get("GIST_ID")
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


def save_trading_state(state: dict):
    gist_token = os.environ.get("GIST_TOKEN")
    gist_id = os.environ.get("GIST_ID")
    if not gist_token or not gist_id:
        return
    requests.patch(
        f"https://api.github.com/gists/{gist_id}",
        headers={"Authorization": f"token {gist_token}"},
        json={"files": {STATE_FILENAME: {"content": json.dumps(state, indent=2)}}},
        timeout=15,
    )


# ── Hyperliquid Client ──────────────────────────────────────────────────────

def get_client():
    """Return (info, exchange, account_address)."""
    priv_key = os.environ.get("HL_PRIVATE_KEY")
    account_address = os.environ.get("HL_ACCOUNT_ADDRESS")
    is_testnet = os.environ.get("HL_TESTNET", "true").lower() == "true"

    if not priv_key or not account_address:
        raise RuntimeError("HL_PRIVATE_KEY and HL_ACCOUNT_ADDRESS required")

    base_url = constants.TESTNET_API_URL if is_testnet else constants.MAINNET_API_URL
    wallet = Account.from_key(priv_key)
    info = Info(base_url, skip_ws=True)
    exchange = Exchange(wallet, base_url, account_address=account_address)

    return info, exchange, account_address


def get_account_equity(info, address: str) -> float:
    """Total account value in USDC."""
    state = info.user_state(address)
    return float(state["marginSummary"]["accountValue"])


def get_open_positions(info, address: str) -> dict:
    """Return {coin: {size, entry_px, unrealized_pnl}} for open positions."""
    state = info.user_state(address)
    positions = {}
    for p in state.get("assetPositions", []):
        pos = p["position"]
        size = float(pos["szi"])
        if size == 0:
            continue
        positions[pos["coin"]] = {
            "size": size,  # signed: + long, - short
            "entry_px": float(pos["entryPx"]),
            "unrealized_pnl": float(pos["unrealizedPnl"]),
        }
    return positions


def get_mid_price(info, coin: str) -> float:
    return float(info.all_mids()[coin])


# ── Signal Computation (reuse notifier logic) ───────────────────────────────

def compute_all_signals() -> dict:
    """Return {ticker: {action_key, regime, price, bull_conf, signal}} — aggressive mode."""
    all_data = fetch_data(tickers=list(ASSETS.keys()))
    current = {}

    for ticker in ASSETS:
        try:
            raw = all_data.get(ticker)
            if raw is None or raw.empty:
                continue

            df = compute_all(raw)
            regimes, bull_probs, bear_probs = causal_hmm_regimes(df)
            profile = get_asset_profile(ticker)
            regime = regimes.iloc[-1] if len(regimes) > 0 else "Unknown"
            price = float(df["Close"].iloc[-1])
            bull_conf = float(bull_probs.iloc[-1]) if len(bull_probs) > 0 else 0.0
            bear_conf = float(bear_probs.iloc[-1]) if len(bear_probs) > 0 else 0.0

            # Aggressive mode only (per call decision)
            sig = generate_signals(
                df, regimes, bull_probs=bull_probs, bear_probs=bear_probs,
                aggressive=True, bull_leverage=profile["max_bull_leverage"],
                allow_short=profile["allow_short"], atr_mult=profile["atr_mult"],
            )
            last = int(sig["Signal"].iloc[-1])
            prev = int(sig["Signal"].iloc[-2]) if len(sig) >= 2 else last
            action_key = classify_signal(last, prev, regime)

            current[ticker] = {
                "signal": last,
                "action": action_key,
                "regime": regime,
                "price": price,
                "bull_conf": bull_conf,
                "bear_conf": bear_conf,
                "leverage": float(sig["Leverage"].iloc[-1]) if "Leverage" in sig.columns else 1.0,
            }
        except Exception as e:
            print(f"Error computing signal for {ticker}: {e}")
            continue

    return current


# ── Trade Decisions ─────────────────────────────────────────────────────────

def decide_trades(signals: dict, open_positions: dict, max_positions: int) -> list[dict]:
    """
    Reconcile signals vs current positions and return list of trade intents.

    Each intent: {ticker, hl_coin, action, side, reason}
    action: "open_long" | "open_short" | "close"
    """
    trades = []

    # Step 1: Determine which current positions need to be closed
    for ticker, info in signals.items():
        hl_coin = HL_TICKER_MAP[ticker]
        current_pos = open_positions.get(hl_coin)
        action_key = info["action"]

        if current_pos is None:
            continue

        is_long = current_pos["size"] > 0
        is_short = current_pos["size"] < 0

        # Close conditions
        should_close = False
        reason = ""
        if action_key == "sell_exit" and is_long:
            should_close = True
            reason = "SELL / EXIT signal"
        elif action_key == "liquidate" and (is_long or is_short):
            should_close = True
            reason = "LIQUIDATE TO CASH signal"
        elif action_key == "cover_short" and is_short:
            should_close = True
            reason = "COVER SHORT signal"
        # If signal flipped direction, close existing
        elif action_key == "buy" and is_short:
            should_close = True
            reason = "Signal flipped long while short"
        elif action_key == "enter_short" and is_long:
            should_close = True
            reason = "Signal flipped short while long"

        if should_close:
            trades.append({
                "ticker": ticker,
                "hl_coin": hl_coin,
                "action": "close",
                "side": "long" if is_long else "short",
                "reason": reason,
            })

    # Step 2: Determine which new positions to open
    # Count positions we'll have AFTER closes
    closes_by_coin = {t["hl_coin"] for t in trades if t["action"] == "close"}
    remaining_positions = {
        c: p for c, p in open_positions.items() if c not in closes_by_coin
    }
    slots_available = max_positions - len(remaining_positions)

    # Candidate opens, sorted by confidence (highest first)
    open_candidates = []
    for ticker, info in signals.items():
        hl_coin = HL_TICKER_MAP[ticker]
        action_key = info["action"]

        # Skip if we already have a position in the right direction
        existing = remaining_positions.get(hl_coin)
        if existing:
            continue

        if action_key == "buy":
            open_candidates.append({
                "ticker": ticker,
                "hl_coin": hl_coin,
                "action": "open_long",
                "side": "long",
                "reason": "BUY signal",
                "confidence": info["bull_conf"],
            })
        elif action_key == "enter_short":
            open_candidates.append({
                "ticker": ticker,
                "hl_coin": hl_coin,
                "action": "open_short",
                "side": "short",
                "reason": "ENTER SHORT signal",
                "confidence": info["bear_conf"],
            })

    open_candidates.sort(key=lambda x: x["confidence"], reverse=True)
    trades.extend(open_candidates[:slots_available])

    return trades


# ── Order Execution ─────────────────────────────────────────────────────────

def round_size(size: float, sz_decimals: int) -> float:
    """Round position size down to the coin's size decimals."""
    if sz_decimals <= 0:
        return float(int(size))
    q = Decimal("1").scaleb(-sz_decimals)
    return float(Decimal(str(size)).quantize(q, rounding=ROUND_DOWN))


def get_size_decimals(info, coin: str) -> int:
    meta = info.meta()
    for universe in meta.get("universe", []):
        if universe["name"] == coin:
            return int(universe["szDecimals"])
    return 3  # safe default


def execute_trade(info, exchange, trade: dict, capital: float, leverage: float) -> dict:
    """Execute a single trade via Hyperliquid market order. Returns result dict."""
    coin = trade["hl_coin"]

    if trade["action"] == "close":
        resp = exchange.market_close(coin)
        return _parse_response(trade, resp, info, coin)

    # Open new position: size = (capital * 0.01 * leverage) / price
    mid = get_mid_price(info, coin)
    notional = capital * POSITION_SIZE_PCT * leverage
    raw_size = notional / mid
    sz_decimals = get_size_decimals(info, coin)
    size = round_size(raw_size, sz_decimals)

    if size <= 0:
        return {**trade, "status": "skipped", "reason": "Size rounded to zero"}

    # Set leverage before opening (cross margin)
    try:
        exchange.update_leverage(int(leverage), coin, True)
    except Exception as e:
        print(f"Warning: could not set leverage for {coin}: {e}")

    is_buy = trade["action"] == "open_long"
    resp = exchange.market_open(coin, is_buy, size)
    return _parse_response(trade, resp, info, coin)


def _parse_response(trade: dict, resp: dict, info, coin: str) -> dict:
    """Extract fill info from Hyperliquid response."""
    result = {**trade}
    try:
        if resp.get("status") == "ok":
            statuses = resp["response"]["data"]["statuses"]
            for s in statuses:
                if "filled" in s:
                    f = s["filled"]
                    result["status"] = "filled"
                    result["fill_size"] = float(f["totalSz"])
                    result["fill_price"] = float(f["avgPx"])
                    result["oid"] = f.get("oid")
                    return result
                elif "error" in s:
                    result["status"] = "error"
                    result["error"] = s["error"]
                    return result
            result["status"] = "unknown"
            result["raw"] = resp
        else:
            result["status"] = "error"
            result["error"] = resp.get("response", str(resp))
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"Parse error: {e} | raw={resp}"
    return result


# ── Guardrails ──────────────────────────────────────────────────────────────

def check_kill_switch() -> bool:
    """Return True if trading should halt."""
    return os.environ.get("KILL_SWITCH", "ON").upper() == "OFF"


def check_daily_drawdown(state: dict, current_equity: float, threshold_pct: float) -> tuple[bool, dict]:
    """
    Check if today's drawdown exceeds threshold.
    Returns (should_halt, updated_state_fragment).
    """
    today = dt.date.today().isoformat()
    day_key = f"day_start_{today}"
    day_start = state.get(day_key)

    update = {}
    if day_start is None:
        update[day_key] = current_equity
        return False, update

    drawdown_pct = (current_equity - day_start) / day_start * 100 if day_start > 0 else 0

    if drawdown_pct <= -threshold_pct:
        update["halted_today"] = today
        update["halt_reason"] = f"Daily DD {drawdown_pct:.2f}% exceeded {-threshold_pct}%"
        return True, update

    return False, update


# ── Notifications ───────────────────────────────────────────────────────────

def send_execution_notifications(results: list[dict], status_summary: str):
    """Send email + telegram notifications for trade executions."""
    if not results and not status_summary:
        return

    # Email
    try:
        _send_email(results, status_summary)
    except Exception as e:
        print(f"Email send failed: {e}")

    # Telegram
    try:
        _send_telegram(results, status_summary)
    except Exception as e:
        print(f"Telegram send failed: {e}")


def _send_email(results: list[dict], status_summary: str):
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    user = os.environ.get("GMAIL_USER")
    password = os.environ.get("GMAIL_APP_PASSWORD")
    recipients = os.environ.get("NOTIFY_EMAILS", "")
    if not user or not password or not recipients:
        return

    recipient_list = [e.strip() for e in recipients.split(",")]

    rows = ""
    for r in results:
        status_color = "#3fb950" if r.get("status") == "filled" else "#f85149"
        fill_px = f"${r.get('fill_price', 0):,.2f}" if r.get("status") == "filled" else "—"
        fill_sz = f"{r.get('fill_size', 0):.6g}" if r.get("status") == "filled" else "—"
        rows += f"""
        <tr>
            <td style="padding:8px;border-bottom:1px solid #30363d;">{r['ticker']}</td>
            <td style="padding:8px;border-bottom:1px solid #30363d;">{r['action']}</td>
            <td style="padding:8px;border-bottom:1px solid #30363d;color:{status_color};font-weight:bold;">{r.get('status', '?').upper()}</td>
            <td style="padding:8px;border-bottom:1px solid #30363d;">{fill_sz}</td>
            <td style="padding:8px;border-bottom:1px solid #30363d;">{fill_px}</td>
            <td style="padding:8px;border-bottom:1px solid #30363d;">{r.get('reason', '')}</td>
        </tr>"""

    html = f"""
    <div style="font-family:monospace;background:#0e1117;color:#c9d1d9;padding:20px;border-radius:12px;">
        <h2 style="color:#58a6ff;">Crypto Y'all Trade Execution</h2>
        <p>{dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</p>
        <p><strong>Status:</strong> {status_summary}</p>
        {f'<table style="width:100%;border-collapse:collapse;margin-top:16px;"><tr style="color:#8b949e;"><th style="padding:8px;text-align:left;">Asset</th><th style="padding:8px;text-align:left;">Action</th><th style="padding:8px;text-align:left;">Status</th><th style="padding:8px;text-align:left;">Size</th><th style="padding:8px;text-align:left;">Fill Price</th><th style="padding:8px;text-align:left;">Reason</th></tr>{rows}</table>' if results else '<p>No trades executed this cycle.</p>'}
    </div>
    """

    msg = MIMEMultipart("alternative")
    summary = f"{len(results)} trade(s)" if results else "No trades"
    msg["Subject"] = f"[Crypto Y'all] Execution: {summary}"
    msg["From"] = user
    msg["To"] = ", ".join(recipient_list)
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(user, password)
        server.send_message(msg)
    print(f"Email sent to {recipient_list}")


def _send_telegram(results: list[dict], status_summary: str):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_ids_raw = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_ids_raw:
        return

    chat_ids = [c.strip() for c in chat_ids_raw.split(",") if c.strip()]

    lines = ["*Crypto Y'all Trade Execution*", "", f"Status: {status_summary}", ""]
    for r in results:
        status = r.get("status", "?").upper()
        lines.append(f"*{r['ticker']}* — {r['action']} [{status}]")
        if r.get("status") == "filled":
            lines.append(f"  Size: {r.get('fill_size', 0):.6g} @ ${r.get('fill_price', 0):,.2f}")
        elif r.get("error"):
            lines.append(f"  Error: {r['error']}")
        lines.append(f"  Reason: {r.get('reason', '')}")
        lines.append("")

    lines.append(f"_{dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}_")
    text = "\n".join(lines)

    for chat_id in chat_ids:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=15,
        )


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print(f"Trade executor started at {dt.datetime.utcnow().isoformat()}Z")

    # Kill switch check
    if check_kill_switch():
        print("KILL_SWITCH is OFF — halting all trading")
        send_execution_notifications([], "KILL SWITCH ACTIVE — no trades executed")
        sys.exit(0)

    try:
        info, exchange, address = get_client()
    except Exception as e:
        print(f"Failed to init Hyperliquid client: {e}")
        send_execution_notifications([], f"CLIENT INIT FAILED: {e}")
        sys.exit(1)

    # Load state and check drawdown
    state = load_trading_state()
    equity = get_account_equity(info, address)
    dd_threshold = float(os.environ.get("DAILY_DD_PCT", "5"))
    halted, state_update = check_daily_drawdown(state, equity, dd_threshold)
    state.update(state_update)

    if halted:
        msg = f"Daily drawdown triggered — halting today. {state_update.get('halt_reason')}"
        print(msg)
        send_execution_notifications([], msg)
        save_trading_state(state)
        sys.exit(0)

    # Check if already halted today
    today = dt.date.today().isoformat()
    if state.get("halted_today") == today:
        print(f"Already halted today: {state.get('halt_reason')}")
        sys.exit(0)

    # Compute signals and decide trades
    signals = compute_all_signals()
    open_positions = get_open_positions(info, address)
    capital = float(os.environ.get("SEGREGATED_CAPITAL", "10000"))
    max_positions = int(os.environ.get("MAX_POSITIONS", "4"))

    trades = decide_trades(signals, open_positions, max_positions)
    print(f"Decided on {len(trades)} trade(s)")

    results = []
    for trade in trades:
        sig_info = signals.get(trade["ticker"], {})
        leverage = max(1.0, min(sig_info.get("leverage", 1.0), 3.0))
        result = execute_trade(info, exchange, trade, capital, leverage)
        results.append(result)
        print(f"  {result['ticker']} {result['action']}: {result.get('status')} "
              f"{result.get('fill_size', '')} @ {result.get('fill_price', '')}")

    # Append to trade history
    history = state.get("history", [])
    for r in results:
        history.append({
            "timestamp": dt.datetime.utcnow().isoformat() + "Z",
            **{k: v for k, v in r.items() if k not in ("raw",)},
        })
    state["history"] = history[-500:]  # keep last 500 trades
    state["last_equity"] = equity
    state["last_run"] = dt.datetime.utcnow().isoformat() + "Z"
    state["open_positions"] = {
        c: p for c, p in get_open_positions(info, address).items()
    }

    save_trading_state(state)

    summary = f"{len(results)} trade(s) executed | Equity: ${equity:,.2f}"
    send_execution_notifications(results, summary)
    print("Done")


if __name__ == "__main__":
    main()
