"""
notifier.py — Standalone signal checker with email + Telegram alerts.

Runs via GitHub Actions cron. Checks all assets for signal transitions
and sends notifications when trades should be entered or exited.

Required environment variables:
    GMAIL_USER          – sender Gmail address
    GMAIL_APP_PASSWORD  – Gmail App Password (not regular password)
    NOTIFY_EMAILS       – comma-separated recipient emails
    TELEGRAM_BOT_TOKEN  – Telegram bot token from @BotFather
    TELEGRAM_CHAT_ID    – Telegram chat ID to send to
    GIST_TOKEN          – GitHub PAT with gist scope
    GIST_ID             – ID of the private Gist for state persistence
"""

import json
import os
import sys
import datetime as dt

import requests

from data_loader import fetch_data
from indicators import compute_all
from hmm_engine import causal_hmm_regimes
from strategy import generate_signals
from backtester import get_asset_profile
from signal_utils import classify_signal, SIGNAL_ACTIONS, TRADE_ACTIONS

ASSETS = {
    "BTC-USD": "Bitcoin (BTC)",
    "ETH-USD": "Ethereum (ETH)",
    "SOL-USD": "Solana (SOL)",
    "AVAX-USD": "Avalanche (AVAX)",
    "LINK-USD": "Chainlink (LINK)",
    "SUI20947-USD": "Sui (SUI)",
    "XRP-USD": "XRP",
}

STATE_FILENAME = "signal_state.json"


# ── State Persistence (GitHub Gist) ─────────────────────────────────────────

def load_state() -> dict:
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
        print(f"Warning: Could not load state from Gist (HTTP {resp.status_code})")
        return {}
    files = resp.json().get("files", {})
    if STATE_FILENAME not in files:
        return {}
    return json.loads(files[STATE_FILENAME]["content"])


def save_state(state: dict):
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


# ── Signal Checking ─────────────────────────────────────────────────────────

def check_all_signals() -> dict:
    """Run pipeline for all assets, return current signal state."""
    current = {}
    all_data = fetch_data(tickers=list(ASSETS.keys()))

    for ticker, name in ASSETS.items():
        try:
            raw = all_data[ticker]
            if raw.empty:
                print(f"Skipping {ticker}: no data")
                continue

            df = compute_all(raw)
            regimes, bull_probs, bear_probs = causal_hmm_regimes(df)
            profile = get_asset_profile(ticker)

            regime = regimes.iloc[-1] if len(regimes) > 0 else "Unknown"
            price = float(df["Close"].iloc[-1])
            bull_conf = float(bull_probs.iloc[-1]) if len(bull_probs) > 0 else 0.0

            for mode, aggressive in [("standard", False), ("aggressive", True)]:
                lev = profile["max_bull_leverage"] if aggressive else 1.0
                sig = generate_signals(
                    df, regimes, bull_probs=bull_probs, bear_probs=bear_probs,
                    aggressive=aggressive, bull_leverage=lev,
                    allow_short=profile["allow_short"], atr_mult=profile["atr_mult"],
                )
                last = int(sig["Signal"].iloc[-1])
                prev = int(sig["Signal"].iloc[-2]) if len(sig) >= 2 else last
                action_key = classify_signal(last, prev, regime)

                current.setdefault(ticker, {})[mode] = {
                    "signal": last,
                    "action": action_key,
                    "regime": regime,
                    "price": price,
                    "bull_conf": bull_conf,
                }

        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

    return current


def find_transitions(previous: dict, current: dict) -> list[dict]:
    """Compare states and return list of actionable transitions."""
    transitions = []
    prev_signals = previous.get("signals", {})

    for ticker, modes in current.items():
        for mode, info in modes.items():
            prev_info = prev_signals.get(ticker, {}).get(mode, {})
            prev_action = prev_info.get("action", "")

            if info["action"] != prev_action and info["action"] in TRADE_ACTIONS:
                action_text = SIGNAL_ACTIONS[info["action"]][0]
                transitions.append({
                    "ticker": ticker,
                    "name": ASSETS[ticker],
                    "mode": mode,
                    "action": action_text,
                    "regime": info["regime"],
                    "price": info["price"],
                    "bull_conf": info["bull_conf"],
                    "prev_action": SIGNAL_ACTIONS.get(prev_action, (prev_action, ""))[0],
                })

    return transitions


# ── Notification Senders ─────────────────────────────────────────────────────

def send_email(transitions: list[dict]):
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    user = os.environ.get("GMAIL_USER")
    password = os.environ.get("GMAIL_APP_PASSWORD")
    recipients = os.environ.get("NOTIFY_EMAILS", "")
    if not user or not password or not recipients:
        print("Email not configured, skipping")
        return

    recipient_list = [e.strip() for e in recipients.split(",")]

    rows = ""
    for t in transitions:
        mode_label = "Aggressive" if t["mode"] == "aggressive" else "Standard"
        rows += f"""
        <tr>
            <td style="padding:8px;border-bottom:1px solid #30363d;">{t['name']}</td>
            <td style="padding:8px;border-bottom:1px solid #30363d;font-weight:bold;color:#3fb950;">{t['action']}</td>
            <td style="padding:8px;border-bottom:1px solid #30363d;">{mode_label}</td>
            <td style="padding:8px;border-bottom:1px solid #30363d;">{t['regime']} ({t['bull_conf']:.0%})</td>
            <td style="padding:8px;border-bottom:1px solid #30363d;">${t['price']:,.2f}</td>
            <td style="padding:8px;border-bottom:1px solid #30363d;">{t['prev_action'] or 'N/A'}</td>
        </tr>"""

    html = f"""
    <div style="font-family:monospace;background:#0e1117;color:#c9d1d9;padding:20px;border-radius:12px;">
        <h2 style="color:#58a6ff;">Crypto Y'all Signal Alert</h2>
        <p>{dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</p>
        <table style="width:100%;border-collapse:collapse;margin-top:16px;">
            <tr style="color:#8b949e;text-transform:uppercase;font-size:0.8em;">
                <th style="padding:8px;text-align:left;">Asset</th>
                <th style="padding:8px;text-align:left;">Signal</th>
                <th style="padding:8px;text-align:left;">Mode</th>
                <th style="padding:8px;text-align:left;">Regime</th>
                <th style="padding:8px;text-align:left;">Price</th>
                <th style="padding:8px;text-align:left;">Previous</th>
            </tr>
            {rows}
        </table>
        <p style="color:#484f58;margin-top:20px;font-size:0.85em;">
            Crypto Y'all — Strictly causal — No look-ahead bias
        </p>
    </div>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[Crypto Y'all] Signal Alert: {', '.join(t['name'].split(' (')[0] + ' ' + t['action'] for t in transitions)}"
    msg["From"] = user
    msg["To"] = ", ".join(recipient_list)
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(user, password)
        server.send_message(msg)
    print(f"Email sent to {recipient_list}")


def send_telegram(transitions: list[dict]):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram not configured, skipping")
        return

    lines = ["*Crypto Y'all Signal Alert*", ""]
    for t in transitions:
        mode_label = "Aggressive" if t["mode"] == "aggressive" else "Standard"
        lines.append(f"*{t['name']}* — {t['action']}")
        lines.append(f"  Mode: {mode_label} | Regime: {t['regime']} ({t['bull_conf']:.0%})")
        lines.append(f"  Price: ${t['price']:,.2f} | Previous: {t['prev_action'] or 'N/A'}")
        lines.append("")

    lines.append(f"_{dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}_")

    text = "\n".join(lines)
    resp = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
        timeout=15,
    )
    if resp.status_code == 200:
        print(f"Telegram message sent to chat {chat_id}")
    else:
        print(f"Telegram error: {resp.status_code} {resp.text}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Signal check started at {dt.datetime.utcnow().isoformat()}Z")

    previous = load_state()
    current = check_all_signals()

    if not current:
        print("No signals computed, exiting")
        sys.exit(1)

    transitions = find_transitions(previous, current)

    if transitions:
        print(f"Found {len(transitions)} signal transition(s):")
        for t in transitions:
            print(f"  {t['name']} [{t['mode']}]: {t['prev_action']} -> {t['action']}")
        send_email(transitions)
        send_telegram(transitions)
    else:
        print("No signal transitions detected")

    # Save current state
    state = {
        "last_checked": dt.datetime.utcnow().isoformat() + "Z",
        "signals": current,
    }
    save_state(state)
    print("State saved")


if __name__ == "__main__":
    main()
