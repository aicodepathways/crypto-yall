"""
signal_utils.py — Shared signal-to-action mapping for dashboard and notifier.
"""

# CSS class is only used by the dashboard; notifier ignores it.
SIGNAL_ACTIONS = {
    "enter_short":    ("ENTER SHORT",       "signal-short"),
    "hold_short":     ("HOLD SHORT",        "signal-short"),
    "cover_short":    ("COVER SHORT",       "signal-buy"),
    "liquidate":      ("LIQUIDATE TO CASH", "signal-liq"),
    "buy":            ("BUY",               "signal-buy"),
    "hold_long":      ("HOLD LONG",         "signal-hold"),
    "sell_exit":      ("SELL / EXIT",        "signal-liq"),
    "flat":           ("FLAT",              "signal-flat"),
}


def classify_signal(last_signal: int, prev_signal: int, regime: str) -> str:
    """Return a key from SIGNAL_ACTIONS describing the current action."""
    if last_signal == -1 and prev_signal != -1:
        return "enter_short"
    elif last_signal == -1:
        return "hold_short"
    elif last_signal == 0 and prev_signal == -1:
        return "cover_short"
    elif regime == "Bear" and last_signal == 0:
        return "liquidate"
    elif last_signal == 1 and prev_signal == 0:
        return "buy"
    elif last_signal == 1:
        return "hold_long"
    elif last_signal == 0 and prev_signal == 1:
        return "sell_exit"
    else:
        return "flat"


def signal_to_action(last_signal: int, prev_signal: int, regime: str):
    """Return (action_text, css_class) tuple."""
    key = classify_signal(last_signal, prev_signal, regime)
    return SIGNAL_ACTIONS[key]


# Actions that represent a trade entry/exit (worth notifying about)
TRADE_ACTIONS = {"enter_short", "cover_short", "buy", "sell_exit", "liquidate"}
