"""
trading_state.py — Read-only access to live trading state for the dashboard.

The executor writes state to a private Gist; this module fetches it
for display without any trading capability.
"""

import json
import os
import requests


STATE_FILENAME = "trading_state.json"


def load_trading_state() -> dict:
    """Fetch live trading state from Gist. Returns {} if unavailable."""
    gist_token = os.environ.get("GIST_TOKEN") or _streamlit_secret("GIST_TOKEN")
    gist_id = os.environ.get("TRADING_GIST_ID") or _streamlit_secret("TRADING_GIST_ID")
    if not gist_token or not gist_id:
        return {}
    try:
        resp = requests.get(
            f"https://api.github.com/gists/{gist_id}",
            headers={"Authorization": f"token {gist_token}"},
            timeout=10,
        )
        if resp.status_code != 200:
            return {}
        files = resp.json().get("files", {})
        if STATE_FILENAME not in files:
            return {}
        return json.loads(files[STATE_FILENAME]["content"])
    except Exception:
        return {}


def _streamlit_secret(key: str):
    """Try to read from Streamlit secrets if available, else None."""
    try:
        import streamlit as st
        return st.secrets.get(key)
    except Exception:
        return None
