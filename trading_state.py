"""
trading_state.py — Read-only access to live trading state for the dashboard.

The executors write state to private Gists; this module fetches them
for display without any trading capability.
"""

import json
import os
import requests


DAILY_STATE_FILE = "trading_state.json"
INTRADAY_STATE_FILE = "intraday_state.json"


def _fetch_gist(gist_id: str, filename: str) -> dict:
    gist_token = os.environ.get("GIST_TOKEN") or _streamlit_secret("GIST_TOKEN")
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
        if filename not in files:
            return {}
        return json.loads(files[filename]["content"])
    except Exception:
        return {}


def load_trading_state() -> dict:
    """Daily bot state."""
    gist_id = os.environ.get("TRADING_GIST_ID") or _streamlit_secret("TRADING_GIST_ID")
    return _fetch_gist(gist_id, DAILY_STATE_FILE)


def load_intraday_state() -> dict:
    """Intraday bot state."""
    gist_id = os.environ.get("INTRADAY_GIST_ID") or _streamlit_secret("INTRADAY_GIST_ID")
    return _fetch_gist(gist_id, INTRADAY_STATE_FILE)


def _streamlit_secret(key: str):
    try:
        import streamlit as st
        return st.secrets.get(key)
    except Exception:
        return None
