"""
intraday_data_loader.py — Fetch intraday OHLCV candles from Hyperliquid.

Uses Hyperliquid's public candleSnapshot endpoint — no auth required.
"""

import datetime as dt
import time

import pandas as pd
import requests


HL_CANDLE_URL = "https://api.hyperliquid.xyz/info"

# Map yfinance-style tickers → Hyperliquid symbols
HL_SYMBOL_MAP = {
    "BTC-USD": "BTC",
    "ETH-USD": "ETH",
    "SOL-USD": "SOL",
    "AVAX-USD": "AVAX",
    "LINK-USD": "LINK",
    "SUI20947-USD": "SUI",
    "XRP-USD": "XRP",
}


def fetch_candles(
    ticker: str,
    interval: str = "1h",
    lookback_hours: int = 2000,  # ~83 days of 1h candles
) -> pd.DataFrame:
    """
    Fetch historical candles for a single asset from Hyperliquid.

    Parameters
    ----------
    ticker : "BTC-USD" style ticker (from our existing system)
    interval : "1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d" etc.
    lookback_hours : how many hours back to fetch

    Returns DataFrame with columns [Open, High, Low, Close, Volume] and a
    DatetimeIndex (UTC, timezone-naive to match daily loader).
    """
    coin = HL_SYMBOL_MAP.get(ticker, ticker)
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - lookback_hours * 3600 * 1000

    body = {
        "type": "candleSnapshot",
        "req": {
            "coin": coin,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
        },
    }

    for attempt in range(3):
        try:
            resp = requests.post(HL_CANDLE_URL, json=body, timeout=15)
            if resp.status_code == 200:
                candles = resp.json()
                break
        except Exception as e:
            print(f"Candle fetch attempt {attempt + 1} failed for {coin}: {e}")
        time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"Failed to fetch candles for {coin} after 3 attempts")

    if not candles:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    df = pd.DataFrame(candles)
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
    df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    df = df[["timestamp", "Open", "High", "Low", "Close", "Volume"]]
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)
    df = df.set_index("timestamp").sort_index()
    df.index.name = "Date"
    return df


def fetch_all_intraday(
    tickers: list[str],
    interval: str = "1h",
    lookback_hours: int = 2000,
) -> dict[str, pd.DataFrame]:
    """Fetch candles for multiple tickers."""
    data = {}
    for t in tickers:
        try:
            data[t] = fetch_candles(t, interval=interval, lookback_hours=lookback_hours)
        except Exception as e:
            print(f"Error fetching {t}: {e}")
            data[t] = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    return data
