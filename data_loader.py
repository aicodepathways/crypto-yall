"""
data_loader.py — Institutional-grade crypto data loader.
Fetches 4 years of daily OHLCV data for BTC-USD and ETH-USD via yfinance.
"""

import datetime as dt

import pandas as pd
import yfinance as yf

TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD", "SUI20947-USD", "XRP-USD"]
ANCHOR_START = dt.date(2022, 3, 1)  # Fixed start date — never shifts


def fetch_data(
    tickers: list[str] = TICKERS,
) -> dict[str, pd.DataFrame]:
    """
    Download daily OHLCV data for each ticker.

    Uses a fixed start date so adding new days never drops early data,
    keeping walk-forward fold boundaries and HMM training windows stable.

    Returns
    -------
    dict mapping ticker -> DataFrame with columns
    [Open, High, Low, Close, Volume] and a DatetimeIndex.
    """
    end = dt.date.today()
    start = ANCHOR_START

    data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = yf.download(
            ticker,
            start=start.isoformat(),
            end=end.isoformat(),
            auto_adjust=True,
            progress=False,
        )
        # yfinance may return MultiIndex columns; flatten if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.index.name = "Date"
        data[ticker] = df

    return data


if __name__ == "__main__":
    for sym, df in fetch_data().items():
        print(f"{sym}: {len(df)} rows  [{df.index[0].date()} → {df.index[-1].date()}]")
