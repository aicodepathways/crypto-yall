"""
data_loader.py — Institutional-grade crypto data loader.
Fetches 4 years of daily OHLCV data for BTC-USD and ETH-USD via yfinance.
"""

import datetime as dt
import pandas as pd
import yfinance as yf

TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD", "SUI20947-USD", "XRP-USD"]
LOOKBACK_YEARS = 4

def fetch_data(
    tickers: list[str] = TICKERS,
    lookback_years: int = LOOKBACK_YEARS,
) -> dict[str, pd.DataFrame]:
    """
    Download daily OHLCV data for each ticker.

    Returns
    -------
    dict mapping ticker -> DataFrame with columns
    [Open, High, Low, Close, Volume] and a DatetimeIndex.
    """
    # HARDCODED: Freeze the end date to the last known good state (March 7)
    end = dt.date(2026, 3, 7)
    start = end - dt.timedelta(days=lookback_years * 365)

    data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        # Let yfinance handle the session natively
        df = yf.download(
            ticker,
            start=start.isoformat(),
            end=end.isoformat(),
            auto_adjust=True,
            progress=False
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
