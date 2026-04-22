"""
intraday_strategy.py — 1-hour 2-pole oscillator signal generator.

No regime filter. Pure mean-reversion around the Butterworth filter trend,
tuned for 1-hour bars. Matches the manual approach Josh runs with members.
"""

import pandas as pd
import numpy as np

from indicators import butterworth_lowpass, two_pole_oscillator, average_true_range


# Parameters tuned for 1h bars
BW_CUTOFF_1H = 0.1       # Same cutoff, but on faster bars → faster response
SMA_PERIOD_1H = 20       # 20-hour moving average
OSC_UPPER = 0.5          # Overbought threshold
OSC_LOWER = -0.5         # Oversold threshold
ATR_PERIOD = 14
ATR_STOP_MULT = 2.0      # Tighter stop on intraday


def generate_intraday_signals(
    df: pd.DataFrame,
    allow_short: bool = True,
    bw_cutoff: float = BW_CUTOFF_1H,
    sma_period: int = SMA_PERIOD_1H,
    osc_upper: float = OSC_UPPER,
    osc_lower: float = OSC_LOWER,
    atr_stop_mult: float = ATR_STOP_MULT,
) -> pd.DataFrame:
    """
    Generate 2-pole oscillator signals on intraday data.

    Signal column: 1 = long, -1 = short, 0 = flat.

    Entry rules:
      - Long:   oscillator crosses up through osc_lower (from below)
      - Short:  oscillator crosses down through osc_upper (from above)
    Exit rules:
      - Long:   oscillator crosses down through 0, or ATR stop hit
      - Short:  oscillator crosses up through 0, or ATR stop hit
    """
    out = df.copy()

    bw = butterworth_lowpass(df["Close"], cutoff=bw_cutoff)
    osc_raw = two_pole_oscillator(df["Close"], cutoff=bw_cutoff, sma_period=sma_period)
    atr = average_true_range(df["High"], df["Low"], df["Close"], period=ATR_PERIOD)

    # Normalize oscillator to z-score so ±0.5 thresholds apply across all assets.
    # Uses a rolling window; strictly causal.
    zscore_window = 100
    osc_mean = osc_raw.rolling(zscore_window, min_periods=20).mean()
    osc_std = osc_raw.rolling(zscore_window, min_periods=20).std()
    osc = (osc_raw - osc_mean) / osc_std.replace(0, np.nan)
    osc = osc.fillna(0)

    out["BW_Filter"] = bw
    out["TwoPole_Osc"] = osc
    out["ATR"] = atr

    signals = np.zeros(len(df), dtype=int)
    entry_price = np.nan
    position = 0  # -1 short, 0 flat, 1 long

    osc_vals = osc.values
    close_vals = df["Close"].values
    atr_vals = atr.values

    for i in range(1, len(df)):
        prev_osc = osc_vals[i - 1]
        curr_osc = osc_vals[i]
        price = close_vals[i]
        atr_now = atr_vals[i] if not np.isnan(atr_vals[i]) else 0

        # Stop-loss check first
        if position == 1 and not np.isnan(entry_price):
            stop = entry_price - atr_stop_mult * atr_now
            if price <= stop:
                position = 0
                entry_price = np.nan
                signals[i] = 0
                continue
        elif position == -1 and not np.isnan(entry_price):
            stop = entry_price + atr_stop_mult * atr_now
            if price >= stop:
                position = 0
                entry_price = np.nan
                signals[i] = 0
                continue

        # Exit rules via oscillator crossing zero
        if position == 1 and prev_osc > 0 >= curr_osc:
            position = 0
            entry_price = np.nan
            signals[i] = 0
            continue
        if position == -1 and prev_osc < 0 <= curr_osc:
            position = 0
            entry_price = np.nan
            signals[i] = 0
            continue

        # Entry rules
        if position == 0:
            if np.isnan(prev_osc) or np.isnan(curr_osc):
                signals[i] = 0
                continue
            # Long entry: oscillator crossed up through lower threshold
            if prev_osc <= osc_lower < curr_osc:
                position = 1
                entry_price = price
                signals[i] = 1
                continue
            # Short entry: oscillator crossed down through upper threshold
            if allow_short and prev_osc >= osc_upper > curr_osc:
                position = -1
                entry_price = price
                signals[i] = -1
                continue

        # Hold the current position
        signals[i] = position

    out["Signal"] = signals
    return out


def classify_intraday_signal(last_signal: int, prev_signal: int) -> str:
    """Map signal values to human-readable action keys."""
    if last_signal == 1 and prev_signal != 1:
        return "buy"
    if last_signal == -1 and prev_signal != -1:
        return "enter_short"
    if last_signal == 0 and prev_signal == 1:
        return "sell_exit"
    if last_signal == 0 and prev_signal == -1:
        return "cover_short"
    if last_signal == 1:
        return "hold_long"
    if last_signal == -1:
        return "hold_short"
    return "flat"
