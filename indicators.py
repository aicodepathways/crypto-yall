"""
indicators.py — Advanced DSP and statistical indicators.

All functions are strictly causal (no look-ahead bias).
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt


# ── Butterworth Low-Pass Filter (2-pole, causal) ────────────────────────────

def butterworth_lowpass(series: pd.Series, cutoff: float = 0.1, order: int = 2) -> pd.Series:
    """
    Apply a 2-pole Butterworth low-pass filter to *series*.

    Uses `sosfilt` (forward-only) so the filter is strictly causal —
    no future data leaks into the smoothed output.

    Parameters
    ----------
    series : pd.Series   – raw price series (Close)
    cutoff : float        – normalised cut-off frequency  (0 < cutoff < 1)
    order  : int          – filter order (2 = two-pole)
    """
    sos = butter(order, cutoff, btype="low", output="sos")
    filtered = sosfilt(sos, series.values)
    return pd.Series(filtered, index=series.index, name="BW_Filter")


# ── 2-Pole Oscillator ───────────────────────────────────────────────────────

def two_pole_oscillator(
    close: pd.Series,
    cutoff: float = 0.1,
    sma_period: int = 20,
) -> pd.Series:
    """
    2-Pole Oscillator = Butterworth-smoothed price  −  SMA(period).

    Measures the deviation of the filtered trend from its own moving
    average, producing a zero-centred oscillator.
    """
    bw = butterworth_lowpass(close, cutoff=cutoff)
    sma = bw.rolling(window=sma_period, min_periods=sma_period).mean()
    osc = bw - sma
    osc.name = "TwoPole_Osc"
    return osc


# ── Average True Range (causal) ─────────────────────────────────────────────

def average_true_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Wilder-style ATR (exponential smoothing)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    atr.name = "ATR"
    return atr


# ── Volatility-Scaled Momentum ──────────────────────────────────────────────

def volatility_scaled_momentum(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    mom_period: int = 14,
    atr_period: int = 14,
) -> pd.Series:
    """
    Returns  =  pct-change(mom_period) / ATR_ratio

    ATR_ratio = ATR / Close, so the metric is dimensionless.
    """
    returns = close.pct_change(periods=mom_period)
    atr = average_true_range(high, low, close, period=atr_period)
    atr_ratio = atr / close
    vsm = returns / atr_ratio.replace(0, np.nan)
    vsm.name = "VolScaled_Mom"
    return vsm


# ── Z-Score ──────────────────────────────────────────────────────────────────

def zscore(series: pd.Series, period: int = 20) -> pd.Series:
    """Rolling Z-Score = (x − mean) / std   over *period* bars."""
    mu = series.rolling(window=period, min_periods=period).mean()
    sigma = series.rolling(window=period, min_periods=period).std()
    z = (series - mu) / sigma.replace(0, np.nan)
    z.name = "ZScore"
    return z


# ── VWAP Deviation ───────────────────────────────────────────────────────────

def vwap_deviation(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Rolling VWAP deviation = (Close − VWAP) / Close.

    Positive → price above VWAP (demand), negative → below (supply).
    Used as a volume-flow feature for the HMM.
    """
    typical = (high + low + close) / 3
    tp_vol = typical * volume
    roll_tp_vol = tp_vol.rolling(window=period, min_periods=period).sum()
    roll_vol = volume.rolling(window=period, min_periods=period).sum()
    vwap = roll_tp_vol / roll_vol.replace(0, np.nan)
    dev = (close - vwap) / close
    dev.name = "VWAP_Dev"
    return dev


# ── Convenience: compute all indicators at once ─────────────────────────────

def compute_all(
    df: pd.DataFrame,
    bw_cutoff: float = 0.1,
    osc_sma: int = 20,
    atr_period: int = 14,
    mom_period: int = 14,
    zscore_period: int = 20,
) -> pd.DataFrame:
    """
    Attach every indicator to a copy of *df* and return it.

    New columns: BW_Filter, TwoPole_Osc, ATR, VolScaled_Mom, ZScore
    """
    out = df.copy()
    out["BW_Filter"] = butterworth_lowpass(df["Close"], cutoff=bw_cutoff)
    out["TwoPole_Osc"] = two_pole_oscillator(df["Close"], cutoff=bw_cutoff, sma_period=osc_sma)
    out["ATR"] = average_true_range(df["High"], df["Low"], df["Close"], period=atr_period)
    out["VolScaled_Mom"] = volatility_scaled_momentum(
        df["Close"], df["High"], df["Low"],
        mom_period=mom_period, atr_period=atr_period,
    )
    out["ZScore"] = zscore(df["Close"], period=zscore_period)
    out["VWAP_Dev"] = vwap_deviation(df["Close"], df["High"], df["Low"], df["Volume"])
    return out
