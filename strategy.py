"""
strategy.py — Regime-adaptive trading strategy.

Standard Mode
-------------
Bear  → Liquidate to cash (position = 0).
Chop  → Mean-reversion via the 2-Pole Oscillator + Z-Score.  1x leverage.
Bull  → Trend-following via Vol-Scaled Momentum + Oscillator pullback.  1x leverage.

Smart Aggressive Mode
---------------------
Bear  → Smart shorting (prob-scaled leverage, inverted Chandelier Exit).
Chop  → Relaxed mean-reversion + opportunistic shorts on strong negative momentum.
Bull  → Trend-following + pyramiding on new 2-day highs.
        Leverage = bull_leverage × bull_probability (probabilistic sizing).
All positions protected by a Chandelier Exit (ATR-based trailing stop):
    Stop = Highest-High since entry  −  (atr_mult × ATR_14)
"""

import numpy as np
import pandas as pd


def generate_signals(
    df: pd.DataFrame,
    regimes: pd.Series,
    bull_probs: pd.Series = None,
    bear_probs: pd.Series = None,
    osc_lower: float = -1.0,
    osc_upper: float = 1.0,
    zscore_entry: float = -2.0,
    zscore_exit: float = 2.0,
    mom_threshold: float = 0.0,
    aggressive: bool = False,
    atr_mult: float = 3.0,
    bull_leverage: float = 3.0,
    allow_short: bool = True,
) -> pd.DataFrame:
    """
    Produce signal + leverage Series aligned to *df.index*.

    Signal values:
        +1  = long
         0  = flat / cash
        -1  = short (Smart Aggressive Bear regime only, if allow_short=True)

    Leverage column carries the effective multiplier per bar.

    Parameters
    ----------
    bull_probs : pd.Series or None
        HMM posterior probability of the Bull state (0–1).
    bear_probs : pd.Series or None
        HMM posterior probability of the Bear state (0–1).
        Used to scale short leverage in aggressive mode.
    aggressive : bool
        Enable probabilistic leverage, pyramiding, Chandelier Exit, and Bear shorting.
    atr_mult : float
        ATR multiplier for the Chandelier Exit (default 3.0).
    bull_leverage : float
        Maximum leverage multiplier during Bull regime (aggressive only).
    allow_short : bool
        If False, disable all short positions (used for volatile mid-cap alts).
    """
    osc = df["TwoPole_Osc"]
    zscore_s = df["ZScore"]
    mom = df["VolScaled_Mom"]
    close = df["Close"]
    high = df["High"]
    atr = df["ATR"]

    n = len(df)
    signal = pd.Series(0, index=df.index, name="Signal", dtype=int)
    leverage = pd.Series(0.0, index=df.index, name="Leverage", dtype=float)

    position = 0            # +1 long, 0 flat, -1 short
    highest_since = np.nan  # Chandelier Exit tracking (longs)
    lowest_since = np.nan   # Inverted Chandelier Exit tracking (shorts)
    pyramid_bonus = 0.0     # extra leverage from pyramiding
    day_high_streak = 0     # consecutive 2-day highs for pyramiding
    low = df["Low"]

    for i in range(1, n):
        regime = regimes.iloc[i]
        prev_osc = osc.iloc[i - 1] if not np.isnan(osc.iloc[i - 1]) else 0.0
        curr_osc = osc.iloc[i] if not np.isnan(osc.iloc[i]) else 0.0
        curr_z = zscore_s.iloc[i] if not np.isnan(zscore_s.iloc[i]) else 0.0
        curr_mom = mom.iloc[i] if not np.isnan(mom.iloc[i]) else 0.0
        curr_close = close.iloc[i]
        curr_high = high.iloc[i]
        curr_low = low.iloc[i]
        curr_atr = atr.iloc[i] if not np.isnan(atr.iloc[i]) else 0.0

        # Bull probability for this bar (default 0.5 if unavailable)
        if bull_probs is not None and not np.isnan(bull_probs.iloc[i]):
            bp = float(bull_probs.iloc[i])
        else:
            bp = 0.5

        # Bear probability for this bar (default 0.5 if unavailable)
        if bear_probs is not None and not np.isnan(bear_probs.iloc[i]):
            bear_p = float(bear_probs.iloc[i])
        else:
            bear_p = 0.5

        # ── Chandelier Exit check for LONGS (Smart Aggressive only) ──
        if aggressive and position == 1:
            highest_since = max(highest_since, curr_high) if not np.isnan(highest_since) else curr_high
            stop_level = highest_since - atr_mult * curr_atr
            if curr_close <= stop_level:
                position = 0
                highest_since = np.nan
                pyramid_bonus = 0.0
                day_high_streak = 0
                signal.iloc[i] = 0
                leverage.iloc[i] = 0.0
                continue

        # ── Inverted Chandelier Exit check for SHORTS (Smart Aggressive only) ──
        if aggressive and position == -1:
            lowest_since = min(lowest_since, curr_low) if not np.isnan(lowest_since) else curr_low
            stop_level = lowest_since + atr_mult * curr_atr
            if curr_close >= stop_level:
                position = 0
                lowest_since = np.nan
                signal.iloc[i] = 0
                leverage.iloc[i] = 0.0
                continue

        # ── Regime logic ────────────────────────────────────────────
        if regime == "Bear":
            if aggressive:
                # Smart shorting: enter/exit shorts based on oscillator
                if position == 1:
                    # Close any long immediately
                    position = 0
                    highest_since = np.nan
                    pyramid_bonus = 0.0
                    day_high_streak = 0
                if position == 0 and allow_short:
                    # Enter short on oscillator crossing below upper band
                    if prev_osc >= osc_upper > curr_osc and curr_mom < mom_threshold:
                        position = -1
                        lowest_since = curr_low
                elif position == -1:
                    # Cover short on oscillator crossing above lower band (mean reversion)
                    if prev_osc <= osc_lower < curr_osc:
                        position = 0
                        lowest_since = np.nan
            else:
                # Standard mode: go to cash
                position = 0
                highest_since = np.nan
                pyramid_bonus = 0.0
                day_high_streak = 0

        elif regime == "Chop":
            if position == 0:
                if aggressive:
                    # Relaxed long entry: oscillator crossover only (no Z-Score gate)
                    if prev_osc <= osc_lower < curr_osc:
                        position = 1
                        highest_since = curr_high
                        pyramid_bonus = 0.0
                        day_high_streak = 0
                    # Opportunistic short: strong negative momentum in Chop
                    elif allow_short and prev_osc >= osc_upper > curr_osc and curr_mom < -0.5:
                        position = -1
                        lowest_since = curr_low
                else:
                    # Standard: require Z-Score confirmation
                    if prev_osc <= osc_lower < curr_osc and curr_z < zscore_entry:
                        position = 1
            elif position == 1:
                if aggressive:
                    # Tighter exit: oscillator crosses below upper band OR Z > exit
                    if (prev_osc >= osc_upper > curr_osc) or curr_z > zscore_exit:
                        position = 0
                        highest_since = np.nan
                        pyramid_bonus = 0.0
                        day_high_streak = 0
                else:
                    if (prev_osc >= osc_upper > curr_osc) or curr_z > zscore_exit:
                        position = 0
            elif position == -1:
                # Cover Chop short on oscillator bounce or positive momentum
                if prev_osc <= osc_lower < curr_osc or curr_mom > 0:
                    position = 0
                    lowest_since = np.nan

        elif regime == "Bull":
            # Close any short when entering Bull
            if position == -1:
                position = 0
                lowest_since = np.nan
            if position == 0:
                if curr_mom > mom_threshold and prev_osc < 0 and curr_osc > prev_osc:
                    position = 1
                    highest_since = curr_high if aggressive else np.nan
                    pyramid_bonus = 0.0
                    day_high_streak = 0
            elif position == 1:
                # ── Pyramiding: add leverage on new 2-day highs ────
                if aggressive and i >= 2:
                    prev_high = high.iloc[i - 1] if i >= 1 else 0
                    prev2_high = high.iloc[i - 2] if i >= 2 else 0
                    if curr_high > max(prev_high, prev2_high):
                        day_high_streak += 1
                        # Add 0.3x per pyramid level, capped at 50% of max leverage
                        pyramid_bonus = min(day_high_streak * 0.3, bull_leverage * 0.5)
                    else:
                        # Decay pyramid bonus if momentum stalls
                        pyramid_bonus = max(0.0, pyramid_bonus - 0.1)

                # Exit on momentum reversal
                if curr_mom < 0:
                    position = 0
                    highest_since = np.nan
                    pyramid_bonus = 0.0
                    day_high_streak = 0

        else:
            position = 0
            highest_since = np.nan
            lowest_since = np.nan
            pyramid_bonus = 0.0
            day_high_streak = 0

        signal.iloc[i] = position

        # ── Dynamic leverage ────────────────────────────────────────
        if aggressive and position == 1:
            if regime == "Bull":
                # Probabilistic sizing: scale max leverage by bull confidence
                prob_lev = bull_leverage * bp
                # Add pyramid bonus
                total_lev = prob_lev + pyramid_bonus
                # Cap at bull_leverage * 1.5 (safety ceiling)
                leverage.iloc[i] = min(total_lev, bull_leverage * 1.5)
            else:
                # Chop: 1x base, no probability scaling
                leverage.iloc[i] = 1.0
        elif aggressive and position == -1:
            if regime == "Bear":
                # Bear short leverage: scale by bear probability, cap at 2x
                leverage.iloc[i] = min(2.0 * bear_p, 2.0)
            else:
                # Chop short leverage: lighter, scale by bear prob, cap at 1.5x
                leverage.iloc[i] = min(1.5 * bear_p, 1.5)
        elif not aggressive and position == 1:
            leverage.iloc[i] = 1.0
        else:
            leverage.iloc[i] = 0.0

    result = df[["Close"]].copy()
    result["Signal"] = signal
    result["Leverage"] = leverage
    result["Regime"] = regimes
    return result
