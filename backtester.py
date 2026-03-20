"""
backtester.py — Walk-Forward Optimization engine.

Prevents curve-fitting by:
  1. Splitting the dataset into rolling 1-year IN-SAMPLE (IS) windows
     followed by 3-month OUT-OF-SAMPLE (OOS) windows.
  2. Optimising strategy parameters on the IS window via grid search.
  3. Applying those parameters to the OOS window.
  4. Stitching **only** the OOS equity curves together.

Optimises for **Sortino Ratio** (penalises downside vol only) so the
optimizer doesn't fear large upside moves.

Supports both Standard and Smart Aggressive modes.
"""

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from indicators import compute_all
from hmm_engine import causal_hmm_regimes
from strategy import generate_signals


# ── Configuration ────────────────────────────────────────────────────────────

IS_DAYS = 252          # ~1 year of trading days
OOS_DAYS = 63          # ~3 months of trading days

# Parameter grid for walk-forward optimisation
PARAM_GRID = {
    "osc_lower":    [-2.0, -1.5, -1.0, -0.5],
    "osc_upper":    [0.5, 1.0, 1.5, 2.0],
    "zscore_entry": [-2.5, -2.0, -1.5],
    "zscore_exit":  [1.5, 2.0, 2.5],
    "mom_threshold": [-0.5, 0.0, 0.5],
}

# ── Asset-Class Profiles ─────────────────────────────────────────────────────
# Large Cap: BTC/ETH — higher leverage, shorting allowed
# Mid Cap: SOL/AVAX/LINK/SUI/XRP — lower leverage, no shorts, wider stops

ASSET_PROFILES = {
    "large_cap": {
        "label": "Large Cap",
        "tickers": {"BTC-USD", "ETH-USD"},
        "max_bull_leverage": 3.0,   # aggressive mode cap
        "allow_short": True,
        "atr_mult": 3.0,
    },
    "mid_cap": {
        "label": "Mid Cap",
        "tickers": {"SOL-USD", "AVAX-USD", "LINK-USD", "SUI20947-USD", "XRP-USD"},
        "max_bull_leverage": 1.5,   # reduced leverage for volatile alts
        "allow_short": False,       # no shorting — too volatile
        "atr_mult": 4.0,           # wider trailing stop
    },
}


def get_asset_profile(ticker: str) -> dict:
    """Return the profile dict for a given ticker."""
    for profile in ASSET_PROFILES.values():
        if ticker in profile["tickers"]:
            return profile
    # Default to large_cap if unknown
    return ASSET_PROFILES["large_cap"]


@dataclass
class WFResult:
    """Container for walk-forward results."""
    oos_equity: pd.Series          # stitched OOS equity curve
    oos_returns: pd.Series         # stitched OOS daily returns
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    signal_series: pd.Series       # full OOS signal series
    regime_series: pd.Series       # full OOS regime series
    best_params_per_fold: list[dict]
    mode: str                      # "standard" or "aggressive"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sharpe(returns: pd.Series, annual_factor: float = 252) -> float:
    if returns.std() == 0 or len(returns) < 2:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(annual_factor))


def _sortino(returns: pd.Series, annual_factor: float = 252) -> float:
    """Sortino ratio — only penalises downside deviation."""
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float(returns.mean() * np.sqrt(annual_factor)) if returns.mean() > 0 else 0.0
    return float(returns.mean() / downside.std() * np.sqrt(annual_factor))


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.expanding().max()
    dd = (equity - peak) / peak
    return float(dd.min())


def _compute_strategy_returns(
    df: pd.DataFrame,
    regimes: pd.Series,
    bull_probs: pd.Series,
    bear_probs: pd.Series,
    params: dict,
    aggressive: bool,
    bull_leverage: float = 3.0,
    allow_short: bool = True,
    atr_mult: float = 3.0,
) -> pd.Series:
    """Run strategy and compute daily returns with leverage applied."""
    sig_df = generate_signals(
        df, regimes, bull_probs=bull_probs, bear_probs=bear_probs,
        aggressive=aggressive, bull_leverage=bull_leverage,
        allow_short=allow_short, atr_mult=atr_mult, **params,
    )
    daily_ret = df["Close"].pct_change().fillna(0)

    pos = sig_df["Signal"].shift(1).fillna(0)
    lev = sig_df["Leverage"].shift(1).fillna(0)
    strat_ret = pos * lev * daily_ret
    return strat_ret


def _evaluate_params(
    df: pd.DataFrame,
    regimes: pd.Series,
    bull_probs: pd.Series,
    bear_probs: pd.Series,
    params: dict,
    aggressive: bool,
    bull_leverage: float = 3.0,
    allow_short: bool = True,
    atr_mult: float = 3.0,
) -> float:
    """Run strategy with *params* and return the **Sortino** ratio."""
    strat_ret = _compute_strategy_returns(df, regimes, bull_probs, bear_probs, params,
                                          aggressive, bull_leverage, allow_short, atr_mult)
    return _sortino(strat_ret)


def _empty_result(mode: str) -> WFResult:
    empty = pd.Series(dtype=float)
    return WFResult(
        oos_equity=empty,
        oos_returns=empty,
        total_return=0.0,
        max_drawdown=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        signal_series=empty,
        regime_series=pd.Series(dtype=object),
        best_params_per_fold=[],
        mode=mode,
    )


# ── Walk-Forward Engine ─────────────────────────────────────────────────────

def walk_forward(
    df_raw: pd.DataFrame,
    is_days: int = IS_DAYS,
    oos_days: int = OOS_DAYS,
    aggressive: bool = False,
    bull_leverage: float = 3.0,
    ticker: str = "",
) -> WFResult:
    """
    Run walk-forward optimisation on *df_raw* (raw OHLCV DataFrame).

    Parameters
    ----------
    aggressive : bool
        If True, use probabilistic leverage, pyramiding, and Chandelier Exit.
    bull_leverage : float
        Maximum leverage multiplier during Bull regime (aggressive only).
    ticker : str
        Ticker symbol — used to look up asset-class profile.

    Returns a WFResult with OOS-only performance metrics.
    """
    mode = "aggressive" if aggressive else "standard"

    # Apply asset-class profile overrides
    profile = get_asset_profile(ticker)
    if aggressive:
        bull_leverage = min(bull_leverage, profile["max_bull_leverage"])
    allow_short = profile["allow_short"]
    atr_mult = profile["atr_mult"]

    # Pre-compute indicators on the full dataset (indicators are causal)
    df = compute_all(df_raw)

    # Pre-compute causal HMM regimes + bull/bear probabilities
    regimes, bull_probs, bear_probs = causal_hmm_regimes(df)

    n = len(df)
    step = oos_days

    oos_returns_list: list[pd.Series] = []
    oos_signals_list: list[pd.Series] = []
    oos_regimes_list: list[pd.Series] = []
    best_params_list: list[dict] = []

    # Build parameter combinations once
    keys = list(PARAM_GRID.keys())
    combos = [dict(zip(keys, v)) for v in itertools.product(*PARAM_GRID.values())]

    fold = 0
    cursor = is_days

    while cursor + oos_days <= n:
        is_start = cursor - is_days
        is_end = cursor
        oos_start = cursor
        oos_end = min(cursor + oos_days, n)

        is_slice = df.iloc[is_start:is_end]
        is_regimes = regimes.iloc[is_start:is_end]
        is_bull_probs = bull_probs.iloc[is_start:is_end]
        is_bear_probs = bear_probs.iloc[is_start:is_end]
        oos_slice = df.iloc[oos_start:oos_end]
        oos_regimes = regimes.iloc[oos_start:oos_end]
        oos_bull_probs = bull_probs.iloc[oos_start:oos_end]
        oos_bear_probs = bear_probs.iloc[oos_start:oos_end]

        # ── Optimise on IS window (Sortino) ─────────────────────────
        best_score = -np.inf
        best_params: dict = combos[0]

        for combo in combos:
            s = _evaluate_params(is_slice, is_regimes, is_bull_probs, is_bear_probs,
                                 combo, aggressive, bull_leverage, allow_short, atr_mult)
            if s > best_score:
                best_score = s
                best_params = combo

        best_params_list.append(best_params)

        # ── Apply best params to OOS window ──────────────────────────
        sig_df = generate_signals(
            oos_slice, oos_regimes, bull_probs=oos_bull_probs,
            bear_probs=oos_bear_probs,
            aggressive=aggressive, bull_leverage=bull_leverage,
            allow_short=allow_short, atr_mult=atr_mult, **best_params,
        )
        daily_ret = oos_slice["Close"].pct_change().fillna(0)
        pos = sig_df["Signal"].shift(1).fillna(0)
        lev = sig_df["Leverage"].shift(1).fillna(0)
        strat_ret = pos * lev * daily_ret

        oos_returns_list.append(strat_ret)
        oos_signals_list.append(sig_df["Signal"])
        oos_regimes_list.append(oos_regimes)

        fold += 1
        cursor += step

    # ── Stitch OOS results ───────────────────────────────────────────
    if not oos_returns_list:
        return _empty_result(mode)

    oos_returns = pd.concat(oos_returns_list)
    oos_returns = oos_returns[~oos_returns.index.duplicated(keep="first")]
    oos_equity = (1 + oos_returns).cumprod()

    signal_series = pd.concat(oos_signals_list)
    signal_series = signal_series[~signal_series.index.duplicated(keep="first")]
    regime_series = pd.concat(oos_regimes_list)
    regime_series = regime_series[~regime_series.index.duplicated(keep="first")]

    total_ret = float(oos_equity.iloc[-1] / oos_equity.iloc[0] - 1) if len(oos_equity) > 1 else 0.0
    mdd = _max_drawdown(oos_equity)
    sr = _sharpe(oos_returns)
    so = _sortino(oos_returns)

    return WFResult(
        oos_equity=oos_equity,
        oos_returns=oos_returns,
        total_return=total_ret,
        max_drawdown=mdd,
        sharpe_ratio=sr,
        sortino_ratio=so,
        signal_series=signal_series,
        regime_series=regime_series,
        best_params_per_fold=best_params_list,
        mode=mode,
    )
