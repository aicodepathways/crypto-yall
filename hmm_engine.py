"""
hmm_engine.py — Strictly causal Hidden Markov Model regime detection.

Uses an expanding-window approach so that predictions are always
out-of-sample:  train on days [0 … t-1], predict day t.

Returns both the regime label AND the Bull-state posterior probability
so that downstream strategy code can scale leverage by confidence.
"""

import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

MIN_TRAIN_DAYS = 365
MAX_TRAIN_DAYS = 756  # ~3 years rolling cap to keep Bear sensitivity


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the observation matrix for the HMM.

    Features (all strictly causal):
        1. Daily log-return
        2. 14-day rolling volatility of log-returns
        3. Volatility-Scaled Momentum (pre-computed column)
        4. VWAP deviation (pre-computed column, if available)
    """
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    roll_vol = log_ret.rolling(14, min_periods=14).std()
    vsm = df["VolScaled_Mom"] if "VolScaled_Mom" in df.columns else pd.Series(0, index=df.index)

    feat = {"ret": log_ret, "vol": roll_vol, "vsm": vsm}

    if "VWAP_Dev" in df.columns:
        feat["vwap_dev"] = df["VWAP_Dev"]

    return pd.DataFrame(feat)


def _label_regimes_by_realized_returns(
    model: GaussianHMM,
    X_scaled: np.ndarray,
    raw_returns: np.ndarray,
) -> dict[int, str]:
    """
    Map HMM hidden-state indices to regime labels using **actual
    realised mean returns** per state.
    """
    states = model.predict(X_scaled)
    n_states = model.n_components

    state_mean_ret = {}
    for s in range(n_states):
        mask = states == s
        if mask.sum() > 0:
            state_mean_ret[s] = float(np.mean(raw_returns[mask]))
        else:
            state_mean_ret[s] = 0.0

    ordered = sorted(state_mean_ret.keys(), key=lambda s: state_mean_ret[s])

    mapping = {}
    mapping[ordered[0]] = "Bear"
    mapping[ordered[-1]] = "Bull"
    for s in ordered[1:-1]:
        mapping[s] = "Chop"

    return mapping


def causal_hmm_regimes(
    df: pd.DataFrame,
    n_components: int = 3,
    min_train: int = MIN_TRAIN_DAYS,
    covariance_type: str = "diag",
    n_iter: int = 100,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Expanding-window HMM regime detection — **strictly causal**.

    Returns
    -------
    regimes : pd.Series[str]
        Regime label per day ('Bull', 'Bear', 'Chop').
    bull_probs : pd.Series[float]
        Posterior probability of the Bull state per day (0–1).
    bear_probs : pd.Series[float]
        Posterior probability of the Bear state per day (0–1).
    """
    feat_df = _build_features(df)
    n = len(feat_df)

    regimes = pd.Series(np.nan, index=df.index, name="Regime", dtype=object)
    bull_probs = pd.Series(np.nan, index=df.index, name="Bull_Prob", dtype=float)
    bear_probs = pd.Series(np.nan, index=df.index, name="Bear_Prob", dtype=float)

    valid_mask = ~feat_df.isna().any(axis=1)
    valid_start = valid_mask.idxmax()
    valid_start_idx = df.index.get_loc(valid_start)
    effective_start = max(valid_start_idx + min_train, min_train)

    refit_interval = 21
    model = None
    scaler = None
    last_mapping: dict[int, str] = {}
    bull_state_idx: int = 0  # which HMM state index maps to Bull
    bear_state_idx: int = 0  # which HMM state index maps to Bear

    for t in range(effective_start, n):
        # Rolling window: use at most MAX_TRAIN_DAYS of recent data
        window_start = max(valid_start_idx, t - MAX_TRAIN_DAYS)
        train_df = feat_df.iloc[window_start:t]
        train_clean = train_df.dropna()

        if len(train_clean) < min_train:
            continue

        if model is None or (t - effective_start) % refit_interval == 0:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(train_clean.values)
            raw_returns = train_clean["ret"].values

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    model = GaussianHMM(
                        n_components=n_components,
                        covariance_type=covariance_type,
                        n_iter=n_iter,
                        random_state=random_state,
                    )
                    model.fit(X_train)
                    last_mapping = _label_regimes_by_realized_returns(
                        model, X_train, raw_returns,
                    )
                    # Identify which internal state index = Bull / Bear
                    for idx, label in last_mapping.items():
                        if label == "Bull":
                            bull_state_idx = idx
                        elif label == "Bear":
                            bear_state_idx = idx
                except (ValueError, np.linalg.LinAlgError):
                    if model is None:
                        continue

        if model is None or scaler is None:
            continue

        obs_raw = feat_df.iloc[t].values.reshape(1, -1)
        if np.isnan(obs_raw).any():
            continue

        try:
            obs_scaled = scaler.transform(obs_raw)
            state = model.predict(obs_scaled)[0]
            regimes.iloc[t] = last_mapping.get(state, "Chop")

            # Posterior probabilities via predict_proba
            posteriors = model.predict_proba(obs_scaled)[0]
            bull_probs.iloc[t] = float(posteriors[bull_state_idx])
            bear_probs.iloc[t] = float(posteriors[bear_state_idx])
        except (ValueError, np.linalg.LinAlgError):
            continue

    # ── Smooth regimes: require N consecutive days to confirm switch ──
    smoothed = regimes.copy()
    confirm_days = 5
    current = None
    pending = None
    pending_count = 0

    for t in range(len(smoothed)):
        raw_regime = smoothed.iloc[t]
        if pd.isna(raw_regime):
            continue
        if current is None:
            current = raw_regime
            continue
        if raw_regime == current:
            pending = None
            pending_count = 0
        elif raw_regime == pending:
            pending_count += 1
            if pending_count >= confirm_days:
                current = pending
                pending = None
                pending_count = 0
            else:
                smoothed.iloc[t] = current
        else:
            pending = raw_regime
            pending_count = 1
            smoothed.iloc[t] = current

    return smoothed, bull_probs, bear_probs
