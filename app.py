"""
app.py — Institutional-Grade Crypto Trading Dashboard.

Run with:  streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from data_loader import fetch_data
from indicators import compute_all, butterworth_lowpass
from hmm_engine import causal_hmm_regimes
from strategy import generate_signals
from backtester import walk_forward, get_asset_profile

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Crypto Y'all — Regime Dashboard",
    page_icon="assets/cryptoyall-main-3d-inverted-rgb-775px@72ppi.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark-mode CSS
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-card.aggressive {
        border: 1px solid #f0883e;
        background: linear-gradient(135deg, #2a1a0e 0%, #1e1608 100%);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 4px 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .pos { color: #3fb950; }
    .neg { color: #f85149; }
    .neu { color: #58a6ff; }
    .mode-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .mode-standard { background: #1a1a2e; border: 1px solid #58a6ff; color: #58a6ff; }
    .mode-aggressive { background: #2a1a0e; border: 1px solid #f0883e; color: #f0883e; }
    .live-card {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
    }
    .live-card.regime-bull { border-color: #3fb950; }
    .live-card.regime-bear { border-color: #f85149; }
    .live-card.regime-chop { border-color: #8b949e; }
    .live-label {
        font-size: 0.75rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 4px;
    }
    .live-value {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 2px 0;
    }
    .signal-buy   { color: #3fb950; }
    .signal-hold  { color: #58a6ff; }
    .signal-flat  { color: #8b949e; }
    .signal-liq   { color: #f85149; }
    .signal-short { color: #da3633; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.image("assets/cryptoyall-main-3d-inverted-rgb-775px@72ppi.png", width="stretch")
st.sidebar.markdown("")  # spacer
ASSETS = {
    "BTC-USD": "Bitcoin (BTC)",
    "ETH-USD": "Ethereum (ETH)",
    "SOL-USD": "Solana (SOL)",
    "AVAX-USD": "Avalanche (AVAX)",
    "LINK-USD": "Chainlink (LINK)",
    "SUI20947-USD": "Sui (SUI)",
    "XRP-USD": "XRP",
}
ticker = st.sidebar.selectbox(
    "Asset",
    options=list(ASSETS.keys()),
    format_func=lambda t: ASSETS[t],
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### Strategy Mode")
aggressive = st.sidebar.toggle("Smart Aggressive", value=False)

if aggressive:
    st.sidebar.markdown(
        '<span class="mode-badge mode-aggressive">Smart Aggressive</span>',
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        '<span class="mode-badge mode-standard">Standard</span>',
        unsafe_allow_html=True,
    )
    st.sidebar.caption(
        "1x leverage  ·  Bear = cash  ·  Long only  ·  No trailing stop"
    )

# ── Capital & Leverage Simulator ─────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.markdown("#### Capital & Leverage Simulator")
starting_capital = st.sidebar.number_input(
    "Starting Capital ($)", min_value=100, max_value=10_000_000,
    value=10_000, step=1_000, format="%d",
)
bull_leverage = st.sidebar.slider(
    "Bull Regime Leverage", min_value=1.0, max_value=5.0, value=3.0, step=0.5,
)

profile = get_asset_profile(ticker)
if aggressive:
    eff_lev = min(bull_leverage, profile["max_bull_leverage"])
    short_label = "Bear shorting (prob-scaled)" if profile["allow_short"] else "No shorting"
    st.sidebar.caption(
        f"Bull up to {eff_lev:.1f}x (prob-scaled) + pyramiding  ·  {short_label}  ·  Chandelier Exit (ATR×{profile['atr_mult']:.0f})"
    )

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Profile**: {profile['label']}  \n"
    f"**Engine**: Causal HMM (rolling window)  \n"
    "**Backtest**: Walk-Forward OOS only  \n"
    "**Filter**: 2-Pole Butterworth LP"
)


# ── Data pipeline (cached) ──────────────────────────────────────────────────

@st.cache_data(show_spinner="Fetching market data …", ttl="24h")
def load_data(sym: str):
    data = fetch_data(tickers=[sym])
    return data[sym]


@st.cache_data(show_spinner="Computing indicators & HMM regimes …", ttl="24h")
def run_pipeline(sym: str):
    raw = load_data(sym)
    df = compute_all(raw)
    regimes, bull_probs, bear_probs = causal_hmm_regimes(df)
    return df, regimes, bull_probs, bear_probs


@st.cache_data(show_spinner="Running walk-forward backtest …", ttl="24h")
def run_backtest(sym: str, is_aggressive: bool, lev: float,
                 _df=None, _regimes=None, _bull_probs=None, _bear_probs=None):
    raw = load_data(sym)
    precomputed = (_df, _regimes, _bull_probs, _bear_probs) if _df is not None else None
    result = walk_forward(raw, aggressive=is_aggressive, bull_leverage=lev,
                          ticker=sym, precomputed=precomputed)
    return result


# ── Run ──────────────────────────────────────────────────────────────────────

mode_label = "Smart Aggressive" if aggressive else "Standard"
st.title(f"Regime Dashboard  —  {ASSETS[ticker]}")

df, regimes, bull_probs, bear_probs = run_pipeline(ticker)

# Generate live signal directly (fast, no walk-forward needed)
_profile = get_asset_profile(ticker)
_live_sig = generate_signals(
    df, regimes, bull_probs=bull_probs, bear_probs=bear_probs,
    aggressive=aggressive, bull_leverage=min(bull_leverage, _profile["max_bull_leverage"]),
    allow_short=_profile["allow_short"], atr_mult=_profile["atr_mult"],
)


# ── Live Market Status ───────────────────────────────────────────────────────

st.markdown("### Live Market Status")

# Latest data points
latest_close = float(df["Close"].iloc[-1])
latest_regime = regimes.iloc[-1] if not pd.isna(regimes.iloc[-1]) else "Unknown"

# Determine strategy description
if aggressive:
    STRATEGY_MAP = {
        "Bull": "Volatility-Scaled Momentum (Trend Following)",
        "Bear": "Smart Shorting (Prob-Scaled)",
        "Chop": "Relaxed Mean Reversion (Osc Only)",
    }
else:
    STRATEGY_MAP = {
        "Bull": "Volatility-Scaled Momentum (Trend Following)",
        "Bear": "Cash Preservation (Risk-Off)",
        "Chop": "2-Pole Oscillator (Mean Reversion)",
    }
active_strategy = STRATEGY_MAP.get(latest_regime, "Waiting for signal")

# Determine current signal/action from live signal
last_signal = int(_live_sig["Signal"].iloc[-1])

# Previous signal for transition detection
if len(_live_sig) >= 2:
    prev_signal = int(_live_sig["Signal"].iloc[-2])
else:
    prev_signal = last_signal

if last_signal == -1 and prev_signal != -1:
    action_text = "ENTER SHORT"
    action_cls = "signal-short"
elif last_signal == -1:
    action_text = "HOLD SHORT"
    action_cls = "signal-short"
elif last_signal == 0 and prev_signal == -1:
    action_text = "COVER SHORT"
    action_cls = "signal-buy"
elif latest_regime == "Bear" and last_signal == 0:
    action_text = "LIQUIDATE TO CASH"
    action_cls = "signal-liq"
elif last_signal == 1 and prev_signal == 0:
    action_text = "BUY"
    action_cls = "signal-buy"
elif last_signal == 1:
    action_text = "HOLD LONG"
    action_cls = "signal-hold"
elif last_signal == 0 and prev_signal == 1:
    action_text = "SELL / EXIT"
    action_cls = "signal-liq"
else:
    action_text = "FLAT"
    action_cls = "signal-flat"

# Bull / Bear probability for display
latest_bp = float(bull_probs.iloc[-1]) if not pd.isna(bull_probs.iloc[-1]) else 0.0
latest_bear_p = float(bear_probs.iloc[-1]) if not pd.isna(bear_probs.iloc[-1]) else 0.0
bp_cls = "pos" if latest_bp >= 0.7 else ("neu" if latest_bp >= 0.4 else "neg")
bear_p_cls = "neg" if latest_bear_p >= 0.7 else ("neu" if latest_bear_p >= 0.4 else "pos")

regime_cls = f"regime-{latest_regime.lower()}" if latest_regime in ("Bull", "Bear", "Chop") else ""
regime_color_map = {"Bull": "pos", "Bear": "neg", "Chop": "neu"}
regime_val_cls = regime_color_map.get(latest_regime, "neu")

lc1, lc2, lc3, lc4, lc5 = st.columns(5)
with lc1:
    st.markdown(
        f"""<div class="live-card">
            <div class="live-label">Current Price</div>
            <div class="live-value neu">${latest_close:,.2f}</div>
        </div>""",
        unsafe_allow_html=True,
    )
with lc2:
    st.markdown(
        f"""<div class="live-card {regime_cls}">
            <div class="live-label">Current Regime</div>
            <div class="live-value {regime_val_cls}">{latest_regime}</div>
        </div>""",
        unsafe_allow_html=True,
    )
with lc3:
    st.markdown(
        f"""<div class="live-card">
            <div class="live-label">Active Strategy</div>
            <div class="live-value" style="font-size:1rem; color:#c9d1d9;">{active_strategy}</div>
        </div>""",
        unsafe_allow_html=True,
    )
with lc4:
    st.markdown(
        f"""<div class="live-card {regime_cls}">
            <div class="live-label">Current Signal</div>
            <div class="live-value {action_cls}">{action_text}</div>
        </div>""",
        unsafe_allow_html=True,
    )
with lc5:
    if latest_regime == "Bear" and aggressive:
        st.markdown(
            f"""<div class="live-card">
                <div class="live-label">Bear Confidence</div>
                <div class="live-value {bear_p_cls}">{latest_bear_p:.0%}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""<div class="live-card">
                <div class="live-label">Bull Confidence</div>
                <div class="live-value {bp_cls}">{latest_bp:.0%}</div>
            </div>""",
            unsafe_allow_html=True,
        )


# ── Walk-Forward Backtests (deferred for faster initial render) ──────────────

wf_std = run_backtest(ticker, False, 1.0,
                      _df=df, _regimes=regimes, _bull_probs=bull_probs, _bear_probs=bear_probs)
wf_agg = run_backtest(ticker, True, bull_leverage,
                      _df=df, _regimes=regimes, _bull_probs=bull_probs, _bear_probs=bear_probs)
wf = wf_agg if aggressive else wf_std

# ── Metric Cards ─────────────────────────────────────────────────────────────

st.markdown(f"### Walk-Forward OOS Performance  —  {mode_label} Mode")

card_cls = "metric-card aggressive" if aggressive else "metric-card"

c1, c2, c3, c4, c5 = st.columns(5)

ret_class = "pos" if wf.total_return >= 0 else "neg"
dd_class = "neg"
sr_class = "pos" if wf.sharpe_ratio >= 1 else ("neu" if wf.sharpe_ratio >= 0 else "neg")
so_class = "pos" if wf.sortino_ratio >= 1.5 else ("neu" if wf.sortino_ratio >= 0 else "neg")

with c1:
    st.markdown(
        f"""<div class="{card_cls}">
            <div class="metric-label">Total Return (OOS)</div>
            <div class="metric-value {ret_class}">{wf.total_return:+.2%}</div>
        </div>""",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""<div class="{card_cls}">
            <div class="metric-label">Max Drawdown</div>
            <div class="metric-value {dd_class}">{wf.max_drawdown:.2%}</div>
        </div>""",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""<div class="{card_cls}">
            <div class="metric-label">Sharpe Ratio</div>
            <div class="metric-value {sr_class}">{wf.sharpe_ratio:.2f}</div>
        </div>""",
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f"""<div class="{card_cls}">
            <div class="metric-label">Sortino Ratio</div>
            <div class="metric-value {so_class}">{wf.sortino_ratio:.2f}</div>
        </div>""",
        unsafe_allow_html=True,
    )
with c5:
    n_folds = len(wf.best_params_per_fold)
    st.markdown(
        f"""<div class="{card_cls}">
            <div class="metric-label">WF Folds</div>
            <div class="metric-value neu">{n_folds}</div>
        </div>""",
        unsafe_allow_html=True,
    )


# ── Regime colour map ───────────────────────────────────────────────────────

REGIME_COLORS = {
    "Bull": "rgba(0,200,80,0.12)",
    "Bear": "rgba(255,60,60,0.12)",
    "Chop": "rgba(160,160,160,0.12)",
}


def _add_regime_bands(fig, regimes_s: pd.Series, row: int = 1):
    """Shade chart background with regime-coloured vertical rectangles."""
    if regimes_s.dropna().empty:
        return
    prev_regime = None
    band_start = None
    dates = regimes_s.index

    for i, (dt_idx, regime) in enumerate(regimes_s.items()):
        if pd.isna(regime):
            continue
        if regime != prev_regime:
            if prev_regime is not None and band_start is not None:
                fig.add_vrect(
                    x0=band_start, x1=dt_idx,
                    fillcolor=REGIME_COLORS.get(prev_regime, "rgba(0,0,0,0)"),
                    layer="below", line_width=0, row=row, col=1,
                )
            band_start = dt_idx
            prev_regime = regime

    # Close the last band
    if prev_regime is not None and band_start is not None:
        fig.add_vrect(
            x0=band_start, x1=dates[-1],
            fillcolor=REGIME_COLORS.get(prev_regime, "rgba(0,0,0,0)"),
            layer="below", line_width=0, row=row, col=1,
        )


# ── Main chart: Price + BW Filter + Regime Bands ────────────────────────────

st.markdown("---")
st.markdown("### Price & Butterworth Filter with HMM Regimes")

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    vertical_spacing=0.06,
    row_heights=[0.65, 0.35],
    subplot_titles=("Price & 2-Pole Butterworth Filter", "2-Pole Oscillator"),
)

# Price candlestick
fig.add_trace(
    go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price",
        increasing_line_color="#3fb950",
        decreasing_line_color="#f85149",
    ),
    row=1, col=1,
)

# Butterworth filter line
fig.add_trace(
    go.Scatter(
        x=df.index, y=df["BW_Filter"],
        mode="lines", name="BW Filter",
        line=dict(color="#58a6ff", width=2),
    ),
    row=1, col=1,
)

# Regime background bands on price chart
_add_regime_bands(fig, regimes, row=1)

# ── Sub-chart: 2-Pole Oscillator ────────────────────────────────────────────

osc = df["TwoPole_Osc"]
fig.add_trace(
    go.Scatter(
        x=osc.index, y=osc.values,
        mode="lines", name="2-Pole Oscillator",
        line=dict(color="#d2a8ff", width=1.5),
    ),
    row=2, col=1,
)

# Zero line
fig.add_hline(y=0, line_dash="dot", line_color="#484f58", row=2, col=1)

# Regime bands on oscillator chart too
_add_regime_bands(fig, regimes, row=2)

# Layout
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    height=800,
    margin=dict(l=60, r=30, t=50, b=40),
    legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
    xaxis_rangeslider_visible=False,
    font=dict(family="JetBrains Mono, monospace", size=12),
)

st.plotly_chart(fig, width="stretch")


# ── OOS Equity Curve — overlaid comparison ──────────────────────────────────

st.markdown("### Walk-Forward OOS Equity Curve")

eq_fig = go.Figure()

has_std = not wf_std.oos_equity.empty
has_agg = not wf_agg.oos_equity.empty

if has_std:
    eq_fig.add_trace(
        go.Scatter(
            x=wf_std.oos_equity.index,
            y=(wf_std.oos_equity * starting_capital).values,
            mode="lines",
            name="Standard",
            line=dict(color="#58a6ff", width=2.5 if not aggressive else 1.5,
                      dash=None if not aggressive else "dot"),
            fill="tozeroy" if not aggressive else None,
            fillcolor="rgba(88,166,255,0.06)" if not aggressive else None,
        )
    )

if has_agg:
    eq_fig.add_trace(
        go.Scatter(
            x=wf_agg.oos_equity.index,
            y=(wf_agg.oos_equity * starting_capital).values,
            mode="lines",
            name="Smart Aggressive",
            line=dict(color="#f0883e", width=2.5 if aggressive else 1.5,
                      dash=None if aggressive else "dot"),
            fill="tozeroy" if aggressive else None,
            fillcolor="rgba(240,136,62,0.06)" if aggressive else None,
        )
    )

if has_std or has_agg:
    eq_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=400,
        margin=dict(l=60, r=30, t=30, b=40),
        yaxis_title="Portfolio Value ($)",
        yaxis_tickprefix="$",
        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
        font=dict(family="JetBrains Mono, monospace", size=12),
    )
    st.plotly_chart(eq_fig, width="stretch")
else:
    st.info("Not enough data for walk-forward analysis.")


# ── Side-by-side metrics comparison ─────────────────────────────────────────

st.markdown("### Standard vs Smart Aggressive — Head to Head")

std_final = starting_capital * (1 + wf_std.total_return)
agg_final = starting_capital * (1 + wf_agg.total_return)

cmp_left, cmp_right = st.columns(2)

with cmp_left:
    st.markdown(
        '<span class="mode-badge mode-standard">Standard</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<div class="metric-card" style="margin-top:8px;">
            <div class="metric-label">Total Return</div>
            <div class="metric-value {'pos' if wf_std.total_return >= 0 else 'neg'}">{wf_std.total_return:+.2%}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<div class="metric-card" style="margin-top:8px;">
            <div class="metric-label">Max Drawdown</div>
            <div class="metric-value neg">{wf_std.max_drawdown:.2%}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<div class="metric-card" style="margin-top:8px;">
            <div class="metric-label">Sharpe / Sortino</div>
            <div class="metric-value {'pos' if wf_std.sharpe_ratio >= 1 else ('neu' if wf_std.sharpe_ratio >= 0 else 'neg')}">{wf_std.sharpe_ratio:.2f} / {wf_std.sortino_ratio:.2f}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<div class="metric-card" style="margin-top:8px;">
            <div class="metric-label">Final Balance</div>
            <div class="metric-value {'pos' if std_final >= starting_capital else 'neg'}">${std_final:,.2f}</div>
        </div>""",
        unsafe_allow_html=True,
    )

with cmp_right:
    st.markdown(
        '<span class="mode-badge mode-aggressive">Smart Aggressive</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<div class="metric-card aggressive" style="margin-top:8px;">
            <div class="metric-label">Total Return</div>
            <div class="metric-value {'pos' if wf_agg.total_return >= 0 else 'neg'}">{wf_agg.total_return:+.2%}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<div class="metric-card aggressive" style="margin-top:8px;">
            <div class="metric-label">Max Drawdown</div>
            <div class="metric-value neg">{wf_agg.max_drawdown:.2%}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<div class="metric-card aggressive" style="margin-top:8px;">
            <div class="metric-label">Sharpe / Sortino</div>
            <div class="metric-value {'pos' if wf_agg.sharpe_ratio >= 1 else ('neu' if wf_agg.sharpe_ratio >= 0 else 'neg')}">{wf_agg.sharpe_ratio:.2f} / {wf_agg.sortino_ratio:.2f}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<div class="metric-card aggressive" style="margin-top:8px;">
            <div class="metric-label">Final Balance</div>
            <div class="metric-value {'pos' if agg_final >= starting_capital else 'neg'}">${agg_final:,.2f}</div>
        </div>""",
        unsafe_allow_html=True,
    )


# ── Regime Distribution ─────────────────────────────────────────────────────

st.markdown("### Regime Distribution")

if not regimes.dropna().empty:
    regime_counts = regimes.dropna().value_counts()
    pie_fig = go.Figure(
        go.Pie(
            labels=regime_counts.index,
            values=regime_counts.values,
            marker=dict(colors=["#3fb950", "#f85149", "#8b949e"]),
            hole=0.45,
            textinfo="label+percent",
            textfont=dict(size=14),
        )
    )
    pie_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
        font=dict(family="JetBrains Mono, monospace", size=12),
        showlegend=False,
    )
    st.plotly_chart(pie_fig, width="stretch")


# ── Walk-Forward Fold Details ────────────────────────────────────────────────

with st.expander(f"Walk-Forward Fold Parameters ({mode_label})"):
    if wf.best_params_per_fold:
        st.dataframe(
            pd.DataFrame(wf.best_params_per_fold).rename(
                index=lambda i: f"Fold {i + 1}"
            ),
            width="stretch",
        )
    else:
        st.write("No folds completed.")

st.markdown(
    "<div style='text-align:center; color:#484f58; margin-top:40px;'>"
    "Crypto Y'all  ·  Strictly causal  ·  No look-ahead bias  ·  Walk-forward OOS only"
    "</div>",
    unsafe_allow_html=True,
)
