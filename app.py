"""
====================================================
App    : app.py
Project: SkyForecast — Airline Passenger Forecasting
Purpose: Streamlit front-end for the RNN/LSTM forecaster
====================================================
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src.data_loader import DataLoader
from src.evaluation import Evaluator
from src.forecast import Forecaster

# ==================================================
# PATHS  (relative to project root — run streamlit
# from the folder that contains this file)
# ==================================================
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Data" / "airline-passengers.csv"

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="SkyForecast · RNN Passenger Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================================
# PALETTE  — "flight-log" theme
# cream paper + navy ink + brass instrument + amber flare
# ==================================================
CREAM_0   = "#FAF6EC"
CREAM_1   = "#F1E9D6"
PAPER_LN  = "#E1D4AE"
INK       = "#2B2A25"
INK_SOFT  = "#6B6250"
NAVY      = "#243349"
NAVY_SOFT = "#3C5170"
BRASS     = "#B08D4F"
BRASS_DK  = "#8A6C3A"
AMBER     = "#E08A3C"
TEAL      = "#4F8A80"

# ==================================================
# GLOBAL STYLE
# ==================================================
st.html(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,500;0,9..144,600;1,9..144,500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root{{
    --cream-0:{CREAM_0}; --cream-1:{CREAM_1}; --paper-line:{PAPER_LN};
    --ink:{INK}; --ink-soft:{INK_SOFT};
    --navy:{NAVY}; --navy-soft:{NAVY_SOFT};
    --brass:{BRASS}; --brass-dk:{BRASS_DK};
    --amber:{AMBER}; --teal:{TEAL};
    --card-shadow: 0 10px 26px rgba(43,42,37,0.08);
}}

html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--ink);
}}

.stApp {{
    background:
        repeating-linear-gradient(180deg, transparent 0 39px, rgba(180,160,110,0.07) 39px 40px),
        linear-gradient(180deg, var(--cream-0) 0%, #F7F0DE 100%);
}}

h1, h2, h3, h4 {{
    font-family: 'Fraunces', serif !important;
    color: var(--navy) !important;
    letter-spacing: 0.2px;
}}

code, .stMetricValue, div[data-testid="stMetricValue"] {{
    font-family: 'IBM Plex Mono', monospace !important;
}}

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"]{{
    background: linear-gradient(180deg, var(--navy) 0%, #1B2738 100%);
    border-right: 1px solid rgba(0,0,0,0.2);
}}
section[data-testid="stSidebar"] * {{ color: #EFE7D2 !important; }}
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{
    color: #F6EFDD !important;
}}
section[data-testid="stSidebar"] .footnote {{ color: #B9AF95 !important; }}
section[data-testid="stSidebar"] hr {{ border-top: 1px dashed rgba(255,255,255,0.18); }}

/* ---------- METRIC / STAT CARDS (native fallback) ---------- */
div[data-testid="stMetric"]{{
    background: #FFFDF6;
    border: 1px solid var(--paper-line);
    padding: 18px 16px;
    border-radius: 14px;
    box-shadow: var(--card-shadow);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}}
div[data-testid="stMetric"]:hover{{
    transform: translateY(-3px);
    box-shadow: 0 16px 30px rgba(43,42,37,0.14);
}}
div[data-testid="stMetricValue"]{{ color: var(--brass-dk) !important; font-weight: 600; }}
div[data-testid="stMetricLabel"]{{ color: var(--ink-soft) !important; }}

/* ---------- TABS ---------- */
.stTabs [data-baseweb="tab-list"]{{ gap: 6px; background: transparent; }}
.stTabs [data-baseweb="tab"]{{
    background: var(--cream-1);
    border-radius: 10px 10px 0 0;
    padding: 10px 18px;
    color: var(--ink-soft);
    font-weight: 500;
}}
.stTabs [aria-selected="true"]{{
    background: #FFFDF6 !important;
    color: var(--navy) !important;
    box-shadow: 0 -2px 8px rgba(43,42,37,0.08);
}}

/* ---------- BUTTON ---------- */
div.stButton > button:first-child{{
    background: linear-gradient(120deg, var(--brass) 0%, var(--amber) 100%);
    color: #FFFDF6;
    border: none;
    width: 100%;
    border-radius: 12px;
    height: 3.1em;
    font-weight: 600;
    letter-spacing: 0.4px;
    box-shadow: 0 8px 20px rgba(176,141,79,0.32);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}}
div.stButton > button:first-child:hover{{
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0 12px 24px rgba(176,141,79,0.4);
}}

/* ---------- DATAFRAMES / CHARTS ---------- */
div[data-testid="stDataFrame"], .stPlotlyChart{{
    background: #FFFDF6;
    border-radius: 16px;
    padding: 6px;
    box-shadow: var(--card-shadow);
}}

hr{{ border: none; border-top: 1px dashed var(--paper-line); margin: 30px 0; }}

.section-badge{{
    display:inline-block;
    background: var(--cream-1);
    color: var(--brass-dk);
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    margin-bottom: 8px;
    border: 1px solid var(--paper-line);
}}

.footnote {{ color: var(--ink-soft); font-size: 0.82rem; }}

iframe {{ border: none !important; }}
</style>
""")


# ==================================================
# COMPONENT: HERO — GSAP flight-path animation
# ==================================================
def render_hero():
    html = f"""
    <div id="hero-wrap" style="font-family:'IBM Plex Sans',sans-serif;">
      <style>
        #hero {{
            position:relative; width:100%; height:300px;
            background: linear-gradient(120deg, #FDF7EA 0%, #F3E7C9 55%, #ECDFBD 100%);
            border-radius:22px; overflow:hidden;
            box-shadow: 0 10px 30px rgba(43,42,37,0.14);
            border: 1px solid {PAPER_LN};
        }}
        .grid-line {{ stroke: rgba(36,51,73,0.12); stroke-width:1; }}
        .flight-path {{ fill:none; stroke:{BRASS}; stroke-width:2.5; stroke-dasharray: 3 7; }}
        .flight-path-solid {{ fill:none; stroke:{AMBER}; stroke-width:3; }}
        .altitude-dot {{ fill:{NAVY}; opacity:0; }}
        .altitude-label {{
            font-family:'IBM Plex Mono',monospace; font-size:11px; fill:{NAVY_SOFT}; opacity:0;
        }}
        .cloud {{ fill: rgba(255,255,255,0.7); opacity:0; }}
        #hero-title {{
            position:absolute; top:26px; left:36px; z-index:5; opacity:0;
        }}
        #hero-title h1 {{
            font-family:'Fraunces',serif; font-weight:600; font-size:2.4rem; margin:0;
            color:{NAVY}; letter-spacing:0.2px;
        }}
        #hero-title p {{
            font-size:1rem; color:{INK_SOFT}; margin-top:8px; max-width:460px; line-height:1.5;
        }}
        #hero-badge {{
            position:absolute; top:26px; right:36px; z-index:5; opacity:0;
            font-family:'IBM Plex Mono',monospace; font-size:11px; color:{NAVY_SOFT};
            border:1px solid {NAVY_SOFT}; border-radius:999px; padding:5px 14px;
            letter-spacing:0.6px; text-transform:uppercase;
        }}
        #plane-icon {{ transform-box: fill-box; transform-origin: center; }}
      </style>

      <div id="hero">
        <div id="hero-badge">LSTM · Sequence-to-One</div>
        <div id="hero-title">
          <h1>SkyForecast</h1>
          <p>Forecasting global air travel with a recurrent neural network —
          trained on historical passenger volumes, deployed for tomorrow's numbers.</p>
        </div>
        <svg id="hero-svg" viewBox="0 0 1000 300" width="100%" height="100%" preserveAspectRatio="none">
          <line class="grid-line" x1="0" y1="230" x2="1000" y2="230"></line>
          <line class="grid-line" x1="0" y1="255" x2="1000" y2="255"></line>
          <ellipse class="cloud" cx="120" cy="70" rx="46" ry="14"></ellipse>
          <ellipse class="cloud" cx="180" cy="82" rx="30" ry="10"></ellipse>
          <ellipse class="cloud" cx="860" cy="60" rx="52" ry="16"></ellipse>
          <ellipse class="cloud" cx="800" cy="50" rx="28" ry="9"></ellipse>

          <path id="flightPath" class="flight-path"
                d="M 40,240 C 250,240 300,120 460,150 C 620,180 680,90 960,60"></path>
          <path id="flightPathSolid" class="flight-path-solid"
                d="M 40,240 C 250,240 300,120 460,150 C 620,180 680,90 960,60"
                stroke-dasharray="1200" stroke-dashoffset="1200"></path>

          <circle class="altitude-dot" cx="40" cy="240" r="4"></circle>
          <text class="altitude-label" x="20" y="264">MONTH 1</text>

          <circle class="altitude-dot" cx="460" cy="150" r="4"></circle>
          <text class="altitude-label" x="425" y="140">TRAINING</text>

          <circle class="altitude-dot" cx="960" cy="60" r="4"></circle>
          <text class="altitude-label" x="880" y="45">FORECAST</text>

          <g id="plane-icon" transform="translate(40,240) rotate(0)">
            <text x="-11" y="7" font-size="26">✈️</text>
          </g>
        </svg>
      </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/MotionPathPlugin.min.js"></script>
    <script>
      gsap.registerPlugin(MotionPathPlugin);

      const tl = gsap.timeline({{ defaults: {{ ease: "power2.out" }} }});

      tl.to("#hero-badge", {{ opacity: 1, y: 0, duration: 0.6 }}, 0.1)
        .to("#hero-title", {{ opacity: 1, duration: 0.7 }}, 0.15)
        .to(".cloud", {{ opacity: 1, duration: 1.2, stagger: 0.15 }}, 0.2)
        .to("#flightPathSolid", {{ strokeDashoffset: 0, duration: 2.2, ease: "power1.inOut" }}, 0.6)
        .to("#plane-icon", {{
            duration: 2.2,
            ease: "power1.inOut",
            motionPath: {{
                path: "#flightPathSolid",
                align: "#flightPathSolid",
                autoRotate: true,
                alignOrigin: [0.5, 0.5]
            }}
        }}, 0.6)
        .to(".altitude-dot", {{ opacity: 1, duration: 0.4, stagger: 0.7 }}, 0.6)
        .to(".altitude-label", {{ opacity: 1, duration: 0.4, stagger: 0.7 }}, 0.75);

      // gentle continuous cloud drift
      gsap.to(".cloud", {{ x: "+=14", duration: 6, ease: "sine.inOut", yoyo: true, repeat: -1, stagger: 0.4 }});
    </script>
    """
    components.html(html, height=320)


# ==================================================
# COMPONENT: instrument-style stat cards with count-up
# ==================================================
def render_stat_cards(cards, height=150):
    """
    cards: list of dicts -> {label, value, prefix, suffix, decimals, accent, delta}
    """
    card_html = ""
    for i, c in enumerate(cards):
        delta_html = ""
        if c.get("delta") is not None:
            sign = "+" if c["delta"] >= 0 else ""
            color = TEAL if c["delta"] >= 0 else "#B24C3A"
            delta_html = f'<div class="delta" style="color:{color}">{sign}{c["delta"]:.1f}%</div>'
        card_html += f"""
        <div class="stat-card" style="--accent:{c.get('accent', BRASS)}">
            <div class="stat-label">{c['label']}</div>
            <div class="stat-value">
                <span>{c.get('prefix','')}</span><span class="count" data-target="{c['value']}"
                     data-decimals="{c.get('decimals',0)}">0</span><span>{c.get('suffix','')}</span>
            </div>
            {delta_html}
        </div>
        """

    html = f"""
    <style>
      .stat-grid {{
          display:grid; grid-template-columns: repeat({len(cards)}, 1fr); gap:16px;
          font-family:'IBM Plex Sans',sans-serif;
      }}
      .stat-card {{
          background:#FFFDF6; border:1px solid {PAPER_LN}; border-radius:14px;
          padding:16px 18px; box-shadow:0 8px 20px rgba(43,42,37,0.08);
          border-top: 3px solid var(--accent);
          opacity:0; transform: translateY(14px);
      }}
      .stat-label {{
          font-size:0.78rem; color:{INK_SOFT}; text-transform:uppercase;
          letter-spacing:0.5px; margin-bottom:6px;
      }}
      .stat-value {{
          font-family:'IBM Plex Mono',monospace; font-size:1.7rem; font-weight:600; color:{NAVY};
      }}
      .delta {{ font-family:'IBM Plex Mono',monospace; font-size:0.85rem; margin-top:4px; }}
    </style>
    <div class="stat-grid">{card_html}</div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
    <script>
      gsap.utils.toArray(".stat-card").forEach((card, i) => {{
          gsap.to(card, {{ opacity:1, y:0, duration:0.6, delay: i*0.12, ease:"power2.out" }});
      }});
      gsap.utils.toArray(".count").forEach((el) => {{
          const target = parseFloat(el.dataset.target);
          const decimals = parseInt(el.dataset.decimals);
          const obj = {{ val: 0 }};
          gsap.to(obj, {{
              val: target, duration: 1.4, delay: 0.3, ease: "power2.out",
              onUpdate: () => {{
                  el.textContent = decimals > 0
                      ? obj.val.toFixed(decimals)
                      : Math.round(obj.val).toLocaleString();
              }}
          }});
      }});
    </script>
    """
    components.html(html, height=height)


# ==================================================
# DATA
# ==================================================
loader = DataLoader(str(DATA_PATH))
df = loader.load_data()

# Be resilient to either column naming (CSV ships lowercase "passengers")
PASS_COL = "Passengers" if "Passengers" in df.columns else "passengers"

# ==================================================
# HERO
# ==================================================
render_hero()

# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.markdown("### ⚙️ Forecast Settings")
    future_months = st.slider("Forecast Horizon (Months)", 1, 24, 12)
    st.markdown(
        '<div class="footnote">Longer horizons widen the approximate '
        'uncertainty band in the combined projection chart.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("### 📊 Dataset")
    st.markdown(
        f'<div class="footnote">{len(df):,} monthly observations · '
        f'{df.index.min().strftime("%b %Y")} → {df.index.max().strftime("%b %Y")}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("### 🧭 Pipeline")
    for i, step in enumerate(
        ["Load data", "Scale (MinMax)", "Sequence (12mo window)", "LSTM inference", "Inverse-scale forecast"], 1
    ):
        st.markdown(f'<div class="footnote">{i:02d} · {step}</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Model: LSTM · pre-trained, loaded for inference only")

# ==================================================
# TABS — MODEL PERFORMANCE / EDA
# ==================================================
st.markdown('<div class="section-badge">Model Diagnostics</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🚀 Model Performance", "🔎 Exploratory Data Analysis"])

with tab1:
    st.subheader("Model Accuracy Metrics")
    with st.spinner("Scoring held-out predictions..."):
        mae, mse, rmse = Evaluator().evaluate()

    render_stat_cards([
        {"label": "Mean Absolute Error", "value": mae, "decimals": 2, "accent": BRASS},
        {"label": "Mean Squared Error", "value": mse, "decimals": 2, "accent": AMBER},
        {"label": "Root Mean Squared Error", "value": rmse, "decimals": 2, "accent": TEAL},
    ])

with tab2:
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.subheader("Raw Data")
        st.dataframe(df, height=380, use_container_width=True)

    with col_b:
        st.subheader("Historical Trend")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=df.index, y=df[PASS_COL],
            mode="lines",
            line=dict(color=BRASS, width=2.4),
            fill="tozeroy",
            fillcolor="rgba(176,141,79,0.12)",
            name="Passengers",
        ))
        fig_hist.update_layout(
            template="plotly_white",
            plot_bgcolor="#FFFDF6",
            paper_bgcolor="#FFFDF6",
            margin=dict(l=0, r=0, t=20, b=0),
            font=dict(family="IBM Plex Sans, sans-serif", color=INK),
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="rgba(36,51,73,0.10)"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# ==================================================
# FORECAST SECTION
# ==================================================
st.markdown("---")
st.markdown('<div class="section-badge">Prediction Engine</div>', unsafe_allow_html=True)
st.header("🔮 Generate Future Forecast")

run = st.button("✈️  Run RNN Model")

if run:
    progress = st.progress(0, text="Warming up the LSTM...")
    for pct, label in [
        (25, "Loading pre-trained weights..."),
        (55, "Reading temporal patterns..."),
        (80, "Projecting future passenger volume..."),
        (100, "Finalizing forecast..."),
    ]:
        time.sleep(0.25)
        progress.progress(pct, text=label)
    time.sleep(0.2)
    progress.empty()

    forecaster = Forecaster()
    future = forecaster.forecast(future_months)

    last_date = df.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=future_months,
        freq="MS",
    )

    predicted = np.asarray(future).flatten()

    # Illustrative uncertainty band, scaled by RMSE and growing with horizon.
    # This is a heuristic visual aid, not a statistically calibrated interval.
    horizon_idx = np.arange(1, future_months + 1)
    band_width = rmse * (1 + 0.06 * horizon_idx)
    upper = predicted + band_width
    lower = np.clip(predicted - band_width, a_min=0, a_max=None)

    forecast_df = pd.DataFrame({
        "Month": future_dates,
        "Predicted Passengers": predicted.round(1),
        "Lower Bound (approx.)": lower.round(1),
        "Upper Bound (approx.)": upper.round(1),
    })

    st.success(f"Forecast generated for the next {future_months} month(s). ✈️")

    # ---------- Headline stat cards ----------
    last_known = float(df[PASS_COL].iloc[-1])
    growth_pct = ((predicted[-1] - last_known) / last_known) * 100

    render_stat_cards([
        {"label": "Last Known Volume", "value": last_known, "decimals": 0, "accent": NAVY_SOFT},
        {
            "label": f'Predicted ({future_dates[-1].strftime("%b %Y")})',
            "value": float(predicted[-1]), "decimals": 0, "accent": AMBER, "delta": growth_pct,
        },
        {"label": "Avg. Predicted Volume", "value": float(predicted.mean()), "decimals": 0, "accent": TEAL},
    ])

    st.markdown("<br>", unsafe_allow_html=True)
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.subheader("Forecasted Values")
        st.dataframe(forecast_df, use_container_width=True, height=360)

        csv = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv,
            file_name="forecast_results.csv",
            mime="text/csv",
        )

    with res_col2:
        st.subheader("Combined Projection")

        fig_combined = go.Figure()

        # Uncertainty band
        fig_combined.add_trace(go.Scatter(
            x=list(forecast_df["Month"]) + list(forecast_df["Month"][::-1]),
            y=list(upper) + list(lower[::-1]),
            fill="toself",
            fillcolor="rgba(224,138,60,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            name="Approx. uncertainty",
            showlegend=True,
        ))

        # Historical
        fig_combined.add_trace(go.Scatter(
            x=df.index, y=df[PASS_COL],
            name="Historical",
            line=dict(color=NAVY_SOFT, width=2.2),
        ))

        # Forecast
        fig_combined.add_trace(go.Scatter(
            x=forecast_df["Month"], y=forecast_df["Predicted Passengers"],
            name="Forecast",
            line=dict(color=AMBER, width=3, dash="dot"),
            marker=dict(size=6, color=BRASS),
            mode="lines+markers",
        ))

        fig_combined.update_layout(
            template="plotly_white",
            plot_bgcolor="#FFFDF6",
            paper_bgcolor="#FFFDF6",
            hovermode="x unified",
            font=dict(family="IBM Plex Sans, sans-serif", color=INK),
            margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="rgba(36,51,73,0.10)"),
        )
        st.plotly_chart(fig_combined, use_container_width=True)

    st.markdown(
        '<p class="footnote">The shaded band is an illustrative uncertainty range '
        "derived from the model's historical RMSE — it widens with the forecast "
        "horizon and is not a formally calibrated confidence interval.</p>",
        unsafe_allow_html=True,
    )
else:
    st.info("Set your horizon in the sidebar, then click **Run RNN Model** to generate a forecast.")
