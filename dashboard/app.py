"""
dashboard/app.py
IMD Aviation Weather Forecast Dashboard
Connects to FastAPI, displays 6-hour multi-horizon forecast.
"""
import streamlit as st
import requests
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, timezone
import json

# ── CONFIG ─────────────────────────────────────────────
API_URL    = "http://localhost:8000"
STATION    = "VABB — CSMI Airport, Mumbai"
REFRESH_S  = 1800   # 30 minutes

st.set_page_config(
    page_title = "IMD Aviation Forecast — CSMI",
    page_icon  = "🛬",
    layout     = "wide"
)

# ── HELPER FUNCTIONS ───────────────────────────────────
def check_api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

# Add this function after imports
def load_live_forecast():
    """Read latest forecast from Airflow pipeline output."""
    import json
    forecast_path = Path("data/processed/latest_forecast.json")
    if forecast_path.exists():
        with open(forecast_path) as f:
            return json.load(f)
    return None

def make_dummy_observations(n=54):
    """Generate realistic dummy METAR for demo when live data unavailable."""
    import numpy as np
    base = datetime.now(timezone.utc).replace(
        minute=0, second=0, microsecond=0
    ) - timedelta(hours=n//2)
    obs = []
    for i in range(n):
        ts = base + timedelta(minutes=30*i)
        hr = ts.hour
        obs.append({
            "timestamp":  ts.strftime('%Y-%m-%d %H:%M:%S'),
            "wind_dir":   int(260 + 20*np.sin(i/8)),
            "wind_speed": int(8  + 5*np.sin(i/6)),
            "gust":       int(14 + 6*np.sin(i/6)),
            "visibility": int(4000 + 1000*np.cos(i/10)),
            "temp":       round(26 + 3*np.sin(i/12), 1),
            "dewpoint":   round(21 + 2*np.sin(i/12), 1),
            "pressure":   round(1008 - 0.5*np.cos(i/8), 1)
        })
    return obs

def fog_color(prob: float) -> str:
    if prob >= 0.5:  return "🔴"
    if prob >= 0.3:  return "🟡"
    return "🟢"

# ── SIDEBAR ────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Emblem_of_India.svg/240px-Emblem_of_India.svg.png",
        width=60
    )
    st.title("IMD Mumbai")
    st.caption("Aviation Weather Forecast System")
    st.divider()

    # API status
    health = check_api_health()
    if health and health.get("models_loaded"):
        st.success("🟢 API Online")
    else:
        st.error("🔴 API Offline")
        st.info("Start API: `uvicorn api.main:app --port 8000`")

    st.divider()
    st.subheader("Data Source")

    data_source = st.radio(
        "Select input",
        ["Demo data", "Upload CSV"],
        help="Demo uses synthetic data. Upload real METAR CSV for live forecast."
    )

    # ── Live Forecast Status ───────────────────────────
    live_forecast = load_live_forecast()

    if live_forecast:
        forecast_age = (
            pd.Timestamp.now()
            - pd.Timestamp(
                live_forecast["forecast_time"].replace("+00:00", "")
            )
        )
        mins = int(forecast_age.total_seconds() / 60)
        st.success(f"📡 Live data available ({mins} min ago)")
    else:
        st.warning("⏳ No live forecast yet — using demo data")

    # ───────────────────────────────────────────────────

    if data_source == "Upload CSV":
        uploaded = st.file_uploader(
            "Upload feature-engineered METAR CSV",
            type=["csv"]
        )
    else:
        uploaded = None

    st.divider()

    if st.button("🔄 Refresh Forecast", use_container_width=True):
        st.rerun()

    st.caption("Auto-refresh every 30 min")
    st.caption(f"Station: {STATION}")
    st.caption("Version: 1.0.0")


# ── MAIN HEADER ────────────────────────────────────────
st.title("🛬 Aviation Weather Forecast — CSMI Airport")
st.caption(f"India Meteorological Department | Mumbai | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
st.divider()

# ── GET FORECAST ───────────────────────────────────────
with st.spinner("Generating forecast..."):
    if uploaded is not None:
        df_upload = pd.read_csv(uploaded)
        # Use raw columns if available
        raw_cols = ['timestamp','wind_dir','wind_speed','gust',
                    'visibility','temp','dewpoint','pressure']
        if all(c in df_upload.columns for c in raw_cols):
            obs = df_upload[raw_cols].tail(54).to_dict('records')
        else:
            st.error("CSV must contain: timestamp, wind_dir, wind_speed, gust, visibility, temp, dewpoint, pressure")
            st.stop()
    else:
        obs = make_dummy_observations(54)

    if data_source == "Demo data" and live_forecast:
        forecast = live_forecast
        st.info("📡 Showing live forecast from Airflow pipeline")
    elif data_source == "Demo data":
        forecast = get_forecast(obs)
        st.info("🔄 Showing demo forecast — Airflow pipeline not yet run")
    else:
        forecast = get_forecast(obs)

if not forecast:
    st.error("Could not retrieve forecast. Check API is running.")
    st.stop()

horizons = forecast['horizons']
fog      = forecast['fog_alert']
hours    = [f"T+{i}hr" for i in range(1, 7)]

# ── FOG ALERT BANNER ───────────────────────────────────
if fog['low_vis_flag']:
    st.error(f"⚠️ LOW VISIBILITY ALERT — Fog probability: {fog['probability']*100:.1f}% | Visibility may drop below {fog['threshold_m']}m | ICAO {fog['icao_category']}")
else:
    st.success(f"✅ Normal Visibility — Fog probability: {fog['probability']*100:.1f}% | ICAO NORMAL")

st.divider()

# ── CURRENT CONDITIONS ROW ────────────────────────────
st.subheader("📊 6-Hour Forecast Summary")

cols = st.columns(6)
vars_display = [
    ("🌡️ Temp", "temperature_c", "°C"),
    ("💨 Wind", "wind_speed_kt", "kt"),
    ("🧭 Dir",  "wind_dir_deg",  "°"),
    ("💥 Gust", "gust_kt",       "kt"),
    ("📊 QNH",  "pressure_hpa",  "hPa"),
    ("👁️ Vis",  "visibility_m",  "m"),
]

for col, (label, key, unit) in zip(cols, vars_display):
    t1_val = horizons['T+1hr'][key]
    t6_val = horizons['T+6hr'][key]
    delta  = round(t6_val - t1_val, 1)
    col.metric(
        label    = f"{label} (T+1)",
        value    = f"{t1_val}{unit}",
        delta    = f"{delta:+.1f} by T+6"
    )

st.divider()

# ── FORECAST TABLE ─────────────────────────────────────
st.subheader("📋 Hourly Forecast Table")

table_data = []
now = datetime.now(timezone.utc)
for hr in range(1, 7):
    key  = f"T+{hr}hr"
    pred = horizons[key]
    valid_time = (now + timedelta(hours=hr)).strftime('%H:%M UTC')
    table_data.append({
        "Horizon":    key,
        "Valid":      valid_time,
        "Temp (°C)":  pred['temperature_c'],
        "Wind (kt)":  pred['wind_speed_kt'],
        "Dir (°)":    pred['wind_dir_deg'],
        "Gust (kt)":  pred['gust_kt'],
        "QNH (hPa)":  pred['pressure_hpa'],
        "Vis (m)":    pred['visibility_m'],
    })

table_df = pd.DataFrame(table_data)
st.dataframe(table_df, use_container_width=True, hide_index=True)

st.divider()

# ── CHARTS ─────────────────────────────────────────────
st.subheader("📈 Forecast Trends")

col1, col2 = st.columns(2)

# Temperature + Visibility chart
with col1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x    = hours,
        y    = [horizons[h]['temperature_c'] for h in hours],
        name = "Temperature (°C)",
        mode = "lines+markers",
        line = dict(color="#FF6B6B", width=2),
        marker = dict(size=8)
    ))
    fig.update_layout(
        title  = "Temperature Forecast",
        xaxis_title = "Horizon",
        yaxis_title = "°C",
        height = 300,
        margin = dict(t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig2 = go.Figure()
    vis_vals = [horizons[h]['visibility_m'] for h in hours]
    colors   = ["red" if v < 1000 else "orange" if v < 3000 else "green"
                for v in vis_vals]
    fig2.add_trace(go.Bar(
        x      = hours,
        y      = vis_vals,
        marker_color = colors,
        name   = "Visibility (m)"
    ))
    fig2.add_hline(y=1000, line_dash="dash", line_color="red",
                   annotation_text="CAT I limit (1000m)")
    fig2.update_layout(
        title  = "Visibility Forecast",
        xaxis_title = "Horizon",
        yaxis_title = "metres",
        height = 300,
        margin = dict(t=40, b=20)
    )
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x    = hours,
        y    = [horizons[h]['wind_speed_kt'] for h in hours],
        name = "Wind speed",
        fill = "tozeroy",
        line = dict(color="#4ECDC4", width=2)
    ))
    fig3.add_trace(go.Scatter(
        x    = hours,
        y    = [horizons[h]['gust_kt'] for h in hours],
        name = "Gust",
        line = dict(color="#45B7D1", width=2, dash="dash")
    ))
    fig3.update_layout(
        title  = "Wind Speed & Gust",
        xaxis_title = "Horizon",
        yaxis_title = "knots",
        height = 300,
        margin = dict(t=40, b=20)
    )
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x    = hours,
        y    = [horizons[h]['pressure_hpa'] for h in hours],
        name = "Pressure QNH",
        line = dict(color="#96CEB4", width=2),
        marker = dict(size=8)
    ))
    fig4.update_layout(
        title  = "Pressure QNH",
        xaxis_title = "Horizon",
        yaxis_title = "hPa",
        height = 300,
        margin = dict(t=40, b=20)
    )
    st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ── FOG PROBABILITY GAUGE ──────────────────────────────
st.subheader("🌫️ Fog / Low-Visibility Alert")

col5, col6 = st.columns([1, 2])
with col5:
    fig5 = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = fog['probability'] * 100,
        title = {"text": "Fog Probability (%)"},
        gauge = {
            "axis":  {"range": [0, 100]},
            "bar":   {"color": "darkred" if fog['low_vis_flag'] else "green"},
            "steps": [
                {"range": [0,  30],  "color": "#d4edda"},
                {"range": [30, 50],  "color": "#fff3cd"},
                {"range": [50, 100], "color": "#f8d7da"},
            ],
            "threshold": {
                "line":  {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 50
            }
        }
    ))
    fig5.update_layout(height=300, margin=dict(t=40, b=20))
    st.plotly_chart(fig5, use_container_width=True)

with col6:
    st.markdown("### Alert Status")
    icon = fog_color(fog['probability'])
    st.markdown(f"## {icon} {fog['icao_category']}")
    st.markdown(f"**Fog probability:** {fog['probability']*100:.1f}%")
    st.markdown(f"**Threshold:** {fog['threshold_m']}m (ICAO CAT I)")
    st.markdown(f"**Low-vis flag:** {'YES ⚠️' if fog['low_vis_flag'] else 'NO ✅'}")
    st.info("Recall = 0.812 — model catches 81% of real fog events. "
            "High recall prioritised over precision for aviation safety.")

st.divider()

# ── MODEL INFO ─────────────────────────────────────────
with st.expander("ℹ️ Model Information"):
    mi = forecast['model_info']
    col7, col8, col9 = st.columns(3)
    col7.metric("Temp/Wind/Gust/Vis", mi['other_variables'])
    col8.metric("Pressure", mi['pressure_model'])
    col9.metric("Fog Alert", mi['fog_classifier'])

    st.caption("Selective ensemble — XGBoost wins on pressure (RMSE 0.752 hPa), "
               "TFT wins on all other variables. "
               "Random Forest with SMOTE for fog binary classification.")
    st.caption(f"Horizon method: {mi['horizon_method']}")
    st.caption(f"Forecast issued: {forecast['forecast_time']}")
