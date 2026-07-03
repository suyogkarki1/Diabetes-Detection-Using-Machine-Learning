import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Glucose Guard | Diabetes Risk Screener",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Design tokens & global styling
# ----------------------------------------------------------------------------
PRIMARY_TEAL = "#0F766E"
DEEP_BLUE = "#155E75"
ALERT_CORAL = "#DC5F45"
SAFE_GREEN = "#15803D"
INK = "#1E293B"
MIST = "#F4F8F8"
CARD_BG = "#FFFFFF"

st.markdown(
    f"""
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"]  {{
            font-family: 'Inter', sans-serif;
            color: {INK};
        }}
        .stApp {{
            background: linear-gradient(180deg, {MIST} 0%, #EAF3F2 100%);
        }}
        h1, h2, h3, .hero-title {{
            font-family: 'Space Grotesk', sans-serif !important;
        }}
        .hero-banner {{
            background: linear-gradient(120deg, {PRIMARY_TEAL} 0%, {DEEP_BLUE} 100%);
            padding: 2.2rem 2.5rem;
            border-radius: 18px;
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 30px rgba(15, 118, 110, 0.25);
        }}
        .hero-title {{
            font-size: 2.1rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }}
        .hero-sub {{
            font-size: 1rem;
            opacity: 0.9;
            font-weight: 400;
        }}
        .metric-card {{
            background: {CARD_BG};
            border-radius: 14px;
            padding: 1rem 1.2rem;
            box-shadow: 0 4px 14px rgba(30, 41, 59, 0.06);
            border-left: 5px solid {PRIMARY_TEAL};
        }}
        .result-card {{
            border-radius: 18px;
            padding: 1.8rem 2rem;
            box-shadow: 0 10px 26px rgba(30, 41, 59, 0.1);
            text-align: center;
        }}
        .badge {{
            display: inline-block;
            padding: 0.2rem 0.7rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }}
        section[data-testid="stSidebar"] {{
            background-color: #0B3B36;
        }}
        section[data-testid="stSidebar"] * {{
            color: #F4F8F8 !important;
        }}
        section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {{
            padding-top: 6px;
        }}
        div.stButton > button {{
            background: linear-gradient(120deg, {PRIMARY_TEAL}, {DEEP_BLUE});
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.7rem 1.4rem;
            font-weight: 600;
            font-size: 1rem;
            width: 100%;
            box-shadow: 0 6px 16px rgba(15, 118, 110, 0.3);
        }}
        div.stButton > button:hover {{
            opacity: 0.92;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Load model
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("diabetes_non_diabetes_pipeline.pkl")

model = load_model()

# Normal/reference ranges used purely for the visual comparison (radar + metric deltas)
NORMAL_RANGES = {
    "Blood Glucose Reading": (70, 100),
    "Diastolic Blood Pressure": (60, 80),
    "Systolic Blood Pressure": (90, 120),
    "Heart Rate": (60, 100),
    "Body Temperature": (97.0, 99.0),
    "SPO2": (95, 100),
}

# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">🩸 Glucose Guard</div>
        <div class="hero-sub">An interactive diabetes risk screener powered by a tuned Gradient Boosting model.
        Adjust the vitals in the sidebar and press Predict to see a full breakdown.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Sidebar — patient inputs
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧍 Patient Profile")

    with st.expander("👤 Demographics", expanded=True):
        age = st.slider("Age", 0, 100, 30)
        gender = st.radio("Gender", ["M", "F"], horizontal=True)

    with st.expander("💉 Vitals", expanded=True):
        blood_glucose = st.slider("Blood Glucose Reading (mg/dL)", 0.0, 400.0, 90.0)
        systolic_bp = st.slider("Systolic Blood Pressure", 0.0, 220.0, 120.0)
        diastolic_bp = st.slider("Diastolic Blood Pressure", 0.0, 150.0, 80.0)
        bp_level = st.select_slider(
            "Diastolic BP Level (category)",
            options=["Low", "Normal", "High"],
            value="Normal",
        )
        heart_rate = st.slider("Heart Rate (bpm)", 0.0, 200.0, 90.0)
        body_temp = st.slider("Body Temperature (°F)", 90.0, 110.0, 98.0)
        spo2 = st.slider("SPO2 (%)", 0.0, 100.0, 98.0)

    with st.expander("🩹 Symptoms", expanded=True):
        sweating = st.toggle("Sweating")
        shivering = st.toggle("Shivering")

    st.markdown("---")
    predict_clicked = st.button("🔍 Predict")

# ----------------------------------------------------------------------------
# Helper: build input row
# ----------------------------------------------------------------------------
def build_input_df():
    return pd.DataFrame([{
        "Age": age,
        "Blood Glucose Reading": blood_glucose,
        "Diastolic Blood Pressure": diastolic_bp,
        "Systolic Blood Pressure": systolic_bp,
        "Heart Rate": heart_rate,
        "Body Temperature": body_temp,
        "SPO2": spo2,
        "Gender": gender,
        "Diastolic Blood Pressure Level": bp_level,
        "Sweating  (Y/N)": int(sweating),
        "Shivering (Y/N)": int(shivering),
    }])

# ----------------------------------------------------------------------------
# Vitals snapshot (always visible) — quick-glance metric cards with deltas
# ----------------------------------------------------------------------------
st.markdown("### 📊 Vitals Snapshot")
vitals_now = {
    "Blood Glucose": (blood_glucose, "Blood Glucose Reading", "mg/dL"),
    "Systolic BP": (systolic_bp, "Systolic Blood Pressure", "mmHg"),
    "Diastolic BP": (diastolic_bp, "Diastolic Blood Pressure", "mmHg"),
    "Heart Rate": (heart_rate, "Heart Rate", "bpm"),
    "Body Temp": (body_temp, "Body Temperature", "°F"),
    "SPO2": (spo2, "SPO2", "%"),
}
cols = st.columns(6)
for col, (label, (val, key, unit)) in zip(cols, vitals_now.items()):
    lo, hi = NORMAL_RANGES[key]
    if val < lo:
        status = "Low"
    elif val > hi:
        status = "High"
    else:
        status = "Normal"
    col.metric(label, f"{val:g} {unit}", status)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Tabs: Prediction | Vitals Radar
# ----------------------------------------------------------------------------
tab_predict, tab_radar = st.tabs(["🔮 Prediction", "🕸️ Vitals Radar"])

with tab_radar:
    st.markdown("Comparison of entered vitals against typical healthy reference midpoints (normalized).")
    categories = list(NORMAL_RANGES.keys())
    patient_vals = []
    normal_vals = []
    for key in categories:
        lo, hi = NORMAL_RANGES[key]
        mid = (lo + hi) / 2
        span = hi - lo if hi != lo else 1
        raw = {
            "Blood Glucose Reading": blood_glucose,
            "Systolic Blood Pressure": systolic_bp,
            "Diastolic Blood Pressure": diastolic_bp,
            "Heart Rate": heart_rate,
            "Body Temperature": body_temp,
            "SPO2": spo2,
        }[key]
        patient_vals.append((raw - mid) / span)
        normal_vals.append(0)

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=normal_vals + [normal_vals[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Healthy midpoint',
        line=dict(color=SAFE_GREEN, width=1, dash="dot"),
        fillcolor="rgba(21,128,61,0.08)",
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=patient_vals + [patient_vals[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Patient (deviation)',
        line=dict(color=PRIMARY_TEAL, width=2),
        fillcolor="rgba(15,118,110,0.25)",
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        margin=dict(l=40, r=40, t=30, b=30),
        height=460,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    st.caption("Values further from the center line indicate a larger deviation from the healthy reference midpoint — this is illustrative, not a clinical scale.")

with tab_predict:
    if predict_clicked:
        input_data = build_input_df()
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        p_non, p_dia = probability[0], probability[1]

        col_gauge, col_result = st.columns([1.2, 1])

        with col_gauge:
            gauge_color = ALERT_CORAL if prediction == 1 else SAFE_GREEN
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=p_dia * 100,
                number={"suffix": "%", "font": {"size": 40}},
                title={"text": "Diabetes Risk Probability", "font": {"size": 16}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": gauge_color},
                    "steps": [
                        {"range": [0, 33], "color": "#DCFCE7"},
                        {"range": [33, 66], "color": "#FEF9C3"},
                        {"range": [66, 100], "color": "#FEE2E2"},
                    ],
                    "threshold": {
                        "line": {"color": INK, "width": 3},
                        "thickness": 0.8,
                        "value": p_dia * 100,
                    },
                },
            ))
            fig_gauge.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_result:
            if prediction == 1:
                st.markdown(
                    f"""
                    <div class="result-card" style="background:#FEF2F0; border:2px solid {ALERT_CORAL};">
                        <div style="font-size:2.5rem;">⚠️</div>
                        <div style="font-size:1.3rem; font-weight:700; color:{ALERT_CORAL};">Diabetic Signal Detected</div>
                        <div class="badge" style="background:{ALERT_CORAL}; color:white; margin-top:0.5rem;">Confidence {p_dia*100:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-card" style="background:#F0FDF4; border:2px solid {SAFE_GREEN};">
                        <div style="font-size:2.5rem;">✅</div>
                        <div style="font-size:1.3rem; font-weight:700; color:{SAFE_GREEN};">Non-Diabetic Signal</div>
                        <div class="badge" style="background:{SAFE_GREEN}; color:white; margin-top:0.5rem;">Confidence {p_non*100:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.balloons()

        st.markdown("<br>", unsafe_allow_html=True)

        # Probability comparison bar
        fig_bar = go.Figure(go.Bar(
            x=[p_non * 100, p_dia * 100],
            y=["Non-Diabetic", "Diabetic"],
            orientation="h",
            marker_color=[SAFE_GREEN, ALERT_CORAL],
            text=[f"{p_non*100:.1f}%", f"{p_dia*100:.1f}%"],
            textposition="auto",
        ))
        fig_bar.update_layout(
            title="Class Probability Breakdown",
            xaxis_title="Probability (%)",
            height=260,
            margin=dict(l=10, r=10, t=50, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        with st.expander("🧾 View submitted patient data"):
            st.dataframe(input_data, use_container_width=True)

        st.caption("⚕️ This tool is for educational/demo purposes only and is not a substitute for professional medical diagnosis.")
    else:
        st.info("👈 Set the patient's vitals in the sidebar and click **Predict** to see the risk assessment.")