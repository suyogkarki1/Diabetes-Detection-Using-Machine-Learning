import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="VitalCheck — Diabetes Risk Screening",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# DESIGN SYSTEM
# Palette: deep clinical teal (trust) + warm amber (attention)
#          + soft sage backgrounds. Deliberately not the
#          cream/terracotta or dark/neon defaults.
# Type: "Fraunces" (display, has real character for a health
#        brand) + "Manrope" (body/UI, humanist and legible)
# Signature: the pulse-line logo mark + radial risk gauge
# ============================================================

TEAL_DARK = "#0B3D3E"
TEAL = "#12595B"
TEAL_LIGHT = "#E7F1EF"
SAGE_BG = "#F6F8F5"
AMBER = "#E8A33D"
AMBER_DARK = "#C97F1D"
CORAL = "#D9634B"
INK = "#16221F"
MUTED = "#5B6C67"
CARD_BG = "#FFFFFF"

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Manrope:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Manrope', sans-serif;
}}

.stApp {{
    background: {SAGE_BG};
}}

/* Hide default streamlit chrome */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}

/* ---- Hero ---- */
.vc-hero {{
    background: linear-gradient(135deg, {TEAL_DARK} 0%, {TEAL} 100%);
    border-radius: 20px;
    padding: 2.6rem 2.8rem;
    margin-bottom: 1.8rem;
    color: white;
    position: relative;
    overflow: hidden;
}}
.vc-hero::after {{
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 240px; height: 240px;
    border-radius: 50%;
    background: rgba(232, 163, 61, 0.14);
}}
.vc-hero-row {{
    display: flex;
    align-items: center;
    gap: 1rem;
    position: relative;
    z-index: 1;
}}
.vc-eyebrow {{
    font-family: 'Manrope', sans-serif;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-size: 0.72rem;
    color: {AMBER};
    margin-bottom: 0.6rem;
}}
.vc-title {{
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 2.5rem;
    line-height: 1.1;
    margin: 0 0 0.6rem 0;
    color: white;
}}
.vc-subtitle {{
    font-size: 1.02rem;
    color: rgba(255,255,255,0.82);
    max-width: 640px;
    line-height: 1.5;
    margin: 0;
}}

/* ---- Section card ---- */
.vc-card {{
    background: {CARD_BG};
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    border: 1px solid rgba(11, 61, 62, 0.08);
    box-shadow: 0 1px 3px rgba(11, 61, 62, 0.04);
    margin-bottom: 1.2rem;
}}
.vc-card-title {{
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 1.15rem;
    color: {TEAL_DARK};
    margin-bottom: 0.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}
.vc-card-sub {{
    color: {MUTED};
    font-size: 0.86rem;
    margin-bottom: 1.1rem;
}}

/* ---- Result panel ---- */
.vc-result {{
    border-radius: 18px;
    padding: 1.8rem 2rem;
    color: white;
    position: relative;
    overflow: hidden;
}}
.vc-result-label {{
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-size: 0.72rem;
    opacity: 0.85;
    margin-bottom: 0.3rem;
}}
.vc-result-headline {{
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 1.9rem;
    margin: 0 0 0.5rem 0;
}}
.vc-result-body {{
    font-size: 0.95rem;
    line-height: 1.55;
    opacity: 0.95;
    max-width: 560px;
}}

/* ---- Badges / pills ---- */
.vc-pill {{
    display: inline-block;
    padding: 0.3rem 0.75rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    background: {TEAL_LIGHT};
    color: {TEAL_DARK};
}}

/* ---- Footer note ---- */
.vc-disclaimer {{
    background: rgba(217, 99, 75, 0.08);
    border-left: 3px solid {CORAL};
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    font-size: 0.83rem;
    color: {INK};
    line-height: 1.5;
}}

/* Streamlit widget tweaks */
div[data-testid="stSlider"] label, .stSelectbox label, .stRadio label {{
    color: {TEAL_DARK} !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}}

.stButton>button {{
    background: linear-gradient(135deg, {AMBER_DARK}, {AMBER});
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 1.4rem;
    font-weight: 700;
    font-size: 0.95rem;
    width: 100%;
    box-shadow: 0 2px 8px rgba(201, 127, 29, 0.35);
    transition: transform 0.15s ease;
}}
.stButton>button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(201, 127, 29, 0.45);
}}

section[data-testid="stSidebar"] {{
    background: {TEAL_DARK};
}}
section[data-testid="stSidebar"] * {{
    color: rgba(255,255,255,0.92) !important;
}}
section[data-testid="stSidebar"] hr {{
    border-color: rgba(255,255,255,0.15);
}}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# LOGO — inline SVG pulse mark (no external image dependency)
# ============================================================
def render_logo(size=40):
    st.markdown(f"""
    <svg width="{size}" height="{size}" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="32" cy="32" r="30" fill="{TEAL_DARK}"/>
        <path d="M14 34 H24 L28 22 L34 44 L38 34 H50"
              stroke="{AMBER}" stroke-width="3.4" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
    </svg>
    """, unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_pipeline():
    return joblib.load("diabetes_non_diabetes_pipeline_2.pkl")

try:
    model = load_pipeline()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)


# ============================================================
# FEATURE ENGINEERING — must exactly mirror training-time logic
# ============================================================
def engineer_features(raw: dict) -> pd.DataFrame:
    age = raw["Age"]
    systolic = raw["Systolic Blood Pressure"]
    diastolic = raw["Diastolic Blood Pressure"]
    heart_rate = raw["Heart Rate"]
    body_temp = raw["Body Temperature"]
    spo2 = raw["SPO2"]

    if age <= 18:
        age_group = "Child_Teen"
    elif age <= 40:
        age_group = "Adult"
    elif age <= 60:
        age_group = "Middle_Age"
    else:
        age_group = "Senior"

    if heart_rate <= 60:
        hr_category = "Bradycardia"
    elif heart_rate <= 100:
        hr_category = "Normal"
    else:
        hr_category = "Tachycardia"

    pulse_pressure = systolic - diastolic
    map_value = diastolic + (pulse_pressure / 3)
    low_spo2_flag = int(spo2 < 95)
    fever_flag = int(body_temp > 100.4)
    symptom_count = raw["Sweating  (Y/N)"] + raw["Shivering (Y/N)"]

    row = {
        "Age": age,
        "Diastolic Blood Pressure": diastolic,
        "Systolic Blood Pressure": systolic,
        "Heart Rate": heart_rate,
        "Body Temperature": body_temp,
        "SPO2": spo2,
        "Pulse_Pressure": pulse_pressure,
        "MAP": map_value,
        "Symptom_Count": symptom_count,
        "Gender": raw["Gender"],
        "Diastolic Blood Pressure Level": raw["Diastolic Blood Pressure Level"],
        "Age_Group": age_group,
        "HR_Category": hr_category,
        "Sweating  (Y/N)": raw["Sweating  (Y/N)"],
        "Shivering (Y/N)": raw["Shivering (Y/N)"],
        "Low_SPO2_Flag": low_spo2_flag,
        "Fever_Flag": fever_flag,
    }
    return pd.DataFrame([row])


def risk_bucket(prob_diabetic: float):
    if prob_diabetic >= 0.66:
        return "high", CORAL, "Elevated signal"
    elif prob_diabetic >= 0.33:
        return "moderate", AMBER_DARK, "Mixed signal"
    else:
        return "low", TEAL, "Low signal"


def render_gauge(prob_diabetic: float, color: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_diabetic * 100,
        number={"suffix": "%", "font": {"size": 42, "family": "Fraunces", "color": INK}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": MUTED, "showticklabels": False},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 33], "color": "#E7F1EF"},
                {"range": [33, 66], "color": "#FBEBD3"},
                {"range": [66, 100], "color": "#F6DFDA"},
            ],
        },
        domain={"x": [0, 1], "y": [0, 1]}
    ))
    fig.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Manrope"}
    )
    return fig


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    col_a, col_b = st.columns([1, 4])
    with col_a:
        render_logo(38)
    with col_b:
        st.markdown("<div style='font-family:Fraunces,serif; font-weight:700; font-size:1.25rem; padding-top:4px;'>VitalCheck</div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("##### How this works")
    st.markdown(
        "This screening tool estimates diabetes risk from **vital signs and symptoms alone** — "
        "**no blood glucose test required.** It's built on a Gradient Boosting model trained on "
        "clinical vitals, tuned and cross-validated for class imbalance."
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("##### Model card")
    st.markdown(
        "- **Algorithm:** Gradient Boosting\n"
        "- **Features:** vitals + engineered clinical indicators\n"
        "- **Balanced accuracy:** ~0.99 (holdout test)\n"
        "- **Excludes:** direct blood glucose reading"
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("Built as a portfolio project. Not a certified medical device.")


# ============================================================
# HERO
# ============================================================
st.markdown(f"""
<div class="vc-hero">
    <div class="vc-eyebrow">Vitals-only screening</div>
    <div class="vc-title">Check your diabetes risk<br/>without a glucose test.</div>
    <p class="vc-subtitle">
        Enter your vital signs and a few symptoms. A Gradient Boosting model — trained and
        validated on clinical data — estimates your likelihood of diabetes from vitals alone,
        the same signals a nurse would check at a routine visit.
    </p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"Could not load the model file `diabetes_non_diabetes_pipeline_2.pkl`. "
             f"Make sure it's in the same folder as this app.\n\nDetails: {load_error}")
    st.stop()


# ============================================================
# INPUT FORM
# ============================================================
left, right = st.columns([1.15, 1], gap="large")

with left:
    st.markdown("""
    <div class="vc-card">
        <div class="vc-card-title">🧍 About you</div>
        <div class="vc-card-sub">Basic demographics used to contextualize your vitals.</div>
    """, unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age (years)", min_value=1, max_value=100, value=32)
    with c2:
        gender = st.selectbox("Gender", ["F", "M"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="vc-card">
        <div class="vc-card-title">❤️ Cardiovascular vitals</div>
        <div class="vc-card-sub">Blood pressure and heart rate readings.</div>
    """, unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        systolic = st.slider("Systolic blood pressure (mmHg)", 70, 200, 118)
        heart_rate = st.slider("Heart rate (bpm)", 40, 180, 78)
    with c2:
        diastolic = st.slider("Diastolic blood pressure (mmHg)", 40, 130, 76)
        bp_level = st.selectbox("Diastolic BP level (clinical reading)", ["Low", "Medium", "High"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="vc-card">
        <div class="vc-card-title">🌡️ Other vitals</div>
        <div class="vc-card-sub">Body temperature and oxygen saturation.</div>
    """, unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        body_temp = st.slider("Body temperature (°F)", 95.0, 104.0, 98.2, step=0.1)
    with c2:
        spo2 = st.slider("SPO2 — oxygen saturation (%)", 85, 100, 97)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="vc-card">
        <div class="vc-card-title">🩹 Symptoms</div>
        <div class="vc-card-sub">Present in the last few hours?</div>
    """, unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        sweating = st.radio("Sweating", ["No", "Yes"], horizontal=True)
    with c2:
        shivering = st.radio("Shivering", ["No", "Yes"], horizontal=True)
    st.markdown("</div>", unsafe_allow_html=True)

    predict_clicked = st.button("Run screening →", use_container_width=True)

with right:
    st.markdown("""
    <div class="vc-card" style="min-height: 100%;">
        <div class="vc-card-title">📊 Your result</div>
        <div class="vc-card-sub">Appears here after you run the screening.</div>
    """, unsafe_allow_html=True)

    if predict_clicked:
        raw_input = {
            "Age": age,
            "Systolic Blood Pressure": systolic,
            "Diastolic Blood Pressure": diastolic,
            "Heart Rate": heart_rate,
            "Body Temperature": body_temp,
            "SPO2": spo2,
            "Gender": gender,
            "Diastolic Blood Pressure Level": bp_level,
            "Sweating  (Y/N)": 1 if sweating == "Yes" else 0,
            "Shivering (Y/N)": 1 if shivering == "Yes" else 0,
        }

        input_df = engineer_features(raw_input)
        proba = model.predict_proba(input_df)[0]
        prob_diabetic = float(proba[1])

        level, color, tag = risk_bucket(prob_diabetic)

        st.markdown(f'<span class="vc-pill">{tag}</span>', unsafe_allow_html=True)
        st.plotly_chart(render_gauge(prob_diabetic, color), use_container_width=True, config={"displayModeBar": False})

        if level == "high":
            headline = "Elevated risk signal"
            body = ("Your vitals pattern is more consistent with the diabetic profile the model "
                    "learned from clinical data. This is not a diagnosis — consider getting an "
                    "actual blood glucose test and speaking with a healthcare provider.")
        elif level == "moderate":
            headline = "Mixed risk signal"
            body = ("Your vitals show some overlap with both profiles. A lab glucose test would "
                    "give a much clearer answer than vitals alone ever can.")
        else:
            headline = "Low risk signal"
            body = ("Your vitals pattern is more consistent with the non-diabetic profile. "
                    "This isn't a clearance — routine checkups still matter.")

        st.markdown(f"""
        <div class="vc-result" style="background: linear-gradient(135deg, {color}, {color}CC); margin-top:1rem;">
            <div class="vc-result-label">Model estimate</div>
            <div class="vc-result-headline">{headline}</div>
            <div class="vc-result-body">{body}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("P(Non-Diabetic)", f"{proba[0]*100:.1f}%")
        m2.metric("P(Diabetic)", f"{proba[1]*100:.1f}%")

    else:
        st.markdown(
            f"<div style='padding: 2.4rem 0; text-align:center; color:{MUTED};'>"
            f"Fill in the form and click <b>Run screening</b> to see your result here."
            f"</div>", unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="vc-disclaimer">
        <b>Not a medical device.</b> This tool is a machine learning demo built for a portfolio
        project. It does not diagnose diabetes and should never replace a real blood glucose test
        or professional medical advice. If you have health concerns, please consult a doctor.
    </div>
    """, unsafe_allow_html=True)