
import os
import re
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Diabetes Risk Detector", page_icon="🩺", layout="wide")

MODEL_PATH = "diabetes_non_diabetes_pipeline.pkl"

# ----------------------------------------------------------------------
# THEME
# ----------------------------------------------------------------------
PRIMARY = "#146C6C"       # deep teal - trust / clinical
PRIMARY_LIGHT = "#E4F3F1"
WARN = "#E4572E"          # warm coral - diabetic result
GOOD = "#3B8B6E"          # muted green - non-diabetic result
TEXT = "#1C2B2A"
MUTED = "#6B7B79"
SURFACE = "#FFFFFF"
BG = "#F6F8F7"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

* {{ box-sizing: border-box; }}

/* ---- hide default Streamlit chrome so it reads as a real site ---- */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {TEXT};
}}
.stApp {{ background-color: {BG}; overflow-x: hidden; }}
h1, h2, h3, h4 {{ font-family: 'Space Grotesk', sans-serif !important; color: {TEXT}; }}

/* ---- constrain + center content, equal gutters on every screen size ---- */
.block-container {{
    max-width: 1080px;
    width: 100%;
    padding-top: clamp(0.8rem, 3vw, 1.2rem);
    padding-bottom: clamp(1.6rem, 5vw, 3rem);
    padding-left: clamp(1rem, 4vw, 2rem);
    padding-right: clamp(1rem, 4vw, 2rem);
    margin: 0 auto;
}}

/* ---- navbar ---- */
.navbar {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 0.6rem;
    padding: 0.9rem 0.1rem;
    margin-bottom: clamp(1rem, 3vw, 1.4rem);
    border-bottom: 1px solid #E4EBE8;
}}
.navbar .brand {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: clamp(1.05rem, 3.5vw, 1.25rem);
    color: {TEXT};
}}
.navbar .brand span {{ color: {PRIMARY}; }}
.navbar .badge {{
    background: {PRIMARY_LIGHT};
    color: {PRIMARY};
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.3rem 0.75rem;
    border-radius: 999px;
    white-space: nowrap;
}}

/* ---- hero ---- */
.hero {{
    background: linear-gradient(135deg, {PRIMARY} 0%, #1F8C7A 100%);
    padding: clamp(1.6rem, 5vw, 2.6rem) clamp(1.4rem, 5vw, 2.6rem) clamp(2.2rem, 7vw, 3.4rem);
    border-radius: 20px;
    color: white;
    margin-bottom: 0;
    box-shadow: 0 12px 30px rgba(20,108,108,0.25);
    width: 100%;
}}
.hero h1 {{ color: white !important; margin-bottom: 0.4rem; font-size: clamp(1.5rem, 5vw, 2.3rem); line-height: 1.2; }}
.hero p {{ color: #DFF3EF; font-size: clamp(0.88rem, 2.3vw, 1.05rem); margin: 0; max-width: 640px; }}

/* ---- stat strip that overlaps the hero, SaaS-landing style ---- */
.stat-strip {{
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    margin: clamp(-2.2rem, -5vw, -1.2rem) 0 clamp(1.2rem, 4vw, 2rem) 0;
    position: relative;
    z-index: 2;
}}
.stat-card {{
    flex: 1 1 200px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 0.15rem;
    background: {SURFACE};
    border-radius: 14px;
    padding: 1rem 1.2rem;
    box-shadow: 0 8px 24px rgba(28,43,42,0.08);
    border: 1px solid #EEF3F1;
}}
.stat-card .stat-value {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: clamp(1.05rem, 2.6vw, 1.3rem);
    color: {PRIMARY};
}}
.stat-card .stat-label {{
    font-size: 0.78rem;
    color: {MUTED};
    text-transform: uppercase;
    letter-spacing: 0.03em;
}}

/* ---- verdict card ---- */
.verdict-card {{
    padding: clamp(1.2rem, 4vw, 1.6rem) clamp(1.2rem, 4vw, 1.8rem);
    border-radius: 16px;
    color: white;
    text-align: center;
    margin-bottom: 1rem;
    box-shadow: 0 10px 26px rgba(0,0,0,0.12);
    animation: fadeUp 0.35s ease-out;
    width: 100%;
}}
@keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}
.verdict-diabetic {{ background: linear-gradient(135deg, {WARN}, #C9421D); }}
.verdict-nondiabetic {{ background: linear-gradient(135deg, {GOOD}, #276E56); }}
.verdict-card h2 {{ color: white !important; margin: 0 0 0.3rem 0; font-size: clamp(1.2rem, 4vw, 1.5rem); }}
.verdict-card p {{ margin: 0; opacity: 0.92; }}

.section-label {{
    display: inline-block;
    background: {PRIMARY_LIGHT};
    color: {PRIMARY};
    font-weight: 600;
    font-size: 0.78rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    margin-bottom: 0.6rem;
}}

/* ---- card containers (st.container(border=True)) ---- */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {SURFACE};
    border-radius: 16px !important;
    border: 1px solid #E9EFEC !important;
    box-shadow: 0 4px 16px rgba(28,43,42,0.05);
    padding: 0.4rem 0.2rem;
    transition: box-shadow 0.2s ease;
    width: 100%;
}}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
    box-shadow: 0 8px 22px rgba(28,43,42,0.09);
}}

/* ---- keep Streamlit's own column gutters even on all widths ---- */
div[data-testid="stHorizontalBlock"] {{
    gap: clamp(0.8rem, 3vw, 1.5rem);
    align-items: stretch;
}}
div[data-testid="column"] {{ min-width: 0; }}

/* ---- buttons ---- */
.stButton>button {{
    background-color: {PRIMARY};
    color: white;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.6rem 1.2rem;
    border: none;
    width: 100%;
    box-shadow: 0 6px 16px rgba(20,108,108,0.25);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}}
.stButton>button:hover {{
    background-color: #0F5555;
    color: white;
    transform: translateY(-1px);
    box-shadow: 0 8px 20px rgba(20,108,108,0.32);
}}

/* ---- metrics ---- */
div[data-testid="stMetric"] {{
    background: {SURFACE};
    border: 1px solid #E4EBE8;
    border-radius: 12px;
    padding: 0.8rem 1rem;
    box-shadow: 0 4px 12px rgba(28,43,42,0.04);
    height: 100%;
}}

/* ---- tabs styled as an underline nav ---- */
.stTabs [data-baseweb="tab-list"] {{
    gap: clamp(0.8rem, 3vw, 1.6rem);
    border-bottom: 1px solid #E4EBE8;
    flex-wrap: wrap;
}}
.stTabs [data-baseweb="tab"] {{
    height: 42px;
    background: transparent;
    font-weight: 600;
    color: {MUTED};
    padding: 0 0.2rem;
    font-size: clamp(0.82rem, 2.2vw, 1rem);
}}
.stTabs [aria-selected="true"] {{
    color: {PRIMARY} !important;
    border-bottom: 2px solid {PRIMARY} !important;
}}

/* ---- footer ---- */
.site-footer {{
    margin-top: 2.5rem;
    padding-top: 1.2rem;
    border-top: 1px solid #E4EBE8;
    color: {MUTED};
    font-size: 0.82rem;
    text-align: center;
    line-height: 1.5;
}}

/* ---- small-screen tightening ---- */
@media (max-width: 480px) {{
    .stat-strip {{ margin-top: -0.8rem; }}
    .verdict-card h2 {{ font-size: 1.15rem; }}
}}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# MODEL LOADING
# ----------------------------------------------------------------------
@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def engineer_features(raw: dict) -> pd.DataFrame:
    """Mirrors engineer_features() in train_model.py exactly."""
    age = raw["Age"]
    systolic = raw["Systolic Blood Pressure"]
    diastolic = raw["Diastolic Blood Pressure"]
    spo2 = raw["SPO2"]
    temp = raw["Body Temperature"]
    hr = raw["Heart Rate"]
    sweating = raw["Sweating  (Y/N)"]
    shivering = raw["Shivering (Y/N)"]

    if age <= 18:
        age_group = "Child_Teen"
    elif age <= 40:
        age_group = "Adult"
    elif age <= 60:
        age_group = "Middle_Age"
    else:
        age_group = "Senior"

    pulse_pressure = systolic - diastolic
    map_value = diastolic + (pulse_pressure / 3)
    low_spo2_flag = int(spo2 < 95)
    fever_flag = int(temp > 100.4)

    if hr <= 60:
        hr_category = "Bradycardia"
    elif hr <= 100:
        hr_category = "Normal"
    else:
        hr_category = "Tachycardia"

    symptom_count = sweating + shivering

    row = {
        "Age": age,
        "Blood Glucose Reading": raw["Blood Glucose Reading"],
        "Diastolic Blood Pressure": diastolic,
        "Systolic Blood Pressure": systolic,
        "Heart Rate": hr,
        "Body Temperature": temp,
        "SPO2": spo2,
        "Pulse_Pressure": pulse_pressure,
        "MAP": map_value,
        "Symptom_Count": symptom_count,
        "Gender": raw["Gender"],
        "Diastolic Blood Pressure Level": raw["Diastolic Blood Pressure Level"],
        "Age_Group": age_group,
        "HR_Category": hr_category,
        "Sweating  (Y/N)": sweating,
        "Shivering (Y/N)": shivering,
        "Low_SPO2_Flag": low_spo2_flag,
        "Fever_Flag": fever_flag,
    }
    return pd.DataFrame([row])


def _normalize_col(name: str) -> str:
    """Collapse whitespace/case so 'Sweating  (Y/N)' == 'sweating (y/n)'."""
    return re.sub(r"\s+", "", str(name)).lower()


def get_expected_columns(fitted_model):
    """Introspect the fitted pipeline's ColumnTransformer to find out what
    input column names it actually expects — instead of guessing."""
    try:
        preprocessor = fitted_model.named_steps["preprocessor"]
    except Exception:
        return None
    expected = []
    for name, _trans, columns in preprocessor.transformers_:
        if name == "remainder":
            continue
        if isinstance(columns, (list, tuple, pd.Index)):
            expected.extend(list(columns))
        else:
            expected.append(columns)
    return expected


def align_to_model(df: pd.DataFrame, expected_cols):
    """Rename df's columns to match expected_cols wherever they differ only
    by whitespace/case, so naming quirks in the saved pipeline (e.g. a
    double space in 'Sweating  (Y/N)') never crash prediction. Returns the
    aligned dataframe plus any columns that genuinely couldn't be matched."""
    if not expected_cols:
        return df, []
    norm_to_actual = {_normalize_col(c): c for c in df.columns}
    rename_map, missing = {}, []
    for exp in expected_cols:
        norm = _normalize_col(exp)
        if norm in norm_to_actual:
            actual = norm_to_actual[norm]
            if actual != exp:
                rename_map[actual] = exp
        else:
            missing.append(exp)
    return df.rename(columns=rename_map), missing


# Reference "typical normal" ranges - illustrative only, for the radar chart.
REFERENCE_RANGES = {
    "Blood Glucose": {"min": 0, "max": 300, "normal": (70, 140)},
    "Systolic BP":   {"min": 60, "max": 200, "normal": (90, 120)},
    "Diastolic BP":  {"min": 40, "max": 120, "normal": (60, 80)},
    "Heart Rate":    {"min": 40, "max": 160, "normal": (60, 100)},
    "Body Temp":     {"min": 95, "max": 104, "normal": (97, 99)},
    "SPO2":          {"min": 80, "max": 100, "normal": (95, 100)},
}


def normalize(value, lo, hi):
    return float(np.clip((value - lo) / (hi - lo), 0, 1)) * 100


def make_gauge(probability_pct, is_diabetic):
    bar_color = WARN if is_diabetic else GOOD
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability_pct,
        number={"suffix": "%", "font": {"size": 46, "color": TEXT, "family": "Space Grotesk"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": MUTED, "tickfont": {"color": MUTED}},
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "#DDEEE9"},
                {"range": [30, 70], "color": "#FBE7C6"},
                {"range": [70, 100], "color": "#F8D3C5"},
            ],
        },
        title={"text": "Diabetic Probability", "font": {"size": 15, "color": MUTED, "family": "Inter"}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
        height=280,
        margin=dict(t=50, b=10, l=30, r=30),
    )
    return fig


def make_radar(raw):
    labels = ["Blood Glucose", "Systolic BP", "Diastolic BP", "Heart Rate", "Body Temp", "SPO2"]
    patient_raw = [
        raw["Blood Glucose Reading"], raw["Systolic Blood Pressure"],
        raw["Diastolic Blood Pressure"], raw["Heart Rate"],
        raw["Body Temperature"], raw["SPO2"],
    ]
    patient_vals, normal_vals = [], []
    for label, val in zip(labels, patient_raw):
        rng = REFERENCE_RANGES[label]
        patient_vals.append(normalize(val, rng["min"], rng["max"]))
        mid = (rng["normal"][0] + rng["normal"][1]) / 2
        normal_vals.append(normalize(mid, rng["min"], rng["max"]))

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=normal_vals + [normal_vals[0]], theta=labels + [labels[0]],
        name="Typical normal", fill="toself",
        line=dict(color=MUTED, dash="dot"), opacity=0.4,
    ))
    fig.add_trace(go.Scatterpolar(
        r=patient_vals + [patient_vals[0]], theta=labels + [labels[0]],
        name="Patient", fill="toself",
        line=dict(color=PRIMARY, width=2.5), fillcolor="rgba(20,108,108,0.25)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], showticklabels=False)),
        showlegend=True, legend=dict(orientation="h", y=-0.1),
        font={"family": "Inter", "color": TEXT},
        paper_bgcolor="rgba(0,0,0,0)",
        height=380, margin=dict(t=30, b=30, l=40, r=40),
    )
    return fig


# ----------------------------------------------------------------------
# NAVBAR
# ----------------------------------------------------------------------
st.markdown("""
<div class="navbar">
    <div class="brand">Glucose<span>Sense</span></div>
    <div class="badge">Educational Demo</div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# HERO
# ----------------------------------------------------------------------
st.markdown(f"""
<div class="hero">
    <h1>🩺 Diabetes Risk Detector</h1>
    <p>A Gradient Boosting model that reads vitals + symptoms and estimates diabetic risk — built for exploration and learning, not diagnosis.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="stat-strip">
    <div class="stat-card"><div class="stat-value">⚡ Gradient Boosting</div><div class="stat-label">Model type</div></div>
    <div class="stat-card"><div class="stat-value">🎯 PR-AUC</div><div class="stat-label">Optimized for imbalance</div></div>
    <div class="stat-card"><div class="stat-value">🧬 17 features</div><div class="stat-label">Vitals + engineered</div></div>
</div>
""", unsafe_allow_html=True)

model = load_model(MODEL_PATH)

if model is None:
    st.error(
        f"Couldn't find **{MODEL_PATH}**. Run `python train_model.py` first "
        "(with `Blood_Glucose.csv` in the same folder) to generate it, then "
        "place the .pkl next to this app.py."
    )
    st.stop()

tab_predict, tab_insights, tab_about = st.tabs(["🔍 Check Risk", "📊 Model Insights", "ℹ️ About"])

# ----------------------------------------------------------------------
# TAB 1 - PREDICT
# ----------------------------------------------------------------------
with tab_predict:
    left, right = st.columns([1, 1], gap="large")

    with left:
        with st.container(border=True):
            st.markdown('<span class="section-label">Vitals</span>', unsafe_allow_html=True)
            age = st.slider("Age", 0, 100, 35)
            glucose = st.number_input("Blood Glucose Reading (mg/dL)", min_value=0.0, value=100.0)
            c1, c2 = st.columns(2)
            with c1:
                systolic = st.number_input("Systolic BP (mmHg)", min_value=0.0, value=120.0)
            with c2:
                diastolic = st.number_input("Diastolic BP (mmHg)", min_value=0.0, value=80.0)
            bp_level = st.select_slider("Diastolic BP Level", options=["Low", "Normal", "High"], value="Normal")
            c3, c4 = st.columns(2)
            with c3:
                heart_rate = st.number_input("Heart Rate (bpm)", min_value=0.0, value=75.0)
            with c4:
                body_temp = st.number_input("Body Temp (°F)", min_value=90.0, max_value=110.0, value=98.6)
            spo2 = st.slider("SPO2 (%)", 0, 100, 98)
            gender = st.radio("Gender", ["Male", "Female"], horizontal=True)

        with st.container(border=True):
            st.markdown('<span class="section-label">Symptoms</span>', unsafe_allow_html=True)
            s1, s2 = st.columns(2)
            with s1:
                sweating = st.toggle("💧 Sweating")
            with s2:
                shivering = st.toggle("🥶 Shivering")

        predict_clicked = st.button("Predict Risk", type="primary", use_container_width=True)

    with right:
        if not predict_clicked:
            st.markdown(f"""
            <div style="
                border: 2px dashed #D7E4E0;
                border-radius: 16px;
                padding: clamp(2rem, 8vw, 3rem) clamp(1.2rem, 5vw, 2rem);
                text-align: center;
                color: {MUTED};
                background: {SURFACE};
                height: 100%;
                min-height: 260px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                gap: 0.5rem;
            ">
                <div style="font-size: clamp(2rem, 6vw, 2.6rem);">🩺📊</div>
                <div style="font-family:'Space Grotesk', sans-serif; font-weight:600; font-size:clamp(1rem, 3vw, 1.1rem); color:{TEXT};">
                    Your results will appear here
                </div>
                <div style="font-size:clamp(0.82rem, 2.2vw, 0.9rem);">Fill in the vitals on the left and click <b>Predict Risk</b></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            raw_input = {
                "Age": age, "Blood Glucose Reading": glucose,
                "Systolic Blood Pressure": systolic, "Diastolic Blood Pressure": diastolic,
                "Diastolic Blood Pressure Level": bp_level, "Heart Rate": heart_rate,
                "Body Temperature": body_temp, "SPO2": spo2, "Gender": gender,
                "Sweating  (Y/N)": int(sweating), "Shivering (Y/N)": int(shivering),
            }
            X_input = engineer_features(raw_input)

            expected_cols = get_expected_columns(model)
            X_input, missing_cols = align_to_model(X_input, expected_cols)

            if missing_cols:
                st.error(
                    "Your saved model expects columns this app doesn't generate: "
                    f"**{', '.join(missing_cols)}**. This usually means the .pkl was "
                    "trained with a different feature set than this app builds. "
                    "Retrain with the matching script, or update `engineer_features()` "
                    "to produce these exact columns."
                )
                st.stop()

            try:
                with st.spinner("Running the model..."):
                    prediction = model.predict(X_input)[0]
                    proba = model.predict_proba(X_input)[0]
            except Exception as e:
                st.error(
                    "The model couldn't process this input. Technical details below "
                    "may help pin down the mismatch."
                )
                with st.expander("Show technical error"):
                    st.code(str(e))
                st.stop()

            is_diabetic = prediction == 1

            card_class = "verdict-diabetic" if is_diabetic else "verdict-nondiabetic"
            label = "Diabetic" if is_diabetic else "Non-Diabetic"
            icon = "⚠️" if is_diabetic else "✅"
            st.markdown(f"""
            <div class="verdict-card {card_class}">
                <h2>{icon} {label}</h2>
                <p>Model confidence: {max(proba)*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

            st.plotly_chart(make_gauge(proba[1] * 100, is_diabetic), use_container_width=True)

            m1, m2 = st.columns(2)
            m1.metric("Probability Non-Diabetic", f"{proba[0]*100:.1f}%")
            m2.metric("Probability Diabetic", f"{proba[1]*100:.1f}%")

            st.markdown("**Vitals vs. typical normal range**")
            st.plotly_chart(make_radar(raw_input), use_container_width=True)

            with st.expander("See engineered feature values sent to the model"):
                st.dataframe(X_input.T.rename(columns={0: "value"}), use_container_width=True)

# ----------------------------------------------------------------------
# TAB 2 - INSIGHTS
# ----------------------------------------------------------------------
with tab_insights:
    st.markdown('<span class="section-label">What drives the model</span>', unsafe_allow_html=True)
    try:
        classifier = model.named_steps["classifier"]
        preprocessor = model.named_steps["preprocessor"]
        feature_names = preprocessor.get_feature_names_out()
        importances = classifier.feature_importances_
        imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        imp_df = imp_df.sort_values("Importance", ascending=True).tail(15)

        fig = go.Figure(go.Bar(
            x=imp_df["Importance"], y=imp_df["Feature"], orientation="h",
            marker=dict(color=imp_df["Importance"], colorscale=[[0, PRIMARY_LIGHT], [1, PRIMARY]]),
        ))
        fig.update_layout(
            height=500, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter", "color": TEXT},
            margin=dict(t=20, b=20, l=10, r=10),
            xaxis_title="Importance", yaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.warning("This model doesn't expose feature importances (e.g. not a tree-based classifier).")

# ----------------------------------------------------------------------
# TAB 3 - ABOUT
# ----------------------------------------------------------------------
with tab_about:
    st.markdown('<span class="section-label">Read before you rely on this</span>', unsafe_allow_html=True)

    caveats = [
        ("⚖️", "Imbalanced training data",
         "~98% diabetic / ~2% non-diabetic in the source data. Very high accuracy on "
         "such a split should be read cautiously, not as proof of clinical-grade reliability."),
        ("🔄", "Circular-ish signal",
         "Blood Glucose Reading is itself a value clinically used to diagnose diabetes, so "
         "this tool is closer to a rules-assist calculator than a model finding hidden patterns."),
        ("🚫", "Not a diagnostic tool",
         "This app is for learning and demo purposes only. Consult a healthcare "
         "professional for real diagnoses."),
    ]
    cols = st.columns(3)
    for col, (icon, title, body) in zip(cols, caveats):
        with col:
            st.markdown(f"""
            <div style="background:{SURFACE}; border:1px solid #E9EFEC; border-radius:14px;
                        padding:1.2rem; height:100%; display:flex; flex-direction:column;
                        justify-content:flex-start; gap:0.5rem;
                        box-shadow:0 4px 14px rgba(28,43,42,0.05);">
                <div style="font-size:1.6rem;">{icon}</div>
                <div style="font-family:'Space Grotesk', sans-serif; font-weight:600;">{title}</div>
                <div style="font-size:0.85rem; color:{MUTED}; line-height:1.5;">{body}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="section-label">How it works</span>', unsafe_allow_html=True)
    st.markdown("""
    1. You enter vitals and symptoms.
    2. The app derives the same engineered features used at training time
       (Pulse Pressure, MAP, Age Group, Heart-Rate Category, symptom flags).
    3. A Gradient Boosting pipeline (scaling + one-hot encoding + classifier) predicts
       the probability of each class.
    """)

st.markdown("""
<div class="site-footer">
    GlucoSense · Model: Gradient Boosting (scikit-learn pipeline) · For educational/demo purposes only, not medical advice.
</div>
""", unsafe_allow_html=True)