import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Sleep Predictor", layout="wide", page_icon="🌙")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #f8f9fc;
}

/* Fix white text on inputs */
input[type="number"], .stNumberInput input {
    color: #1a1d23 !important;
    background-color: #ffffff !important;
}
label, .stSelectbox label, .stSlider label, .stNumberInput label {
    color: #374151 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}
.stSlider p {
    color: #374151 !important;
}

/* Fix dropdown/selectbox text visibility */
.stSelectbox div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #1a1d23 !important;
}
.stSelectbox div[data-baseweb="select"] span {
    color: #1a1d23 !important;
}
ul[role="listbox"] {
    background-color: #ffffff !important;
}
ul[role="listbox"] li {
    color: #1a1d23 !important;
    background-color: #ffffff !important;
}
ul[role="listbox"] li:hover {
    background-color: #f3f4f6 !important;
    color: #1a1d23 !important;
}
div[data-baseweb="select"] input {
    color: #1a1d23 !important;
}
div[data-baseweb="select"] [class*="placeholder"] {
    color: #6b7280 !important;
}
div[data-baseweb="select"] svg {
    fill: #6b7280 !important;
}
div[data-baseweb="popover"] * {
    color: #1a1d23 !important;
    background-color: #ffffff !important;
}
div[data-baseweb="popover"] li:hover {
    background-color: #f3f4f6 !important;
}

p, div, span {
    color: #1a1d23;
}

/* Hero */
.hero-title {
    font-size: 58px;   /* increased from 36px */
    font-weight: 700;  /* slightly bolder */
    color: #1a1d23;
    text-align: center;
    letter-spacing: -0.5px;
}

.hero-sub {
    text-align: center;
    color: #6b7280;
    font-size: 15px;
    margin-bottom: 2rem;
}

/* Cards */
.card {
    background: #ffffff;
    border-radius: 16px;
    padding: 28px;
    border: 1px solid #e5e7eb;
}

/* Section label */
.section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #9ca3af;
    margin-bottom: 1rem;
}

/* Result cards */
.result-good {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin-bottom: 1.2rem;
}
.result-good-title {
    font-size: 20px;
    font-weight: 600;
    color: #166534;
    margin-bottom: 4px;
}
.result-good-sub {
    font-size: 13px;
    color: #16a34a;
}

.result-bad {
    background: #fff7ed;
    border: 1px solid #fdba74;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin-bottom: 1.2rem;
}
.result-bad-title {
    font-size: 20px;
    font-weight: 600;
    color: #9a3412;
    margin-bottom: 4px;
}
.result-bad-sub {
    font-size: 13px;
    color: #ea580c;
}

/* Metric chips */
.metric-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 1.2rem;
}
.metric-chip {
    background: #f3f4f6;
    border-radius: 8px;
    padding: 10px 14px;
    flex: 1;
    min-width: 80px;
}
.metric-chip-label {
    font-size: 11px;
    color: #9ca3af;
    font-weight: 500;
    margin-bottom: 4px;
}
.metric-chip-value {
    font-size: 18px;
    font-weight: 600;
    color: #1a1d23;
}

/* Score bar */
.score-bar-wrap {
    margin-bottom: 1rem;
}
.score-bar-header {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: #6b7280;
    margin-bottom: 6px;
}
.score-bar-track {
    height: 6px;
    background: #e5e7eb;
    border-radius: 3px;
    overflow: hidden;
}
.score-bar-fill-good {
    height: 100%;
    background: linear-gradient(90deg, #34d399, #059669);
    border-radius: 3px;
}
.score-bar-fill-bad {
    height: 100%;
    background: linear-gradient(90deg, #fb923c, #dc2626);
    border-radius: 3px;
}

/* Tips */
.tip-item {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    padding: 8px 0;
    border-bottom: 1px solid #f3f4f6;
    font-size: 13px;
    color: #4b5563;
}
.tip-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #34d399;
    margin-top: 5px;
    flex-shrink: 0;
}
.tip-dot-bad {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #fb923c;
    margin-top: 5px;
    flex-shrink: 0;
}

.divider {
    height: 1px;
    background: #f3f4f6;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ------------------ LOAD MODEL ------------------
model = pickle.load(open('sleep_model.pkl', 'rb'))

# Uncomment to check exact feature names your model expects:
# st.write(model.feature_names_in_)

# ------------------ HEADER ------------------
st.markdown('<p class="hero-title">🌙 Sleep Efficiency Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Analyze your lifestyle and predict sleep quality using machine learning</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

# ------------------ INPUT CARD ------------------
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Your details</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", 10, 100, 28)
    with c2:
        gender = st.selectbox("Gender", ["Male", "Female"])

    sleep_duration = st.slider("Sleep duration (hours)", 0.0, 12.0, 7.0, step=0.5)

    c3, c4 = st.columns(2)
    with c3:
        awakenings = st.number_input("Awakenings per night", 0, 10, 1)
    with c4:
        caffeine = st.number_input("Caffeine consumption", 0, 10, 2)

    c5, c6 = st.columns(2)
    with c5:
        alcohol = st.number_input("Alcohol consumption", 0, 10, 0)
    with c6:
        exercise = st.number_input("Exercise frequency", 0, 10, 3)

    smoking = st.selectbox("Smoking status", ["No", "Yes"])

    predict_btn = st.button("Predict sleep quality →", use_container_width=True, type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ RESULT CARD ------------------
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Prediction result</div>', unsafe_allow_html=True)

    gender_val = 1 if gender == "Male" else 0
    smoking_val = 1 if smoking == "Yes" else 0

    # ⚠️ If you get a feature names error, uncomment this line to see exact names:
    # st.write(model.feature_names_in_)
    input_data = pd.DataFrame([[
        age, gender_val, sleep_duration, awakenings,
        caffeine, alcohol, exercise, smoking_val
    ]], columns=[
        "Age",
        "Gender",
        "Sleep duration",
        "Awakenings",
        "Caffeine consumption",
        "Alcohol consumption",
        "Exercise frequency",
        "Smoking status"
    ])

    if predict_btn:
        result = model.predict(input_data)
        score = int(model.predict_proba(input_data)[0][1] * 100) if hasattr(model, "predict_proba") else (78 if result[0] == 1 else 35)

        if result[0] == 1:
            st.markdown(f"""
            <div class="result-good">
                <div style="font-size:28px;margin-bottom:8px">✅</div>
                <div class="result-good-title">Good Sleep Quality</div>
                <div class="result-good-sub">Your habits suggest healthy sleep patterns</div>
            </div>
            <div class="score-bar-wrap">
                <div class="score-bar-header">
                    <span style="color:#6b7280;">Sleep efficiency score</span>
                    <span style="color:#166534;font-weight:600;">{score} / 100</span>
                </div>
                <div class="score-bar-track">
                    <div class="score-bar-fill-good" style="width:{score}%"></div>
                </div>
            </div>
            <div class="metric-row">
                <div class="metric-chip">
                    <div class="metric-chip-label">Duration</div>
                    <div class="metric-chip-value">{sleep_duration}h</div>
                </div>
                <div class="metric-chip">
                    <div class="metric-chip-label">Awakenings</div>
                    <div class="metric-chip-value">{awakenings}x</div>
                </div>
                <div class="metric-chip">
                    <div class="metric-chip-label">Caffeine</div>
                    <div class="metric-chip-value">{"Low" if caffeine <= 2 else "High"}</div>
                </div>
                <div class="metric-chip">
                    <div class="metric-chip-label">Activity</div>
                    <div class="metric-chip-value">{"Active" if exercise >= 3 else "Low"}</div>
                </div>
            </div>
            <div class="divider"></div>
            <div class="section-label" style="color:#9ca3af;">Tips to maintain quality</div>
            <div class="tip-item">
                <div class="tip-dot"></div>
                <span style="color:#4b5563;">Keep consistent sleep and wake times daily</span>
            </div>
            <div class="tip-item">
                <div class="tip-dot"></div>
                <span style="color:#4b5563;">Avoid caffeine after 2 PM for deeper sleep</span>
            </div>
            <div class="tip-item">
                <div class="tip-dot"></div>
                <span style="color:#4b5563;">Continue your current exercise routine</span>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="result-bad">
                <div style="font-size:28px;margin-bottom:8px">⚠️</div>
                <div class="result-bad-title">Poor Sleep Detected</div>
                <div class="result-bad-sub">Possible signs of insomnia — review your habits</div>
            </div>
            <div class="score-bar-wrap">
                <div class="score-bar-header">
                    <span style="color:#6b7280;">Sleep efficiency score</span>
                    <span style="color:#9a3412;font-weight:600;">{score} / 100</span>
                </div>
                <div class="score-bar-track">
                    <div class="score-bar-fill-bad" style="width:{score}%"></div>
                </div>
            </div>
            <div class="metric-row">
                <div class="metric-chip">
                    <div class="metric-chip-label">Duration</div>
                    <div class="metric-chip-value">{sleep_duration}h</div>
                </div>
                <div class="metric-chip">
                    <div class="metric-chip-label">Awakenings</div>
                    <div class="metric-chip-value">{awakenings}x</div>
                </div>
                <div class="metric-chip">
                    <div class="metric-chip-label">Caffeine</div>
                    <div class="metric-chip-value">{"Low" if caffeine <= 2 else "High"}</div>
                </div>
                <div class="metric-chip">
                    <div class="metric-chip-label">Activity</div>
                    <div class="metric-chip-value">{"Active" if exercise >= 3 else "Low"}</div>
                </div>
            </div>
            <div class="divider"></div>
            <div class="section-label" style="color:#9ca3af;">Recommendations</div>
            <div class="tip-item">
                <div class="tip-dot-bad"></div>
                <span style="color:#4b5563;">Aim for 7–9 hours of uninterrupted sleep</span>
            </div>
            <div class="tip-item">
                <div class="tip-dot-bad"></div>
                <span style="color:#4b5563;">Reduce alcohol — it disrupts sleep architecture</span>
            </div>
            <div class="tip-item">
                <div class="tip-dot-bad"></div>
                <span style="color:#4b5563;">Try light exercise like walking 30 min daily</span>
            </div>
            <div class="tip-item">
                <div class="tip-dot-bad"></div>
                <span style="color:#4b5563;">Consider consulting a sleep specialist</span>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;">
            <div style="font-size:40px;margin-bottom:12px">🌙</div>
            <div style="font-size:15px;font-weight:500;color:#6b7280;margin-bottom:6px">Ready to analyze</div>
            <div style="font-size:13px;color:#9ca3af;">Fill in your details and click predict</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)