"""HeartGuard — Streamlit UI-only interface.

This phase intentionally contains no model, dataset, training, or real prediction.
The future ML integration boundary is prediction/predictor.py.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from prediction.predictor import predict_heart_disease


# -----------------------------------------------------------------------------
# Page configuration and design system
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=DM+Serif+Display&display=swap');

:root {
    --teal: #087f8c;
    --teal-dark: #075985;
    --ink: #10243e;
    --muted: #64748b;
    --line: #dbe7ef;
    --lavender: #6d5dfc;
}

.stApp { background: #f4f7fb; color: var(--ink); }
.block-container { max-width: 1240px; padding-top: 1.4rem; padding-bottom: 4rem; }
h1, h2, h3 { font-family: 'DM Serif Display', Georgia, serif !important; letter-spacing: -.04em; }

.brand { display: flex; align-items: center; gap: 12px; margin-bottom: 1.5rem; padding: 10px 0; }
.logo {
    display: flex; align-items: center; justify-content: center;
    width: 40px; height: 40px; border-radius: 13px;
    background: linear-gradient(145deg, #0a96a4, #075985);
    color: #fff; font-size: 21px; box-shadow: 0 8px 18px #0f766e33;
}
.brand-title { font-family: 'DM Serif Display', Georgia, serif; font-size: 22px; font-weight: 700; line-height: 1; }
.brand-sub { margin-top: 4px; color: #64748b; font-size: 10px; font-weight: 800; letter-spacing: .18em; text-transform: uppercase; }

.hero { position: relative; overflow: hidden; padding: 3.2rem; border-radius: 30px; background: linear-gradient(125deg, #10243e 0%, #173a5b 58%, #0b7280 100%); box-shadow: 0 25px 65px #10243e2c; }
.hero::before { position: absolute; right: -90px; top: -150px; width: 460px; height: 460px; content: ''; border: 1px solid #ffffff20; border-radius: 50%; box-shadow: 0 0 0 32px #ffffff08, 0 0 0 64px #ffffff05; }
.hero::after { position: absolute; right: 135px; bottom: -140px; width: 260px; height: 260px; content: ''; border: 1px solid #76e0de55; border-radius: 50%; }
.hero-layout { position: relative; z-index: 1; display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(260px, .8fr); align-items: center; gap: 3rem; }
.eyebrow { color: #36c7c2; font-size: 11px; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }
.hero h1 { max-width: 760px; margin: .9rem 0 1.4rem; color: #fff; font-size: clamp(3rem, 7vw, 5.7rem); line-height: .95; }
.gradient { background: linear-gradient(120deg, #63e4dd 9%, #b8b0ff 94%); -webkit-background-clip: text; background-clip: text; color: transparent; }
.hero p { max-width: 620px; color: #c8d8e7; font-size: 17px; line-height: 1.8; }
.hero-panel { padding: 1.3rem; border: 1px solid #ffffff22; border-radius: 22px; background: #ffffff12; backdrop-filter: blur(10px); }
.hero-panel-label { color: #9ab4c9; font-size: 10px; font-weight: 800; letter-spacing: .15em; text-transform: uppercase; }
.hero-panel-title { margin-top: .45rem; color: #fff; font-family: 'DM Serif Display', Georgia, serif; font-size: 29px; }
.hero-panel-line { display: flex; align-items: center; gap: 10px; margin-top: 1.2rem; color: #d9e8f2; font-size: 12px; }
.hero-panel-dot { width: 9px; height: 9px; border-radius: 50%; background: #55dbcc; box-shadow: 0 0 0 6px #55dbcc22; }

.card { padding: 1.65rem; border: 1px solid var(--line); border-radius: 24px; background: rgba(255,255,255,.88); box-shadow: 0 16px 38px #20415c0d; backdrop-filter: blur(12px); }
.choice { min-height: 240px; transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease; }
.choice:hover { transform: translateY(-5px); border-color: #a8dce2; box-shadow: 0 22px 46px #20415c18; }
.choice h3 { margin: .8rem 0 .3rem; color: #10243e; font-size: 25px; }
.choice p { color: var(--muted); line-height: 1.7; }
.chip { display: inline-block; padding: .4rem .75rem; border: 1px solid #c6e9ec; border-radius: 999px; background: #e8f9fa; color: var(--teal-dark); font-size: 11px; font-weight: 800; letter-spacing: .1em; text-transform: uppercase; }
.section { margin-top: 3.5rem; }
.section-title-row { display: flex; align-items: end; justify-content: space-between; gap: 1rem; }
.section-kicker { color: #8ca0b3; font-size: 12px; font-weight: 700; }
.small { color: #64748b; font-size: 12px; line-height: 1.6; }
.review-row { display: flex; justify-content: space-between; gap: 1rem; padding: .8rem 0; border-bottom: 1px solid #eef2f4; color: #64748b; font-size: 13px; }
.review-row strong { color: #1e293b; text-align: right; }
div.stButton > button { padding: .75rem 1.15rem; border: 1px solid #d5e2ea; border-radius: 13px; color: #18324e; background: rgba(255,255,255,.9); font-weight: 800; transition: transform .18s ease, box-shadow .18s ease, background .18s ease; }
div.stButton > button:hover { transform: translateY(-2px); border-color: #9ed6dd; box-shadow: 0 10px 22px #20415c14; }
div.stButton > button[kind='primary'] { border-color: var(--teal); background: linear-gradient(135deg, #087f8c, #075985); color: #fff; box-shadow: 0 10px 22px #087f8c2e; }
div[data-testid="stForm"] { border: 0; }
div[data-testid="stFormSubmitButton"] button { min-height: 48px; }
div[data-testid="stInfo"] { border-radius: 15px; border-left-color: var(--teal); background: #eaf9fb; }
div[data-testid="stAlert"] { border-radius: 15px; }
@media (max-width: 760px) { .hero { padding: 2rem 1.4rem; } .hero-layout { grid-template-columns: 1fr; gap: 1.5rem; } .hero-panel { max-width: 360px; } }
</style>
"""


# -----------------------------------------------------------------------------
# Application constants
# -----------------------------------------------------------------------------

FIELD_LABELS = {
    "age": "Age",
    "sex": "Biological sex",
    "resting_blood_pressure": "Resting blood pressure",
    "cholestoral": "Serum cholesterol",
    "fasting_blood_sugar": "Fasting blood sugar",
    "chest_pain_type": "Chest pain type",
    "rest_ecg": "Resting ECG",
    "max_heart_rate": "Maximum heart rate",
    "exercise_induced_angina": "Exercise-induced angina",
    "oldpeak": "Oldpeak (ST depression)",
    "slope": "ST slope",
    "vessels_colored_by_flourosopy": "Major vessels (fluoroscopy)",
    "thalassemia": "Thalassemia status",
}

CHEST_PAIN_TYPES = [
    "Typical angina",
    "Atypical angina",
    "Non-anginal pain",
    "Asymptomatic",
]

REST_ECG_TYPES = [
    "Normal",
    "ST-T wave abnormality",
    "Left ventricular hypertrophy",
]

ST_SLOPES = ["Upsloping", "Flat", "Downsloping"]
THALASSEMIA_TYPES = ["Normal", "Fixed defect", "Reversible defect"]


# -----------------------------------------------------------------------------
# Session state and shared UI helpers
# -----------------------------------------------------------------------------


def initialize_session() -> None:
    """Create the small amount of state needed for the multi-step UI."""
    defaults = {
        "screen": "home",
        "role": None,
        "inputs": None,
        "placeholder_result": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_application() -> None:
    """Return the user to the landing page and clear the current assessment."""
    st.session_state.screen = "home"
    st.session_state.role = None
    st.session_state.inputs = None
    st.session_state.placeholder_result = None
    st.rerun()


def render_brand() -> None:
    st.markdown(
        """
        <div class="brand">
            <div class="logo">♥</div>
            <div>
                <div class="brand-title">HeartGuard</div>
                <div class="brand-sub">Clinical intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f'<div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p class="small">{description}</p>',
        unsafe_allow_html=True,
    )


def render_section_card(eyebrow: str, title: str) -> None:
    st.markdown(
        f'<div class="card"><div class="eyebrow">{eyebrow}</div><h3>{title}</h3>',
        unsafe_allow_html=True,
    )


def close_card() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Input collection and review
# -----------------------------------------------------------------------------


def collect_inputs(role: str) -> dict[str, Any]:
    """Render the role-specific fields and return the UI values."""
    values: dict[str, Any] = {}

    render_section_card("01 / Patient profile", "Basic inputs")
    first_row = st.columns(2)
    with first_row[0]:
        values["age"] = st.number_input(
            "Age", min_value=18, max_value=100, value=55, help="Patient age in years."
        )
    with first_row[1]:
        values["sex"] = st.selectbox(
            "Biological sex", ["Female", "Male"], help="Notebook category: 0=Female, 1=Male."
        )

    second_row = st.columns(3)
    with second_row[0]:
        values["resting_blood_pressure"] = st.number_input(
            "Resting blood pressure",
            min_value=80,
            max_value=220,
            value=130,
            help="Systolic blood pressure at rest, in mm Hg.",
        )
    with second_row[1]:
        values["cholestoral"] = st.number_input(
            "Serum cholesterol",
            min_value=80,
            max_value=650,
            value=240,
            help="Total serum cholesterol, in mg/dl.",
        )
    with second_row[2]:
        values["fasting_blood_sugar"] = st.selectbox(
            "Fasting blood sugar",
            ["≤ 120 mg/dl", "> 120 mg/dl"],
            help="Whether fasting blood sugar exceeds 120 mg/dl.",
        )
    close_card()

    if role == "normal":
        st.info("This pathway intentionally shows only the five Basic Inputs identified for Normal Users in the supplied feature table.")
        return values

    render_section_card("02 / Symptoms", "Clinical symptoms")
    symptom_row = st.columns(2)
    with symptom_row[0]:
        values["chest_pain_type"] = st.selectbox(
            "Chest pain type", CHEST_PAIN_TYPES, help="The notebook uses four chest-pain categories."
        )
    with symptom_row[1]:
        values["exercise_induced_angina"] = st.selectbox(
            "Exercise-induced angina", ["No", "Yes"], help="Whether exercise testing triggered chest pain."
        )
    close_card()

    render_section_card("03 / Clinical measurements", "Diagnostic measurements")
    measurement_row = st.columns(3)
    with measurement_row[0]:
        values["rest_ecg"] = st.selectbox("Resting ECG", REST_ECG_TYPES)
    with measurement_row[1]:
        values["max_heart_rate"] = st.number_input(
            "Maximum heart rate",
            min_value=60,
            max_value=230,
            value=150,
            help="Peak heart rate achieved during exercise testing, in bpm.",
        )
    with measurement_row[2]:
        values["oldpeak"] = st.number_input(
            "Oldpeak (ST depression)",
            min_value=0.0,
            max_value=7.0,
            value=1.0,
            step=0.1,
            help="ST segment depression induced by exercise relative to rest.",
        )

    measurement_row_two = st.columns(2)
    with measurement_row_two[0]:
        values["slope"] = st.selectbox("ST slope", ST_SLOPES)
    with measurement_row_two[1]:
        values["vessels_colored_by_flourosopy"] = st.number_input(
            "Major vessels (fluoroscopy)",
            min_value=0,
            max_value=4,
            value=0,
            help="Number of major vessels colored by fluoroscopy.",
        )
    close_card()

    render_section_card("04 / Medical history", "Blood disorder status")
    values["thalassemia"] = st.selectbox(
        "Thalassemia status", THALASSEMIA_TYPES, help="Blood disorder status from the input table."
    )
    close_card()

    return values


def render_input_form(role: str) -> None:
    with st.form("heartguard_inputs"):
        values = collect_inputs(role)
        submitted = st.form_submit_button("Review inputs →", type="primary", use_container_width=True)

    if submitted:
        st.session_state.inputs = values
        st.session_state.screen = "review"
        st.rerun()


def render_review() -> None:
    render_page_header(
        "Review before prediction",
        "Everything looks ready.",
        "This is the final UI review step. No model will run until your final model is connected.",
    )

    if st.button("← Edit inputs"):
        st.session_state.screen = "form"
        st.rerun()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    for key, value in st.session_state.inputs.items():
        label = FIELD_LABELS[key]
        st.markdown(
            f'<div class="review-row"><span>{label}</span><strong>{value}</strong></div>',
            unsafe_allow_html=True,
        )
    close_card()

    if st.button("Predict →", type="primary", use_container_width=True):
        with st.spinner("Preparing prediction interface…"):
            st.session_state.placeholder_result = predict_heart_disease(st.session_state.inputs)
        st.session_state.screen = "result"
        st.rerun()


# -----------------------------------------------------------------------------
# Individual screens
# -----------------------------------------------------------------------------


def render_home() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-layout">
                <div>
                    <div class="eyebrow">AI-assisted cardiovascular screening</div>
                    <h1>A clearer view of your <span class="gradient">heart health.</span></h1>
                    <p>A calm, guided interface for the supplied 13-feature Heart Disease Prediction project. The model connection is intentionally reserved for your final trained artifacts.</p>
                </div>
                <div class="hero-panel">
                    <div class="hero-panel-label">Assessment workspace</div>
                    <div class="hero-panel-title">Ready when you are.</div>
                    <div class="hero-panel-line"><span class="hero-panel-dot"></span> Two guided pathways</div>
                    <div class="hero-panel-line"><span class="hero-panel-dot"></span> 13 documented features</div>
                    <div class="hero-panel-line"><span class="hero-panel-dot"></span> Private UI-only session</div>
                </div>
            </div>
        </div>
        <div class="section">
            <div class="section-title-row">
                <div><div class="eyebrow">Start here</div><h2>Choose your user type</h2></div>
                <div class="section-kicker">Select a pathway to begin</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(
            '<div class="card choice"><span class="chip">For everyday screening</span><h3>👤 Normal User</h3><p>Five basic inputs only. Specialist-only clinical fields remain hidden.</p><p class="small"><b>5 inputs</b> · Simple guided flow</p></div>',
            unsafe_allow_html=True,
        )
        if st.button("Continue as Normal User →", type="primary", use_container_width=True):
            st.session_state.role = "normal"
            st.session_state.screen = "form"
            st.rerun()

    with right:
        st.markdown(
            '<div class="card choice"><span class="chip" style="background:#efedff;color:#4f46b5">For clinical use</span><h3>🩺 Medical Specialist</h3><p>All feature-table inputs organized into symptoms, measurements, and medical history.</p><p class="small"><b>13 inputs</b> · Complete clinical flow</p></div>',
            unsafe_allow_html=True,
        )
        if st.button("Continue as Medical Specialist →", use_container_width=True):
            st.session_state.role = "specialist"
            st.session_state.screen = "form"
            st.rerun()

    st.markdown(
        '<div class="section card"><div class="eyebrow">Integration boundary</div><h3>Model connection reserved</h3><p class="small">No model, dataset, training code, prediction, confidence score, or ML artifact is used in this UI-only version. The separate <code>prediction/predictor.py</code> module is ready for your final model later.</p></div>',
        unsafe_allow_html=True,
    )


def render_form() -> None:
    role = st.session_state.role
    role_label = "Normal user" if role == "normal" else "Medical specialist"
    render_page_header(
        f"{role_label} pathway",
        "Tell us what you know.",
        "Use the exact feature names and categories from the supplied notebook and input table.",
    )

    if st.button("← Change user type"):
        reset_application()

    render_input_form(role)


def render_result() -> None:
    render_page_header(
        "Prediction result",
        "Model connection placeholder",
        "The interface is ready for your final trained model and preprocessing files.",
    )
    st.markdown(
        '<div class="card"><h2>Prediction Result</h2><p style="font-size:20px;font-weight:700;color:#0f766e">Model will be connected here.</p><p class="small">The interface is complete, but no prediction was performed. Connect your final trained model and preprocessing objects inside <code>prediction/predictor.py</code>.</p></div>',
        unsafe_allow_html=True,
    )
    st.info("This placeholder is intentional. No fake prediction, confidence percentage, dataset, or ML artifact is being used.")

    if st.button("Start a new assessment", type="primary"):
        reset_application()
    if st.button("← Edit inputs"):
        st.session_state.screen = "form"
        st.rerun()


# -----------------------------------------------------------------------------
# App entry point
# -----------------------------------------------------------------------------


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    initialize_session()
    render_brand()

    screens = {
        "home": render_home,
        "form": render_form,
        "review": render_review,
        "result": render_result,
    }
    screens[st.session_state.screen]()


if __name__ == "__main__":
    main()
