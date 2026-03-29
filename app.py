"""
FemScan AI — Streamlit Dashboard
=================================
Entry point. Handles page routing and the landing hero.

Run:
    streamlit run app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="FemScan AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Hide default Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }

    /* Tighten top padding */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    /* Streamlit button override */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: opacity 0.15s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Slider label colour */
    .stSlider label { color: #c0c0d0 !important; }

    /* Toggle label */
    .stCheckbox label, .stToggle label { color: #c0c0d0 !important; }

    /* File uploader */
    [data-testid="stFileUploadDropzone"] {
        background: #1a1a2e !important;
        border: 1px dashed #3a3a5e !important;
        border-radius: 12px !important;
    }

    /* Expander header */
    .streamlit-expanderHeader {
        background: #1a1a2e !important;
        border-radius: 8px !important;
        color: #E8E8E8 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state defaults ────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state["page"] = "home"


# ── Top navigation bar ────────────────────────────────────────────────────────
def _navbar():
    cols = st.columns([2, 1, 1, 1, 1])
    with cols[0]:
        st.markdown(
            '<span style="color:#9C27B0; font-weight:700; font-size:16px;">🔬 FemScan AI</span>',
            unsafe_allow_html=True,
        )

    nav_items = [("step1", "1 · Screen"), ("step2", "2 · Analyse"), ("step3", "3 · Report")]
    for i, (key, label) in enumerate(nav_items, start=2):
        with cols[i]:
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state["page"] = key
                st.rerun()

    st.markdown(
        '<hr style="border:none; border-top:1px solid #2a2a3e; margin:8px 0 24px;">',
        unsafe_allow_html=True,
    )


# ── Landing page ──────────────────────────────────────────────────────────────
def _landing():
    st.markdown(
        """
        <div style="text-align:center; padding: 40px 0 32px;">
            <p style="color:#9C27B0; font-size:13px; font-weight:600;
                       text-transform:uppercase; letter-spacing:2px; margin-bottom:12px;">
                AI-Powered Women's Health
            </p>
            <h1 style="color:#E8E8E8; font-size:42px; font-weight:700; margin:0 0 16px;
                        line-height:1.15;">
                Cervical Cancer Screening<br>
                <span style="color:#9C27B0;">Decision Support</span>
            </h1>
            <p style="color:#a0a0b0; font-size:16px; max-width:580px;
                       margin:0 auto 32px; line-height:1.6;">
                FemScan AI combines cytology image analysis, HPV risk scoring,
                and endometriosis symptom screening into a single triage workflow
                for clinical decision support.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col = st.columns([2, 1, 2])[1]
    with col:
        if st.button("Start Assessment →", type="primary", use_container_width=True):
            st.session_state["page"] = "step1"
            st.rerun()

    st.markdown('<div style="height:40px"/>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    cards = [
        ("1", "🧪", "Screen & Score",
         "Enter patient history and symptoms to get instant HPV acquisition risk and "
         "endometriosis likelihood scores, powered by XGBoost and SHAP explainability."),
        ("2", "🔬", "Capture & Analyse",
         "Upload a cervical cytology (Pap smear) image. Our EfficientNetV2-S model "
         "classifies CIN grade and generates a Grad-CAM attention overlay to show "
         "which regions drove the prediction."),
        ("3", "📋", "Clinical Report",
         "Receive a consolidated triage report combining all findings — with a traffic-light "
         "urgency system, ranked action checklist, and one-click text export."),
    ]
    for col, (step, icon, title, desc) in zip([c1, c2, c3], cards):
        with col:
            st.markdown(
                f"""
                <div style="
                    background:#1a1a2e; border:1px solid #2a2a4e;
                    border-radius:16px; padding:28px 24px;
                ">
                    <div style="
                        width:44px; height:44px; border-radius:12px;
                        background:#9C27B022; border:1px solid #9C27B066;
                        display:flex; align-items:center; justify-content:center;
                        font-size:20px; margin-bottom:16px;
                    ">{icon}</div>
                    <p style="color:#9C27B0; font-size:11px; font-weight:600;
                               text-transform:uppercase; letter-spacing:1px; margin:0 0 8px;">
                        Step {step}
                    </p>
                    <h3 style="color:#E8E8E8; font-size:16px; margin:0 0 10px;">{title}</h3>
                    <p style="color:#a0a0b0; font-size:13px; line-height:1.6; margin:0;">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div style="height:40px"/>', unsafe_allow_html=True)

    st.markdown(
        '<p style="color:#606070; font-size:12px; text-align:center; '
        'text-transform:uppercase; letter-spacing:1.5px; margin-bottom:16px;">Models</p>',
        unsafe_allow_html=True,
    )
    m1, m2, m3, m4 = st.columns(4, gap="small")
    modules = [
        ("#4CAF50", "Module 1", "HPV Risk Scorer",  "XGBoost · SHAP"),
        ("#FF9800", "Module 2", "CIN Classifier",   "EfficientNetV2-S · Grad-CAM"),
        ("#9C27B0", "Module 3", "Endo Scorer",      "XGBoost · SHAP"),
        ("#2196F3", "Module 4", "Triage Report",    "Rules · Synthesis"),
    ]
    for col, (color, mod, name, tech) in zip([m1, m2, m3, m4], modules):
        with col:
            st.markdown(
                f"""
                <div style="
                    background:#1a1a2e; border:1px solid {color}33;
                    border-radius:10px; padding:14px 16px; text-align:center;
                ">
                    <p style="color:{color}; font-size:10px; font-weight:600;
                               text-transform:uppercase; letter-spacing:1px; margin:0 0 6px;">{mod}</p>
                    <p style="color:#E8E8E8; font-size:13px; font-weight:600; margin:0 0 4px;">{name}</p>
                    <p style="color:#606070; font-size:11px; margin:0;">{tech}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div style="height:32px"/>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#505060; font-size:12px; text-align:center;">'
        'No model weights required for demo — run '
        '<code style="color:#9C27B0;">python scripts/prepare_demo.py</code> '
        'to generate demo assets.</p>',
        unsafe_allow_html=True,
    )


# ── Router ────────────────────────────────────────────────────────────────────
def main():
    _navbar()
    page = st.session_state.get("page", "home")

    if page == "home":
        _landing()
    elif page == "step1":
        from dashboard.pages.step1_screen import render
        render()
    elif page == "step2":
        from dashboard.pages.step2_analyse import render
        render()
    elif page == "step3":
        from dashboard.pages.step3_report import render
        render()
    else:
        st.session_state["page"] = "home"
        st.rerun()


main()
