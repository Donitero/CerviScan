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
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&display=swap');

    :root {
        --bg: #0b0d0e;
        --surface: #141a1b;
        --surface-2: #101516;
        --accent: #1BAE77;
        --accent-strong: #22c986;
        --text: #F2F7F3;
        --muted: #9BB3A7;
        --border: #1f2a25;
    }

    html, body, [class*="stApp"] {
        font-family: "Sora", "Segoe UI", system-ui, sans-serif;
        color: var(--text);
    }

    .stApp {
        background:
            radial-gradient(1100px 700px at 75% -10%, rgba(27,174,119,0.18), transparent 60%),
            radial-gradient(800px 600px at 10% 90%, rgba(27,174,119,0.12), transparent 60%),
            #0b0d0e;
    }

    /* Hide default Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }

    /* Tighten top padding */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    /* Streamlit button override */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        color: #06110c;
        border: 1px solid var(--accent);
        background: linear-gradient(180deg, var(--accent-strong), var(--accent));
        box-shadow: 0 10px 30px rgba(27,174,119,0.25);
        transition: transform 0.15s, filter 0.15s, box-shadow 0.15s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        filter: brightness(1.05);
        box-shadow: 0 14px 34px rgba(27,174,119,0.35);
    }

    /* Slider label colour */
    .stSlider label { color: var(--muted) !important; }

    /* Toggle label */
    .stCheckbox label, .stToggle label { color: var(--muted) !important; }

    /* File uploader */
    [data-testid="stFileUploadDropzone"] {
        background: var(--surface) !important;
        border: 1px dashed #214236 !important;
        border-radius: 14px !important;
    }

    /* Expander header */
    .streamlit-expanderHeader {
        background: var(--surface) !important;
        border-radius: 10px !important;
        color: var(--text) !important;
        border: 1px solid var(--border);
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
            '<span style="color:#1BAE77; font-weight:700; font-size:16px;">🔬 FemScan AI</span>',
            unsafe_allow_html=True,
        )

    nav_items = [("step1", "1 · Screen"), ("step2", "2 · Analyse"), ("step3", "3 · Report")]
    for i, (key, label) in enumerate(nav_items, start=2):
        with cols[i]:
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state["page"] = key
                st.rerun()

    st.markdown(
        '<hr style="border:none; border-top:1px solid #1f2a25; margin:8px 0 24px;">',
        unsafe_allow_html=True,
    )


# ── Landing page ──────────────────────────────────────────────────────────────
def _landing():
    asset_dir = Path(__file__).resolve().parent / "dashboard" / "assets" / "preview"

    st.markdown(
        """
        <div style="text-align:center; padding: 40px 0 32px;">
            <p style="color:#1BAE77; font-size:13px; font-weight:600;
                       text-transform:uppercase; letter-spacing:2px; margin-bottom:12px;">
                AI-Powered Women's Health
            </p>
            <h1 style="color:#F2F7F3; font-size:42px; font-weight:700; margin:0 0 16px;
                        line-height:1.15;">
                Cervical Cancer Screening<br>
                <span style="color:#1BAE77;">Decision Support</span>
            </h1>
            <p style="color:#9BB3A7; font-size:16px; max-width:580px;
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
                    background:#141a1b; border:1px solid #1f2a25;
                    border-radius:16px; padding:28px 24px;
                ">
                    <div style="
                        width:44px; height:44px; border-radius:12px;
                        background:#1BAE7722; border:1px solid #1BAE7766;
                        display:flex; align-items:center; justify-content:center;
                        font-size:20px; margin-bottom:16px;
                    ">{icon}</div>
                    <p style="color:#1BAE77; font-size:11px; font-weight:600;
                               text-transform:uppercase; letter-spacing:1px; margin:0 0 8px;">
                        Step {step}
                    </p>
                    <h3 style="color:#F2F7F3; font-size:16px; margin:0 0 10px;">{title}</h3>
                    <p style="color:#9BB3A7; font-size:13px; line-height:1.6; margin:0;">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div style="height:28px"/>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#1BAE77; font-size:11px; text-align:center; '
        'text-transform:uppercase; letter-spacing:2px; margin-bottom:14px;">Cytology Preview Grid</p>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4, gap="small")
    previews = [
        (asset_dir / "preview1.jpg", "Sample A"),
        (asset_dir / "preview2.jpg", "Sample B"),
        (asset_dir / "preview3.jpg", "Sample C"),
        (asset_dir / "preview4.jpg", "Sample D"),
    ]
    for col, (path, caption) in zip([c1, c2, c3, c4], previews):
        with col:
            if path.exists():
                st.image(str(path), use_container_width=True)
                st.caption(caption)

    st.markdown('<div style="height:40px"/>', unsafe_allow_html=True)

    st.markdown(
        '<p style="color:#6f857a; font-size:12px; text-align:center; '
        'text-transform:uppercase; letter-spacing:1.5px; margin-bottom:16px;">Models</p>',
        unsafe_allow_html=True,
    )
    m1, m2, m3, m4 = st.columns(4, gap="small")
    modules = [
        ("#4CAF50", "Module 1", "HPV Risk Scorer",  "XGBoost · SHAP"),
        ("#FF9800", "Module 2", "CIN Classifier",   "EfficientNetV2-S · Grad-CAM"),
        ("#1BAE77", "Module 3", "Endo Scorer",      "XGBoost · SHAP"),
        ("#2196F3", "Module 4", "Triage Report",    "Rules · Synthesis"),
    ]
    for col, (color, mod, name, tech) in zip([m1, m2, m3, m4], modules):
        with col:
            st.markdown(
                f"""
                <div style="
                    background:#141a1b; border:1px solid {color}33;
                    border-radius:10px; padding:14px 16px; text-align:center;
                ">
                    <p style="color:{color}; font-size:10px; font-weight:600;
                               text-transform:uppercase; letter-spacing:1px; margin:0 0 6px;">{mod}</p>
                    <p style="color:#F2F7F3; font-size:13px; font-weight:600; margin:0 0 4px;">{name}</p>
                    <p style="color:#6f857a; font-size:11px; margin:0;">{tech}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div style="height:32px"/>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#6f857a; font-size:12px; text-align:center;">'
        'No model weights required for demo — run '
        '<code style="color:#1BAE77;">python scripts/prepare_demo.py</code> '
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
