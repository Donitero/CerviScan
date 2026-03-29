import sys
from pathlib import Path
import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.hpv_risk_scorer import HPVRiskScorer
from models.cancer_scorer import CancerRiskScorer

_hpv_scorer = HPVRiskScorer()       # <-- instantiate HPV scorer
_cancer_scorer = CancerRiskScorer() # <-- instantiate Cancer scorer


from models.hpv_risk_scorer import HPVRiskScorer
from models.cancer_scorer import CancerRiskScorer
_cancer_scorer = CancerRiskScorer()

from dashboard.components.factor_chart import factor_chart


# ─────────────────────────────────────────────
# SAFE HEADER
# ─────────────────────────────────────────────
def _section_header(step: str, title: str, subtitle: str):
    st.markdown(f"### {step}")
    st.title(title)
    st.caption(subtitle)

def _divider():
    st.divider()

# ─────────────────────────────────────────────
# Risk Card Renderer (HTML + SVG)
# ─────────────────────────────────────────────
def risk_card(title, score, level, color, recommendation):
    html = f"""
    <div style="
        background:white;
        border-radius:12px;
        padding:16px;
        margin-bottom:20px;
        box-shadow:0px 4px 12px rgba(0,0,0,0.1);
        text-align:center;
    ">
        <h4 style="margin-bottom:10px; color:#003366;">{title}</h4>
        <svg width="110" height="110" viewBox="0 0 110 110">
          <circle cx="55" cy="55" r="45" fill="none" stroke="#e5e7eb" stroke-width="10"/>
          <circle cx="55" cy="55" r="45" fill="none" stroke="{color}" stroke-width="10"
            stroke-dasharray="{score*2.83} 283"
            stroke-dashoffset="71"
            stroke-linecap="round"
            transform="rotate(-90 55 55)"/>
          <text x="55" y="52" text-anchor="middle" fill="#111827" font-size="22" font-weight="700">{score}</text>
          <text x="55" y="68" text-anchor="middle" fill="#6b7280" font-size="11">/100</text>
        </svg>
        <div style="
          display:inline-block;
          background:{color}22;
          border:1px solid {color};
          color:{color};
          border-radius:20px;
          padding:4px 16px;
          font-size:13px;
          font-weight:600;
          margin-top:10px;
        ">
          {level}
        </div>
        <p style="margin-top:12px; font-size:13px; color:#374151;">{recommendation}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN RENDER
# ─────────────────────────────────────────────
def render():

    _section_header(
        "Step 1 of 3",
        "Screen & Score",
        "Enter patient information to assess HPV acquisition risk and endometriosis likelihood.",
    )

    left, right = st.columns(2, gap="large")

    # ── HPV SECTION ───────────────────────────
    with left:
        st.markdown("#### HPV Risk Assessment")

        age = st.slider("Age", 15, 65, 30)
        num_partners = st.slider("Number of sexual partners", 0, 20, 2)
        age_first = st.slider("Age at first intercourse", 10, 40, 18)
        last_pap = st.slider("Years since last Pap smear", 0, 20, 2)
        stds_count = st.number_input("Prior STDs diagnosed", 0, 10, 0)
        oral_contra = st.slider("Oral contraceptive use (years)", 0, 20, 0)
        iud_yrs = st.slider("IUD use (years)", 0, 15, 0)

        c1, c2 = st.columns(2)
        with c1:
            smoking = st.toggle("Smoker")
            immunocompromised = st.toggle("Immunocompromised")
        with c2:
            hiv = st.toggle("HIV positive")
            prev_biopsy = st.toggle("Previous cervical biopsy")
            family_hx_cc = st.toggle("Family history of cervical cancer")

        hpv_data = {
            "age": float(age),
            "num_sexual_partners": float(num_partners),
            "age_first_intercourse": float(age_first),
            "smoking": float(smoking),
            "oral_contraceptives_yrs": float(oral_contra),
            "iud_yrs": float(iud_yrs),
            "stds_count": float(stds_count),
            "prev_cervical_biopsies": float(prev_biopsy),
            "immunocompromised": float(immunocompromised),
            "hiv_positive": float(hiv),
            "family_hx_cervical_cancer": float(family_hx_cc),
            "last_pap_years_ago": float(last_pap),
        }

        hpv_result = _hpv_scorer.predict(hpv_data)
        st.session_state["hpv_result"] = hpv_result

        risk_card(
            title="HPV Acquisition Risk",
            score=hpv_result["hpv_risk_score"],
            level=hpv_result["risk_level"],
            color=hpv_result["risk_color"],
            recommendation=hpv_result["recommendation"],
        )

        st.caption(f"Strain risk: {hpv_result['hpv_strain_risk']}")

        _divider()
        factor_chart(hpv_result["top_risk_factors"], "Top HPV Risk Factors")

        if hpv_result["escalate_to_imaging"]:
            st.warning("High HPV risk — proceed to Step 2 for cytology analysis.")

  # ── CANCER SECTION ──────────────────────────
    with right:
        st.markdown("#### Cancer Risk Assessment")

        cancer_age = st.slider("Patient age", 20, 80, 42)
        family_hx = st.toggle("Family history of cancer")
        prev_cancer = st.toggle("Previous cancer diagnosis")
        radiation = st.toggle("History of radiation exposure")
        chronic_inflam = st.toggle("Chronic inflammation")
        lifestyle = st.slider("Lifestyle risk (0–10)", 0, 10, 5)
        diet = st.slider("Diet quality (0–10)", 0, 10, 5)
        exercise = st.slider("Exercise frequency (0–10)", 0, 10, 5)

        cancer_data = {
            "age": float(cancer_age),
            "family_history": float(family_hx),
            "previous_cancer": float(prev_cancer),
            "radiation_exposure": float(radiation),
            "chronic_inflammation": float(chronic_inflam),
            "lifestyle_score": float(lifestyle),
            "diet_score": float(diet),
            "exercise_score": float(exercise),
        }

        cancer_result = _cancer_scorer.predict(cancer_data)
        st.session_state["cancer_result"] = cancer_result

        risk_card(
            title="Cancer Risk",
            score=cancer_result["risk_score"],
            level=cancer_result["risk_level"],
            color=cancer_result["risk_color"],
            recommendation=cancer_result["recommendation"],
        )

        if cancer_result["imaging_needed"]:
            st.info("Imaging recommended: biopsy or advanced imaging suggested.")

        _divider()
        factor_chart(cancer_result["top_factors"], "Top Cancer Risk Drivers")
    # ── CTA ─────────────────────────────────
    _divider()
    col_cta = st.columns([3, 1])[1]
    with col_cta:
        if st.button("Continue to Image Analysis →", type="primary", use_container_width=True):
            st.session_state["page"] = "step2"
            st.rerun()
