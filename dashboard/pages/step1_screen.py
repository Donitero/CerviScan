"""
Step 1 — Screen & Score
HPV risk assessment (left) + Endometriosis symptom scoring (right).
Calls heuristic scorers directly (no trained weights needed).
"""
import sys
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.hpv_risk_scorer import HPVRiskScorer
from models.endo_scorer import EndoSymptomScorer
from dashboard.components.risk_card import risk_card, mini_badge
from dashboard.components.factor_chart import factor_chart

_hpv_scorer  = HPVRiskScorer()    # heuristic mode (no model path)
_endo_scorer = EndoSymptomScorer()


def _section_header(step: str, title: str, subtitle: str):
    st.markdown(
        f"""
        <div style="margin-bottom:20px;">
            <span style="color:#1BAE77; font-size:12px; font-weight:600;
                          text-transform:uppercase; letter-spacing:1.5px;">{step}</span>
            <h2 style="color:#F2F7F3; margin:4px 0 6px; font-size:22px;">{title}</h2>
            <p style="color:#9BB3A7; margin:0; font-size:14px;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _divider():
    st.markdown('<hr style="border:none; border-top:1px solid #1f2a25; margin:20px 0;">', unsafe_allow_html=True)


def render():
    _section_header(
        "Step 1 of 3",
        "Screen & Score",
        "Enter patient information to assess HPV acquisition risk and endometriosis likelihood.",
    )

    left, right = st.columns(2, gap="large")

    # ── HPV Risk Form ──────────────────────────────────────────────────────────
    with left:
        st.markdown(
            '<p style="color:#1BAE77; font-size:13px; font-weight:600; '
            'text-transform:uppercase; letter-spacing:1px; margin-bottom:16px;">HPV Risk Assessment</p>',
            unsafe_allow_html=True,
        )

        age = st.slider("Age", 15, 65, 30, key="hpv_age")
        num_partners = st.slider("Number of sexual partners", 0, 20, 2, key="hpv_partners")
        age_first = st.slider("Age at first intercourse", 10, 40, 18, key="hpv_first")
        last_pap = st.slider("Years since last Pap smear", 0, 20, 2, key="hpv_pap")
        stds_count = st.number_input("Prior STDs diagnosed", 0, 10, 0, key="hpv_stds")
        oral_contra = st.slider("Oral contraceptive use (years)", 0, 20, 0, key="hpv_oc")
        iud_yrs = st.slider("IUD use (years)", 0, 15, 0, key="hpv_iud")

        c1, c2 = st.columns(2)
        with c1:
            smoking            = st.toggle("Smoker",                    key="hpv_smoke")
            immunocompromised  = st.toggle("Immunocompromised",         key="hpv_imm")
        with c2:
            hiv                = st.toggle("HIV positive",              key="hpv_hiv")
            prev_biopsy        = st.toggle("Prev. cervical biopsy",     key="hpv_bx")
            family_hx_cc       = st.toggle("Family Hx cervical cancer", key="hpv_fam")

        hpv_data = {
            "age":                       float(age),
            "num_sexual_partners":       float(num_partners),
            "age_first_intercourse":     float(age_first),
            "smoking":                   float(smoking),
            "oral_contraceptives_yrs":   float(oral_contra),
            "iud_yrs":                   float(iud_yrs),
            "stds_count":                float(stds_count),
            "prev_cervical_biopsies":    float(prev_biopsy),
            "immunocompromised":         float(immunocompromised),
            "hiv_positive":              float(hiv),
            "family_hx_cervical_cancer": float(family_hx_cc),
            "last_pap_years_ago":        float(last_pap),
        }

        st.markdown('<div style="height:12px"/>', unsafe_allow_html=True)
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

    # ── Endo Symptom Form ──────────────────────────────────────────────────────
    with right:
        st.markdown(
            '<p style="color:#1BAE77; font-size:13px; font-weight:600; '
            'text-transform:uppercase; letter-spacing:1px; margin-bottom:16px;">Endometriosis Symptom Scoring</p>',
            unsafe_allow_html=True,
        )

        endo_age  = st.slider("Patient age", 15, 55, 30, key="endo_age")
        dysm      = st.slider("Dysmenorrhea severity (0–10)", 0, 10, 3, key="endo_dysm")
        dysp      = st.slider("Dyspareunia — pain during sex (0–10)", 0, 10, 0, key="endo_dysp")
        pelvic    = st.slider("Chronic pelvic pain (0–10)", 0, 10, 2, key="endo_pelv")
        fatigue   = st.slider("Fatigue severity (0–10)", 0, 10, 3, key="endo_fat")
        bloating  = st.slider("Bloating / GI symptoms (0–10)", 0, 10, 2, key="endo_bloat")
        urinary   = st.slider("Urinary symptom severity (0–10)", 0, 10, 0, key="endo_uri")
        duration  = st.slider("Symptom duration (years)", 0, 30, 1, key="endo_dur")

        c3, c4 = st.columns(2)
        with c3:
            irreg_cycles  = st.toggle("Irregular cycles",        key="endo_irr")
            heavy_bleed   = st.toggle("Heavy menstrual bleeding", key="endo_hb")
        with c4:
            infertility   = st.toggle("History of infertility",   key="endo_inf")
            family_hx_e   = st.toggle("Family Hx endometriosis",  key="endo_fam")

        endo_data = {
            "age":                  float(endo_age),
            "dysmenorrhea_score":   float(dysm),
            "dyspareunia_score":    float(dysp),
            "chronic_pelvic_pain":  float(pelvic),
            "cycle_regularity":     float(irreg_cycles),
            "heavy_bleeding":       float(heavy_bleed),
            "infertility_hx":       float(infertility),
            "family_hx":            float(family_hx_e),
            "fatigue_score":        float(fatigue),
            "bloating_score":       float(bloating),
            "urinary_symptoms":     float(urinary),
            "symptom_duration_yrs": float(duration),
        }

        st.markdown('<div style="height:12px"/>', unsafe_allow_html=True)
        endo_result = _endo_scorer.predict(endo_data)
        st.session_state["endo_result"] = endo_result

        risk_card(
            title="Endometriosis Risk",
            score=endo_result["risk_score"],
            level=endo_result["risk_level"],
            color=endo_result["risk_color"],
            recommendation=endo_result["recommendation"],
        )

        if endo_result["imaging_needed"]:
            st.info("Imaging recommended: transvaginal ultrasound suggested.")

        _divider()
        factor_chart(endo_result["top_factors"], "Top Symptom Drivers")

    # -- Sample imagery --
    st.markdown('<div style="height:18px"/>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#1BAE77; font-size:11px; text-transform:uppercase; '        'letter-spacing:2px; margin-bottom:12px;">Additional Cytology Samples</p>',
        unsafe_allow_html=True,
    )
    asset_dir = Path(__file__).resolve().parents[2] / "dashboard" / "assets" / "preview"
    s1, s2 = st.columns(2, gap="large")
    samples = [
        (asset_dir / "preview5.jpg", "Sample E"),
        (asset_dir / "preview6.jpg", "Sample F"),
    ]
    for col, (path, caption) in zip([s1, s2], samples):
        with col:
            if path.exists():
                st.image(str(path), use_container_width=True)
                st.caption(caption)

    # ── Continue CTA ──────────────────────────────────────────────────────────
    st.markdown('<div style="height:16px"/>', unsafe_allow_html=True)
    _divider()
    col_cta = st.columns([3, 1])[1]
    with col_cta:
        if st.button("Continue to Image Analysis →", type="primary", use_container_width=True):
            st.session_state["page"] = "step2"
            st.rerun()
