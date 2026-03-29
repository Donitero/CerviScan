"""
Step 3 — Act & Record
Consolidated triage report from all prior steps.
"""
import sys
from pathlib import Path
from datetime import datetime

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dashboard.components.report_card import report_card
from dashboard.components.risk_card import triage_banner


def render():
    st.markdown(
        """
        <div style="margin-bottom:20px;">
            <span style="color:#1BAE77; font-size:12px; font-weight:600;
                          text-transform:uppercase; letter-spacing:1.5px;">Step 3 of 3</span>
            <h2 style="color:#F2F7F3; margin:4px 0 6px; font-size:22px;">Clinical Report</h2>
            <p style="color:#9BB3A7; margin:0; font-size:14px;">
                Consolidated findings and recommended actions across all modules.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    hpv_result  = st.session_state.get("hpv_result")
    endo_result = st.session_state.get("endo_result")
    cin_result  = st.session_state.get("cin_result")

    # Warn if nothing has been run yet
    if not any([hpv_result, endo_result, cin_result]):
        st.warning("No results yet. Complete Steps 1 and 2 first.")
        if st.button("← Go to Step 1"):
            st.session_state["page"] = "step1"
            st.rerun()
        return

    # ── Overall triage colour (worst of all results) ─────────────────────────
    overall_color, overall_label = _overall_triage(hpv_result, endo_result, cin_result)
    st.markdown(
        f"""
        <div style="
            background:{overall_color}15;
            border:1px solid {overall_color}66;
            border-radius:12px;
            padding:16px 24px;
            margin-bottom:20px;
            display:flex;
            align-items:center;
            gap:16px;
        ">
            <div style="
                width:48px; height:48px; border-radius:50%;
                background:{overall_color}33;
                border:2px solid {overall_color};
                display:flex; align-items:center; justify-content:center;
                font-size:20px; flex-shrink:0;
            ">{'🔴' if overall_color == '#F44336' else ('🟡' if overall_color == '#FF9800' else '🟢')}</div>
            <div>
                <p style="color:#9BB3A7; font-size:12px; margin:0 0 2px; text-transform:uppercase; letter-spacing:1px;">Overall Triage Status</p>
                <p style="color:{overall_color}; font-size:18px; font-weight:700; margin:0;">{overall_label}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Full report card ─────────────────────────────────────────────────────
    report_card(
        hpv_result=hpv_result,
        endo_result=endo_result,
        cin_result=cin_result,
    )

    st.markdown('<div style="height:20px"/>', unsafe_allow_html=True)

    # ── Detailed findings expanders ──────────────────────────────────────────
    if hpv_result:
        with st.expander("HPV Risk — Detailed Findings"):
            _detail_row("Risk Score",   f"{hpv_result['hpv_risk_score']:.0f} / 100")
            _detail_row("Risk Level",   hpv_result["risk_level"])
            _detail_row("Strain Risk",  hpv_result["hpv_strain_risk"])
            _detail_row("Imaging",      "Required" if hpv_result["escalate_to_imaging"] else "Not required")
            st.markdown('<hr style="border:none;border-top:1px solid #1f2a25;margin:12px 0"/>', unsafe_allow_html=True)
            st.caption("Top contributing factors:")
            for f in hpv_result.get("top_risk_factors", []):
                arrow = "▲" if f["direction"] == "increases_risk" else "▼"
                color = "#F44336" if f["direction"] == "increases_risk" else "#4CAF50"
                st.markdown(
                    f'<span style="color:{color};">{arrow}</span> '
                    f'<span style="color:#F2F7F3; font-size:13px;">{f.get("factor","")}</span> '
                    f'<span style="color:#6f857a; font-size:12px;">(impact: {f["impact"]:.3f})</span>',
                    unsafe_allow_html=True,
                )

    if endo_result:
        with st.expander("Endometriosis — Detailed Findings"):
            _detail_row("Risk Score",  f"{endo_result['risk_score']:.0f} / 100")
            _detail_row("Risk Level",  endo_result["risk_level"])
            _detail_row("Imaging",     "Recommended" if endo_result["imaging_needed"] else "Not required")
            st.markdown('<hr style="border:none;border-top:1px solid #1f2a25;margin:12px 0"/>', unsafe_allow_html=True)
            st.caption("Top symptom drivers:")
            for f in endo_result.get("top_factors", []):
                arrow = "▲" if f["direction"] == "increases_risk" else "▼"
                color = "#F44336" if f["direction"] == "increases_risk" else "#4CAF50"
                st.markdown(
                    f'<span style="color:{color};">{arrow}</span> '
                    f'<span style="color:#F2F7F3; font-size:13px;">{f.get("feature_label","")}</span> '
                    f'<span style="color:#6f857a; font-size:12px;">(impact: {f["impact"]:.3f})</span>',
                    unsafe_allow_html=True,
                )

    if cin_result:
        with st.expander("CIN Analysis — Detailed Findings"):
            _detail_row("Class",       cin_result["class_name"].replace("im_", ""))
            _detail_row("CIN Grade",   cin_result["cin_grade"])
            _detail_row("Category",    cin_result["category"])
            _detail_row("Confidence",  f"{cin_result['confidence']*100:.1f}%")
            _detail_row("Urgency",     cin_result["urgency"].capitalize())

    # ── Export button ─────────────────────────────────────────────────────────
    st.markdown('<div style="height:12px"/>', unsafe_allow_html=True)
    report_text = _build_text_report(hpv_result, endo_result, cin_result)
    st.download_button(
        label="Download Report (.txt)",
        data=report_text,
        file_name=f"femscan_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
        use_container_width=False,
    )

    # ── Nav ───────────────────────────────────────────────────────────────────
    st.markdown('<hr style="border:none; border-top:1px solid #1f2a25; margin:20px 0 16px;">', unsafe_allow_html=True)
    back_col, _, restart_col = st.columns([1, 3, 1])
    with back_col:
        if st.button("← Back to Analysis", use_container_width=True):
            st.session_state["page"] = "step2"
            st.rerun()
    with restart_col:
        if st.button("New Assessment", use_container_width=True):
            for key in ["hpv_result", "endo_result", "cin_result", "page"]:
                st.session_state.pop(key, None)
            st.session_state["page"] = "step1"
            st.rerun()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detail_row(label: str, value: str):
    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between;
                    padding:6px 0; border-bottom:1px solid #1f2a25;">
            <span style="color:#9BB3A7; font-size:13px;">{label}</span>
            <span style="color:#F2F7F3; font-size:13px; font-weight:500;">{value}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _overall_triage(hpv, endo, cin):
    """Return (hex_color, label) for the most urgent finding."""
    levels = []
    if cin:
        c = cin.get("triage_color", "green")
        levels.append({"green": 0, "amber": 1, "red": 2}.get(c, 0))
    if hpv:
        l = hpv.get("risk_level", "Low")
        levels.append({"Low": 0, "Medium": 1, "High": 2}.get(l, 0))
    if endo:
        l = endo.get("risk_level", "Low")
        levels.append({"Low": 0, "Moderate": 1, "High": 2, "Very High": 2}.get(l, 0))

    worst = max(levels) if levels else 0
    if worst == 2:
        return "#F44336", "High Priority — Urgent Clinical Review Required"
    if worst == 1:
        return "#FF9800", "Moderate Risk — Follow-up Recommended"
    return "#4CAF50", "Low Risk — Routine Monitoring"


def _build_text_report(hpv, endo, cin) -> str:
    now  = datetime.now().strftime("%d %B %Y, %H:%M")
    lines = [
        "=" * 56,
        "  FEMSCAN AI — CLINICAL TRIAGE REPORT",
        f"  Generated: {now}",
        "=" * 56,
        "",
    ]
    if hpv:
        lines += [
            "HPV RISK ASSESSMENT",
            f"  Score     : {hpv['hpv_risk_score']:.0f} / 100",
            f"  Level     : {hpv['risk_level']}",
            f"  Strain    : {hpv['hpv_strain_risk']}",
            f"  Action    : {hpv['recommendation']}",
            "",
        ]
    if endo:
        lines += [
            "ENDOMETRIOSIS RISK",
            f"  Score     : {endo['risk_score']:.0f} / 100",
            f"  Level     : {endo['risk_level']}",
            f"  Imaging   : {'Recommended' if endo['imaging_needed'] else 'Not required'}",
            f"  Action    : {endo['recommendation']}",
            "",
        ]
    if cin:
        lines += [
            "CIN / CYTOLOGY ANALYSIS",
            f"  Grade     : {cin['cin_grade']}",
            f"  Category  : {cin['category']}",
            f"  Confidence: {cin['confidence']*100:.1f}%",
            f"  Action    : {cin['action']}",
            "",
        ]
    lines += [
        "-" * 56,
        "For clinical decision support only.",
        "Not a substitute for professional medical judgement.",
        "=" * 56,
    ]
    return "\n".join(lines)
