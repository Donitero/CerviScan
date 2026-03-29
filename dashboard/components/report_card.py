"""
Final consolidated report panel.
Combines all three model outputs into a single triage summary.
"""
import streamlit as st
from datetime import datetime


def report_card(
    hpv_result:   dict | None,
    endo_result:  dict | None,
    cin_result:   dict | None,
    patient_age:  int | None = None,
):
    """Render the full triage report from all available results."""
    now = datetime.now().strftime("%d %b %Y, %H:%M")

    st.markdown(
        f"""
        <div style="
            background:#141a1b;
            border:1px solid #1BAE7733;
            border-radius:16px;
            padding:28px 28px 20px;
        ">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:20px;">
                <div>
                    <p style="color:#1BAE77; font-size:12px; text-transform:uppercase;
                               letter-spacing:1.5px; margin:0 0 4px;">FemScan AI</p>
                    <h2 style="color:#F2F7F3; margin:0; font-size:20px;">Clinical Triage Report</h2>
                </div>
                <p style="color:#6f857a; font-size:12px; margin:0;">{now}</p>
            </div>
        """,
        unsafe_allow_html=True,
    )

    # — Module results grid —
    cols = st.columns(3)

    with cols[0]:
        _result_tile(
            "HPV Risk",
            hpv_result.get("hpv_risk_score", 0) if hpv_result else None,
            hpv_result.get("risk_level", "—") if hpv_result else "—",
            hpv_result.get("risk_color", "#6f857a") if hpv_result else "#6f857a",
            hpv_result.get("hpv_strain_risk", "") if hpv_result else "",
        )

    with cols[1]:
        _result_tile(
            "Endometriosis Risk",
            endo_result.get("risk_score", 0) if endo_result else None,
            endo_result.get("risk_level", "—") if endo_result else "—",
            endo_result.get("risk_color", "#6f857a") if endo_result else "#6f857a",
            "Imaging recommended" if (endo_result or {}).get("imaging_needed") else "",
        )

    with cols[2]:
        if cin_result:
            _cin_tile(cin_result)
        else:
            _result_tile("CIN Analysis", None, "Not run", "#6f857a", "Upload image in Step 2")

    st.markdown("<div style='height:20px'/>", unsafe_allow_html=True)

    # — Recommended actions —
    actions = _collect_actions(hpv_result, endo_result, cin_result)
    if actions:
        st.markdown(
            '<p style="color:#9BB3A7; font-size:12px; text-transform:uppercase; '
            'letter-spacing:1px; margin:0 0 10px;">Recommended Actions</p>',
            unsafe_allow_html=True,
        )
        for urgency, text in actions:
            icon  = {"high": "🔴", "moderate": "🟡", "low": "🟢"}.get(urgency, "•")
            st.markdown(
                f'<div style="padding:8px 0; border-bottom:1px solid #1f2a25; '
                f'color:#F2F7F3; font-size:13px;">{icon}&nbsp; {text}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # Disclaimer
    st.markdown(
        '<p style="color:#404050; font-size:11px; text-align:center; margin-top:10px;">'
        'For clinical decision support only. Not a substitute for professional medical judgement.</p>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _result_tile(title, score, level, color, sub):
    score_str = f"{score:.0f}/100" if score is not None else "—"
    st.markdown(
        f"""
        <div style="
            background:#101516;
            border:1px solid {color}33;
            border-radius:12px;
            padding:16px;
            text-align:center;
        ">
            <p style="color:#9BB3A7; font-size:11px; text-transform:uppercase;
                       letter-spacing:1px; margin:0 0 8px;">{title}</p>
            <p style="color:{color}; font-size:28px; font-weight:700; margin:0 0 4px;">{score_str}</p>
            <span style="
                background:{color}22; border:1px solid {color};
                color:{color}; border-radius:10px; padding:2px 12px;
                font-size:12px; font-weight:600;
            ">{level}</span>
            <p style="color:#6f857a; font-size:11px; margin:8px 0 0;">{sub}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _cin_tile(cin_result):
    color_map = {"green": "#4CAF50", "amber": "#FF9800", "red": "#F44336"}
    color = color_map.get(cin_result.get("triage_color", "green"), "#4CAF50")
    grade = cin_result.get("cin_grade", "—")
    conf  = cin_result.get("confidence", 0)
    cat   = cin_result.get("category", "")

    st.markdown(
        f"""
        <div style="
            background:#101516;
            border:1px solid {color}33;
            border-radius:12px;
            padding:16px;
            text-align:center;
        ">
            <p style="color:#9BB3A7; font-size:11px; text-transform:uppercase;
                       letter-spacing:1px; margin:0 0 8px;">CIN Analysis</p>
            <p style="color:{color}; font-size:18px; font-weight:700; margin:0 0 4px;">{grade}</p>
            <span style="
                background:{color}22; border:1px solid {color};
                color:{color}; border-radius:10px; padding:2px 12px;
                font-size:12px; font-weight:600;
            ">{cat}</span>
            <p style="color:#6f857a; font-size:11px; margin:8px 0 0;">{conf*100:.0f}% confidence</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _collect_actions(hpv, endo, cin) -> list[tuple[str, str]]:
    actions = []

    if cin:
        urgency = cin.get("urgency", "low")
        actions.append((urgency, cin.get("action", "")))

    if hpv:
        level = hpv.get("risk_level", "Low")
        urgency = "high" if level == "High" else ("moderate" if level == "Medium" else "low")
        actions.append((urgency, hpv.get("recommendation", "")))

    if endo:
        level = endo.get("risk_level", "Low")
        urgency = "high" if level in ("High", "Very High") else ("moderate" if level == "Moderate" else "low")
        actions.append((urgency, endo.get("recommendation", "")))

    # Sort: high first
    order = {"high": 0, "moderate": 1, "low": 2}
    actions.sort(key=lambda x: order.get(x[0], 3))
    return actions
