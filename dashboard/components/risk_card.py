"""
Reusable risk score card component.
Renders a circular gauge + risk band label in a styled card.
"""
import streamlit as st


def risk_card(
    title: str,
    score: float,          # 0-100
    level: str,            # e.g. "High", "Low"
    color: str,            # hex
    recommendation: str,
    key: str = "",
):
    """Render a self-contained risk score card."""
    # Circular gauge via HTML/CSS — no external deps
    pct = int(score)
    # Stroke dasharray for SVG circle (circumference = 2*pi*45 ≈ 283)
    dash = int(283 * pct / 100)

    st.markdown(
        f"""
        <div style="
            background: #1a1a2e;
            border: 1px solid {color}44;
            border-radius: 16px;
            padding: 24px 20px 20px;
            text-align: center;
            height: 100%;
        ">
            <p style="color:#a0a0b0; font-size:13px; margin:0 0 12px; text-transform:uppercase; letter-spacing:1px;">{title}</p>

            <svg width="110" height="110" viewBox="0 0 110 110">
                <circle cx="55" cy="55" r="45"
                    fill="none" stroke="#2a2a3e" stroke-width="10"/>
                <circle cx="55" cy="55" r="45"
                    fill="none" stroke="{color}" stroke-width="10"
                    stroke-dasharray="{dash} 283"
                    stroke-dashoffset="71"
                    stroke-linecap="round"
                    transform="rotate(-90 55 55)"/>
                <text x="55" y="52" text-anchor="middle"
                    fill="white" font-size="22" font-weight="700">{pct}</text>
                <text x="55" y="68" text-anchor="middle"
                    fill="#a0a0b0" font-size="11">/100</text>
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
                margin:10px 0 14px;
            ">{level}</div>

            <p style="color:#c0c0d0; font-size:12px; line-height:1.5; margin:0; text-align:left;">
                {recommendation}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def mini_badge(label: str, color: str):
    """Inline coloured pill badge."""
    st.markdown(
        f"""<span style="
            background:{color}22; border:1px solid {color};
            color:{color}; border-radius:12px; padding:2px 12px;
            font-size:12px; font-weight:600;
        ">{label}</span>""",
        unsafe_allow_html=True,
    )


def triage_banner(triage_color: str, cin_grade: str, action: str):
    """
    Full-width coloured banner for CIN triage result.
    triage_color: "green" | "amber" | "red"
    """
    color_map = {"green": "#4CAF50", "amber": "#FF9800", "red": "#F44336"}
    bg_map    = {"green": "#4CAF5015", "amber": "#FF980015", "red": "#F4433615"}
    hex_c = color_map.get(triage_color, "#9C27B0")
    hex_bg = bg_map.get(triage_color, "#9C27B015")

    icon_map = {"green": "✓", "amber": "!", "red": "!!"}
    icon = icon_map.get(triage_color, "•")

    st.markdown(
        f"""
        <div style="
            background:{hex_bg};
            border-left: 4px solid {hex_c};
            border-radius: 8px;
            padding: 16px 20px;
            margin: 8px 0;
        ">
            <span style="color:{hex_c}; font-weight:700; font-size:15px;">
                {icon}&nbsp; {cin_grade}
            </span>
            <p style="color:#c0c0d0; margin:6px 0 0; font-size:13px;">{action}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
