"""
Horizontal SHAP factor bar chart component.
Works for both HPV (top_risk_factors) and Endo (top_factors) outputs.
"""
import streamlit as st
import plotly.graph_objects as go


def factor_chart(factors: list, title: str = "Key Risk Factors"):
    """
    Render a horizontal Plotly bar chart of SHAP-ranked factors.

    factors: list of dicts with keys:
      - "factor" or "feature_label"  (label string)
      - "impact"                      (float, magnitude)
      - "direction"                   ("increases_risk" | "decreases_risk")
    """
    if not factors:
        st.caption("No factor data available.")
        return

    labels  = []
    values  = []
    colors  = []

    for f in reversed(factors):   # reversed so highest is at top
        label = f.get("factor") or f.get("feature_label", "Unknown")
        impact = f.get("impact", 0)
        direction = f.get("direction", "increases_risk")

        labels.append(label)
        values.append(impact)
        colors.append("#F44336" if direction == "increases_risk" else "#4CAF50")

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(color=colors, opacity=0.85),
        hovertemplate="<b>%{y}</b><br>Impact: %{x:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#F2F7F3")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(l=10, r=10, t=36, b=10),
        xaxis=dict(
            showgrid=True, gridcolor="#1f2a25",
            zeroline=False, tickfont=dict(color="#9BB3A7", size=11),
            title=dict(text="SHAP Impact", font=dict(color="#9BB3A7", size=11)),
        ),
        yaxis=dict(
            tickfont=dict(color="#F2F7F3", size=11),
            automargin=True,
        ),
        font=dict(color="#F2F7F3"),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
