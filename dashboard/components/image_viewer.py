"""
Cytology image viewer component.
Shows original image alongside Grad-CAM overlay with confidence bars.
"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np


def image_viewer(
    original_np: np.ndarray,        # (224,224,3) uint8 RGB
    overlay_np:  np.ndarray | None, # (224,224,3) uint8 RGB Grad-CAM overlay
    class_name:  str,
    confidence:  float,
    all_probs:   dict,
    attention_pct: float | None = None,
):
    """
    Two-column layout: original image | Grad-CAM overlay.
    Below: probability bars for all classes.
    """
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown(
            '<p style="color:#9BB3A7; font-size:12px; text-align:center; margin-bottom:4px;">CYTOLOGY IMAGE</p>',
            unsafe_allow_html=True,
        )
        st.image(original_np, use_container_width=True)

    with col2:
        if overlay_np is not None:
            label = "GRAD-CAM ATTENTION"
            if attention_pct is not None:
                label += f"  ·  {attention_pct:.1f}% focus area"
            st.markdown(
                f'<p style="color:#9BB3A7; font-size:12px; text-align:center; margin-bottom:4px;">{label}</p>',
                unsafe_allow_html=True,
            )
            st.image(overlay_np, use_container_width=True)
        else:
            st.markdown(
                '<p style="color:#555; font-size:12px; text-align:center;">Grad-CAM not available<br>(no model weights)</p>',
                unsafe_allow_html=True,
            )

    # Probability distribution bar chart
    if all_probs:
        _prob_bars(all_probs, class_name)


# ---------------------------------------------------------------------------
# Probability bars
# ---------------------------------------------------------------------------

# Clean display names for the im_ prefixed class names
_DISPLAY = {
    "im_Superficial-Intermediate": "Superficial-Intermediate",
    "im_Parabasal":    "Parabasal",
    "im_Metaplastic":  "Metaplastic",
    "im_Koilocytotic": "Koilocytotic",
    "im_Dyskeratotic": "Dyskeratotic",
}

_CLASS_COLORS = {
    "im_Superficial-Intermediate": "#4CAF50",
    "im_Parabasal":    "#8BC34A",
    "im_Metaplastic":  "#CDDC39",
    "im_Koilocytotic": "#FF9800",
    "im_Dyskeratotic": "#F44336",
}

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return f"rgba(27,174,119,{alpha})"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _prob_bars(all_probs: dict, predicted: str):
    labels = []
    values = []
    colors = []

    for cls, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
        labels.append(_DISPLAY.get(cls, cls))
        values.append(round(prob * 100, 1))
        base = _CLASS_COLORS.get(cls, "#1BAE77")
        colors.append(base if cls == predicted else _hex_to_rgba(base, 0.35))

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(color=colors),
        hovertemplate="<b>%{y}</b>: %{x:.1f}%<extra></extra>",
        text=[f"{v:.0f}%" for v in values],
        textposition="outside",
        textfont=dict(color="#9BB3A7", size=11),
    ))

    fig.update_layout(
        title=dict(text="Class Probabilities", font=dict(size=13, color="#F2F7F3")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=200,
        margin=dict(l=10, r=40, t=36, b=10),
        xaxis=dict(
            range=[0, 110],
            showgrid=False, zeroline=False,
            tickfont=dict(color="#9BB3A7", size=10),
        ),
        yaxis=dict(
            tickfont=dict(color="#F2F7F3", size=11),
            automargin=True,
        ),
        font=dict(color="#F2F7F3"),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
