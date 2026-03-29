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
            '<p style="color:#a0a0b0; font-size:12px; text-align:center; margin-bottom:4px;">CYTOLOGY IMAGE</p>',
            unsafe_allow_html=True,
        )
        st.image(original_np, use_container_width=True)

    with col2:
        if overlay_np is not None:
            label = "GRAD-CAM ATTENTION"
            if attention_pct is not None:
                label += f"  ·  {attention_pct:.1f}% focus area"
            st.markdown(
                f'<p style="color:#a0a0b0; font-size:12px; text-align:center; margin-bottom:4px;">{label}</p>',
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


def _prob_bars(all_probs: dict, predicted: str):
    labels = []
    values = []
    colors = []

    for cls, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
        labels.append(_DISPLAY.get(cls, cls))
        values.append(round(prob * 100, 1))
        base = _CLASS_COLORS.get(cls, "#9C27B0")
        colors.append(base if cls == predicted else base + "55")

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(color=colors),
        hovertemplate="<b>%{y}</b>: %{x:.1f}%<extra></extra>",
        text=[f"{v:.0f}%" for v in values],
        textposition="outside",
        textfont=dict(color="#a0a0b0", size=11),
    ))

    fig.update_layout(
        title=dict(text="Class Probabilities", font=dict(size=13, color="#E8E8E8")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=200,
        margin=dict(l=10, r=40, t=36, b=10),
        xaxis=dict(
            range=[0, 110],
            showgrid=False, zeroline=False,
            tickfont=dict(color="#a0a0b0", size=10),
        ),
        yaxis=dict(
            tickfont=dict(color="#E8E8E8", size=11),
            automargin=True,
        ),
        font=dict(color="#E8E8E8"),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
