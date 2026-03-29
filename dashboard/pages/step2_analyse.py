"""
Step 2 — Capture & Analyse
Cytology image upload → CIN classification + Grad-CAM explainability.
Falls back to demo assets when no model weights are present.
"""
import sys
import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dashboard.components.risk_card import triage_banner
from dashboard.components.image_viewer import image_viewer

PROJ      = Path(__file__).resolve().parents[2]
DEMO_DIR  = PROJ / "data" / "demo"
MODEL_DIR = PROJ / "trained_models"


def _section_header():
    st.markdown(
        """
        <div style="margin-bottom:20px;">
            <span style="color:#9C27B0; font-size:12px; font-weight:600;
                          text-transform:uppercase; letter-spacing:1.5px;">Step 2 of 3</span>
            <h2 style="color:#E8E8E8; margin:4px 0 6px; font-size:22px;">Capture &amp; Analyse</h2>
            <p style="color:#a0a0b0; margin:0; font-size:14px;">
                Upload a cervical cytology image for CIN classification and AI attention mapping.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _load_model():
    """Try to load trained model. Returns (model, explainer) or (None, None)."""
    checkpoint = MODEL_DIR / "cervical_best.pt"
    if not checkpoint.exists():
        return None, None
    try:
        import torch
        from models.cervical_classifier import CervicalClassifier
        from models.gradcam import GradCAMExplainer
        model    = CervicalClassifier(pretrained=False, checkpoint_path=str(checkpoint))
        model.eval()
        explainer = GradCAMExplainer(model)
        return model, explainer
    except Exception:
        return None, None


def _preprocess(img: Image.Image):
    """Resize + normalise → tensor and uint8 numpy."""
    import torch
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img_224   = img.resize((224, 224))
    img_np    = np.array(img_224.convert("RGB"))
    tensor    = tf(img.convert("RGB")).unsqueeze(0)
    return tensor, img_np


def _demo_result() -> dict:
    """Return a hardcoded demo result when no model weights exist."""
    return {
        "class_name":   "im_Koilocytotic",
        "category":     "Low-grade",
        "confidence":   0.74,
        "cin_grade":    "LSIL / CIN1",
        "triage_color": "amber",
        "action":       "Evidence of HPV infection. Repeat cytology in 6-12 months.",
        "urgency":      "moderate",
        "description":  "Koilocytes indicate HPV-related changes. Consistent with LSIL.",
        "color":        "#FF9800",
        "all_probs": {
            "im_Superficial-Intermediate": 0.06,
            "im_Parabasal":    0.04,
            "im_Metaplastic":  0.08,
            "im_Koilocytotic": 0.74,
            "im_Dyskeratotic": 0.08,
        },
    }


def render():
    _section_header()

    model, explainer = _load_model()
    demo_mode = model is None

    if demo_mode:
        st.info(
            "Running in **demo mode** — no trained weights found in `trained_models/`. "
            "Showing pre-computed example results. Train the model via `notebooks/train_colab.ipynb` "
            "to enable live inference.",
            icon="ℹ️",
        )

    # ── Upload ──────────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload cervical cytology image",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        help="Supported formats: PNG, JPG, BMP, TIFF. Ideal size: 224×224 or larger.",
    )

    # Use demo image if nothing uploaded
    demo_img_path = DEMO_DIR / "cell_im_Koilocytotic.png"
    if uploaded is None:
        if demo_img_path.exists():
            img = Image.open(demo_img_path)
            st.caption("Showing demo cytology image.")
        else:
            st.markdown(
                '<div style="background:#1a1a2e; border:1px dashed #3a3a5e; border-radius:12px; '
                'padding:40px; text-align:center; color:#606070;">Upload an image above to begin analysis</div>',
                unsafe_allow_html=True,
            )
            # Still store empty state and return
            st.session_state.setdefault("cin_result", None)
            _nav_buttons()
            return
    else:
        img = Image.open(uploaded)

    # ── Run inference / load demo ────────────────────────────────────────────
    tensor, img_np = _preprocess(img)

    if demo_mode:
        result        = _demo_result()
        overlay_np    = _load_demo_overlay()
        attention_pct = 18.4
    else:
        with st.spinner("Analysing image..."):
            result     = model.predict(tensor)
            exp_out    = explainer.explain(tensor, img_np)
            overlay_np = exp_out["overlay"]
            attention_pct = exp_out["attention_focus_pct"]

    st.session_state["cin_result"] = result

    # ── Result layout ───────────────────────────────────────────────────────
    st.markdown('<div style="height:12px"/>', unsafe_allow_html=True)

    res_col, detail_col = st.columns([1, 1], gap="large")

    with res_col:
        # Triage banner
        triage_banner(
            triage_color=result["triage_color"],
            cin_grade=result["cin_grade"],
            action=result["action"],
        )

        # Key stats
        st.markdown('<div style="height:8px"/>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="background:#1a1a2e; border-radius:12px; padding:16px 20px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                    <span style="color:#a0a0b0; font-size:13px;">Predicted class</span>
                    <span style="color:#E8E8E8; font-size:13px; font-weight:600;">
                        {result['class_name'].replace('im_', '')}
                    </span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                    <span style="color:#a0a0b0; font-size:13px;">Category</span>
                    <span style="color:#E8E8E8; font-size:13px;">{result['category']}</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                    <span style="color:#a0a0b0; font-size:13px;">Confidence</span>
                    <span style="color:{result['color']}; font-size:13px; font-weight:700;">
                        {result['confidence']*100:.1f}%
                    </span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:#a0a0b0; font-size:13px;">Urgency</span>
                    <span style="color:#E8E8E8; font-size:13px; text-transform:capitalize;">
                        {result['urgency']}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div style="height:8px"/>', unsafe_allow_html=True)
        st.markdown(
            f'<p style="color:#a0a0b0; font-size:12px; font-style:italic; line-height:1.5;">'
            f'{result["description"]}</p>',
            unsafe_allow_html=True,
        )

    with detail_col:
        image_viewer(
            original_np=img_np,
            overlay_np=overlay_np,
            class_name=result["class_name"],
            confidence=result["confidence"],
            all_probs=result.get("all_probs", {}),
            attention_pct=attention_pct,
        )

    _nav_buttons()


def _load_demo_overlay() -> np.ndarray | None:
    path = DEMO_DIR / "gradcam_im_Koilocytotic.png"
    if path.exists():
        return np.array(Image.open(path).resize((224, 224)).convert("RGB"))
    return None


def _nav_buttons():
    st.markdown('<div style="height:16px"/>', unsafe_allow_html=True)
    st.markdown('<hr style="border:none; border-top:1px solid #2a2a3e; margin:0 0 16px;">', unsafe_allow_html=True)
    back_col, _, next_col = st.columns([1, 3, 1])
    with back_col:
        if st.button("← Back to Screening", use_container_width=True):
            st.session_state["page"] = "step1"
            st.rerun()
    with next_col:
        if st.button("View Report →", type="primary", use_container_width=True):
            st.session_state["page"] = "step3"
            st.rerun()
