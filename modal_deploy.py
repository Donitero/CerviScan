"""
Modal Deployment — FemScan AI
==============================
Osborn   → train_* functions (fine-tuning pipelines)
Person D → serve_* functions (inference endpoints)

Deploy:
    modal deploy modal_deploy.py
"""

import modal

app = modal.App("femscan-ai")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
)

# ── Training functions (Osborn) ────────────────────────────────────────────

@app.function(image=image, gpu="A10G", timeout=3600)
def train_cervical_classifier():
    """Fine-tune EfficientNetV2-S on SIPaKMeD. Saves to trained_models/cervical_best.pt"""
    raise NotImplementedError("Training pipeline — Osborn implements this.")


@app.function(image=image, timeout=1800)
def train_hpv_scorer():
    """Train XGBoost HPV risk scorer. Saves to trained_models/hpv_xgb.pkl"""
    raise NotImplementedError("Training pipeline — Osborn implements this.")


@app.function(image=image, timeout=1800)
def train_endo_scorer():
    """Train XGBoost endo scorer. Saves to trained_models/endo_xgb.pkl"""
    raise NotImplementedError("Training pipeline — Osborn implements this.")


# ── Inference endpoints (Person D) ────────────────────────────────────────

@app.function(image=image, gpu="T4", timeout=120)
def serve_cervical(image_bytes: bytes) -> dict:
    """
    Run Module 2: CervicalClassifier + GradCAM.
    Returns Contracts 2A + 2B (heatmap/overlay as nested lists for JSON transport).
    """
    # Person D implements this
    raise NotImplementedError("Inference endpoint — Person D implements this.")


@app.function(image=image, timeout=60)
def serve_hpv(patient_data: dict) -> dict:
    """Run Module 1: HPVRiskScorer. Returns Contract 1A."""
    # Person D implements this
    raise NotImplementedError("Inference endpoint — Person D implements this.")


@app.function(image=image, timeout=60)
def serve_endo(symptoms: dict) -> dict:
    """Run Module 3: EndoSymptomScorer. Returns Contract 1B."""
    # Person D implements this
    raise NotImplementedError("Inference endpoint — Person D implements this.")
