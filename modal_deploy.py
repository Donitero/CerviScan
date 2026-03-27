"""
Modal deployment configuration for FemScan-AI inference endpoints.

Deploy with:
    modal deploy modal_deploy.py
"""

import modal

app = modal.App("femscan-ai")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
)


@app.function(image=image, gpu="T4", timeout=120)
def classify_cervical(image_bytes: bytes) -> dict:
    """Run CervicalClassifier + GradCAM on raw image bytes. Returns CONTRACT 1 + 2."""
    import io
    import torch
    import numpy as np
    from PIL import Image
    import torchvision.transforms as T
    from models.cervical_classifier import CervicalClassifier
    from models.gradcam import GradCAMExplainer

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_np = np.array(pil_img.resize((224, 224))).astype(np.uint8)

    tensor = transform(pil_img).unsqueeze(0)

    model = CervicalClassifier(checkpoint_path="trained_models/cervical_best.pt")
    explainer = GradCAMExplainer(model)

    classify_result = model.predict(tensor)
    explain_result = explainer.explain(tensor, original_np)

    # Serialise numpy arrays to lists for JSON transport
    explain_result["heatmap"] = explain_result["heatmap"].tolist()
    explain_result["overlay"] = explain_result["overlay"].tolist()

    return {"classify": classify_result, "explain": explain_result}


@app.function(image=image, timeout=60)
def score_endo(symptoms_dict: dict) -> dict:
    """Run EndoSymptomScorer. Returns CONTRACT 3."""
    from models.endo_scorer import EndoSymptomScorer

    scorer = EndoSymptomScorer(model_path="trained_models/endo_xgb.pkl")
    return scorer.predict(symptoms_dict)
