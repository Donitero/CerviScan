"""
Prepare demo assets for CONTRACT 4.

Creates synthetic/placeholder demo images and pre-computed result JSONs
so the Streamlit app can run without trained model weights.

Usage:
    python scripts/prepare_demo.py [--model-dir trained_models/]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.cervical_classifier import CLASS_NAMES, CLASS_META
from models.endo_scorer import EndoSymptomScorer

DEMO_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "demo")

# Synthetic symptom profiles for each endo risk level
ENDO_PROFILES = {
    "low": {
        "dysmenorrhea_score": 2.0, "dyspareunia_score": 1.0, "chronic_pelvic_pain": 1.0,
        "cycle_regularity": 0.0, "heavy_bleeding": 0.0, "infertility_hx": 0.0,
        "family_hx": 0.0, "fatigue_score": 2.0, "bloating_score": 1.0,
        "urinary_symptoms": 0.0, "symptom_duration_yrs": 0.5, "age": 24.0,
    },
    "moderate": {
        "dysmenorrhea_score": 5.0, "dyspareunia_score": 4.0, "chronic_pelvic_pain": 4.0,
        "cycle_regularity": 1.0, "heavy_bleeding": 1.0, "infertility_hx": 0.0,
        "family_hx": 0.0, "fatigue_score": 5.0, "bloating_score": 4.0,
        "urinary_symptoms": 2.0, "symptom_duration_yrs": 2.0, "age": 30.0,
    },
    "high": {
        "dysmenorrhea_score": 8.0, "dyspareunia_score": 7.0, "chronic_pelvic_pain": 7.0,
        "cycle_regularity": 1.0, "heavy_bleeding": 1.0, "infertility_hx": 1.0,
        "family_hx": 1.0, "fatigue_score": 7.0, "bloating_score": 6.0,
        "urinary_symptoms": 4.0, "symptom_duration_yrs": 5.0, "age": 33.0,
    },
    "very_high": {
        "dysmenorrhea_score": 10.0, "dyspareunia_score": 9.0, "chronic_pelvic_pain": 9.0,
        "cycle_regularity": 1.0, "heavy_bleeding": 1.0, "infertility_hx": 1.0,
        "family_hx": 1.0, "fatigue_score": 9.0, "bloating_score": 8.0,
        "urinary_symptoms": 7.0, "symptom_duration_yrs": 10.0, "age": 36.0,
    },
}

# Synthetic cell colours for placeholder images
CELL_COLOURS = {
    "Superficial-Intermediate": (180, 220, 180),
    "Parabasal":               (160, 200, 160),
    "Koilocyte":               (220, 180, 100),
    "Dyskeratocyte":           (220, 140,  80),
    "Metaplastic":             (220,  80,  80),
}


def make_synthetic_cell_image(class_name: str, size: int = 224) -> np.ndarray:
    """Generate a simple synthetic circular-cell placeholder image."""
    bg_color = (20, 20, 30)
    cell_color = CELL_COLOURS.get(class_name, (200, 200, 200))

    img = Image.new("RGB", (size, size), bg_color)
    draw = ImageDraw.Draw(img)
    cx, cy, r = size // 2, size // 2, size // 3
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=cell_color, outline=(255, 255, 255), width=2)
    # nucleus
    nr = r // 3
    draw.ellipse([cx - nr, cy - nr, cx + nr, cy + nr], fill=(40, 40, 80))
    return np.array(img)


def make_synthetic_gradcam_overlay(class_name: str, size: int = 224) -> np.ndarray:
    """Generate a simple heatmap overlay placeholder."""
    base = make_synthetic_cell_image(class_name, size)
    heatmap = np.zeros((size, size, 3), dtype=np.uint8)
    cx, cy, r = size // 2, size // 2, size // 3
    for y in range(size):
        for x in range(size):
            d = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if d < r:
                intensity = int(255 * (1 - d / r))
                heatmap[y, x] = (intensity, 0, 255 - intensity)  # red-blue gradient
    overlay = (base.astype(np.float32) * 0.6 + heatmap.astype(np.float32) * 0.4).clip(0, 255).astype(np.uint8)
    return overlay


def main(model_dir: str):
    os.makedirs(DEMO_DIR, exist_ok=True)
    scorer = EndoSymptomScorer()  # heuristic mode — no trained weights needed

    manifest = {"cervical": [], "endo": []}

    # --- Cervical demo images ---
    for class_name in CLASS_NAMES:
        safe_name = class_name.replace(" ", "-")
        img_path = os.path.join(DEMO_DIR, f"demo_{safe_name}.png")
        cam_path = os.path.join(DEMO_DIR, f"demo_{safe_name}_gradcam.png")

        cell_img = make_synthetic_cell_image(class_name)
        cam_img = make_synthetic_gradcam_overlay(class_name)

        Image.fromarray(cell_img).save(img_path)
        Image.fromarray(cam_img).save(cam_path)

        meta = CLASS_META[class_name]
        probs = {n: 0.02 for n in CLASS_NAMES}
        probs[class_name] = 0.90

        predict_result = {
            "class_name":  class_name,
            "category":    meta["category"],
            "confidence":  0.90,
            "all_probs":   probs,
            "action":      meta["action"],
            "urgency":     meta["urgency"],
            "description": meta["description"],
            "color":       meta["color"],
        }

        manifest["cervical"].append({
            "class_name":   class_name,
            "image_path":   os.path.relpath(img_path),
            "gradcam_path": os.path.relpath(cam_path),
            "predict_result": predict_result,
            "explain_result": {"attention_focus_pct": 18.5},
        })
        print(f"[cervical] {class_name} -> {img_path}")

    # --- Endo demo JSONs ---
    for risk_key, symptoms in ENDO_PROFILES.items():
        result = scorer.predict(symptoms)
        result_path = os.path.join(DEMO_DIR, f"demo_endo_{risk_key}.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

        manifest["endo"].append({
            "risk_level":    result["risk_level"],
            "result_path":   os.path.relpath(result_path),
            "predict_result": result,
        })
        print(f"[endo] {risk_key} -> {result_path}")

    # --- Write manifest ---
    manifest_path = os.path.join(DEMO_DIR, "demo_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {manifest_path}")
    print("Demo assets ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare FemScan-AI demo assets (CONTRACT 4)")
    parser.add_argument("--model-dir", default="trained_models/", help="Path to trained model weights")
    args = parser.parse_args()
    main(args.model_dir)
