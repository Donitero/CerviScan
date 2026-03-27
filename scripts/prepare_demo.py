"""
Prepare demo assets — CONTRACT 4 (data/demo/)
==============================================
Generates synthetic placeholder images and pre-computed result JSONs
so the Streamlit dashboard runs without any trained model weights.

Usage
-----
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
from models.hpv_risk_scorer import HPVRiskScorer
from models.endo_scorer import EndoSymptomScorer

DEMO_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "demo")

# ── Synthetic patient profiles ────────────────────────────────────────────

HPV_PROFILES = {
    "high": {
        "age": 28, "num_sexual_partners": 8, "age_first_intercourse": 15,
        "smoking": 1, "oral_contraceptives_yrs": 3, "iud_yrs": 0,
        "stds_count": 3, "prev_cervical_biopsies": 1, "immunocompromised": 0,
        "hiv_positive": 0, "family_hx_cervical_cancer": 1, "last_pap_years_ago": 5,
    },
    "medium": {
        "age": 35, "num_sexual_partners": 3, "age_first_intercourse": 20,
        "smoking": 0, "oral_contraceptives_yrs": 5, "iud_yrs": 0,
        "stds_count": 1, "prev_cervical_biopsies": 0, "immunocompromised": 0,
        "hiv_positive": 0, "family_hx_cervical_cancer": 0, "last_pap_years_ago": 2,
    },
    "low": {
        "age": 25, "num_sexual_partners": 1, "age_first_intercourse": 22,
        "smoking": 0, "oral_contraceptives_yrs": 0, "iud_yrs": 0,
        "stds_count": 0, "prev_cervical_biopsies": 0, "immunocompromised": 0,
        "hiv_positive": 0, "family_hx_cervical_cancer": 0, "last_pap_years_ago": 1,
    },
}

ENDO_PROFILES = {
    "high": {
        "dysmenorrhea_score": 8, "dyspareunia_score": 7, "chronic_pelvic_pain": 7,
        "cycle_regularity": 1, "heavy_bleeding": 1, "infertility_hx": 1,
        "family_hx": 1, "fatigue_score": 7, "bloating_score": 6,
        "urinary_symptoms": 4, "symptom_duration_yrs": 5, "age": 33,
    },
    "moderate": {
        "dysmenorrhea_score": 5, "dyspareunia_score": 4, "chronic_pelvic_pain": 4,
        "cycle_regularity": 1, "heavy_bleeding": 1, "infertility_hx": 0,
        "family_hx": 0, "fatigue_score": 5, "bloating_score": 4,
        "urinary_symptoms": 2, "symptom_duration_yrs": 2, "age": 30,
    },
    "low": {
        "dysmenorrhea_score": 2, "dyspareunia_score": 1, "chronic_pelvic_pain": 1,
        "cycle_regularity": 0, "heavy_bleeding": 0, "infertility_hx": 0,
        "family_hx": 0, "fatigue_score": 2, "bloating_score": 1,
        "urinary_symptoms": 0, "symptom_duration_yrs": 0.5, "age": 24,
    },
}

CELL_COLOURS = {
    "Superficial-Intermediate": (180, 220, 180),
    "Parabasal":               (160, 200, 160),
    "Koilocyte":               (220, 180, 100),
    "Dyskeratocyte":           (220, 140,  80),
    "Metaplastic":             (220,  80,  80),
}


# ── Image helpers ─────────────────────────────────────────────────────────

def _cell_image(class_name: str, size: int = 224) -> np.ndarray:
    bg, cell = (20, 20, 30), CELL_COLOURS.get(class_name, (200, 200, 200))
    img = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(img)
    cx, cy, r = size // 2, size // 2, size // 3
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=cell, outline=(255, 255, 255), width=2)
    nr = r // 3
    d.ellipse([cx - nr, cy - nr, cx + nr, cy + nr], fill=(40, 40, 80))
    return np.array(img)


def _gradcam_overlay(class_name: str, size: int = 224) -> np.ndarray:
    base = _cell_image(class_name, size).astype(np.float32)
    heat = np.zeros((size, size, 3), dtype=np.float32)
    cx, cy, r = size // 2, size // 2, size // 3
    Y, X = np.ogrid[:size, :size]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    mask = dist < r
    intensity = np.clip(1 - dist / r, 0, 1) * 255
    heat[:, :, 0] = mask * intensity          # red channel
    heat[:, :, 2] = mask * (255 - intensity)  # blue channel
    overlay = (base * 0.6 + heat * 0.4).clip(0, 255).astype(np.uint8)
    return overlay


# ── Main ──────────────────────────────────────────────────────────────────

def main(model_dir: str) -> None:
    os.makedirs(DEMO_DIR, exist_ok=True)

    hpv_scorer  = HPVRiskScorer()   # heuristic mode
    endo_scorer = EndoSymptomScorer()

    manifest: dict = {"cervical": [], "hpv": [], "endo": []}

    # Cervical images
    for cls in CLASS_NAMES:
        safe = cls.replace(" ", "-")
        img_path = os.path.join(DEMO_DIR, f"demo_{safe}.png")
        cam_path = os.path.join(DEMO_DIR, f"demo_{safe}_gradcam.png")

        Image.fromarray(_cell_image(cls)).save(img_path)
        Image.fromarray(_gradcam_overlay(cls)).save(cam_path)

        meta  = CLASS_META[cls]
        probs = {n: round(0.02 if n != cls else 0.90, 2) for n in CLASS_NAMES}
        predict_result = {
            "class_name":   cls,
            "category":     meta["category"],
            "confidence":   0.90,
            "all_probs":    probs,
            "cin_grade":    meta["cin_grade"],
            "triage_color": meta["triage_color"],
            "action":       meta["action"],
            "urgency":      meta["urgency"],
            "description":  meta["description"],
            "color":        meta["color"],
        }
        manifest["cervical"].append({
            "class_name":    cls,
            "image_path":    f"data/demo/demo_{safe}.png",
            "gradcam_path":  f"data/demo/demo_{safe}_gradcam.png",
            "predict_result": predict_result,
            "explain_result": {"attention_focus_pct": 16.0},
        })
        print(f"[cervical] {cls}")

    # HPV JSONs
    for level, profile in HPV_PROFILES.items():
        result = hpv_scorer.predict(profile)
        path   = os.path.join(DEMO_DIR, f"demo_hpv_{level}.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        manifest["hpv"].append({
            "risk_level":    result["risk_level"],
            "result_path":   f"data/demo/demo_hpv_{level}.json",
            "predict_result": result,
        })
        print(f"[hpv] {level} → score={result['hpv_risk_score']}")

    # Endo JSONs
    for level, profile in ENDO_PROFILES.items():
        result = endo_scorer.predict(profile)
        path   = os.path.join(DEMO_DIR, f"demo_endo_{level}.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        manifest["endo"].append({
            "risk_level":    result["risk_level"],
            "result_path":   f"data/demo/demo_endo_{level}.json",
            "predict_result": result,
        })
        print(f"[endo] {level} → score={result['risk_score']}")

    # Manifest
    manifest_path = os.path.join(DEMO_DIR, "demo_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest → {manifest_path}")
    print("Demo assets ready.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate FemScan-AI demo assets")
    ap.add_argument("--model-dir", default="trained_models/")
    main(ap.parse_args().model_dir)
