"""
MODULE 3 — Cancer Risk Scorer
======================================================
Heuristic/XGBoost-ready risk predictor for cancer likelihood
based on patient questionnaire and lifestyle factors.

Output contract → docs/OUTPUT_CONTRACTS.md § Contract 1C
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

FEATURE_SCHEMA: Dict[str, Dict[str, Any]] = {
    "age":                 {"label": "Patient age (years)",                "range": (20, 80)},
    "family_history":      {"label": "Family history of cancer (0=No,1=Yes)", "range": (0, 1)},
    "previous_cancer":     {"label": "Previous cancer diagnosis (0=No,1=Yes)", "range": (0, 1)},
    "radiation_exposure":  {"label": "History of radiation exposure (0=No,1=Yes)", "range": (0, 1)},
    "chronic_inflammation":{"label": "Chronic inflammation (0=No,1=Yes)", "range": (0, 1)},
    "lifestyle_score":     {"label": "Lifestyle risk (0–10)",              "range": (0, 10)},
    "diet_score":          {"label": "Diet quality (0–10)",                "range": (0, 10)},
    "exercise_score":      {"label": "Exercise frequency (0–10)",          "range": (0, 10)},
}

FEATURE_COLUMNS = list(FEATURE_SCHEMA.keys())

RISK_BANDS = [
    (80.0, "Critical", "#dc2626"),
    (50.0, "High",     "#ea580c"),
    (25.0, "Moderate", "#f59e0b"),
    (0.0,  "Low",      "#16a34a"),
]

def _classify(score: float):
    for threshold, level, color in RISK_BANDS:
        if score >= threshold:
            return level, color
    return "Low", "#16a34a"

# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class CancerRiskScorer:
    """
    Usage
    -----
    scorer = CancerRiskScorer()
    result = scorer.predict(cancer_dict)   # → CONTRACT 1C
    """

    def __init__(self, model_path: str | None = None):
        self.model = None
        if model_path:
            import joblib
            self.model = joblib.load(model_path)

    def predict(self, cancer_data: Dict[str, float]) -> Dict[str, Any]:
        row = {col: float(cancer_data.get(col, 0.0)) for col in FEATURE_COLUMNS}
        X   = pd.DataFrame([row], columns=FEATURE_COLUMNS)

        if self.model is not None:
            prob       = float(self.model.predict_proba(X)[0, 1])
            risk_score = round(prob * 100, 1)
            top_factors = [{"feature_label": f, "impact": 0.0, "direction": "increases_risk"} for f in FEATURE_COLUMNS]
        else:
            risk_score, top_factors = _heuristic_cancer(row)

        risk_level, risk_color = _classify(risk_score)

        return {
            "risk_score":    risk_score,
            "risk_level":    risk_level,
            "risk_color":    risk_color,
            "recommendation": _cancer_recommendation(risk_level),
            "imaging_needed": risk_score >= 80,
            "top_factors":   top_factors,
        }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _heuristic_cancer(row: Dict[str, float]) -> tuple[float, List[Dict[str, Any]]]:
    weights = {
        "age":                 0.25,
        "family_history":      0.15,
        "previous_cancer":     0.20,
        "radiation_exposure":  0.10,
        "chronic_inflammation":0.10,
        "lifestyle_score":     0.10,
        "diet_score":          -0.05,  # good diet lowers risk
        "exercise_score":      -0.05,  # exercise lowers risk
    }
    score = 0.0
    factors = []
    for col, w in weights.items():
        lo, hi = FEATURE_SCHEMA[col]["range"]
        norm = ((row[col] - lo) / (hi - lo)) if hi > lo else row[col]
        norm = max(0.0, min(1.0, norm))
        contribution = w * norm * 100
        score += contribution
        factors.append({
            "feature_label": FEATURE_SCHEMA[col]["label"],
            "impact":        round(abs(contribution), 4),
            "direction":     "increases_risk" if contribution > 0 else "decreases_risk",
        })
    factors.sort(key=lambda x: x["impact"], reverse=True)
    return round(min(max(score, 0), 100), 1), factors[:5]

def _cancer_recommendation(risk_level: str) -> str:
    return {
        "Low":       "Low cancer risk. Continue routine screening.",
        "Moderate":  "Moderate risk. Clinical consultation and imaging recommended within 6 months.",
        "High":      "High risk. Specialist referral and biopsy advised.",
        "Critical":  "Critical risk. 🚨 Immediate oncological referral required.",
    }.get(risk_level, "Consult a healthcare professional.")
