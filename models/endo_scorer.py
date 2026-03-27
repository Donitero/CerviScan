"""
MODULE 3 — Endometriosis Symptom Risk Scorer (Osborn)
======================================================
XGBoost + SHAP risk predictor for endometriosis from patient symptom questionnaire.

Output contract → docs/OUTPUT_CONTRACTS.md § Contract 1B
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
import shap
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

FEATURE_SCHEMA: Dict[str, Dict[str, Any]] = {
    "dysmenorrhea_score":    {"label": "Dysmenorrhea (pain) severity (0-10)",        "range": (0, 10)},
    "dyspareunia_score":     {"label": "Dyspareunia — pain during sex (0-10)",        "range": (0, 10)},
    "chronic_pelvic_pain":   {"label": "Chronic pelvic pain score (0-10)",            "range": (0, 10)},
    "cycle_regularity":      {"label": "Irregular cycles (0=regular, 1=irregular)",   "range": (0, 1)},
    "heavy_bleeding":        {"label": "Heavy menstrual bleeding (0=No, 1=Yes)",       "range": (0, 1)},
    "infertility_hx":        {"label": "History of infertility (0=No, 1=Yes)",         "range": (0, 1)},
    "family_hx":             {"label": "Family history of endometriosis (0=No, 1=Yes)","range": (0, 1)},
    "fatigue_score":         {"label": "Fatigue severity (0-10)",                     "range": (0, 10)},
    "bloating_score":        {"label": "Bloating / GI symptoms (0-10)",               "range": (0, 10)},
    "urinary_symptoms":      {"label": "Urinary symptom severity (0-10)",             "range": (0, 10)},
    "symptom_duration_yrs":  {"label": "Duration of symptoms (years)",               "range": (0, 30)},
    "age":                   {"label": "Patient age (years)",                          "range": (15, 55)},
}

FEATURE_COLUMNS = list(FEATURE_SCHEMA.keys())

RISK_BANDS = [
    (75.0, "Very High", "#F44336"),
    (50.0, "High",      "#FF5722"),
    (25.0, "Moderate",  "#FF9800"),
    (0.0,  "Low",       "#4CAF50"),
]


def _classify(score: float):
    for threshold, level, color in RISK_BANDS:
        if score >= threshold:
            return level, color
    return "Low", "#4CAF50"


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class EndoSymptomScorer:
    """
    Usage
    -----
    scorer = EndoSymptomScorer("trained_models/endo_xgb.pkl")
    result = scorer.predict(symptoms_dict)   # → CONTRACT 1B
    """

    def __init__(self, model_path: str | None = None):
        self.model = None
        self.explainer = None
        if model_path:
            self.model = joblib.load(model_path)
            self.explainer = shap.TreeExplainer(self.model)

    # ------------------------------------------------------------------
    # Inference API — CONTRACT 1B
    # ------------------------------------------------------------------

    def predict(self, symptoms: Dict[str, float]) -> Dict[str, Any]:
        """
        Args
        ----
        symptoms : dict  (keys from FEATURE_SCHEMA; missing keys → 0.0)

        Returns (CONTRACT 1B)
        ----------------------
        {
            "risk_score":    float,      # 0-100
            "risk_level":    str,        # "Low" / "Moderate" / "High" / "Very High"
            "risk_color":    str,        # hex
            "recommendation": str,
            "imaging_needed": bool,      # True when risk_score >= 50
            "top_factors":   list[dict],
        }
        """
        row = {col: float(symptoms.get(col, 0.0)) for col in FEATURE_COLUMNS}
        X   = pd.DataFrame([row], columns=FEATURE_COLUMNS)

        if self.model is not None:
            prob       = float(self.model.predict_proba(X)[0, 1])
            risk_score = round(prob * 100, 1)
            shap_vals  = self.explainer.shap_values(X)
            sv = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
            top_factors = _top_factors(sv)
        else:
            risk_score, top_factors = _heuristic_endo(row)

        risk_level, risk_color = _classify(risk_score)

        return {
            "risk_score":    risk_score,
            "risk_level":    risk_level,
            "risk_color":    risk_color,
            "recommendation": _endo_recommendation(risk_level),
            "imaging_needed": risk_score >= 50,
            "top_factors":   top_factors,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _top_factors(shap_values: np.ndarray, n: int = 5) -> List[Dict[str, Any]]:
    idx = np.argsort(np.abs(shap_values))[::-1][:n]
    return [
        {
            "feature_label": FEATURE_SCHEMA[FEATURE_COLUMNS[i]]["label"],
            "impact":        round(float(abs(shap_values[i])), 4),
            "direction":     "increases_risk" if shap_values[i] > 0 else "decreases_risk",
        }
        for i in idx
    ]


def _heuristic_endo(row: Dict[str, float]) -> tuple[float, List[Dict[str, Any]]]:
    weights = {
        "dysmenorrhea_score":   0.20,
        "dyspareunia_score":    0.15,
        "chronic_pelvic_pain":  0.15,
        "infertility_hx":       0.12,
        "family_hx":            0.10,
        "heavy_bleeding":       0.08,
        "symptom_duration_yrs": 0.07,
        "fatigue_score":        0.06,
        "bloating_score":       0.04,
        "urinary_symptoms":     0.03,
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
            "impact":        round(contribution, 4),
            "direction":     "increases_risk" if norm > 0.4 else "decreases_risk",
        })
    factors.sort(key=lambda x: x["impact"], reverse=True)
    return round(min(score, 100), 1), factors[:5]


def _endo_recommendation(risk_level: str) -> str:
    return {
        "Low":      "Low endometriosis risk. Continue routine gynaecological care.",
        "Moderate": "Moderate risk. Clinical consultation and pelvic ultrasound recommended within 3 months.",
        "High":     "High risk. Specialist referral for transvaginal ultrasound and possible laparoscopy advised.",
        "Very High": "Very high risk. Urgent specialist referral required. Prompt diagnostic laparoscopy recommended.",
    }.get(risk_level, "Consult a healthcare professional.")
