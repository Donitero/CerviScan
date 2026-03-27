"""
Endometriosis Symptom Risk Scorer — XGBoost + SHAP explainability.

Output contract (see docs/OUTPUT_CONTRACTS.md — CONTRACT 3):
    EndoSymptomScorer.predict(symptoms_dict) -> dict
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
import shap
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Feature schema — MUST match training column order
# ---------------------------------------------------------------------------

FEATURE_SCHEMA: Dict[str, Dict[str, Any]] = {
    "dysmenorrhea_score":    {"label": "Dysmenorrhea (pain) severity",        "range": (0, 10)},
    "dyspareunia_score":     {"label": "Dyspareunia (pain during sex)",        "range": (0, 10)},
    "chronic_pelvic_pain":   {"label": "Chronic pelvic pain score",            "range": (0, 10)},
    "cycle_regularity":      {"label": "Cycle regularity (0=regular, 1=irreg)","range": (0, 1)},
    "heavy_bleeding":        {"label": "Heavy menstrual bleeding (0/1)",        "range": (0, 1)},
    "infertility_hx":        {"label": "History of infertility (0/1)",          "range": (0, 1)},
    "family_hx":             {"label": "Family history of endometriosis (0/1)", "range": (0, 1)},
    "fatigue_score":         {"label": "Fatigue severity score",               "range": (0, 10)},
    "bloating_score":        {"label": "Bloating / GI symptoms score",         "range": (0, 10)},
    "urinary_symptoms":      {"label": "Urinary symptom severity",             "range": (0, 10)},
    "symptom_duration_yrs":  {"label": "Duration of symptoms (years)",         "range": (0, 30)},
    "age":                   {"label": "Patient age (years)",                   "range": (15, 55)},
}

FEATURE_COLUMNS = list(FEATURE_SCHEMA.keys())

# Risk level thresholds (inclusive lower bound)
RISK_LEVELS = [
    (75, "Very High", "#F44336"),
    (50, "High",      "#FF5722"),
    (25, "Moderate",  "#FF9800"),
    (0,  "Low",       "#4CAF50"),
]


def _score_to_level(score: float):
    for threshold, label, color in RISK_LEVELS:
        if score >= threshold:
            return label, color
    return "Low", "#4CAF50"


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class EndoSymptomScorer:
    """
    XGBoost-based endometriosis risk scorer with SHAP feature importance.

    Usage:
        scorer = EndoSymptomScorer(model_path="trained_models/endo_xgb.pkl")
        result = scorer.predict(symptoms_dict)
    """

    def __init__(self, model_path: str | None = None):
        self.model = None
        self.explainer = None

        if model_path:
            self.model = joblib.load(model_path)
            self.explainer = shap.TreeExplainer(self.model)

    # ------------------------------------------------------------------
    # Inference API — returns OUTPUT CONTRACT 3
    # ------------------------------------------------------------------

    def predict(self, symptoms_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Args:
            symptoms_dict: keys matching FEATURE_COLUMNS, values as floats.
                           Missing keys default to 0.

        Returns:
            dict matching CONTRACT 3 in docs/OUTPUT_CONTRACTS.md.
        """
        # Build feature row in correct column order
        row = {col: float(symptoms_dict.get(col, 0.0)) for col in FEATURE_COLUMNS}
        X = pd.DataFrame([row], columns=FEATURE_COLUMNS)

        if self.model is not None:
            # Model returns probability of positive class (endometriosis)
            prob = float(self.model.predict_proba(X)[0, 1])
            risk_score = round(prob * 100, 1)

            # SHAP values for this sample
            shap_vals = self.explainer.shap_values(X)
            # For binary XGBoost TreeExplainer, shap_vals may be list[2] or array
            if isinstance(shap_vals, list):
                sv = shap_vals[1][0]  # positive class
            else:
                sv = shap_vals[0]

            top_factors = _build_top_factors(sv, row, n=5)
        else:
            # No trained model — return heuristic score for demo/dev
            risk_score, top_factors = _heuristic_score(row)

        risk_level, risk_color = _score_to_level(risk_score)
        imaging_needed = risk_score >= 50

        recommendation = _recommendation_text(risk_level)

        return {
            "risk_score":      risk_score,
            "risk_level":      risk_level,
            "risk_color":      risk_color,
            "recommendation":  recommendation,
            "imaging_needed":  imaging_needed,
            "top_factors":     top_factors,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_top_factors(
    shap_values: np.ndarray,
    row: Dict[str, float],
    n: int = 5,
) -> List[Dict[str, Any]]:
    indices = np.argsort(np.abs(shap_values))[::-1][:n]
    factors = []
    for i in indices:
        col = FEATURE_COLUMNS[i]
        sv = float(shap_values[i])
        factors.append({
            "feature_label": FEATURE_SCHEMA[col]["label"],
            "impact":        round(abs(sv), 4),
            "direction":     "increases_risk" if sv > 0 else "decreases_risk",
        })
    return factors


def _heuristic_score(row: Dict[str, float]):
    """Simple weighted heuristic used when no trained model is available."""
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
    # Normalise numeric features to [0,1]
    schema = FEATURE_SCHEMA
    score = 0.0
    factor_list = []
    for col, w in weights.items():
        lo, hi = schema[col]["range"]
        norm_val = (row[col] - lo) / (hi - lo) if hi > lo else row[col]
        contribution = w * norm_val * 100
        score += contribution
        factor_list.append({
            "feature_label": schema[col]["label"],
            "impact":        round(abs(contribution), 4),
            "direction":     "increases_risk" if norm_val > 0.5 else "decreases_risk",
        })

    factor_list.sort(key=lambda x: x["impact"], reverse=True)
    return round(min(score, 100), 1), factor_list[:5]


def _recommendation_text(risk_level: str) -> str:
    recs = {
        "Low": (
            "Symptom profile suggests low endometriosis risk. Continue routine gynaecological "
            "care and monitor for symptom changes."
        ),
        "Moderate": (
            "Moderate risk indicators present. A detailed clinical consultation and pelvic "
            "ultrasound are recommended to rule out endometriosis."
        ),
        "High": (
            "High-risk symptom pattern detected. Referral to a specialist gynaecologist for "
            "transvaginal ultrasound and possible laparoscopy is strongly advised."
        ),
        "Very High": (
            "Very high risk. Urgent specialist referral required. Imaging and minimally invasive "
            "diagnostic evaluation (laparoscopy) should be arranged promptly."
        ),
    }
    return recs.get(risk_level, "Please consult a healthcare professional.")
