"""
MODULE 1 — HPV Risk Scorer (Osborn)
====================================
XGBoost + SHAP risk predictor for HPV acquisition / high-risk strain likelihood.

Output contract → docs/OUTPUT_CONTRACTS.md § Contract 1A
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
import shap
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Feature schema  (MUST match training column order)
# ---------------------------------------------------------------------------

FEATURE_SCHEMA: Dict[str, Dict[str, Any]] = {
    "age":                       {"label": "Age (years)",                          "range": (15, 65)},
    "num_sexual_partners":       {"label": "Number of sexual partners",             "range": (0, 20)},
    "age_first_intercourse":     {"label": "Age at first sexual intercourse",       "range": (10, 40)},
    "smoking":                   {"label": "Smoking (0=No, 1=Yes)",                 "range": (0, 1)},
    "oral_contraceptives_yrs":   {"label": "Oral contraceptive use (years)",        "range": (0, 20)},
    "iud_yrs":                   {"label": "IUD use (years)",                       "range": (0, 15)},
    "stds_count":                {"label": "Number of prior STDs",                  "range": (0, 10)},
    "prev_cervical_biopsies":    {"label": "Previous cervical biopsies (0/1)",      "range": (0, 1)},
    "immunocompromised":         {"label": "Immunocompromised status (0/1)",        "range": (0, 1)},
    "hiv_positive":              {"label": "HIV positive (0/1)",                    "range": (0, 1)},
    "family_hx_cervical_cancer": {"label": "Family history of cervical cancer (0/1)","range": (0, 1)},
    "last_pap_years_ago":        {"label": "Years since last Pap smear",            "range": (0, 20)},
}

FEATURE_COLUMNS = list(FEATURE_SCHEMA.keys())

# Thresholds
RISK_BANDS = [
    (66.0, "High",   "#f44336"),
    (33.0, "Medium", "#FF9800"),
    (0.0,  "Low",    "#4CAF50"),
]


def _classify(score: float):
    for threshold, level, color in RISK_BANDS:
        if score >= threshold:
            return level, color
    return "Low", "#4CAF50"


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class HPVRiskScorer:
    """
    Predicts HPV acquisition risk and high-risk strain likelihood from
    demographic and behavioural patient data.

    Usage
    -----
    scorer = HPVRiskScorer("trained_models/hpv_xgb.pkl")
    result = scorer.predict(patient_data_dict)   # → CONTRACT 1A
    """

    def __init__(self, model_path: str | None = None):
        self.model = None
        self.explainer = None
        if model_path:
            self.model = joblib.load(model_path)
            self.explainer = shap.TreeExplainer(self.model)

    # ------------------------------------------------------------------
    # Public API — CONTRACT 1A
    # ------------------------------------------------------------------

    def predict(self, patient_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Args
        ----
        patient_data : dict
            Keys from FEATURE_SCHEMA; missing keys default to 0.

        Returns (CONTRACT 1A)
        ----------------------
        {
            "hpv_risk_score":     float,      # 0-100
            "risk_level":         str,         # "Low" / "Medium" / "High"
            "risk_color":         str,         # hex
            "escalate_to_imaging": bool,       # True when risk_level == "High"
            "top_risk_factors":   list[dict],  # top 5 SHAP factors
            "recommendation":     str,
            "hpv_strain_risk":    str,
        }
        """
        row = {col: float(patient_data.get(col, 0.0)) for col in FEATURE_COLUMNS}
        X = pd.DataFrame([row], columns=FEATURE_COLUMNS)

        if self.model is not None:
            prob = float(self.model.predict_proba(X)[0, 1])
            risk_score = round(prob * 100, 1)

            shap_vals = self.explainer.shap_values(X)
            sv = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
            top_factors = _top_factors(sv, FEATURE_COLUMNS, FEATURE_SCHEMA, n=5)
        else:
            risk_score, top_factors = _heuristic_hpv(row)

        risk_level, risk_color = _classify(risk_score)

        return {
            "hpv_risk_score":      risk_score,
            "risk_level":          risk_level,
            "risk_color":          risk_color,
            "escalate_to_imaging": risk_level == "High",
            "top_risk_factors":    top_factors,
            "recommendation":      _hpv_recommendation(risk_level),
            "hpv_strain_risk":     _strain_risk(risk_score),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _top_factors(
    shap_values: np.ndarray,
    columns: List[str],
    schema: Dict[str, Any],
    n: int = 5,
) -> List[Dict[str, Any]]:
    idx = np.argsort(np.abs(shap_values))[::-1][:n]
    return [
        {
            "factor":    schema[columns[i]]["label"],
            "impact":    round(float(abs(shap_values[i])), 4),
            "direction": "increases_risk" if shap_values[i] > 0 else "decreases_risk",
        }
        for i in idx
    ]


def _heuristic_hpv(row: Dict[str, float]) -> tuple[float, List[Dict[str, Any]]]:
    weights = {
        "num_sexual_partners":     0.20,
        "stds_count":              0.18,
        "age_first_intercourse":   0.12,  # lower age → higher risk (inverted below)
        "smoking":                 0.10,
        "hiv_positive":            0.10,
        "immunocompromised":       0.08,
        "last_pap_years_ago":      0.08,
        "prev_cervical_biopsies":  0.07,
        "oral_contraceptives_yrs": 0.04,
        "family_hx_cervical_cancer": 0.03,
    }
    score = 0.0
    factors = []
    for col, w in weights.items():
        lo, hi = FEATURE_SCHEMA[col]["range"]
        val = row[col]
        if col == "age_first_intercourse":
            # younger first intercourse → higher risk
            norm = 1.0 - ((val - lo) / (hi - lo)) if hi > lo else 0.0
        else:
            norm = (val - lo) / (hi - lo) if hi > lo else val
        norm = max(0.0, min(1.0, norm))
        contribution = w * norm * 100
        score += contribution
        factors.append({
            "factor":    FEATURE_SCHEMA[col]["label"],
            "impact":    round(contribution, 4),
            "direction": "increases_risk" if norm > 0.4 else "decreases_risk",
        })
    factors.sort(key=lambda x: x["impact"], reverse=True)
    return round(min(score, 100), 1), factors[:5]


def _hpv_recommendation(risk_level: str) -> str:
    return {
        "Low":    "Low HPV risk. Continue routine annual screening and ensure HPV vaccination is up to date.",
        "Medium": "Moderate HPV risk factors identified. Schedule a Pap smear / HPV co-test within 6 months and discuss vaccination.",
        "High":   "High HPV risk profile. Immediate referral for colposcopy and HPV genotyping (types 16/18) is recommended.",
    }.get(risk_level, "Consult a healthcare professional.")


def _strain_risk(score: float) -> str:
    if score >= 66:
        return "High-risk strains (16/18) likely"
    if score >= 33:
        return "Intermediate — mixed strain risk"
    return "Low-risk strains likely"
