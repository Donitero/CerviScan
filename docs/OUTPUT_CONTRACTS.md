# FemScan AI — Model Output Contracts
# Aligned to Pitch Deck v3: Slide 7 (3 modules) + Slide 8 (3 steps)

> **This is the single source of truth for all model output shapes.**
> Person B (dashboard) and Person C (components) code against these contracts.
> Osborn (models) guarantees these exact keys and types — no peeking at internals needed.

---

## Step 1: Screen & Score (Modules 1 + 3)

### Contract 1A — HPV Risk Score (Module 1)

**Method:** `HPVRiskScorer.predict(patient_data)`

**Input:**
```python
patient_data: dict  # keys from models/hpv_risk_scorer.py FEATURE_SCHEMA
                    # missing keys default to 0.0
```

**Output:**
```python
{
    "hpv_risk_score": float,          # 0-100 scale

    "risk_level": str,                # "Low" / "Medium" / "High"
                                      # thresholds: Low <33, Medium 33-65, High ≥66

    "risk_color": str,                # "#4CAF50"  (Low)
                                      # "#FF9800"  (Medium)
                                      # "#f44336"  (High)

    "escalate_to_imaging": bool,      # True when risk_level == "High"
                                      # → UI proceeds to Step 2 (Module 2)

    "top_risk_factors": [             # Top 5 SHAP-ranked factors (descending |impact|)
        {
            "factor":    str,         # Human-readable feature name
            "impact":    float,       # |SHAP value| — magnitude of contribution
            "direction": str,         # "increases_risk" | "decreases_risk"
        },
        # … up to 5 items
    ],

    "recommendation": str,            # Plain-language clinical recommendation

    "hpv_strain_risk": str,           # "Low-risk strains likely"
                                      # "Intermediate — mixed strain risk"
                                      # "High-risk strains (16/18) likely"
}
```

**Example:**
```python
{
    "hpv_risk_score":     78.4,
    "risk_level":         "High",
    "risk_color":         "#f44336",
    "escalate_to_imaging": True,
    "top_risk_factors": [
        {"factor": "Number of sexual partners",  "impact": 0.21, "direction": "increases_risk"},
        {"factor": "Number of prior STDs",       "impact": 0.18, "direction": "increases_risk"},
        {"factor": "Smoking (0=No, 1=Yes)",      "impact": 0.10, "direction": "increases_risk"},
        {"factor": "HIV positive (0/1)",          "impact": 0.09, "direction": "increases_risk"},
        {"factor": "Years since last Pap smear", "impact": 0.08, "direction": "increases_risk"},
    ],
    "recommendation":  "High HPV risk profile. Immediate referral for colposcopy...",
    "hpv_strain_risk": "High-risk strains (16/18) likely",
}
```

---

### Contract 1B — Endometriosis Risk Score (Module 3)

**Method:** `EndoSymptomScorer.predict(symptoms)`

**Input:**
```python
symptoms: dict  # keys from models/endo_scorer.py FEATURE_SCHEMA
                # missing keys default to 0.0
```

**Output:**
```python
{
    "risk_score": float,              # 0-100

    "risk_level": str,                # "Low" / "Moderate" / "High" / "Very High"
                                      # thresholds: Low <25, Moderate 25-49,
                                      #             High 50-74, Very High ≥75

    "risk_color": str,                # "#4CAF50"  (Low)
                                      # "#FF9800"  (Moderate)
                                      # "#FF5722"  (High)
                                      # "#F44336"  (Very High)

    "recommendation": str,            # Plain-language clinical recommendation

    "imaging_needed": bool,           # True when risk_score ≥ 50

    "top_factors": [                  # Top 5 SHAP-ranked symptom drivers
        {
            "feature_label": str,     # Human-readable symptom name
            "impact":        float,   # |SHAP value|
            "direction":     str,     # "increases_risk" | "decreases_risk"
        },
        # … up to 5 items
    ],
}
```

**Example:**
```python
{
    "risk_score":    72.3,
    "risk_level":    "High",
    "risk_color":    "#FF5722",
    "recommendation": "High risk. Specialist referral for transvaginal ultrasound...",
    "imaging_needed": True,
    "top_factors": [
        {"feature_label": "Dysmenorrhea (pain) severity (0-10)", "impact": 0.18, "direction": "increases_risk"},
        {"feature_label": "Chronic pelvic pain score (0-10)",    "impact": 0.14, "direction": "increases_risk"},
        {"feature_label": "Family history of endometriosis",     "impact": 0.10, "direction": "increases_risk"},
        {"feature_label": "Duration of symptoms (years)",        "impact": 0.08, "direction": "increases_risk"},
        {"feature_label": "Patient age (years)",                  "impact": 0.04, "direction": "decreases_risk"},
    ],
}
```

---

## Step 2: Capture & Analyse (Module 2)

### Contract 2A — CIN Lesion Detection

**Method:** `CervicalClassifier.predict(image_tensor)`

**Input:**
```python
image_tensor: torch.Tensor  # shape (1, 3, 224, 224), float32, ImageNet-normalised
```

**Output:**
```python
{
    "class_name": str,                # One of 5 SIPaKMeD classes:
                                      #   "Superficial-Intermediate" | "Parabasal"
                                      #   "Koilocyte" | "Dyskeratocyte" | "Metaplastic"

    "category": str,                  # "Normal" | "Benign" | "Abnormal"

    "confidence": float,              # Predicted class probability 0-1

    "all_probs": dict,                # {class_name: float} for all 5 classes (sums to 1)

    "cin_grade": str,                 # "No CIN"
                                      # "CIN1 (low-grade)"
                                      # "CIN2-3 (high-grade)"

    "triage_color": str,              # "green"   → No CIN
                                      # "amber"   → CIN1
                                      # "red"     → CIN2-3
                                      # (matches pitch deck Slide 8 triage traffic-light)

    "action": str,                    # Short clinical recommendation

    "urgency": str,                   # "low" | "moderate" | "high"

    "description": str,               # 1-2 sentence clinical description

    "color": str,                     # Hex colour for UI badge, e.g. "#4CAF50"
}
```

**Example:**
```python
{
    "class_name":   "Koilocyte",
    "category":     "Benign",
    "confidence":   0.87,
    "all_probs": {
        "Superficial-Intermediate": 0.04,
        "Parabasal":               0.02,
        "Koilocyte":               0.87,
        "Dyskeratocyte":           0.05,
        "Metaplastic":             0.02,
    },
    "cin_grade":    "CIN1 (low-grade)",
    "triage_color": "amber",
    "action":       "HPV cytopathic effect noted. Repeat smear in 6–12 months.",
    "urgency":      "moderate",
    "description":  "Koilocytes show perinuclear halos consistent with active HPV infection.",
    "color":        "#FF9800",
}
```

---

### Contract 2B — Grad-CAM Overlay

**Method:** `GradCAMExplainer.explain(image_tensor, original_np)`

**Inputs:**
```python
image_tensor: torch.Tensor   # (1, 3, 224, 224) float32 — same tensor as predict()
original_np:  np.ndarray     # (224, 224, 3) uint8 RGB — raw image resized to 224×224
```

**Output:**
```python
{
    "heatmap": np.ndarray,            # shape (224, 224), float32, values 0-1
                                      # 0 = no attention, 1 = max attention

    "overlay": np.ndarray,            # shape (224, 224, 3), uint8, RGB
                                      # Grad-CAM jet overlay blended with original
                                      # → ready for st.image(result["overlay"])

    "attention_focus_pct": float,     # % of pixels with heatmap > 0.5
                                      # e.g. 12.4 means model focused on 12.4% of image
}
```

**UI usage:**
```python
st.image(result["overlay"], caption="Model Attention Map", use_column_width=True)
st.metric("Attention Focus", f"{result['attention_focus_pct']:.1f}%")
```

---

## Step 3: Act & Record

This step is **UI-only** (Person B builds it in `app.py`).
No model output is needed — the UI synthesises an action plan from the Step 1 + Step 2 outputs.

Suggested UI fields to generate from prior outputs:
- `escalate_to_imaging` (Contract 1A) → show/hide Step 2 module
- `cin_grade` + `triage_color` (Contract 2A) → traffic-light status card
- `imaging_needed` (Contract 1B) → recommend ultrasound banner
- Combined urgency logic → "Next Steps" checklist for clinician

---

## Demo Assets (`data/demo/`)

Generated by `scripts/prepare_demo.py` — no trained weights required.

```
data/demo/
├── demo_manifest.json
│
│   # Cervical (5 classes × 2 images each)
├── demo_Superficial-Intermediate.png
├── demo_Superficial-Intermediate_gradcam.png
├── demo_Parabasal.png
├── demo_Parabasal_gradcam.png
├── demo_Koilocyte.png
├── demo_Koilocyte_gradcam.png
├── demo_Dyskeratocyte.png
├── demo_Dyskeratocyte_gradcam.png
├── demo_Metaplastic.png
├── demo_Metaplastic_gradcam.png
│
│   # HPV risk (pre-computed Contract 1A dicts)
├── demo_hpv_high.json
├── demo_hpv_medium.json
├── demo_hpv_low.json
│
│   # Endo risk (pre-computed Contract 1B dicts)
├── demo_endo_high.json
├── demo_endo_moderate.json
└── demo_endo_low.json
```

**`demo_manifest.json` schema:**
```json
{
  "cervical": [
    {
      "class_name":    "Koilocyte",
      "image_path":    "data/demo/demo_Koilocyte.png",
      "gradcam_path":  "data/demo/demo_Koilocyte_gradcam.png",
      "predict_result": { /* Contract 2A dict */ },
      "explain_result": { "attention_focus_pct": 14.7 }
    }
  ],
  "hpv": [
    {
      "risk_level":     "High",
      "result_path":    "data/demo/demo_hpv_high.json",
      "predict_result": { /* Contract 1A dict */ }
    }
  ],
  "endo": [
    {
      "risk_level":     "High",
      "result_path":    "data/demo/demo_endo_high.json",
      "predict_result": { /* Contract 1B dict */ }
    }
  ]
}
```

---

## Contract Version Table

| Contract | Covers          | Version | Last updated |
|----------|-----------------|---------|--------------|
| 1A       | HPV Risk Score  | 1.0     | 2026-03-27   |
| 1B       | Endo Risk Score | 1.0     | 2026-03-27   |
| 2A       | CIN Detection   | 1.0     | 2026-03-27   |
| 2B       | Grad-CAM        | 1.0     | 2026-03-27   |

> **Breaking changes** to any output shape must increment the version and be
> communicated to Person B and Person C **before** merging to `main`.
