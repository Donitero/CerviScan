# FemScan-AI — Output Contracts

> **This document is the single source of truth for all model output shapes.**
> Person A (model builder) guarantees these exact formats.
> Person B (UI builder) can code against them without seeing model internals.

---

## CONTRACT 1 — Cervical Classification

**Method:** `CervicalClassifier.predict(image_tensor)`

**Input:**
```python
image_tensor: torch.Tensor  # shape (1, 3, 224, 224), float32, ImageNet-normalised
```

**Output:** `dict`

```python
{
    "class_name": str,          # one of the 5 SIPaKMeD classes:
                                #   "Superficial-Intermediate" | "Parabasal"
                                #   "Koilocyte" | "Dyskeratocyte" | "Metaplastic"

    "category":  str,           # coarse clinical category:
                                #   "Normal" | "Benign" | "Abnormal"

    "confidence": float,        # predicted class probability in [0.0, 1.0]

    "all_probs": {              # softmax probability for every class
        "Superficial-Intermediate": float,
        "Parabasal":               float,
        "Koilocyte":               float,
        "Dyskeratocyte":           float,
        "Metaplastic":             float,
    },

    "action":   str,            # short clinical recommendation sentence
    "urgency":  str,            # "low" | "moderate" | "high" | "critical"
    "description": str,         # 1–2 sentence clinical description of the cell type
    "color":    str,            # hex colour for UI badge/chip, e.g. "#4CAF50"
}
```

**Example:**
```python
{
    "class_name":  "Koilocyte",
    "category":    "Benign",
    "confidence":  0.87,
    "all_probs":   {
        "Superficial-Intermediate": 0.04,
        "Parabasal":               0.02,
        "Koilocyte":               0.87,
        "Dyskeratocyte":           0.05,
        "Metaplastic":             0.02,
    },
    "action":      "HPV-associated changes noted. Recommend repeat smear in 6–12 months.",
    "urgency":     "moderate",
    "description": "Koilocytes exhibit perinuclear halos consistent with HPV cytopathic effect...",
    "color":       "#FF9800",
}
```

---

## CONTRACT 2 — Grad-CAM Explainability

**Method:** `GradCAMExplainer.explain(image_tensor, original_np)`

**Inputs:**
```python
image_tensor: torch.Tensor   # shape (1, 3, 224, 224), float32 — same tensor as predict()
original_np:  np.ndarray     # shape (224, 224, 3), uint8, RGB — raw image resized to 224×224
```

**Output:** `dict`

```python
{
    "heatmap": np.ndarray,          # shape (224, 224), float32, values in [0.0, 1.0]
                                    # 0 = no attention, 1 = max attention

    "overlay": np.ndarray,          # shape (224, 224, 3), uint8, RGB
                                    # Grad-CAM jet overlay blended with original image
                                    # ready for: st.image(result["overlay"])

    "attention_focus_pct": float,   # % of image pixels with heatmap > 0.5
                                    # e.g. 12.4 means model focused on 12.4% of pixels
}
```

**Example:**
```python
{
    "heatmap":             np.array(...),   # (224, 224) float32
    "overlay":             np.array(...),   # (224, 224, 3) uint8 RGB
    "attention_focus_pct": 14.7,
}
```

**UI usage:**
```python
st.image(result["overlay"], caption="Model Attention Map", use_column_width=True)
st.metric("Attention Focus", f"{result['attention_focus_pct']:.1f}%")
```

---

## CONTRACT 3 — Endometriosis Risk Score

**Method:** `EndoSymptomScorer.predict(symptoms_dict)`

**Input:**
```python
symptoms_dict: dict  # keys from FEATURE_SCHEMA in endo_scorer.py
                     # all values are float; missing keys default to 0.0
# Example:
{
    "dysmenorrhea_score":   8.0,
    "dyspareunia_score":    6.0,
    "chronic_pelvic_pain":  7.0,
    "cycle_regularity":     1.0,
    "heavy_bleeding":       1.0,
    "infertility_hx":       0.0,
    "family_hx":            1.0,
    "fatigue_score":        5.0,
    "bloating_score":       4.0,
    "urinary_symptoms":     3.0,
    "symptom_duration_yrs": 4.0,
    "age":                  28.0,
}
```

**Output:** `dict`

```python
{
    "risk_score":   float,      # 0.0 – 100.0  (higher = greater risk)

    "risk_level":   str,        # "Low" | "Moderate" | "High" | "Very High"
                                # thresholds: Low <25, Moderate 25–49, High 50–74, Very High ≥75

    "risk_color":   str,        # hex colour matching risk_level:
                                #   "Low"       → "#4CAF50"
                                #   "Moderate"  → "#FF9800"
                                #   "High"      → "#FF5722"
                                #   "Very High" → "#F44336"

    "recommendation": str,      # 1–3 sentence plain-language clinical recommendation

    "imaging_needed": bool,     # True when risk_score ≥ 50

    "top_factors": [            # top 5 features driving this prediction (descending |impact|)
        {
            "feature_label": str,   # human-readable feature name
            "impact":        float, # |SHAP value| — magnitude of contribution
            "direction":     str,   # "increases_risk" | "decreases_risk"
        },
        # … up to 5 items
    ],
}
```

**Example:**
```python
{
    "risk_score":  72.3,
    "risk_level":  "High",
    "risk_color":  "#FF5722",
    "recommendation": "High-risk symptom pattern detected. Referral to a specialist...",
    "imaging_needed": True,
    "top_factors": [
        {"feature_label": "Dysmenorrhea (pain) severity",  "impact": 0.182, "direction": "increases_risk"},
        {"feature_label": "Chronic pelvic pain score",     "impact": 0.141, "direction": "increases_risk"},
        {"feature_label": "Family history of endometriosis","impact": 0.098, "direction": "increases_risk"},
        {"feature_label": "Duration of symptoms (years)",  "impact": 0.076, "direction": "increases_risk"},
        {"feature_label": "Patient age (years)",            "impact": 0.041, "direction": "decreases_risk"},
    ],
}
```

---

## CONTRACT 4 — Demo Assets

**Location:** `data/demo/`

The `scripts/prepare_demo.py` script pre-computes all demo assets so the Streamlit app
can showcase results without requiring trained weights or user uploads.

**Files produced:**

```
data/demo/
├── demo_manifest.json                     # index of all demo assets (see schema below)
├── demo_Superficial-Intermediate.png      # original cell image (224×224 RGB)
├── demo_Parabasal.png
├── demo_Koilocyte.png
├── demo_Dyskeratocyte.png
├── demo_Metaplastic.png
├── demo_Superficial-Intermediate_gradcam.png   # Grad-CAM overlay (224×224 RGB)
├── demo_Parabasal_gradcam.png
├── demo_Koilocyte_gradcam.png
├── demo_Dyskeratocyte_gradcam.png
├── demo_Metaplastic_gradcam.png
├── demo_endo_low.json                     # pre-computed EndoSymptomScorer output
├── demo_endo_moderate.json
├── demo_endo_high.json
└── demo_endo_very_high.json
```

**`demo_manifest.json` schema:**
```json
{
  "cervical": [
    {
      "class_name":    "Koilocyte",
      "image_path":    "data/demo/demo_Koilocyte.png",
      "gradcam_path":  "data/demo/demo_Koilocyte_gradcam.png",
      "predict_result": { /* CONTRACT 1 dict */ },
      "explain_result": {
        "attention_focus_pct": 14.7
        /* heatmap/overlay arrays are not stored in JSON — load from *_gradcam.png */
      }
    }
    // … one entry per class
  ],
  "endo": [
    {
      "risk_level":    "High",
      "result_path":   "data/demo/demo_endo_high.json",
      "predict_result": { /* CONTRACT 3 dict */ }
    }
    // … one entry per risk level
  ]
}
```

**`demo_endo_<risk>.json` schema:** matches CONTRACT 3 output exactly.

---

## Versioning

| Contract | Version | Last updated |
|----------|---------|--------------|
| 1 — Cervical Classification | 1.0 | 2026-03-27 |
| 2 — Grad-CAM               | 1.0 | 2026-03-27 |
| 3 — Endo Risk Score        | 1.0 | 2026-03-27 |
| 4 — Demo Assets            | 1.0 | 2026-03-27 |

Any breaking change to an output shape **must** increment the version and be communicated
to Person B before merging to `main`.
