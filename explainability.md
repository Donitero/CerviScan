# FemScan AI — Explainable AI in Cancer Screening

> An AI-powered cervical cancer screening platform that doesn't just predict risk —
> it **explains every decision** using the same ML techniques that power
> modern credit scoring, alternative lending, and financial inclusion platforms.

---

## 1 · Project Overview

FemScan AI combines **tabular risk scoring**, **medical image classification**,
and **explainability** into a single clinical decision-support pipeline.

### 1.1 · Architecture

```
┌──────────────────────────────────┬─────────────────────────────────────┐
│  Module 1 — HPV Risk Scorer      │  XGBoost  +  SHAP Explainability    │
│  Module 2 — CIN Lesion Detector  │  EfficientNetV2-S  +  Grad-CAM     │
│  Module 3 — Cancer Risk Scorer   │  XGBoost  +  SHAP Explainability    │
├──────────────────────────────────┴─────────────────────────────────────┤
│              ▼  All feed into  ▼                                      │
│         Triage Report  (traffic-light system: 🟢 🟡 🔴)              │
└───────────────────────────────────────────────────────────────────────┘
```

### 1.2 · Pipeline Flow

| Step | What Happens | Models Used |
|:----:|:-------------|:------------|
| **① Screen & Score** | Patient demographics + symptoms → risk prediction | Module 1 (HPV) + Module 3 (Cancer) |
| **② Capture & Analyse** | Pap smear image uploaded → cell classification | Module 2 (CIN classifier + Grad-CAM) |
| **③ Act & Record** | All findings → consolidated triage report | Rules engine (combines prior outputs) |

---

## 2 · Core ML Concepts

### 2.1 · XGBoost — Gradient Boosted Decision Trees

> **What:** An ensemble method where each new decision tree corrects the errors
> of the previous ones. Gold standard for **structured / tabular data**.

**In this project** — XGBoost takes **12 patient features** and outputs a
continuous risk score (0–100):

```python
FEATURE_SCHEMA = {
    "age", "num_sexual_partners", "age_first_intercourse",
    "smoking", "oral_contraceptives_yrs", "iud_yrs",
    "stds_count", "prev_cervical_biopsies", "immunocompromised",
    "hiv_positive", "family_hx_cervical_cancer", "last_pap_years_ago"
}
```

**Why XGBoost over deep learning for tabular data:**

- ✅ Consistently outperforms neural networks on structured data
- ✅ Handles missing values natively
- ✅ Fast training and inference
- ✅ Pairs perfectly with SHAP's `TreeExplainer` for exact, fast explanations

> 💡 **This is the same algorithm** used in production credit scoring at
> companies like Tala, Branch, and Lendable — the difference is only
> the input features, not the methodology.

---

### 2.2 · SHAP — SHapley Additive exPlanations

> **What:** A game-theory-based approach that assigns each input feature a
> **contribution score** to the final prediction.
> It answers: *"Why did this patient get this score?"*

#### How It Works

Imagine **12 players** on a team that scored 78 points.
SHAP calculates exactly how many points each player contributed —
including *negative* contributions from players who hurt the team.

#### Example — Healthcare (This Project)

```
  Patient Input
  ─────────────
    age = 28, partners = 8, smoking = yes, STDs = 3, ...

  Model Output
  ────────────
    HPV Risk Score = 78.4%  →  Risk Level: HIGH

  SHAP Breakdown (top factors)
  ────────────────────────────
    ✚ Number of sexual partners   +21%  ▲ increases risk
    ✚ Number of prior STDs        +18%  ▲ increases risk
    ✚ Smoking status              +10%  ▲ increases risk
    ✚ HIV positive                 +9%  ▲ increases risk
    ⊖ Age                          -3%  ▼ decreases risk
```

The clinician sees not just the score, but **exactly which factors drove it**
and in which direction. Critical for trust and actionable next steps.

#### Example — Credit Scoring (Same Technique)

Swap the features. The pipeline is identical:

```
  Tenant Input
  ────────────
    on_time_payments = 11, avg_days_late = 2, income_rent_ratio = 0.35, ...

  Model Output
  ────────────
    RentScore = 740  →  Rating: GOOD

  SHAP Breakdown (top factors)
  ────────────────────────────
    ✚ On-time payment streak      +15%  ▲ increases score
    ✚ Payment consistency         +12%  ▲ increases score
    ⊖ Recent late payment          -8%  ▼ decreases score
```

> 📌 Same `shap.TreeExplainer`, same output format — different features, different domain.

#### Why Explainability Matters Beyond Healthcare

| Use Case | Why SHAP Is Essential |
|:---------|:---------------------|
| 🏛 **Regulatory compliance** | Regulators require lenders to explain denials — SHAP produces auditable reason codes |
| 👤 **Consumer transparency** | Users see *what to do* to improve — *"pay on time for 3 more months"* |
| ⚖️ **Bias detection** | Reveals if the model unfairly weights proxy features (e.g., neighbourhood → ethnicity) |
| 🔧 **Model debugging** | Engineers verify the model isn't relying on spurious or leaky features |
| 📈 **Actionable insights** | Feature-level explanations turn a black-box score into a concrete improvement plan |

> **In CerviScan**, SHAP tells a doctor *which factors drove the cancer risk*.
> **In credit scoring**, the same approach tells a tenant *which behaviors are
> building — or hurting — their financial profile*. Same math, different domain.

---

### 2.3 · EfficientNetV2-S — Image Classification

> **What:** A CNN architecture optimised for accuracy-per-parameter
> via compound scaling (depth × width × resolution).

- 🔬 Fine-tuned on the **SIPaKMeD** cervical cytology dataset (5,015 images)
- 🏷 Classifies Pap smear cells into **5 categories**:

| Class | Category | Risk |
|:------|:---------|:----:|
| `im_Superficial-Intermediate` | Normal | 🟢 |
| `im_Parabasal` | Normal / Atrophic | 🟢 |
| `im_Metaplastic` | Normal / Reactive | 🟢 |
| `im_Koilocytotic` | Low-grade (LSIL) | 🟡 |
| `im_Dyskeratotic` | High-grade (HSIL) | 🔴 |

- 🧠 Uses **transfer learning** — pretrained ImageNet weights fine-tuned for this medical domain

**Transferable engineering patterns:**

1. **Transfer learning** — pretrained → fine-tuned for a specific context
2. **Class imbalance handling** — rare cell types ≈ rare event prediction in any domain
3. **Production inference pipelines** — efficient, reproducible serving

---

### 2.4 · Grad-CAM — Visual Explainability

> **What:** The image equivalent of SHAP — highlights which pixels
> were most important to the model's classification.

```python
# Output shape (Contract 2B)
{
    "heatmap":             # (224, 224) float32, values 0–1
    "overlay":             # (224, 224, 3) uint8 RGB — heatmap blended with original
    "attention_focus_pct": # % of image the model focused on (e.g., 12.4%)
}
```

The clinician sees the original image alongside a **coloured overlay**
showing exactly *where* the model looked.

> 🔑 **The principle is universal:** any AI system making high-stakes decisions —
> medical, financial, or otherwise — benefits from showing
> *what drove the decision*, not just the outcome.

---

### 2.5 · Class Imbalance Handling

> **The problem:** In medical data, high-grade cancer is rare.
> In credit data, defaults are rare. Both create the same trap —
> a naive model achieves high accuracy by predicting the majority class every time.

**Two complementary strategies used in this project:**

| # | Strategy | How It Works | Credit Scoring Parallel |
|:-:|:---------|:-------------|:------------------------|
| A | **Class-weighted loss** | Rare classes get a higher penalty when misclassified | Defaulters penalised more heavily when missed |
| B | **WeightedRandomSampler** | Each batch sampled inversely proportional to class frequency | Every epoch sees balanced defaults vs. non-defaults |

```python
# Strategy A — weighted loss
class_weights = compute_class_weights(train_ds, device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Strategy B — balanced sampling
sample_weights = compute_sample_weights(train_ds)
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_ds))
```

> ⚠️ The cost of missing a rare but critical case — cancer or loan default —
> is **asymmetrically high**. These strategies encode that asymmetry
> directly into training.

---

## 3 · Production Engineering

### 3.1 · Output Contracts

All model outputs follow **strict JSON API contracts** — decoupling ML models
from the front-end.

```python
# Contract 1A — HPV Risk Score
{
    "hpv_risk_score":      float,    # 0–100
    "risk_level":          str,      # "Low" / "Medium" / "High"
    "risk_color":          str,      # hex colour for UI
    "escalate_to_imaging": bool,     # triggers Step 2
    "top_risk_factors": [            # top 5 SHAP factors
        {"factor": str, "impact": float, "direction": str}
    ],
    "recommendation":      str,      # plain-language clinical advice
    "hpv_strain_risk":     str,      # strain likelihood text
}
```

> Front-end devs build against these contracts — not model internals.
> Same pattern used in production ML APIs everywhere.

---

### 3.2 · Heuristic Fallback (Cold Start)

When no trained model is available, the system falls back to
**rule-based heuristic scoring**:

```python
weights = {
    "num_sexual_partners": 0.20,
    "stds_count":          0.18,
    "smoking":             0.10,
    "hiv_positive":        0.10,
    ...
}
```

> 💡 This solves the **cold start problem** — scoring users with minimal data.
> As more data arrives, the system transitions from heuristic → model-based
> scoring seamlessly.

---

### 3.3 · Risk Band Classification

Continuous scores → discrete, actionable tiers:

```python
RISK_BANDS = [
    (66.0, "High",   "#f44336"),   # 🔴
    (33.0, "Medium", "#FF9800"),   # 🟡
    (0.0,  "Low",    "#4CAF50"),   # 🟢
]
```

> Directly analogous to tiered lending decisions:
> **auto-approve** · **manual review** · **auto-decline**

---

### 3.4 · Multi-Modal Fusion

The most advanced model (`CervicalMultiModal`) fuses **three data modalities**
into a single prediction:

```
  [Cytology image]    → ImageBranch (EfficientNetV2-S)  → 1280-d features
  [Clinical CSV]      → TabularBranch (MLP + embeddings) →   64-d features
  [Colposcopy image]  → ColposcopyBranch (optional)      →  512-d features
                               │
                        FusionHead (concat + dense)
                         ┌──────┴──────┐
                  ProgressionHead   CancerHead
                  (future risk)    (current risk)
```

> 🔗 Richer predictions come from combining multiple data sources —
> directly applicable to alternative credit scoring where payment data,
> landlord feedback, utility records, and mobile signals are all fused
> into a single, more accurate score.

---

## 4 · Skills Transferability

| Capability Demonstrated | Application in Alternative Credit Scoring |
|:------------------------|:------------------------------------------|
| 🌲 **XGBoost on tabular data** | Scoring from payment history, demographics, behavioral features |
| 🔍 **SHAP explainability** | Reason codes, consumer-facing explanations, bias audits |
| ⚙️ **Feature engineering** (12 features) | Engineering features from raw payment timestamps (streak, consistency) |
| 🚦 **Risk band classification** | Mapping continuous scores to actionable tiers |
| 🆕 **Heuristic fallback** | Bootstrapping scoring for new users with minimal history |
| 📦 **Output contracts** (strict APIs) | ML APIs for partner integrations (lenders, property managers) |
| ⚖️ **Class imbalance strategies** | Handling rare defaults in a mostly-paying population |
| 🔗 **Multi-modal data fusion** | Combining payment + reviews + utility data for richer scoring |
| 🧠 **Transfer learning** | Adapting general models to context-specific behaviors |
| 🚀 **Production deployment** | Serving real-time predictions via scalable endpoints |

---

## 5 · Technology Stack

| Component | Technology |
|:----------|:-----------|
| Tabular ML | XGBoost, scikit-learn |
| Explainability | SHAP (`TreeExplainer`), Grad-CAM |
| Deep Learning | PyTorch, timm (EfficientNetV2-S) |
| Data Augmentation | Albumentations |
| Dashboard | Streamlit |
| Deployment | Modal (serverless GPU) |
| Data Processing | Pandas, NumPy |
| Visualisation | Plotly, Matplotlib |
