# FemScan AI

AI-assisted cervical health screening platform.
Three modules. Three steps. One pipeline.

---

## The Three Modules (Pitch Deck Slide 7)

| Module | File | Owner | Description |
|--------|------|-------|-------------|
| 1 — HPV Risk | `models/hpv_risk_scorer.py` | Osborn | XGBoost + SHAP risk scorer from patient data |
| 2 — CIN Detection | `models/cervical_classifier.py` + `models/gradcam.py` | Osborn | EfficientNetV2-S on SIPaKMeD + Grad-CAM |
| 3 — Endo Scoring | `models/endo_scorer.py` | Osborn | XGBoost + SHAP endo risk from symptoms |

## The Three Steps (Pitch Deck Slide 8)

| Step | Modules | Owner |
|------|---------|-------|
| 1 — Screen & Score | Modules 1 + 3 | Person B (UI) |
| 2 — Capture & Analyse | Module 2 | Person B (UI) + Person C (components) |
| 3 — Act & Record | UI-only | Person B (UI) |

## Roles

| Person | Responsibility |
|--------|---------------|
| **Osborn** | All model code (`models/`), training functions in `modal_deploy.py`, `scripts/prepare_demo.py` |
| **Person B** | `app.py` — Streamlit dashboard, 3-step UI flow |
| **Person C** | `dashboard/components/` — reusable Streamlit components |
| **Person D** | `serve_*` functions in `modal_deploy.py` — cloud inference endpoints |

## Output Contracts

All model output shapes: [`docs/OUTPUT_CONTRACTS.md`](docs/OUTPUT_CONTRACTS.md)

Person B and Person C code against these contracts — not model internals.

## Quick Start

```bash
pip install -r requirements.txt

# Generate demo assets (no trained weights needed)
python scripts/prepare_demo.py

# Run the dashboard
streamlit run app.py
```

## Project Structure

```
femscan-ai/
├── app.py                    # Person B — Streamlit entry point
├── modal_deploy.py           # Osborn (train) + Person D (serve)
├── requirements.txt
├── .streamlit/config.toml    # Dark medical theme
├── models/
│   ├── hpv_risk_scorer.py    # MODULE 1 (Osborn)
│   ├── cervical_classifier.py# MODULE 2 (Osborn)
│   ├── gradcam.py            # Grad-CAM for Module 2 (Osborn)
│   └── endo_scorer.py        # MODULE 3 (Osborn)
├── trained_models/           # Weights — not tracked in git
├── data/
│   ├── sipakmed/             # Raw dataset — not tracked
│   ├── hpv/                  # Raw dataset — not tracked
│   ├── endo/                 # Raw dataset — not tracked
│   └── demo/                 # Pre-computed demo assets
├── dashboard/components/     # Person C — reusable UI components
├── docs/
│   └── OUTPUT_CONTRACTS.md   # API contracts (read this first)
├── scripts/
│   └── prepare_demo.py       # Generate demo assets
├── tests/
└── assets/
```
