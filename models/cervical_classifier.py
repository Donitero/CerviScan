"""
MODULE 2 — CIN Lesion Detector (Osborn)
========================================
EfficientNetV2-S fine-tuned on Bethesda-classified cervical cytology data
(6 classes: Negative, ASC-US, LSIL, ASC-H, HSIL, ca).

Output contract -> docs/OUTPUT_CONTRACTS.md § Contract 2A
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm
from typing import Dict, Any


# ---------------------------------------------------------------------------
# Class metadata — Bethesda classification aligned to actual dataset
# Classes: Negative, ASC-US, LSIL, ASC-H, HSIL, ca
# ---------------------------------------------------------------------------

CLASS_META: Dict[str, Dict[str, Any]] = {
    "Negative": {
        "category":     "Normal",
        "cin_grade":    "No CIN",
        "triage_color": "green",
        "action":       "No abnormality detected. Routine screening schedule.",
        "urgency":      "low",
        "description":  "Negative for intraepithelial lesion or malignancy (NILM). Normal squamous cells.",
        "color":        "#4CAF50",
    },
    "ASC-US": {
        "category":     "Borderline",
        "cin_grade":    "Indeterminate",
        "triage_color": "amber",
        "action":       "Atypical cells of undetermined significance. Reflex HPV testing or repeat smear in 12 months.",
        "urgency":      "low",
        "description":  "ASC-US: minor atypia that does not meet criteria for LSIL. Most resolve spontaneously.",
        "color":        "#CDDC39",
    },
    "LSIL": {
        "category":     "Low-grade",
        "cin_grade":    "CIN1 (low-grade)",
        "triage_color": "amber",
        "action":       "Low-grade squamous intraepithelial lesion. Repeat cytology or colposcopy in 6-12 months.",
        "urgency":      "moderate",
        "description":  "LSIL: changes consistent with HPV cytopathic effect (koilocytosis). Usually CIN1.",
        "color":        "#FF9800",
    },
    "ASC-H": {
        "category":     "High-grade suspect",
        "cin_grade":    "CIN2 suspect",
        "triage_color": "red",
        "action":       "Atypical cells -- HSIL cannot be excluded. Colposcopy referral required.",
        "urgency":      "high",
        "description":  "ASC-H: atypical squamous cells where high-grade lesion cannot be excluded.",
        "color":        "#FF5722",
    },
    "HSIL": {
        "category":     "High-grade",
        "cin_grade":    "CIN2-3 (high-grade)",
        "triage_color": "red",
        "action":       "High-grade squamous intraepithelial lesion. Urgent colposcopy and directed biopsy.",
        "urgency":      "high",
        "description":  "HSIL: significant nuclear atypia consistent with CIN2 or CIN3. Requires immediate follow-up.",
        "color":        "#F44336",
    },
    "ca": {
        "category":     "Malignant",
        "cin_grade":    "Carcinoma",
        "triage_color": "red",
        "action":       "Findings suspicious for carcinoma. Immediate oncology referral required.",
        "urgency":      "high",
        "description":  "Cells with features consistent with squamous cell carcinoma. Urgent specialist review.",
        "color":        "#B71C1C",
    },
}

CLASS_NAMES = list(CLASS_META.keys())
NUM_CLASSES  = len(CLASS_NAMES)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CervicalClassifier(nn.Module):
    """EfficientNetV2-S fine-tuned on Bethesda-classified cervical cytology (6 classes)."""

    def __init__(self, pretrained: bool = False, checkpoint_path: str | None = None):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnetv2_s",
            pretrained=pretrained,
            num_classes=NUM_CLASSES,
        )
        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location="cpu")
            self.backbone.load_state_dict(state.get("model_state_dict", state))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    # ------------------------------------------------------------------
    # Inference API — CONTRACT 2A
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Args
        ----
        image_tensor : torch.Tensor  shape (1, 3, 224, 224), ImageNet-normalised.

        Returns (CONTRACT 2A)
        ----------------------
        {
            "class_name":    str,
            "category":      "Normal" | "Borderline" | "Low-grade" | "High-grade suspect"
                             | "High-grade" | "Malignant",
            "confidence":    float 0-1,
            "all_probs":     {class_name: float, ...},
            "cin_grade":     str,
            "triage_color":  "green" | "amber" | "red",
            "action":        str,
            "urgency":       "low" | "moderate" | "high",
            "description":   str,
            "color":         str hex,
        }
        """
        self.eval()
        probs = torch.softmax(self.forward(image_tensor), dim=1)[0]
        idx   = int(probs.argmax())
        name  = CLASS_NAMES[idx]
        meta  = CLASS_META[name]

        return {
            "class_name":   name,
            "category":     meta["category"],
            "confidence":   float(probs[idx]),
            "all_probs":    {n: float(probs[i]) for i, n in enumerate(CLASS_NAMES)},
            "cin_grade":    meta["cin_grade"],
            "triage_color": meta["triage_color"],
            "action":       meta["action"],
            "urgency":      meta["urgency"],
            "description":  meta["description"],
            "color":        meta["color"],
        }
