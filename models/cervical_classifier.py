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
# Class metadata — SIPaKMeD classification 
# Mapped to clinical categories for Contract 2A
# ---------------------------------------------------------------------------

CLASS_META: Dict[str, Dict[str, Any]] = {
    "im_Superficial-Intermediate": {
        "category":     "Normal",
        "cin_grade":    "No CIN",
        "triage_color": "green",
        "action":       "Normal squamous cells detected. Routine screening.",
        "urgency":      "low",
        "description":  "Superficial and intermediate squamous cells. Benign finding.",
        "color":        "#4CAF50",
    },
    "im_Parabasal": {
        "category":     "Normal/Atrophic",
        "cin_grade":    "No CIN",
        "triage_color": "green",
        "action":       "Normal parabasal cells. Common in post-menopausal or atrophic smears.",
        "urgency":      "low",
        "description":  "Normal parabasal squamous cells. Typically benign.",
        "color":        "#8BC34A",
    },
    "im_Metaplastic": {
        "category":     "Normal/Reactive",
        "cin_grade":    "No CIN",
        "triage_color": "green",
        "action":       "Benign metaplastic changes. No follow-up required.",
        "urgency":      "low",
        "description":  "Squamous metaplastic cells. Represents normal transformation zone activity.",
        "color":        "#CDDC39",
    },
    "im_Koilocytotic": {
        "category":     "Low-grade",
        "cin_grade":    "LSIL / CIN1",
        "triage_color": "amber",
        "action":       "Evidence of HPV infection. Repeat cytology in 6-12 months.",
        "urgency":      "moderate",
        "description":  "Koilocytes indicate HPV-related changes. Consistent with LSIL.",
        "color":        "#FF9800",
    },
    "im_Dyskeratotic": {
        "category":     "High-grade",
        "cin_grade":    "HSIL / CIN2-3",
        "triage_color": "red",
        "action":       "High-grade suspicion. Urgent colposcopy and biopsy required.",
        "urgency":      "high",
        "description":  "Dyskeratotic cells associated with high-grade dysplasia or malignancy.",
        "color":        "#F44336",
    },
}

CLASS_NAMES = sorted(list(CLASS_META.keys()))
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
