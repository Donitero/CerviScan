"""
MODULE 2 — CIN Lesion Detector (Osborn)
========================================
EfficientNetV2-S fine-tuned on SIPaKMeD for 5-class cervical cell classification
with CIN grading aligned to pitch deck Slide 8 triage colours.

Output contract → docs/OUTPUT_CONTRACTS.md § Contract 2A
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm
from typing import Dict, Any


# ---------------------------------------------------------------------------
# Class metadata — drives CIN grading + pitch-deck triage colours
# ---------------------------------------------------------------------------

CLASS_META: Dict[str, Dict[str, Any]] = {
    "Superficial-Intermediate": {
        "category":     "Normal",
        "cin_grade":    "No CIN",
        "triage_color": "green",
        "action":       "Routine screening. No immediate follow-up required.",
        "urgency":      "low",
        "description":  "Mature squamous cells from the upper epithelial layers. Normal finding.",
        "color":        "#4CAF50",
    },
    "Parabasal": {
        "category":     "Normal",
        "cin_grade":    "No CIN",
        "triage_color": "green",
        "action":       "Routine screening. Correlate with hormonal status.",
        "urgency":      "low",
        "description":  "Parabasal cells — common in atrophic/post-menopausal smears. Usually benign.",
        "color":        "#8BC34A",
    },
    "Koilocyte": {
        "category":     "Benign",
        "cin_grade":    "CIN1 (low-grade)",
        "triage_color": "amber",
        "action":       "HPV cytopathic effect noted. Repeat smear in 6–12 months; consider HPV co-test.",
        "urgency":      "moderate",
        "description":  "Koilocytes show perinuclear halos consistent with active HPV infection (CIN1 equivalent).",
        "color":        "#FF9800",
    },
    "Dyskeratocyte": {
        "category":     "Benign",
        "cin_grade":    "CIN1 (low-grade)",
        "triage_color": "amber",
        "action":       "Abnormal keratinisation detected. Colposcopy referral recommended.",
        "urgency":      "moderate",
        "description":  "Dyskeratocytes indicate premature keratinisation; may represent low-grade SIL.",
        "color":        "#FF5722",
    },
    "Metaplastic": {
        "category":     "Abnormal",
        "cin_grade":    "CIN2-3 (high-grade)",
        "triage_color": "red",
        "action":       "High-grade changes suspected. Urgent colposcopy and directed biopsy required.",
        "urgency":      "high",
        "description":  "Atypical metaplastic cells from the transformation zone — high-grade SIL cannot be excluded.",
        "color":        "#F44336",
    },
}

CLASS_NAMES = list(CLASS_META.keys())
NUM_CLASSES  = len(CLASS_NAMES)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CervicalClassifier(nn.Module):
    """EfficientNetV2-S fine-tuned on SIPaKMeD (5-class CIN classification)."""

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
            "category":      "Normal" | "Benign" | "Abnormal",
            "confidence":    float 0-1,
            "all_probs":     {class_name: float, ...},
            "cin_grade":     "No CIN" | "CIN1 (low-grade)" | "CIN2-3 (high-grade)",
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
