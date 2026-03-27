"""
Cervical Cell Classifier — SIPaKMeD 5-class model.

Output contract (see docs/OUTPUT_CONTRACTS.md — CONTRACT 1):
    CervicalClassifier.predict(image_tensor) -> dict
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm
from typing import Dict, Any


# ---------------------------------------------------------------------------
# Class metadata — drives UI colours and clinical copy
# ---------------------------------------------------------------------------

CLASS_META: Dict[str, Dict[str, Any]] = {
    "Superficial-Intermediate": {
        "category": "Normal",
        "action": "Routine screening. No immediate follow-up required.",
        "urgency": "low",
        "description": (
            "Superficial and intermediate squamous cells are mature, well-differentiated "
            "cells found on the upper layers of healthy cervical epithelium."
        ),
        "color": "#4CAF50",  # green
    },
    "Parabasal": {
        "category": "Normal",
        "action": "Routine screening. Correlate with hormonal status.",
        "urgency": "low",
        "description": (
            "Parabasal cells originate from the lower epithelial layers and are commonly "
            "seen in post-menopausal or atrophic cervical smears."
        ),
        "color": "#8BC34A",  # light green
    },
    "Koilocyte": {
        "category": "Benign",
        "action": "HPV-associated changes noted. Recommend repeat smear in 6–12 months.",
        "urgency": "moderate",
        "description": (
            "Koilocytes exhibit perinuclear halos consistent with HPV cytopathic effect. "
            "These changes are benign but warrant monitoring for progression."
        ),
        "color": "#FF9800",  # orange
    },
    "Dyskeratocyte": {
        "category": "Benign",
        "action": "Abnormal keratinisation detected. Colposcopy referral recommended.",
        "urgency": "moderate",
        "description": (
            "Dyskeratocytes show premature or abnormal keratinisation. While often benign, "
            "colposcopic evaluation is advised to exclude low-grade dysplasia."
        ),
        "color": "#FF5722",  # deep orange
    },
    "Metaplastic": {
        "category": "Abnormal",
        "action": "Atypical metaplastic changes. Urgent colposcopy and biopsy advised.",
        "urgency": "high",
        "description": (
            "Metaplastic cells from the transformation zone showing atypical features may "
            "indicate squamous intraepithelial lesion (SIL). Prompt clinical evaluation required."
        ),
        "color": "#F44336",  # red
    },
}

CLASS_NAMES = list(CLASS_META.keys())  # order must match training label encoding
NUM_CLASSES = len(CLASS_NAMES)


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

class CervicalClassifier(nn.Module):
    """EfficientNetV2-S fine-tuned on SIPaKMeD for 5-class cervical classification."""

    def __init__(self, pretrained: bool = False, checkpoint_path: str | None = None):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnetv2_s",
            pretrained=pretrained,
            num_classes=NUM_CLASSES,
        )

        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location="cpu")
            # support both raw state-dicts and checkpoint dicts
            sd = state.get("model_state_dict", state)
            self.backbone.load_state_dict(sd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    # ------------------------------------------------------------------
    # Inference API — returns OUTPUT CONTRACT 1
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Args:
            image_tensor: float32 tensor of shape (1, 3, 224, 224),
                          normalised with ImageNet mean/std.

        Returns:
            dict matching CONTRACT 1 in docs/OUTPUT_CONTRACTS.md.
        """
        self.eval()
        logits = self.forward(image_tensor)           # (1, 5)
        probs = torch.softmax(logits, dim=1)[0]       # (5,)

        pred_idx = int(probs.argmax())
        class_name = CLASS_NAMES[pred_idx]
        meta = CLASS_META[class_name]

        return {
            "class_name": class_name,
            "category": meta["category"],
            "confidence": float(probs[pred_idx]),
            "all_probs": {
                name: float(probs[i]) for i, name in enumerate(CLASS_NAMES)
            },
            "action": meta["action"],
            "urgency": meta["urgency"],
            "description": meta["description"],
            "color": meta["color"],
        }
