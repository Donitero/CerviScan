"""
Multi-Modal, Multi-Task Cervical Cancer Screening Model
=======================================================

Architecture
------------

  [Cytology image]  --> ImageBranch (EfficientNetV2-S)  --> 1280-d features
                                                               |
  [Clinical CSV]    --> TabularBranch (MLP + embeddings) -->  64-d features
                                                               |
  [Colposcopy img]  --> ColposcopyBranch (optional)      --> 512-d features
                                                               |
                              FusionHead (concat + dense) --> shared repr
                                                         /          \
                              ProgressionHead (sigmoid)     CancerHead (sigmoid)
                              "future cancer risk"          "cancer present now"

Tasks
-----
  Task 1 - progression_risk  : P(high-grade CIN / future cancer)
  Task 2 - cancer_probability: P(cancer currently present)

Both outputs are sigmoid-activated floats in [0, 1].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm
from typing import Dict, Any, Optional


# ---------------------------------------------------------------------------
# 1. Image branch — EfficientNetV2-S feature extractor
# ---------------------------------------------------------------------------

class ImageBranch(nn.Module):
    """
    Wraps EfficientNetV2-S and removes the classification head,
    returning 1280-dimensional image features.

    Why EfficientNetV2-S over ResNet18:
      - Already in requirements.txt / pretrained weights cached
      - Better accuracy per parameter for medical imaging tasks
      - Compound scaling handles the fine-grained cell morphology better
    """

    OUT_DIM = 1280

    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        backbone = timm.create_model("efficientnetv2_s", pretrained=pretrained, num_classes=0)
        self.features = backbone          # num_classes=0 removes head, outputs pooled features
        self.drop     = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 224, 224) -> (B, 1280)"""
        return self.drop(self.features(x))


# ---------------------------------------------------------------------------
# 2. Tabular branch — MLP with categorical embedding for HPV type
# ---------------------------------------------------------------------------

class TabularBranch(nn.Module):
    """
    Encodes clinical tabular features:
      - HPV_type (categorical: none/other/18/16)  -> 8-d embedding
      - Numerical: age, HPV_status, previous_CIN_stage, persistence (4 features)
      - Total input to MLP: 8 + 4 = 12 dimensions

    Missing values: caller should pass 0.0 for unknowns (handled gracefully).
    """

    # Numerical columns in the order expected by forward()
    NUM_FEATURES = ["age_norm", "HPV_status", "previous_CIN_stage_norm", "persistence"]
    HPV_TYPE_VOCAB_SIZE = 4    # none=0, other=1, 18=2, 16=3
    HPV_TYPE_EMBED_DIM  = 8
    OUT_DIM             = 64

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.hpv_type_embed = nn.Embedding(self.HPV_TYPE_VOCAB_SIZE, self.HPV_TYPE_EMBED_DIM)

        in_dim = self.HPV_TYPE_EMBED_DIM + len(self.NUM_FEATURES)  # 8 + 4 = 12
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, self.OUT_DIM),
            nn.ReLU(),
        )

    def forward(
        self,
        hpv_type_code:      torch.Tensor,   # (B,) int64
        age_norm:           torch.Tensor,   # (B,) float32  age/100
        hpv_status:         torch.Tensor,   # (B,) float32  0/1
        prev_cin_norm:      torch.Tensor,   # (B,) float32  stage/4
        persistence:        torch.Tensor,   # (B,) float32  0/1
    ) -> torch.Tensor:
        """Returns (B, 64) tabular features."""
        emb = self.hpv_type_embed(hpv_type_code)          # (B, 8)
        num = torch.stack(
            [age_norm, hpv_status, prev_cin_norm, persistence], dim=1
        )                                                   # (B, 4)
        x = torch.cat([emb, num], dim=1)                   # (B, 12)
        return self.mlp(x)


# ---------------------------------------------------------------------------
# 3. Optional colposcopy branch
# ---------------------------------------------------------------------------

class ColposcopyBranch(nn.Module):
    """
    Separate image encoder for colposcopy images (optional second modality).
    Uses a lighter EfficientNet-B0 to keep parameter count manageable.
    Returns 512-d features.
    """

    OUT_DIM = 512

    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        backbone = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=0)
        self.features = backbone
        self.proj = nn.Sequential(
            nn.Linear(1280, self.OUT_DIM),
            nn.ReLU(),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 224, 224) -> (B, 512)"""
        feats = self.drop(self.features(x))   # (B, 1280) for B0
        return self.proj(feats)


# ---------------------------------------------------------------------------
# 4. Fusion + multi-task heads
# ---------------------------------------------------------------------------

class CervicalMultiModal(nn.Module):
    """
    Full multi-modal, multi-task model.

    Inputs
    ------
    cyto_img         : (B, 3, 224, 224)   cytology image (required)
    tabular          : dict with keys age_norm, hpv_type_code, hpv_status,
                       prev_cin_norm, persistence
    colpo_img        : (B, 3, 224, 224)   colposcopy image (optional, may be None)

    Outputs (dict)
    --------------
    {
        "progression_risk"   : (B,) sigmoid  -- P(high-grade CIN / future risk)
        "cancer_probability" : (B,) sigmoid  -- P(cancer currently present)
        "image_features"     : (B, 1280)     -- for Grad-CAM / embedding viz
        "tabular_features"   : (B, 64)       -- for SHAP analysis
    }
    """

    def __init__(
        self,
        pretrained:      bool = True,
        use_colposcopy:  bool = False,
        dropout:         float = 0.4,
    ):
        super().__init__()
        self.use_colposcopy = use_colposcopy

        self.image_branch    = ImageBranch(pretrained=pretrained, dropout=dropout)
        self.tabular_branch  = TabularBranch(dropout=dropout)

        fusion_in = ImageBranch.OUT_DIM + TabularBranch.OUT_DIM  # 1280 + 64 = 1344
        if use_colposcopy:
            self.colpo_branch = ColposcopyBranch(pretrained=pretrained, dropout=dropout)
            fusion_in += ColposcopyBranch.OUT_DIM                # 1344 + 512 = 1856

        # Shared fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
        )

        # Task-specific heads — each outputs a single logit (BCEWithLogitsLoss)
        self.progression_head = nn.Linear(128, 1)
        self.cancer_head      = nn.Linear(128, 1)

    def forward(
        self,
        cyto_img:  torch.Tensor,
        tabular:   Dict[str, torch.Tensor],
        colpo_img: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        img_feats = self.image_branch(cyto_img)

        tab_feats = self.tabular_branch(
            hpv_type_code = tabular["hpv_type_code"],
            age_norm      = tabular["age_norm"],
            hpv_status    = tabular["hpv_status"],
            prev_cin_norm = tabular["prev_cin_norm"],
            persistence   = tabular["persistence"],
        )

        parts = [img_feats, tab_feats]

        if self.use_colposcopy and colpo_img is not None:
            parts.append(self.colpo_branch(colpo_img))

        fused = self.fusion(torch.cat(parts, dim=1))

        return {
            "progression_risk":   self.progression_head(fused).squeeze(1),  # (B,) raw logit
            "cancer_probability": self.cancer_head(fused).squeeze(1),       # (B,) raw logit
            "image_features":     img_feats,
            "tabular_features":   tab_feats,
        }

    # ------------------------------------------------------------------
    # Inference API (post-sigmoid scores)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        cyto_img:  torch.Tensor,
        tabular:   Dict[str, torch.Tensor],
        colpo_img: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Returns human-readable prediction dict.

        {
          "progression_risk":    float 0-1,
          "cancer_probability":  float 0-1,
          "progression_level":   "Low" | "Moderate" | "High",
          "cancer_level":        "Unlikely" | "Possible" | "Probable",
          "triage_color":        "green" | "amber" | "red",
          "action":              str,
        }
        """
        self.eval()
        out   = self.forward(cyto_img, tabular, colpo_img)
        p_prog  = float(torch.sigmoid(out["progression_risk"]).item())
        p_cancer = float(torch.sigmoid(out["cancer_probability"]).item())

        prog_level, cancer_level, color, action = _interpret(p_prog, p_cancer)

        return {
            "progression_risk":   round(p_prog,   4),
            "cancer_probability": round(p_cancer, 4),
            "progression_level":  prog_level,
            "cancer_level":       cancer_level,
            "triage_color":       color,
            "action":             action,
        }

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save({
            "model_state_dict": self.state_dict(),
            "use_colposcopy":   self.use_colposcopy,
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "CervicalMultiModal":
        ckpt  = torch.load(path, map_location=device)
        model = cls(pretrained=False, use_colposcopy=ckpt.get("use_colposcopy", False))
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _interpret(p_prog: float, p_cancer: float):
    if p_cancer >= 0.5:
        return "High", "Probable", "red", \
            "Cancer probability high. Immediate oncology referral and biopsy required."
    if p_prog >= 0.6 or p_cancer >= 0.25:
        return "High", "Possible", "red", \
            "High progression risk or elevated cancer signal. Urgent colposcopy and biopsy."
    if p_prog >= 0.3:
        return "Moderate", "Unlikely", "amber", \
            "Moderate progression risk. Colposcopy referral within 3 months recommended."
    return "Low", "Unlikely", "green", \
        "Low risk. Continue routine screening schedule."
