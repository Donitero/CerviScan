"""
Grad-CAM Explainability — Module 2 (Osborn)
=============================================
Wraps pytorch-grad-cam to produce heatmaps compatible with the femscan-ai UI.

Output contract → docs/OUTPUT_CONTRACTS.md § Contract 2B
"""

from __future__ import annotations

import numpy as np
import cv2
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from typing import Dict, Any


class GradCAMExplainer:
    """
    Usage
    -----
    explainer = GradCAMExplainer(model)
    result = explainer.explain(image_tensor, original_np)   # → CONTRACT 2B
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        """
        Args
        ----
        model        : CervicalClassifier instance.
        target_layer : Conv layer to hook. Defaults to conv_head of EfficientNetV2-S.
        """
        if target_layer is None:
            target_layer = model.backbone.conv_head
        self.cam = GradCAM(model=model, target_layers=[target_layer])

    # ------------------------------------------------------------------
    # Explainability API — CONTRACT 2B
    # ------------------------------------------------------------------

    def explain(
        self,
        image_tensor: torch.Tensor,
        original_np: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Args
        ----
        image_tensor : torch.Tensor  (1, 3, 224, 224), same tensor passed to predict().
        original_np  : np.ndarray    (224, 224, 3) uint8 RGB — raw image at model size.

        Returns (CONTRACT 2B)
        ----------------------
        {
            "heatmap":             np.ndarray [224, 224] float32  0-1,
            "overlay":             np.ndarray [224, 224, 3] uint8 RGB,  ← st.image() ready
            "attention_focus_pct": float,   # % of pixels with heatmap > 0.5
        }
        """
        grayscale_cam = self.cam(input_tensor=image_tensor)
        heatmap = grayscale_cam[0]  # (224, 224) float32

        img_float   = original_np.astype(np.float32) / 255.0
        overlay_bgr = show_cam_on_image(img_float, heatmap, use_rgb=False)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        return {
            "heatmap":             heatmap,
            "overlay":             overlay_rgb,
            "attention_focus_pct": float((heatmap > 0.5).mean() * 100),
        }
