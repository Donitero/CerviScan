"""
Grad-CAM explainability wrapper.

Output contract (see docs/OUTPUT_CONTRACTS.md — CONTRACT 2):
    GradCAMExplainer.explain(image_tensor, original_np) -> dict
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
    Wraps pytorch-grad-cam to produce heatmaps and overlays compatible
    with the femscan-ai UI (CONTRACT 2).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        """
        Args:
            model: A CervicalClassifier (or any nn.Module).
            target_layer: The convolutional layer to hook. Defaults to the
                          last conv block of an EfficientNetV2-S backbone.
        """
        if target_layer is None:
            # EfficientNetV2-S via timm: last conv before global pool
            target_layer = model.backbone.conv_head

        self.cam = GradCAM(model=model, target_layers=[target_layer])

    # ------------------------------------------------------------------
    # Explainability API — returns OUTPUT CONTRACT 2
    # ------------------------------------------------------------------

    def explain(
        self,
        image_tensor: torch.Tensor,
        original_np: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Args:
            image_tensor: float32 tensor (1, 3, 224, 224), same as passed to predict().
            original_np:  uint8 RGB ndarray (224, 224, 3) — the raw image
                          resized to model input size, used for overlay blending.

        Returns:
            dict matching CONTRACT 2 in docs/OUTPUT_CONTRACTS.md:
            {
                "heatmap":            np.ndarray [224, 224] float32 in [0, 1],
                "overlay":            np.ndarray [224, 224, 3] uint8 RGB,
                "attention_focus_pct": float  (% of pixels with heatmap > 0.5)
            }
        """
        # grayscale_cam shape: (1, 224, 224)
        grayscale_cam = self.cam(input_tensor=image_tensor)
        heatmap = grayscale_cam[0]  # (224, 224) float32 in [0, 1]

        # Normalise original image to [0, 1] float for blending
        img_float = original_np.astype(np.float32) / 255.0
        overlay_bgr = show_cam_on_image(img_float, heatmap, use_rgb=False)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        attention_focus_pct = float((heatmap > 0.5).mean() * 100)

        return {
            "heatmap": heatmap,
            "overlay": overlay_rgb,
            "attention_focus_pct": attention_focus_pct,
        }
