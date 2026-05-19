# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unified attacker interface (factory + dispatch).

``AdversarialAttacker`` is a thin facade that selects the right implementation
based on ``(mode, aug_type)`` and forwards ``predict_params`` /
``apply_transform`` calls into it. Two-step interface (predict, then apply)
keeps gradient flow stable during the joint min-max backward.
"""

import torch
import torch.nn as nn

from ruac.core.attackers.deform import FeatureLevelDeformationImpl
from ruac.core.attackers.style import ImageLevelStyleImpl


class AdversarialAttacker(nn.Module):
    """
    Unified interface for adversarial attacks.

    Supports style (image-level) and deformation (feature-level) attacks.
    Automatically selects the appropriate implementation based on configuration.

    Args:
        mode: "image_level" (style) or "feature_level" (deformation)
        aug_type: "style" or "deformation"
        **kwargs: Additional arguments passed to the specific implementation
    """

    def __init__(self, mode: str, aug_type: str, **kwargs):
        super().__init__()
        self.mode = mode
        self.aug_type = aug_type

        # Create the appropriate implementation
        self.impl = self._create_impl(**kwargs)

    def _create_impl(self, **kwargs):
        """Factory method to create the appropriate implementation"""
        if self.aug_type == "style":
            if self.mode == "image_level":
                return ImageLevelStyleImpl(**kwargs)
            raise ValueError(f"Unknown mode for style augmentation: {self.mode}")

        if self.aug_type == "deformation":
            if self.mode == "feature_level":
                return FeatureLevelDeformationImpl(**kwargs)
            raise ValueError(f"Unknown mode for deformation: {self.mode}")

        raise ValueError(f"Unknown augmentation type: {self.aug_type}")

    def predict_params(self, clean_features: torch.Tensor, pixel_gt: torch.Tensor, model: nn.Module, **kwargs):
        """
        Predict adversarial parameters (e.g., style codes, deformation offsets).

        Args:
            clean_features: [B, C, H, W] Clean backbone features
            pixel_gt: [B, K, H, W] Ground truth masks
            model: adapter-side module exposing config attributes that the
                concrete implementation reads (duck-typed; any ``nn.Module``
                with the expected attribute surface works).
            **kwargs: Additional arguments

        Returns:
            params: Adversarial parameters (type depends on implementation)
        """
        return self.impl.predict_params(clean_features=clean_features, pixel_gt=pixel_gt, model=model, **kwargs)

    def apply_transform(
        self,
        params: torch.Tensor | dict,
        img_batch: torch.Tensor | None = None,
        clean_features: torch.Tensor | None = None,
        pixel_gt: torch.Tensor | None = None,
        model: nn.Module | None = None,
        **kwargs,
    ):
        """
        Apply the transformation using predicted parameters.

        Args:
            params: Adversarial parameters predicted by predict_params
            img_batch: [B, 3, H, W] Input images (for image-level aug)
            clean_features: [B, C, H, W] Clean features (for feature-level aug)
            pixel_gt: [B, K, H, W] Ground truth masks
            model: adapter-side module; same duck-typed contract as
                ``predict_params``.
            **kwargs: Additional arguments

        Returns:
            transformed: Transformed images or features
        """
        return self.impl.apply_transform(img_batch=img_batch, clean_features=clean_features, params=params, pixel_gt=pixel_gt, model=model, **kwargs)
