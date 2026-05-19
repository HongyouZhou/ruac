# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""SAM2 adapter — AUE bridge.

Bridge layer between SAM2's ``track_step`` hook contract (which calls
``self._aue_module.generate_adversarial_samples(...)``) and the model-agnostic
``ruac.core.pipeline.AdversarialPipeline``. ``initialize()`` reads attackers
and config off ``self._model`` and constructs the pipeline with explicit DI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ruac.core.pipeline import AdversarialPipeline
from ruac.modeling.aue.visualization import AUEVisualizer

if TYPE_CHECKING:
    from sam2.modeling.sam2_base import SAM2Base


class AUEModule:
    """
    SAM2-side AUE orchestrator.

    Holds a reference to the SAM2 model and a (possibly null) core
    ``AdversarialPipeline``. Attached to the model as ``self._aue_module`` so
    the inherited ``track_step`` can drive it via ``generate_adversarial_samples``.

    Args:
        model: Reference to SAM2Base model
    """

    def __init__(self, model: SAM2Base):
        self._model = model
        self._pipeline: AdversarialPipeline | None = None
        self._visualizer = AUEVisualizer()

    def initialize(self) -> None:
        """
        Initialize AUE components after model is fully constructed.

        Collects attacker modules from ``self._model`` attributes and constructs
        the core ``AdversarialPipeline`` with explicit dependency injection.
        This is the SAM2 adapter layer: pure config extraction.
        """
        model = self._model

        if not getattr(model, "use_aue", False):
            return

        if not (
            getattr(model, "use_style_adv", False)
            or getattr(model, "use_deform_adv", False)
            or getattr(model, "use_pgd_adv", False)
            or getattr(model, "use_patch_adv", False)
            or getattr(model, "use_random_noise_adv", False)
        ):
            return

        attackers = {}
        for name in ("style", "deform", "pgd", "patch", "random_noise"):
            attacker = getattr(model, f"{name}_attacker", None)
            if attacker is not None:
                attackers[name] = attacker

        self._pipeline = AdversarialPipeline(
            attackers=attackers,
            attack_order=list(model.adversarial_attack_order),
            backbone_fn=model.forward_image,
            use_high_res_features=model.use_high_res_features_in_sam,
            enable_background=getattr(model, "adv_enable_background", False),
            attack_context=model,
        )

    def generate_adversarial_samples(
        self,
        img_batch: torch.Tensor,
        backbone_features: torch.Tensor,
        high_res_features: list[torch.Tensor],
        pixel_gt: torch.Tensor,
        single_obj_gt: torch.Tensor | None = None,
        enable_vis: bool = False,
    ) -> dict:
        """
        Generate adversarial samples without forward pass or loss computation.

        This is a convenience method that delegates to the pipeline.

        Args:
            img_batch: [B, 3, H, W] input images
            backbone_features: [B, C, H, W] backbone features
            high_res_features: List of high-res features
            pixel_gt: [B, K, H, W] ground truth masks (all objects, for attack generation)
            single_obj_gt: [B, 1, H, W] single object GT (same as clean branch, for SAM task)
            enable_vis: Whether to collect visualization data

        Returns:
            Dict with adv_img, adv_features, adv_high_res, adv_pixel_gt, adv_single_obj_gt, vis_refs
        """
        if self._pipeline is None:
            return {
                "adv_img": img_batch,
                "adv_features": backbone_features,
                "adv_high_res": high_res_features,
                "adv_pixel_gt": pixel_gt,
                "adv_single_obj_gt": single_obj_gt,
                "vis_refs": {},
            }

        return self._pipeline.generate_adversarial_samples(
            img_batch=img_batch,
            backbone_features=backbone_features,
            high_res_features=high_res_features,
            pixel_gt=pixel_gt,
            single_obj_gt=single_obj_gt,
            enable_vis=enable_vis,
        )
