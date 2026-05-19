# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""AUE Adversarial Attack Pipeline.

Orchestrates the cooperative attack:

1. Predict parameters for ALL active attackers from CLEAN backbone features
   (parallel prediction, stable gradients).
2. Apply each attacker's image-space transformation sequentially.
3. Do a SINGLE backbone forward at the end on the final transformed image.

The pipeline is model-agnostic: all the things it used to read off the model
(``attackers``, ``attack_order``, ``backbone_fn``, ``use_high_res_features``,
``enable_background``) are injected at construction time. The ``attack_context``
slot carries a duck-typed handle that attackers may read further config from
(e.g. SAM2's ``style_gcn``); the pipeline itself does not interpret it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdversarialPipeline:
    """
    Manages adversarial attack execution for AUE training.

    Implements the cooperative attack strategy:
    1. Predict parameters for ALL attacks using CLEAN features (parallel)
    2. Apply transformations sequentially using predicted parameters

    This ensures stable gradient flow for joint min-max optimization.

    Args:
        attackers: ``{name: AdversarialAttacker}``. Names are the keys used in
            ``attack_order`` (e.g. ``"style"``, ``"deform"``).
        attack_order: Sequence specifying the order in which attacks apply.
            Must be a subset of ``attackers.keys()``.
        backbone_fn: Callable ``img -> {"backbone_fpn": [...]}``. The pipeline
            invokes it once after all transformations to re-encode the final
            adversarial image.
        use_high_res_features: If ``True``, the pipeline also reads
            ``backbone_fpn[0]`` and ``backbone_fpn[1]`` for high-res levels.
        enable_background: If ``True``, the deformation step treats the
            last channel of ``pixel_gt`` as a real object even when it covers
            more than half the image.
        attack_context: Duck-typed object passed through to attackers as their
            ``model`` argument. Attackers read attributes such as ``style_gcn``
            or ``style_adv_*`` flags off it. May be ``None`` if all attackers
            ignore the ``model`` argument.
    """

    def __init__(
        self,
        *,
        attackers: dict[str, nn.Module],
        attack_order: list[str],
        backbone_fn: Callable[..., dict[str, Any]],
        use_high_res_features: bool = False,
        enable_background: bool = False,
        attack_context: nn.Module | None = None,
    ):
        self.attackers = attackers
        self.attack_order = attack_order
        self.backbone_fn = backbone_fn
        self.use_high_res_features = use_high_res_features
        self.enable_background = enable_background
        self.attack_context = attack_context

    def generate_adversarial_samples(
        self,
        img_batch: torch.Tensor,
        backbone_features: torch.Tensor,
        high_res_features: list[torch.Tensor],
        pixel_gt: torch.Tensor,
        single_obj_gt: torch.Tensor | None = None,
        enable_vis: bool = False,
    ) -> dict[str, Any]:
        """
        Generate adversarial samples without forward pass or loss computation.

        This is a lightweight method that only applies adversarial attacks and returns
        the transformed inputs. The calling code is responsible for:
        1. Running forward pass on adversarial samples
        2. Running iterative refinement (reusing existing code)
        3. Computing task loss and calibration loss

        Args:
            img_batch: [B, 3, H, W] input images
            backbone_features: [B, C, H, W] backbone features (used for attack prediction)
            high_res_features: List of high-res features [stride 4, stride 8]
            pixel_gt: [B, K, H, W] ground truth masks (all objects, for attack generation)
            single_obj_gt: [B, 1, H, W] single object GT (same as clean branch, for SAM task)
            enable_vis: Whether to collect visualization data

        Returns:
            Dict containing:
                - adv_img: Adversarially transformed images
                - adv_features: New backbone features from transformed images
                - adv_high_res: New high-res features from transformed images
                - adv_pixel_gt: Possibly warped GT masks (multi-object, for attack)
                - adv_single_obj_gt: Warped single object GT (for SAM task, matches clean branch)
                - vis_refs: Visualization references (if enable_vis)
        """
        vis_refs = {}
        if enable_vis:
            vis_refs["img_batch"] = img_batch.detach().cpu()
            vis_refs["pixel_gt"] = pixel_gt.detach().cpu()

        # Initialize state for cooperative attack
        state = {
            "img": img_batch,
            "features": backbone_features,
            "high_res": high_res_features,
            "pixel_gt": pixel_gt,
            "single_obj_gt": single_obj_gt,  # Track single object GT separately
        }

        # Apply cooperative attack (parallel predict, sequential apply)
        state = self._apply_cooperative_attack(state, enable_vis, vis_refs)

        # Record attack order for visualization
        if enable_vis:
            vis_refs["attack_order"] = list(self.attack_order)
            vis_refs["adv_img"] = state["img"].detach().cpu()

        return {
            "adv_img": state["img"],
            "adv_features": state["features"],
            "adv_high_res": state.get("high_res"),
            "adv_pixel_gt": state["pixel_gt"],
            "adv_single_obj_gt": state.get("single_obj_gt"),  # Warped single object GT
            "adv_images_for_attacker": state.get("adv_images_for_attacker"),  # For attacker loss
            "deform_offsets": state.get("deform_offsets"),  # For prompt coordinate transformation
            "vis_refs": vis_refs,
        }

    def _apply_cooperative_attack(
        self,
        state: dict,
        enable_vis: bool,
        vis_refs: dict,
    ) -> dict:
        """
        Apply cooperative adversarial attack.

        Strategy:
        1. Predict parameters for all active attackers using CLEAN features.
        2. Apply all image transformations sequentially (NO intermediate backbone forward).
        3. Do a SINGLE backbone forward at the end on the final transformed image.

        Optimization: Removed redundant backbone forwards after each attack.
        Previous: 3x forward (clean + style + deform)
        Current: 2x forward (clean + final transformed)
        """
        aug_params = {}
        clean_features = state["features"].detach()
        pixel_gt = state["pixel_gt"]
        img_batch = state["img"]
        ctx = self.attack_context

        # === Phase 1: Predict (Parallel) ===
        # All predictions use CLEAN features (cooperative attack strategy)
        for aug_name in self.attack_order:
            attacker = self.attackers.get(aug_name)
            if attacker is not None:
                params = attacker.predict_params(
                    clean_features=clean_features,
                    pixel_gt=pixel_gt,
                    model=ctx,
                    img_batch=img_batch,
                )
                aug_params[aug_name] = params

        # === Phase 2: Apply Image Transformations (NO backbone forward here) ===
        any_transform_applied = False
        for aug_name in self.attack_order:
            if aug_name not in aug_params:
                continue

            attacker = self.attackers[aug_name]
            params = aug_params[aug_name]

            if attacker.mode == "image_level":
                # Image level attack (e.g. Style)
                # Save original styles BEFORE applying transform (for visualization)
                if enable_vis and attacker.aug_type == "style":
                    from ruac.modeling.style_utils import extract_gt_region_style

                    # extract_gt_region_style now always returns [B, K, 6]
                    orig_styles = extract_gt_region_style(state["img"], state["pixel_gt"])
                    vis_refs["original_styles"] = orig_styles.detach().cpu()

                styled_images = attacker.apply_transform(
                    img_batch=state["img"],
                    params=params,
                    pixel_gt=state["pixel_gt"],
                    model=ctx,
                )
                state["img"] = styled_images
                any_transform_applied = True

                # NOTE: Removed backbone forward here - will do once at the end

                if enable_vis:
                    vis_refs["styled_images"] = styled_images.detach().cpu()
                    if attacker.aug_type == "style":
                        vis_refs["adversarial_styles"] = params.detach().cpu()

            elif attacker.mode == "feature_level":
                # Feature level attack (e.g. Deform)
                offsets = params["image_offsets"] if isinstance(params, dict) else params

                warped_img, warped_gt, warped_single_gt = self._apply_deformation_to_images(
                    state["img"],
                    state["pixel_gt"],
                    offsets,
                    single_obj_gt=state.get("single_obj_gt"),
                    enable_vis=enable_vis,
                    vis_refs=vis_refs if enable_vis else {},
                )
                state["img"] = warped_img
                state["pixel_gt"] = warped_gt
                if warped_single_gt is not None:
                    state["single_obj_gt"] = warped_single_gt
                # Always save offsets for prompt coordinate transformation
                state["deform_offsets"] = offsets
                any_transform_applied = True

                # NOTE: Removed backbone forward here - will do once at the end

                if enable_vis:
                    vis_refs["deform_offsets"] = offsets.detach().cpu()
                    vis_refs.setdefault("warped_images", warped_img.detach().cpu())
                    vis_refs.setdefault("warped_pixel_gt", warped_gt.detach().cpu())

        # === Phase 3: Single Backbone Forward on Final Transformed Image ===
        # GRADIENT FLOW DESIGN:
        # - adv_images_for_attacker is saved BEFORE any processing
        # - Backbone forward WITHOUT detach to preserve gradient path for attacker
        # - Since backbone is frozen (freeze_image_encoder_epochs), no gradient conflict
        # - Gradient path: -attacker_loss -> pred_masks -> features -> adv_img -> params -> Attacker
        if any_transform_applied:
            # Save adversarial images (with gradient to attacker params via style/deform transforms)
            state["adv_images_for_attacker"] = state["img"]
            # Forward through backbone (keep gradient for attacker, use checkpoint for memory)
            backbone_out = self.backbone_fn(state["img"], use_checkpoint=True)
            state["features"] = backbone_out["backbone_fpn"][-1]
            if self.use_high_res_features:
                state["high_res"] = [backbone_out["backbone_fpn"][0], backbone_out["backbone_fpn"][1]]

        return state

    def _apply_deformation_to_images(
        self,
        img_batch: torch.Tensor,
        pixel_gt: torch.Tensor,
        deform_offsets: torch.Tensor | None,
        single_obj_gt: torch.Tensor | None = None,
        enable_vis: bool = False,
        vis_refs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Apply deformation offsets to images and GT masks (batched vectorized).

        Optimization: Uses batched grid_sample to warp all objects in 2 GPU calls
        instead of 2*K calls (K = number of valid objects).

        Also warps single_obj_gt if provided (for SAM task consistency with clean branch).
        """
        if vis_refs is None:
            vis_refs = {}

        if deform_offsets is None:
            return img_batch, pixel_gt.clone(), single_obj_gt.clone() if single_obj_gt is not None else None

        B, K, _, H_img, W_img = deform_offsets.shape
        device = img_batch.device

        # Identify valid objects (non-empty, non-background)
        masks_float = (pixel_gt > 0.5).float()
        mask_areas = masks_float.sum(dim=(2, 3))
        is_empty = mask_areas.sum(dim=0) == 0
        mask_area_ratio = mask_areas / (H_img * W_img)
        is_bg_per_sample = mask_area_ratio > 0.5

        include_background = bool(self.enable_background)
        is_bg = torch.zeros(K, dtype=torch.bool, device=device)
        if K > 0:
            bg_candidate = is_bg_per_sample[:, -1].all()
            is_bg[-1] = include_background or bg_candidate
        valid_objects = ~(is_empty | is_bg)
        valid_indices = torch.where(valid_objects)[0]
        K_valid = len(valid_indices)

        if K_valid == 0:
            return img_batch, pixel_gt.clone(), single_obj_gt.clone() if single_obj_gt is not None else None

        # Remove valid objects from base image
        valid_masks = masks_float[:, valid_indices, :, :]  # [B, K_valid, H, W]
        valid_masks_union = valid_masks.sum(dim=1, keepdim=True).clamp(0, 1)
        base_img = img_batch * (1 - valid_masks_union)

        # ===== BATCHED WARPING OPTIMIZATION =====
        # Instead of K separate grid_sample calls, we do 2 batched calls:
        # 1. Warp images: [B*K_valid, 3, H, W]
        # 2. Warp masks: [B*K_valid, 1, H, W]

        # Gather offsets for valid objects: [B, K_valid, 2, H, W]
        valid_offsets = deform_offsets[:, valid_indices, :, :, :]

        # Reshape for batched processing
        # Images: expand [B, 3, H, W] -> [B, K_valid, 3, H, W] -> [B*K_valid, 3, H, W]
        img_expanded = img_batch.unsqueeze(1).expand(-1, K_valid, -1, -1, -1)
        img_flat = img_expanded.reshape(B * K_valid, 3, H_img, W_img)

        # Offsets: [B, K_valid, 2, H, W] -> [B*K_valid, 2, H, W]
        offsets_flat = valid_offsets.reshape(B * K_valid, 2, H_img, W_img)

        # Masks: [B, K_valid, H, W] -> [B*K_valid, 1, H, W]
        masks_flat = valid_masks.reshape(B * K_valid, 1, H_img, W_img)

        # Build sample grids once for all objects (batched)
        sample_grids = self._build_sample_grids_batched(offsets_flat, H_img, W_img)

        # Batched warp: 2 grid_sample calls instead of 2*K_valid
        warped_imgs_flat = F.grid_sample(img_flat, sample_grids, mode="bilinear", padding_mode="border", align_corners=True)  # [B*K_valid, 3, H, W]

        warped_masks_flat = F.grid_sample(masks_flat, sample_grids, mode="bilinear", padding_mode="border", align_corners=True)  # [B*K_valid, 1, H, W]

        # Reshape back: [B*K_valid, C, H, W] -> [B, K_valid, C, H, W]
        warped_imgs = warped_imgs_flat.reshape(B, K_valid, 3, H_img, W_img)
        warped_masks = warped_masks_flat.reshape(B, K_valid, 1, H_img, W_img)

        # Binarize masks after warping
        warped_masks_bin = (warped_masks > 0.5).float()  # [B, K_valid, 1, H, W]

        # Apply masks to get per-object warped images
        warped_objs = warped_imgs * warped_masks_bin  # [B, K_valid, 3, H, W]

        # ===== COMPOSITING (already vectorized) =====
        # mask_stack: [B, K_valid, 1, H, W]
        mask_stack = warped_masks_bin
        sum_mask = mask_stack.sum(dim=1).clamp(0, 1)  # [B, 1, H, W]
        background = base_img * (1 - sum_mask)

        overlap_count = mask_stack.sum(dim=1)  # [B, 1, H, W]
        foreground_sum = (warped_objs * mask_stack).sum(dim=1)  # [B, 3, H, W]
        foreground = foreground_sum / overlap_count.clamp(min=1.0)

        augmented_img = background + foreground

        # ===== UPDATE PIXEL_GT (vectorized scatter) =====
        warped_pixel_gt = pixel_gt.clone()
        # warped_masks_bin: [B, K_valid, 1, H, W] -> squeeze to [B, K_valid, H, W]
        warped_masks_squeezed = warped_masks_bin.squeeze(2)
        # Scatter warped masks back to their original positions
        for i, k_idx in enumerate(valid_indices):
            warped_pixel_gt[:, k_idx, :, :] = warped_masks_squeezed[:, i, :, :]

        if enable_vis:
            vis_refs["warped_images"] = augmented_img.detach().cpu()
            vis_refs["warped_pixel_gt"] = warped_pixel_gt.detach().cpu()

        # === WARP SINGLE OBJECT GT (for SAM task) ===
        # Use the combined offset field to warp the single object mask consistently
        warped_single_obj_gt = None
        if single_obj_gt is not None:
            # single_obj_gt: [B, 1, H, W]
            # Use a weighted average offset based on overlap with valid objects.
            # For simplicity, use the offset from the object that has most overlap with single_obj_gt.
            single_mask = (single_obj_gt > 0.5).float()  # [B, 1, H, W]

            # Find which valid object has most overlap with single_obj_gt per sample
            overlaps = (single_mask * valid_masks).sum(dim=(2, 3))  # [B, K_valid]
            best_obj_idx = overlaps.argmax(dim=1)  # [B]

            # Gather the corresponding offsets: [B, 2, H, W]
            batch_indices = torch.arange(B, device=device)
            selected_offsets = valid_offsets[batch_indices, best_obj_idx]  # [B, 2, H, W]

            # Build sample grids and warp single_obj_gt
            single_sample_grids = self._build_sample_grids_batched(selected_offsets, H_img, W_img)
            warped_single = F.grid_sample(
                single_obj_gt.float(),
                single_sample_grids,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            warped_single_obj_gt = (warped_single > 0.5).float()  # Binarize

        # Cleanup intermediate tensors
        del masks_float, img_expanded, img_flat, offsets_flat, masks_flat
        del sample_grids, warped_imgs_flat, warped_masks_flat
        del warped_imgs, warped_masks, warped_masks_bin, warped_objs

        return augmented_img, warped_pixel_gt, warped_single_obj_gt

    def _build_sample_grids_batched(
        self,
        offset_fields: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """
        Build sample grids for batched grid_sample.

        Args:
            offset_fields: [N, 2, H, W] Offset fields (dx, dy) for N samples
            H, W: Spatial dimensions

        Returns:
            sample_grids: [N, H, W, 2] Sample grids for F.grid_sample
        """
        N = offset_fields.shape[0]
        device = offset_fields.device
        dtype = offset_fields.dtype

        # Create base grid once (shared across all samples)
        y_coords = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        x_coords = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        base_grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        base_grid = base_grid.unsqueeze(0).expand(N, -1, -1, -1)  # [N, H, W, 2]

        # Normalize offsets to [-1, 1] range
        offset_normalized = offset_fields.clone()
        offset_normalized[:, 0] = offset_fields[:, 0] / (W / 2)  # x offset
        offset_normalized[:, 1] = offset_fields[:, 1] / (H / 2)  # y offset
        offset_normalized = offset_normalized.permute(0, 2, 3, 1)  # [N, H, W, 2]

        # Combine base grid with offsets
        sample_grids = base_grid + offset_normalized

        return sample_grids
