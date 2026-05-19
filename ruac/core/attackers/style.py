# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Style adversarial attack — image-level AdaIN with GRL.

The style attacker predicts a 6-dim ``(mean, std)`` perturbation for each
object, applies it as Mask-Aware AdaIN on the input image, and reverses the
gradient so the predictor is trained to maximise the downstream loss in the
SAME backward pass.

The optional GCN refinement coordinates style deltas between spatially or
semantically related objects. Both the GCN module (``model.style_gcn``) and
its hyperparameters are read off the model attribute surface that the SAM2
adapter sets up.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from ruac.core.grl import GRL
from ruac.modeling.style_gcn import build_object_graph
from ruac.modeling.style_utils import extract_gt_region_style, extract_style_statistics


class ImageLevelStyleImpl(nn.Module):
    """
    Image-level style augmentation implementation with GRL (self-contained).

    Applies AdaIN-based style transfer on images using neural network + GRL.
    Replaces PGD-based optimization for memory efficiency.

    Note: This requires an additional backbone forward pass.
    All style-related methods are now self-contained within this class.
    """

    def __init__(
        self,
        epsilon: float = 2.0,
        use_multi_object: bool = False,
        use_gcn: bool = False,
        use_gt_region_style: bool = False,
        enable_background: bool = False,
        use_global_local_mix: bool = False,
        global_epsilon: float = 1.5,
        global_weight: float = 0.7,
        feature_dim: int = 256,
        num_objects: int = 11,
        **kwargs,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.use_multi_object = use_multi_object
        self.use_gcn = use_gcn
        self.use_gt_region_style = use_gt_region_style
        self.enable_background = enable_background
        self.use_global_local_mix = use_global_local_mix
        self.global_epsilon = global_epsilon
        self.global_weight = global_weight

        # Create adversarial style network with GRL.
        # If GCN is used, internal GRL is disabled and applied after GCN refinement
        # so both networks receive reversed gradients (collaborative attack).
        self.style_net = StyleAdversarialNetwork(
            feature_dim=feature_dim,
            num_objects=num_objects,
            epsilon=epsilon,
            use_grl=not use_gcn,
        )

        # The actual GCN module lives on the adapter (model.style_gcn) and is
        # accessed via the ``model`` argument in ``predict_params``. We keep
        # this attribute only for API symmetry.
        self.style_gcn = None

    def predict_params(self, clean_features: torch.Tensor, pixel_gt: torch.Tensor, model: nn.Module, img_batch: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """
        Predict adversarial style parameters.

        Args:
            clean_features: [B, C, H, W] Clean backbone features
            pixel_gt: [B, K, H, W] Ground truth masks
            model: adapter-side module exposing ``style_gcn`` + the
                ``style_adv_*`` config attributes; duck-typed (any ``nn.Module``
                with the expected attribute surface).
            img_batch: [B, 3, H, W] Input images (needed for original style extraction)

        Returns:
            adv_styles: [B, K, 6] Adversarial style parameters
        """
        if img_batch is None:
            raise ValueError("ImageLevelStyleImpl.predict_params requires img_batch")

        clean_features_detached = clean_features.detach()

        # 1. Prepare inputs (extract original styles)
        pixel_gt_normalized, original_styles = self._prepare_style_adversary_inputs(img_batch, pixel_gt, model)

        # 2. Predict adversarial styles using neural network + GRL.
        # Detach clean_features to prevent gradient fighting:
        # the backbone should not receive gradients from the GRL that attempt
        # to maximize loss; it should only update on the final loss
        # minimization over the adversarial example.
        adv_styles = self.style_net(clean_features_detached, original_styles, pixel_gt=pixel_gt_normalized)

        # 3. Optional GCN refinement for multi-object coordination
        if self.use_gcn and model.style_gcn is not None:
            # Compute style delta
            style_delta = adv_styles - original_styles

            # Build object graph (img_batch needed for future semantic features)
            # CRITICAL: Detach clean_features to prevent GRL gradients flowing to backbone
            with torch.no_grad():
                edge_index, edge_weight, _ = self._build_object_graph(
                    pixel_gt_normalized,
                    img_batch,
                    clean_features_detached,
                    model,
                )

            # Extract mask features if GCN uses visual features
            mask_features = None
            if model.style_gcn.feature_dim > 0 and clean_features is not None:
                # Explicitly detach to prevent gradient flow from GCN back to backbone
                with torch.no_grad():
                    mask_features = self._extract_mask_features(
                        clean_features_detached,
                        pixel_gt_normalized,
                    )
                mask_features = mask_features.detach()

            # Refine delta using GCN
            if edge_index is not None:
                refined_delta = model.style_gcn(
                    style_delta,
                    edge_index,
                    edge_weight,
                    mask_features=mask_features,
                )
                # Clip refined delta to epsilon ball
                refined_delta = torch.clamp(refined_delta, -self.epsilon, self.epsilon)

                # Apply GRL to the refined delta (since it was skipped in style_net)
                # This ensures gradients flow: Loss -> GRL(neg) -> GCN -> delta -> style_net
                # Both GCN and style_net update to maximize loss.
                grl = GRL()
                refined_delta = grl(refined_delta)

                adv_styles = original_styles + refined_delta

        return adv_styles

    def apply_transform(self, img_batch: torch.Tensor, params: torch.Tensor, model: nn.Module, pixel_gt: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """
        Apply style transformation to images.

        Args:
            img_batch: [B, 3, H, W] Input images (must preserve gradients!)
            params: [B, K, 6] Adversarial style parameters (already passed through GRL)
            model: adapter-side module exposing ``style_adv_use_gt_region_style`` flag.
            pixel_gt: [B, K, H, W] Ground truth masks (optional, for region-based style)

        Returns:
            styled_images: [B, 3, H, W] Styled images with gradients preserved

        Note:
            - params have already been processed by GRL in predict_params
            - img_batch MUST keep gradients for backbone training
        """
        # Determine if we should use GT region style
        use_gt_region = self.use_gt_region_style
        if model is not None:
            use_gt_region = model.style_adv_use_gt_region_style

        # Prepare mask if needed
        apply_mask = None
        if use_gt_region and pixel_gt is not None:
            # Normalize pixel_gt if needed (similar to _prepare_style_adversary_inputs logic)
            if pixel_gt.ndim == 4 and pixel_gt.shape[1] > 1:
                pass
            apply_mask = pixel_gt

        styled_images = self._apply_style_to_images(img_batch, params, gt_mask=apply_mask)
        return styled_images

    def _prepare_style_adversary_inputs(
        self,
        img_batch: torch.Tensor,
        pixel_gt: torch.Tensor,
        model: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare inputs for style-based adversarial generation.

        Args:
            img_batch: [B, 3, H, W] input images (may have gradients from deform AUE)
            pixel_gt: [B, K, H, W] ground truth masks
            model: adapter-side module exposing ``style_adv_use_gt_region_style``.

        Returns:
            pixel_gt: [B, K, H, W] normalized GT masks (possibly merged if single-object)
            original_styles: [B, K, 6] original style statistics (detached)
        """
        # Ensure 4D: [B, K, H, W]
        if pixel_gt.ndim == 3:
            pixel_gt = pixel_gt.unsqueeze(1)

        # Enforce single-object mode if requested (merge all masks)
        if not self.use_multi_object and pixel_gt.shape[1] > 1:
            pixel_gt = pixel_gt.sum(dim=1, keepdim=True).clamp(0, 1)  # [B, 1, H, W]
        else:
            # In multi-object mode, drop background channel when background attacks are disabled.
            # Heuristic: the last channel is background if it covers the majority of pixels.
            if not self.enable_background and pixel_gt.shape[1] > 1:
                mask_area_ratio = pixel_gt.float().mean(dim=(2, 3))  # [B, K]
                bg_is_last = (mask_area_ratio[:, -1] > 0.5).all()
                if bg_is_last:
                    pixel_gt = pixel_gt[:, :-1]  # remove background channel

        B, K, H, W = pixel_gt.shape

        # Extract all objects' styles (vectorized)
        # CRITICAL: Detach to break gradient flow from deform_augmenter
        # Style PGD should only optimize style parameters, not deform offsets
        if model.style_adv_use_gt_region_style:
            # extract_gt_region_style now always returns [B, K, 6]
            original_styles = extract_gt_region_style(img_batch.detach(), pixel_gt)
        else:
            # Global style: extract_style_statistics now returns [B, 1, 6]
            global_style = extract_style_statistics(img_batch.detach())
            original_styles = global_style.expand(-1, K, -1)  # [B, 1, 6] -> [B, K, 6]

        return pixel_gt, original_styles

    def _apply_style_to_images(
        self,
        img_batch: torch.Tensor,
        style_stats: torch.Tensor | None,
        gt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply style statistics to images using Mask-Aware AdaIN.

        Normalization is tied to the specific object region: source statistics
        are computed on the masked area so that ``Tgt ≈ Src`` reduces to the
        identity, preserving consistency between style prediction and
        application.

        Args:
            img_batch: [B, 3, H, W] normalized images
            style_stats: [B, K, 6] or [B, 6] style statistics per object
            gt_mask: [B, K_mask, H, W] GT masks (optional)

        Returns:
            styled_images: [B, 3, H, W] styled images
        """
        # If no style stats provided, return original images
        if style_stats is None:
            return img_batch

        B, C, H, W = img_batch.shape

        # Handle backward compatibility for [B, 6] input
        if style_stats.ndim == 2:
            style_stats = style_stats.unsqueeze(1)  # [B, 1, 6]

        K = style_stats.shape[1]

        # 1. Mask Alignment & Source Statistics Extraction
        source_stats = None

        if gt_mask is not None:
            # Ensure mask acts as a float binary mask
            if gt_mask.shape[2:] != (H, W):
                gt_mask = F.interpolate(gt_mask.float(), size=(H, W), mode="nearest")
            gt_mask = (gt_mask > 0.5).float()

            # Align mask channels with style stats (handle drop/merge from predict pipeline)
            if gt_mask.shape[1] != K:
                if K == 1:
                    # Single-object style applied to Multi-object mask -> Merge all objects
                    gt_mask = gt_mask.max(dim=1, keepdim=True)[0]
                elif gt_mask.shape[1] - 1 == K:
                    # Background drop detected (style has 1 less channel) -> Drop last channel
                    gt_mask = gt_mask[:, :-1]
                else:
                    # Fallback for unexpected mismatch: wrap or slice safely
                    logging.warning(f"Style/Mask channel mismatch: Style={K}, Mask={gt_mask.shape[1]}. Slicing mask.")
                    gt_mask = gt_mask[:, :K]

            # Compute Source Stats using the aligned mask
            # CRITICAL: Compute on img_batch (with gradient) to support full differentiability
            # extract_gt_region_style handles broadcasting and safe division
            # We use min_pixels=100 to MATCH the default used in 'predict_params' (_prepare_style_adversary_inputs)
            # This ensures that fallback-to-global decisions are identical between prediction and application,
            # preventing Global-vs-Local mismatch for small objects.
            source_stats = extract_gt_region_style(img_batch, gt_mask, min_pixels=100)  # [B, K, 6]

        else:
            # Global Application (no mask provided)
            source_stats = extract_style_statistics(img_batch)  # [B, 1, 6]
            if K > 1:
                source_stats = source_stats.expand(-1, K, -1)
            # Create dummy full-image masks for composition loop
            gt_mask = torch.ones(B, K, H, W, device=img_batch.device)

        # 2. Iterate and Apply Style per Object
        styled_regions = []
        masks_list = []

        for k in range(K):
            target_style = style_stats[:, k]  # [B, 6]
            source_style = source_stats[:, k]  # [B, 6]
            mask_k = gt_mask[:, k : k + 1]  # [B, 1, H, W]

            # Extract Means and Stds
            src_mean = source_style[:, :3].view(B, 3, 1, 1)
            src_std = source_style[:, 3:].view(B, 3, 1, 1)
            tgt_mean = target_style[:, :3].view(B, 3, 1, 1)
            tgt_std = target_style[:, 3:].view(B, 3, 1, 1)

            # Mask-Aware AdaIN: (x - mu_src) / sigma_src * sigma_tgt + mu_tgt
            # Using specific source stats ensures that if Tgt ~= Src, result ~= Input (Identity)
            # We add 1e-6 to std for numerical stability
            normalized = (img_batch - src_mean) / (src_std + 1e-6)
            object_styled = normalized * tgt_std + tgt_mean

            # Clamp output to safe image range
            object_styled = object_styled.clamp(min=-3.0, max=3.0)

            styled_regions.append(object_styled)
            masks_list.append(mask_k)

        # 3. Composition
        # Compose objects onto the original image
        # Using simple overwriting (Painter's Algorithm) based on the channel order
        styled_images = img_batch.clone()

        for k in range(K):
            m = masks_list[k]
            # Hard composition masked by region
            styled_images = m * styled_regions[k] + (1 - m) * styled_images

        return styled_images

    def _extract_mask_features(
        self,
        backbone_features: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract visual features for each mask region via masked average pooling.

        Args:
            backbone_features: [B, C, H, W] features from backbone_fpn[-1]
            masks: [B, K, H, W] binary masks (or [B, K, 1, H, W])

        Returns:
            mask_features: [B, K, C] per-mask visual features
        """
        # Handle 5D mask input
        if masks.ndim == 5:
            masks = masks.squeeze(2)  # [B, K, 1, H, W] → [B, K, H, W]

        B, C, fH, fW = backbone_features.shape
        _, K, mH, mW = masks.shape

        # Resize masks to match feature map size if needed
        if (mH, mW) != (fH, fW):
            masks_resized = F.interpolate(masks, size=(fH, fW), mode="bilinear", align_corners=False)
        else:
            masks_resized = masks

        # Binarize masks
        masks_binary = (masks_resized > 0.5).float()  # [B, K, fH, fW]

        # Compute masked average pooling for each mask
        mask_features = []
        for k in range(K):
            mask_k = masks_binary[:, k : k + 1, :, :]  # [B, 1, fH, fW]
            mask_area = mask_k.sum(dim=(2, 3), keepdim=True)  # [B, 1, 1, 1]
            mask_area = mask_area + 1e-6

            # Weighted average of features in mask region
            masked_feat = (backbone_features * mask_k).sum(dim=(2, 3))  # [B, C]
            feat_k = masked_feat / mask_area.view(B, 1)  # [B, C] / [B, 1] -> [B, C]
            mask_features.append(feat_k)

        # Stack features: [K, B, C] → [B, K, C]
        mask_features = torch.stack(mask_features, dim=1)  # [B, K, C]

        return mask_features

    def _build_object_graph(
        self,
        pixel_gt: torch.Tensor | None,
        img_batch: torch.Tensor,
        backbone_features: torch.Tensor | None,
        model: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Build object graph for GCN refinement.

        Args:
            pixel_gt: [B, K, H, W] ground truth masks
            img_batch: [B, 3, H, W] images (for future semantic features)
            backbone_features: [B, C, H, W] backbone features for extracting visual features
            model: adapter-side module exposing ``style_adv_gcn_*`` and
                ``adv_enable_background`` config attributes.

        Returns:
            edge_index: [2, E] edge indices
            edge_weight: [E] edge weights
        """
        if backbone_features is not None and backbone_features.requires_grad:
            backbone_features = backbone_features.detach()

        if pixel_gt is not None and logging.getLogger().isEnabledFor(logging.DEBUG):
            # DEBUG: Log detailed pixel_gt info (avoid any work unless debug is enabled)
            per_channel_sum = [pixel_gt[:, k].sum().item() for k in range(min(pixel_gt.shape[1], 5))]
            per_channel_nonzero = [(pixel_gt[:, k] > 0.5).sum().item() for k in range(min(pixel_gt.shape[1], 5))]
            logging.debug(
                f"DEBUG _build_style_graph: pixel_gt.shape={pixel_gt.shape}, "
                f"dtype={pixel_gt.dtype}, device={pixel_gt.device}, "
                f"min={pixel_gt.min():.3f}, max={pixel_gt.max():.3f}, "
                f"per_channel_sum[0:5]={per_channel_sum}, "
                f"per_channel_nonzero[0:5]={per_channel_nonzero}"
            )

            # Debug: Check input masks
            mask_areas = (pixel_gt > 0.5).float().sum(dim=(2, 3))  # [B, K]
            valid_masks = (mask_areas > 0).sum().item()
            logging.debug(
                f"GCN input: pixel_gt.shape={pixel_gt.shape}, valid_masks={valid_masks}/{pixel_gt.shape[0] * pixel_gt.shape[1]}, "
                f"edge_thresh={model.style_adv_gcn_edge_threshold}, dist_thresh={model.style_adv_gcn_distance_threshold}, "
                f"use_bg={model.adv_enable_background and model.style_adv_gcn_use_background_edges}"
            )

        # Extract mask features if visual features are enabled
        # Features serve two purposes:
        # 1. Build semantic edges in graph (via cosine similarity)
        # 2. Fuse with style deltas in GCN (via MLP projection)
        mask_features_for_graph = None
        if model.style_gcn is not None and model.style_gcn.feature_dim > 0 and backbone_features is not None and pixel_gt is not None:
            with torch.no_grad():
                mask_features_for_graph = self._extract_mask_features(backbone_features, pixel_gt)  # [B, K, 256]

        # Build graph structure using visual features for semantic edges
        edge_index, edge_weight, stats = build_object_graph(
            pixel_gt,
            img_batch,
            edge_threshold=model.style_adv_gcn_edge_threshold,
            use_semantic=model.style_adv_gcn_use_semantic_edges,
            use_background=(model.adv_enable_background and model.style_adv_gcn_use_background_edges),
            distance_threshold=model.style_adv_gcn_distance_threshold,
            use_boundary_distance=model.style_adv_gcn_use_boundary_distance,
            mask_features=mask_features_for_graph,  # Used to build semantic edges
            feature_sim_threshold=model.style_adv_gcn_feature_sim_threshold,
        )

        # Add self-loops to the graph
        num_nodes_total = pixel_gt.shape[0] * pixel_gt.shape[1]  # B * K
        edge_index, edge_weight = model.style_gcn._add_self_loops(edge_index, edge_weight, num_nodes_total)

        model._latest_gcn_stats = stats if stats else None
        if stats and stats.get("graphs", 0) == 0:
            # Check if pixel_gt has any content
            if pixel_gt is not None:
                valid_pixels = (pixel_gt > 0.5).sum().item()
                logging.warning(
                    f"GCN graph built but NO edges: graphs={stats['graphs']}, nodes_fg={stats['nodes_foreground']}, "
                    f"nodes_bg={stats['nodes_background']}, edges_iou={stats['edges_iou']}, edges_dist={stats['edges_distance']}, "
                    f"edges_bg={stats['edges_background']}, edges_semantic={stats.get('edges_semantic', 0)}, valid_pixels={valid_pixels}"
                )
            else:
                logging.warning("GCN graph built but NO edges (pixel_gt is None)")
        elif stats:
            # Show edge type breakdown for non-empty graphs
            logging.debug(
                f"GCN graph built: {stats['graphs']:.0f} graphs, {stats['edges_total']:.0f} edges "
                f"(IoU:{stats['edges_iou']:.0f}, Dist:{stats['edges_distance']:.0f}, Semantic:{stats.get('edges_semantic', 0):.0f}, BG:{stats['edges_background']:.0f}), "
                f"nodes: {stats['nodes_foreground']:.0f}fg+{stats['nodes_background']:.0f}bg, "
                f"avg_degree: {stats['avg_degree']:.2f}"
            )
        else:
            logging.debug("GCN graph build returned None")
        return edge_index, edge_weight, stats


class StyleAdversarialNetwork(nn.Module):
    """
    Neural network that predicts adversarial style transformations with GRL.

    Architecture:
        Features + Masks -> Mask-Aware Pooling -> Object Features -> Shared MLP -> Style Params -> GRL

    Args:
        feature_dim: Feature dimension (default: 256)
        epsilon: Max style perturbation magnitude (default: 2.0)
        use_grl: Whether to apply Gradient Reversal Layer (default: True)
    """

    def __init__(self, feature_dim: int = 256, epsilon: float = 2.0, use_grl: bool = True, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.epsilon = epsilon
        self.use_grl = use_grl

        # Shared MLP: [B, K, C] -> [B, K, 6]
        # LayerNorm normalizes across feature dimension (C), ensuring consistent
        # style residual magnitudes regardless of object size or feature statistics
        self.object_mlp = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6),
        )

        # Initialize ALL layers with small weights for near-zero initial output.
        # This prevents overly strong attacks at the start of training.
        for m in self.object_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        logging.info("StyleAdversarialNetwork: Initialized all linear layers with std=0.01 for near-zero initial output")

        self.grl = GRL()

        # Track if we've logged initial output (for debugging)
        self._logged_initial_output = False

    def forward(
        self,
        features: torch.Tensor,
        original_styles: torch.Tensor,
        pixel_gt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, C, H, W = features.shape
        K_actual = original_styles.shape[1]

        if pixel_gt is not None:
            # CRITICAL: Binarize masks to prevent numerical instability from soft values
            pixel_gt_binary = (pixel_gt > 0.5).float()
            if pixel_gt_binary.shape[-2:] != (H, W):
                masks = F.interpolate(pixel_gt_binary, size=(H, W), mode="nearest")
            else:
                masks = pixel_gt_binary

            # Efficient Masked Pooling
            flat_features = features.flatten(2).transpose(1, 2)  # [B, N, C]
            flat_masks = masks.flatten(2)  # [B, K, N]

            # Use larger clamp value to prevent division instability
            mask_sums = flat_masks.sum(dim=2, keepdim=True).clamp(min=1.0)
            flat_masks_norm = flat_masks / mask_sums

            object_features = torch.bmm(flat_masks_norm, flat_features)  # [B, K, C]

        else:
            # Fallback: Global pooling
            global_feat = features.mean(dim=[2, 3])  # [B, C]
            object_features = global_feat.unsqueeze(1).expand(-1, K_actual, -1)

        style_residuals = self.object_mlp(object_features)

        # Log initial output magnitude once (verify near-zero initialization)
        if not self._logged_initial_output:
            logging.info(
                f"StyleNetwork initial output: mean={style_residuals.mean().item():.6f}, "
                f"std={style_residuals.std().item():.6f}, "
                f"min={style_residuals.min().item():.6f}, max={style_residuals.max().item():.6f}"
            )
            self._logged_initial_output = True

        # Apply GRL if enabled
        if self.use_grl:
            style_residuals_adv = self.grl(style_residuals)
        else:
            style_residuals_adv = style_residuals

        # Handle shape mismatch if pixel_gt K != original_styles K
        if style_residuals_adv.shape[1] != K_actual:
            if style_residuals_adv.shape[1] > K_actual:
                style_residuals_adv = style_residuals_adv[:, :K_actual, :]
            else:
                pad_k = K_actual - style_residuals_adv.shape[1]
                style_residuals_adv = F.pad(style_residuals_adv, (0, 0, 0, pad_k))

        # Relative encoding: all transformations are controlled by epsilon for
        # proper regularization. epsilon=0.0001 -> minimal perturbation;
        # epsilon=0.1 -> stronger perturbation. The 6-dim output splits into
        # scale (first 3) and shift (last 3).
        raw_scale = style_residuals_adv[:, :, :3]  # For multiplicative factor
        raw_shift = style_residuals_adv[:, :, 3:]  # For additive factor

        # Scale factor controlled by epsilon:
        # epsilon=0.0001 -> scale_range=0.02 -> [0.98, 1.02] (+-2% brightness)
        # epsilon=0.01   -> scale_range=0.20 -> [0.80, 1.20] (+-20% brightness)
        # epsilon=0.1    -> scale_range=0.20 -> [0.80, 1.20] (capped at +-20%)
        scale_range = min(0.2, self.epsilon * 200)
        scale_factor = 1.0 - scale_range + 2 * scale_range * torch.sigmoid(raw_scale)

        # Shift factor: tanh outputs [-1, 1], scale by epsilon for small shifts.
        # Max shift is +-epsilon (e.g., +-0.5 for epsilon=0.5)
        shift_factor = self.epsilon * torch.tanh(raw_shift)  # [-epsilon, +epsilon]

        # Apply relative transformation to original styles
        original_means = original_styles[:, :, :3]  # [B, K, 3]
        original_stds = original_styles[:, :, 3:]  # [B, K, 3]

        # Means: scale + shift (both now bounded by epsilon)
        adv_means = original_means * scale_factor + shift_factor

        # Std scale controlled by epsilon:
        # epsilon=0.0001 -> std_range=0.05 -> [0.95, 1.05] (+-5% contrast)
        # epsilon=0.01   -> std_range=0.50 -> [0.50, 1.50] (+-50% contrast)
        # epsilon=0.1    -> std_range=0.50 -> [0.50, 1.50] (capped at +-50%)
        std_range = min(0.5, self.epsilon * 500)
        std_scale = 1.0 - std_range + 2 * std_range * torch.sigmoid(raw_scale)
        adv_stds = original_stds * std_scale
        # Safety: ensure stds stay positive
        adv_stds = adv_stds.clamp(min=0.1)

        adv_styles_out = torch.cat([adv_means, adv_stds], dim=2)

        return adv_styles_out
