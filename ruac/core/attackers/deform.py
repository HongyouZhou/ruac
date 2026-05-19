# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Deformation adversarial attack — DG-Font-style feature warp.

A small per-object offset predictor sits on top of fused (image, mask)
features and emits a dense ``(Δx, Δy)`` field. The GRL flips the sign of its
gradients so the predictor is trained to maximise the downstream task loss
while the rest of the model minimises it.

The mask encoder / projection / fuser modules borrow the *shape* of SAM2's
memory encoder building blocks, but the classes themselves are supplied by
the adapter at construction time (``mask_downsampler_cls``, ``fuser_cls``,
``cxblock_cls``). This keeps ``ruac.core`` free of any SAM2 import.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from ruac.core.grl import GRL


class FeatureLevelDeformationImpl(nn.Module):
    """
    Feature-level deformation using memory encoder components.

    Architecture:
        1. MaskDownSampler: Encode masks to feature space [B, 256, H, W]
        2. Image feature projection: Project image features
        3. Fusion: Combine mask and image features (memory encoder style)
        4. Offset prediction: Generate deformation offsets from fused features

    Args:
        feature_dim: Backbone feature dimension (default: 256)
        epsilon: Max deformation magnitude in feature space (default: 0.15)
        use_soft_composite: Use soft compositing for multi-object overlaps
        temperature: Softmax temperature for soft compositing
        use_gcn: Use GCN for multi-object coordination (not implemented yet)
        gcn_num_layers: Number of GCN layers
        num_deform_groups: Deformable convolution groups
        init_from_memory_encoder: Initialize weights from memory encoder
        freeze_encoder_components: Freeze all encoder components (mask_encoder, img_feat_proj, fuser)
        image_size: Target image resolution (default: 1024)
        mask_downsampler_cls: Class to instantiate for the mask down-sampler
            (e.g. ``sam2.modeling.memory_encoder.MaskDownSampler``). Required
            when ``init_from_memory_encoder=True``.
        fuser_cls: Class to instantiate for the feature fuser. Required when
            ``init_from_memory_encoder=True``.
        cxblock_cls: ConvNeXt-style block class used inside the fuser.
            Required when ``init_from_memory_encoder=True``.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        epsilon: float = 0.15,
        use_soft_composite: bool = True,
        temperature: float = 1.0,
        use_multi_object: bool = False,
        use_gcn: bool = False,
        gcn_num_layers: int = 2,
        num_deform_groups: int = 4,
        init_from_memory_encoder: bool = True,
        freeze_encoder_components: bool = False,
        image_size: int = 1024,
        zero_mean_offsets: bool = False,
        local_offset_gain: float = 1.0,
        mask_downsampler_cls: type | None = None,
        fuser_cls: type | None = None,
        cxblock_cls: type | None = None,
        **kwargs,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.epsilon = epsilon
        self.use_soft_composite = use_soft_composite
        self.temperature = temperature
        self.use_multi_object = use_multi_object
        self.use_gcn = use_gcn
        self.init_from_memory_encoder = init_from_memory_encoder
        self.freeze_encoder_components = freeze_encoder_components

        # Memory-encoder building blocks come in via DI so ruac.core stays
        # free of model-specific imports. The adapter (e.g. SAM2) passes the
        # concrete classes.
        if mask_downsampler_cls is None or fuser_cls is None or cxblock_cls is None:
            raise ValueError(
                "FeatureLevelDeformationImpl requires mask_downsampler_cls, fuser_cls and cxblock_cls. "
                "These should be passed in by the adapter (e.g. SAM2 adapter passes the SAM2 memory-encoder classes)."
            )

        # 1. Mask encoder (memory-encoder mask_downsampler shape)
        self.mask_encoder = mask_downsampler_cls(
            embed_dim=feature_dim,
            kernel_size=3,
            stride=2,
            padding=1,
            total_stride=16,  # 1024 -> 64
        )

        # 2. Image feature projection (memory-encoder pix_feat_proj)
        self.img_feat_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)

        # 3. Feature fusion module (memory-encoder fuser)
        self.fuser = fuser_cls(
            layer=cxblock_cls(dim=feature_dim, kernel_size=7, padding=3, layer_scale_init_value=1e-6, use_dwconv=True),
            num_layers=2,
        )

        # 4. Deformation module (uses fused features)
        # Note: Produces both feature-level deformation and image-level offsets
        self.deform_module = FeatureBasedDeformModule(
            feature_dim=feature_dim,
            epsilon=epsilon,
            image_size=image_size,
            zero_mean_offsets=zero_mean_offsets,
            local_offset_gain=local_offset_gain,
            use_grl=not use_gcn,  # Disable internal GRL if GCN is used (to apply it after GCN)
        )

        # 5. Soft compositor for multi-object overlaps
        if self.use_soft_composite:
            self.compositor = SoftCompositor(temperature=temperature)

        # 6. Optional GCN (placeholder)
        if self.use_gcn:
            logging.warning("GCN coordination not implemented yet")
            self.deform_gcn = None

    def load_memory_encoder_weights(self, memory_encoder):
        """
        Initialize weights from pretrained memory encoder.

        Copies weights from:
            - memory_encoder.mask_downsampler -> self.mask_encoder
            - memory_encoder.pix_feat_proj -> self.img_feat_proj
            - memory_encoder.fuser -> self.fuser

        The deform_module is NOT initialized (trained from scratch).

        Args:
            memory_encoder: Pretrained MemoryEncoder module
        """
        # Copy mask encoder weights
        self.mask_encoder.load_state_dict(memory_encoder.mask_downsampler.state_dict())

        # Copy image feature projection weights
        self.img_feat_proj.load_state_dict(memory_encoder.pix_feat_proj.state_dict())

        # Copy fuser weights
        self.fuser.load_state_dict(memory_encoder.fuser.state_dict())

        logging.info("Initialized deformation network from memory encoder weights")

        # Optionally freeze encoder components (mask_encoder, img_feat_proj, fuser)
        # to prevent GRL gradients from destabilizing pretrained representations
        if self.freeze_encoder_components:
            for param in self.mask_encoder.parameters():
                param.requires_grad = False
            for param in self.img_feat_proj.parameters():
                param.requires_grad = False
            for param in self.fuser.parameters():
                param.requires_grad = False
            logging.info("Frozen all encoder components (mask_encoder, img_feat_proj, fuser)")

    def predict_params(self, clean_features: torch.Tensor, pixel_gt: torch.Tensor, model: nn.Module, img_batch: torch.Tensor | None = None, **kwargs) -> dict[str, torch.Tensor]:
        """
        Predict deformation offsets for all objects.

        Args:
            clean_features: [B, C, H_feat, W_feat] Clean backbone features
            pixel_gt: [B, K, H_img, W_img] Ground truth masks
            model: adapter-side module exposing ``adv_enable_background`` flag.
            img_batch: Optional (not used for feature-level deform)

        Returns:
            params: Dict containing:
                - feature_offsets: [B, K, 2, H_feat, W_feat]
                - image_offsets: [B, K, 2, H_img, W_img]
                - valid_mask: [K] boolean mask of valid objects
        """
        B, C, H_feat, W_feat = clean_features.shape
        _, K, H_img, W_img = pixel_gt.shape
        device = clean_features.device

        # CRITICAL: Binarize masks to prevent numerical instability from soft mask values
        pixel_gt_binary = (pixel_gt > 0.5).float()

        # Resize masks to feature resolution
        pixel_gt_resized = F.interpolate(pixel_gt_binary.flatten(0, 1).unsqueeze(1), size=(H_feat, W_feat), mode="nearest").view(B, K, H_feat, W_feat)

        # Identify valid objects
        mask_areas = pixel_gt_resized.sum(dim=(0, 2, 3))
        is_empty = mask_areas == 0

        # Background detection (only if background channel is enabled)
        area_ratios = mask_areas / (B * H_feat * W_feat)
        is_background = torch.zeros(K, dtype=torch.bool, device=device)
        if self.use_multi_object and getattr(model, "adv_enable_background", False) and K > 0 and area_ratios[-1] > 0.5:
            is_background[-1] = True

        valid_mask = ~(is_empty | is_background)
        valid_indices = torch.where(valid_mask)[0]

        # Initialize outputs
        feature_offsets_all = torch.zeros(B, K, 2, H_feat, W_feat, device=device)
        image_offsets_all = torch.zeros(B, K, 2, H_img, W_img, device=device)

        if len(valid_indices) == 0:
            return {"feature_offsets": feature_offsets_all, "image_offsets": image_offsets_all, "valid_mask": valid_mask}

        # Pre-compute image projection.
        # Detach features to prevent gradient fighting:
        # the backbone should not receive gradients from the GRL (via offset_net)
        # that attempt to maximize loss. Backbone should only adapt to the *result*
        # of the deformation (task loss), not try to fool the offset predictor.
        clean_features_detached = clean_features.detach()

        img_proj = self.img_feat_proj(clean_features_detached)

        if img_proj.abs().max() > 1e3:
            logging.warning(f"FeatureLevelDeformationImpl: img_proj has large values: {img_proj.abs().max().item():.4f}")

        # Process valid objects
        for k_idx in valid_indices.tolist():
            mask_k_original = pixel_gt_binary[:, k_idx : k_idx + 1]
            mask_k_resized = pixel_gt_resized[:, k_idx : k_idx + 1]

            # Encode mask
            mask_emb = self.mask_encoder(mask_k_original)

            # Fuse
            fused = self.fuser(img_proj + mask_emb)

            if fused.abs().max() > 1e3:
                logging.warning(f"FeatureLevelDeformationImpl: fused features has large values: {fused.abs().max().item():.4f}")

            # Predict offsets
            feat_off, img_off = self.deform_module(fused, mask_k_resized)

            # Apply manual GRL if internal GRL was disabled (e.g. for GCN coordination)
            if self.use_gcn:
                grl = GRL()
                feat_off = grl(feat_off)
                img_off = grl(img_off)

            feature_offsets_all[:, k_idx] = feat_off
            image_offsets_all[:, k_idx] = img_off

        return {"feature_offsets": feature_offsets_all, "image_offsets": image_offsets_all, "valid_mask": valid_mask}

    def apply_transform(
        self, img_batch: torch.Tensor | None, clean_features: torch.Tensor, params: dict[str, torch.Tensor], pixel_gt: torch.Tensor | None = None, model: nn.Module | None = None, **kwargs
    ) -> torch.Tensor:
        """
        Apply deformation to features using predicted offsets.

        Args:
            clean_features: [B, C, H_feat, W_feat] Clean features
            params: Dict from predict_params

        Returns:
            deformed_features: [B, C, H_feat, W_feat]
        """
        feature_offsets = params["feature_offsets"]
        valid_mask = params["valid_mask"]

        B, K, _, H_feat, W_feat = feature_offsets.shape
        device = clean_features.device

        valid_indices = torch.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return clean_features

        # Pre-compute base grid
        norm_grid = torch.meshgrid(torch.linspace(-1, 1, H_feat, device=device), torch.linspace(-1, 1, W_feat, device=device), indexing="ij")
        norm_grid = torch.stack(norm_grid[::-1], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        deformed_list = []
        mask_list = []

        # If pixel_gt is provided, resize it for compositing
        pixel_gt_resized = None
        if pixel_gt is not None:
            # CRITICAL: Binarize masks before resize
            pixel_gt_binary = (pixel_gt > 0.5).float()
            pixel_gt_resized = F.interpolate(pixel_gt_binary.flatten(0, 1).unsqueeze(1), size=(H_feat, W_feat), mode="nearest").view(B, K, H_feat, W_feat)

        for k in range(K):
            # If object is not valid, use clean features
            if not valid_mask[k]:
                deformed_list.append(clean_features)
                if pixel_gt_resized is not None:
                    mask_list.append(pixel_gt_resized[:, k : k + 1])
                continue

            # Get offsets
            offset_k = feature_offsets[:, k]  # [B, 2, H, W]

            # Normalize offsets
            offset_norm = offset_k.permute(0, 2, 3, 1).clone()
            offset_norm[..., 0] = offset_norm[..., 0] / (W_feat / 2.0)
            offset_norm[..., 1] = offset_norm[..., 1] / (H_feat / 2.0)

            sampling_grid = norm_grid + offset_norm

            # Warp: CRITICAL - DO NOT detach clean_features here!
            # Offsets have already passed through GRL in predict_params,
            # so clean_features needs gradients for backbone training on adversarial features.
            deformed_k = F.grid_sample(clean_features, sampling_grid, mode="bilinear", padding_mode="border", align_corners=False)

            deformed_list.append(deformed_k)
            if pixel_gt_resized is not None:
                mask_list.append(pixel_gt_resized[:, k : k + 1])

        # Composite
        if self.use_soft_composite and len(deformed_list) > 1:
            return self.compositor(deformed_list, mask_list)
        else:
            # Fallback or single object logic
            if len(deformed_list) > 0:
                return deformed_list[0]
            else:
                return clean_features


class FeatureBasedDeformModule(nn.Module):
    """
    Dense flow deformation module with Gradient Reversal Layer (GRL).

    Architecture:
        Input: Fused features [B, 256, H_feat, W_feat]
        |
        Offset Predictor Network (3 conv layers)
        |
        Offsets [B, 2, H_feat, W_feat] (constrained by epsilon)
        |
        Gradient Reversal Layer (GRL)
        |
        Upsample to image resolution
        |
        Image offsets [B, 2, H_img, W_img]

    Design rationale:
        - Single branch ensures gradients flow from image warping back to offset predictor.
        - GRL enables adversarial training (maximize loss) within a minimization loop.

    Args:
        feature_dim: Feature dimension (default: 256)
        epsilon: Max offset magnitude in feature space (default: 0.15 pixels)
        image_size: Target image resolution for offset prediction (default: 1024)
    """

    def __init__(
        self,
        feature_dim: int = 256,
        epsilon: float = 0.15,
        image_size: int = 1024,
        zero_mean_offsets: bool = False,
        local_offset_gain: float = 1.0,
        use_grl: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.epsilon = epsilon
        self.image_size = image_size
        self.zero_mean_offsets = zero_mean_offsets
        self.local_offset_gain = local_offset_gain
        self.use_grl = use_grl

        # Offset predictor: fused_features -> dense offsets (2 channels)
        self.offset_net = nn.Sequential(
            nn.InstanceNorm2d(feature_dim),  # Normalize input (critical for stability)
            nn.Conv2d(feature_dim, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=3, padding=1),  # 2 channels for (dx, dy)
        )

        # Initialize ALL layers with small weights for near-zero initial output.
        # This prevents overly strong attacks at the start of training.
        for m in self.offset_net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        logging.info("FeatureBasedDeformModule: Initialized all conv layers with std=0.01 for near-zero initial output")

        # Gradient Reversal Layer - pure negation, no scaling
        # Note: Attack strength is controlled via LR and epsilon, not GRL alpha
        self.grl = GRL()

        # Track if we've logged initial output (for debugging)
        self._logged_initial_output = False

    def forward(self, fused_features: torch.Tensor, object_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply deformation prediction with GRL.

        Args:
            fused_features: [B, C, H_feat, W_feat] Pre-fused image+mask features
            object_mask: [B, 1, H_feat, W_feat] Object mask (for masking/weighting)

        Returns:
            feature_offsets: [B, 2, H_feat, W_feat] Offsets in feature resolution
            image_offsets: [B, 2, H_img, W_img] Offsets in image resolution
        """

        # 1. Predict raw offsets (allow gradients to flow to backbone/encoder)
        # Range: (-1, 1) after tanh
        raw_offsets = self.offset_net(fused_features)  # [B, 2, H_feat, W_feat]

        # Log initial output magnitude once (verify near-zero initialization)
        if not self._logged_initial_output:
            logging.info(
                f"DeformModule initial output: mean={raw_offsets.mean().item():.6f}, std={raw_offsets.std().item():.6f}, min={raw_offsets.min().item():.6f}, max={raw_offsets.max().item():.6f}"
            )
            self._logged_initial_output = True

        # 3. Apply GRL to the offsets (OUTPUT side)
        # This ensures OffsetNet receives inverted gradients (Maximize Loss)
        if self.use_grl:
            raw_offsets_adv = self.grl(raw_offsets)
        else:
            raw_offsets_adv = raw_offsets

        # Relative encoding: use sigmoid for naturally bounded output.
        # Sigmoid outputs [0, 1], we shift to [-0.5, 0.5] for symmetric offsets.
        # This is smoother than tanh near the boundaries and avoids saturation.
        offset_ratio = torch.sigmoid(raw_offsets_adv) - 0.5  # [-0.5, 0.5]

        # Remove global shift while keeping local deformation energy
        if self.zero_mean_offsets:
            if object_mask is not None:
                # SAFETY: Ensure mask is binary to prevent numerical instability
                object_mask_binary = (object_mask > 0.5).float()
                mask_sum = object_mask_binary.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
                mean_offset = (offset_ratio * object_mask_binary).sum(dim=(2, 3), keepdim=True) / mask_sum
            else:
                mean_offset = offset_ratio.mean(dim=(2, 3), keepdim=True)
            offset_ratio = offset_ratio - mean_offset

        # After zero-mean, the range is still bounded (approx [-0.5, 0.5] in practice).
        # No need for local_offset_gain or additional clamp with this design.

        # Use actual target resolution instead of a fixed scale factor
        if isinstance(self.image_size, (tuple, list)):
            target_h, target_w = self.image_size
        else:
            target_h = target_w = int(self.image_size)

        scale_y = target_h / float(fused_features.shape[2])
        scale_x = target_w / float(fused_features.shape[3])

        # 1. Compute Image-level Offsets (Target Resolution)
        # offset_ratio is in [-0.5, 0.5], multiply by 2*epsilon to get [-epsilon, +epsilon]
        # Use float32 here for stability under AMP.
        offset_ratio_f32 = offset_ratio.to(dtype=torch.float32)
        image_raw_offsets = F.interpolate(
            offset_ratio_f32,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        # Scale: offset_ratio * 2 * epsilon * scale_factor
        # This gives max pixel shift of +-epsilon * scale_factor
        scale_tensor = torch.tensor([scale_x, scale_y], device=fused_features.device, dtype=torch.float32).view(1, 2, 1, 1)
        image_offsets_f32 = image_raw_offsets * (2.0 * float(self.epsilon) * scale_tensor)

        image_offsets = image_offsets_f32.to(dtype=fused_features.dtype)

        # 2. Feature-level Offsets (Source Resolution)
        # Keep offsets in feature pixel units for direct warping on the feature map
        feature_offsets = (offset_ratio_f32 * 2.0 * float(self.epsilon)).to(dtype=fused_features.dtype)

        return feature_offsets, image_offsets


class SoftCompositor(nn.Module):
    """
    Soft compositing for handling multi-object overlaps.

    Uses softmax weighting to blend features from different objects at overlap regions,
    providing a differentiable solution to the z-order problem.

    Args:
        temperature: Temperature for softmax (lower = sharper boundaries)
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, deformed_features_list: list[torch.Tensor], mask_list: list[torch.Tensor]) -> torch.Tensor:
        """
        Compose multiple deformed features using soft weighting.

        Args:
            deformed_features_list: List of [B, C, H, W] deformed features for each object
            mask_list: List of [B, 1, H, W] masks for each object

        Returns:
            composited: [B, C, H, W] Soft-composited features
        """
        if len(deformed_features_list) == 0:
            raise ValueError("Empty feature list for compositing")

        if len(deformed_features_list) == 1:
            # Single object: no need for compositing
            return deformed_features_list[0]

        # Stack features and masks
        feat_stack = torch.stack(deformed_features_list, dim=1)  # [B, K, C, H, W]
        mask_stack = torch.stack(mask_list, dim=1)  # [B, K, 1, H, W]

        # Compute soft weights using softmax
        # This automatically handles z-order: objects with higher mask values get higher priority
        mask_weights = F.softmax(mask_stack / self.temperature, dim=1)  # [B, K, 1, H, W]

        # Weighted sum
        composited = (feat_stack * mask_weights).sum(dim=1)  # [B, C, H, W]

        return composited
