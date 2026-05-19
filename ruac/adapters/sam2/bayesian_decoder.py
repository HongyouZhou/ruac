# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Bayesian mask decoder for the SAM2 adapter (paper §3.2, UE).

Subclasses SAM2's ``MaskDecoder`` and replaces the final hypernet pixel-mask
computation with a Weibull-Bayesian head (BNDL, Hu et al., ICLR 2025). Image
tokens (after upscaling) and mask tokens are both modeled with Weibull
variational posteriors; mask logits come from a sampled forward in training
and from analytically-summarized features (or Monte Carlo) at inference.
Per-pixel uncertainty is exposed via the ``aux_outputs`` dict so the loss
layer can compute the calibration objective.

This file is the only place in the package that imports from ``sam2.*`` for
the decoder swap; everything else operates on duck-typed ``model`` arguments.
"""

from __future__ import annotations

import torch
from BNDL.BNDL_upload.ViT_Sparse.utils.bndl import (
    BNDL,
    entropy_uncertainty,
    uncertainty_sample_parallel,
)
from sam2.modeling.sam.mask_decoder import MaskDecoder

from ruac.core.bndl import pixel_weibull_to_entropy_uncertainty


class BayesianMaskDecoder(MaskDecoder):
    """SAM2 MaskDecoder with a Weibull-Bayesian pixel head.

    The transformer / upscale / hyper_in path is identical to vanilla SAM2.
    Only the final pixel mask computation is replaced by BNDL, and an
    aux_outputs dict is returned alongside the four standard outputs.
    """

    def __init__(
        self,
        *,
        bndl_factor_z: float = 0.0,
        bndl_factor_w: float = 0.0,
        bndl_force_single_sample: bool = False,
        bndl_sample_num: int = 20,
        **mask_decoder_kwargs,
    ):
        # Force-disable parent's BNDL/UR-ERN paths so this subclass owns pixel_bndl.
        mask_decoder_kwargs.pop("use_bndl_for_pixels", None)
        mask_decoder_kwargs.pop("use_ur_ern_for_pixels", None)
        mask_decoder_kwargs.pop("bndl_factor_z", None)
        mask_decoder_kwargs.pop("bndl_factor_w", None)
        mask_decoder_kwargs.pop("bndl_force_single_sample", None)
        super().__init__(
            use_bndl_for_pixels=False,
            use_ur_ern_for_pixels=False,
            **mask_decoder_kwargs,
        )

        self.bndl_factor_z = float(bndl_factor_z)
        self.bndl_factor_w = float(bndl_factor_w)
        self.bndl_force_single_sample = bool(bndl_force_single_sample)
        self.bndl_sample_num = int(bndl_sample_num)

        pixel_feat_dim = self.transformer_dim // 8
        self.pixel_bndl = BNDL(
            pixel_feat_dim,
            2,
            mask_token_dim=self.transformer_dim,
        )

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # === Vanilla SAM2: token concatenation + transformer ===
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat(
                [
                    self.obj_score_token.weight,
                    self.iou_token.weight,
                    self.mask_tokens.weight,
                ],
                dim=0,
            )
            s = 1
        else:
            output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        src = src + dense_prompt_embeddings
        assert image_pe.size(0) == 1, "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]

        if self.pred_obj_scores:
            assert s == 1
            obj_token = hs[:, 0, :]
            object_score_logits = self.pred_obj_score_head(obj_token)
        else:
            object_score_logits = 10.0 * iou_token_out.new_ones(iou_token_out.shape[0], 1)

        # === Vanilla SAM2: upscale + hypernet weights (kept for parity, unused below) ===
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        # NOTE: parent's `output_hypernetworks_mlps` is left untouched in the
        # state_dict for vanilla-SAM2 checkpoint compatibility, but never invoked:
        # the BNDL head below replaces the `hyper_in @ upscaled` mask computation.

        # === BNDL pixel head (paper §3.2) ===
        pixel_feat = upscaled_embedding.permute(0, 2, 3, 1)  # [B, C', H, W] -> [B, H, W, C']
        force_sample = (not self.training) and self.bndl_force_single_sample
        (
            masks_bndl_sam,
            z_out,
            wei_lambda,
            inv_k,
            wei_lambda_w,
            inv_k_w,
        ) = self.pixel_bndl(
            pixel_feat,
            mask_tokens_out,
            factor_z=self.bndl_factor_z,
            factor_w=self.bndl_factor_w,
            force_sample=force_sample,
        )
        masks_bndl = masks_bndl_sam.permute(0, 3, 1, 2)
        masks = masks_bndl

        # Sampling-based uncertainty: no gradients, used for visualization, logging,
        # and the attacker calibration loss.
        sampled_logits, _mean_logits = uncertainty_sample_parallel(
            self.pixel_bndl,
            pixel_feat,
            mask_tokens_out,
            sample_num=self.bndl_sample_num,
            factor_z=self.bndl_factor_z,
            factor_w=self.bndl_factor_w,
        )
        pixel_uncertainty_sampling = entropy_uncertainty(sampled_logits)

        # Analytic uncertainty: derived in closed form from Weibull parameters and
        # carries gradients into BNDL, so the L_cal_sym calibration loss can train it.
        # Only computed in training mode to save memory.
        pixel_uncertainty_analytic = None
        if self.training:
            pixel_uncertainty_analytic = pixel_weibull_to_entropy_uncertainty(
                pixel_bndl_model=self.pixel_bndl,
                pixel_feat=pixel_feat,
                external_pre_out_w=mask_tokens_out,
                per_channel=True,
            )

        aux_outputs = {
            "bndl": {
                "z_out": z_out,
                "wei_lambda": wei_lambda,
                "inv_k": inv_k,
                "wei_lambda_w": wei_lambda_w,
                "inv_k_w": inv_k_w,
                "masks_bndl_raw": masks_bndl_sam.detach(),
                "pixel_uncertainty_sampling": pixel_uncertainty_sampling,
                "pixel_uncertainty_analytic": pixel_uncertainty_analytic,
                "pixel_uncertainty": pixel_uncertainty_sampling,
                "pixel_logits": masks_bndl_sam if self.training else masks_bndl_sam.detach(),
                "upscaled_shape": (b, c, h, w),
                "mask_tokens_out": mask_tokens_out if self.training else mask_tokens_out.detach(),
                "pixel_feat": pixel_feat.detach(),
                "pixel_feat_grad": pixel_feat if self.training else None,
                "masks_bndl": masks_bndl.detach(),
                "masks_hyper": None,
            }
        }

        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred, mask_tokens_out, object_score_logits, aux_outputs
