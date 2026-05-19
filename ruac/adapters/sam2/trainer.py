# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""SAM2 training-time wrapper with RUAC's Bayesian decoder + AUE branch.

Subclass of ``sam2.training.model.sam2.SAM2Train`` that:

* leaves the SAM2 fork untouched (no monkey-patching, no fork edits),
* swaps the freshly-built vanilla ``MaskDecoder`` for ``BayesianMaskDecoder``,
* builds the RUAC adversarial pipeline (Style + Deform + GCN attackers and
  ``AUEModule``) from a structured ``AUEConfig`` and attaches them onto
  ``self``.

The inherited ``track_step`` already calls
``self._aue_module.generate_adversarial_samples`` during training on initial
conditioning frames. By assigning RUAC's ``AUEModule`` to ``self._aue_module``
here, we exercise RUAC's pipeline, not the SAM2 fork's. The parent's own AUE
wiring is force-disabled via ``use_aue=False`` to ``super()``.
"""

from __future__ import annotations

import logging

# NOTE: SAM2's training scaffold lives at the top-level `training` package
# (sam2 fork lays it out as `sam2/training/`, separate from `sam2/sam2/`).
from training.model.sam2 import SAM2Train

from ruac.adapters.sam2.aue_module import AUEModule
from ruac.modeling.aue.config import AUEConfig

_FLAT_AUE_KWARGS_TO_DROP = (
    "use_aue",
    "use_style_adv",
    "use_deform_adv",
    "use_pgd_adv",
    "use_patch_adv",
    "use_random_noise_adv",
)


class SAM2RUACTrain(SAM2Train):
    """SAM2 training model with the RUAC Bayesian decoder + AUE adversarial branch."""

    def __init__(
        self,
        *args,
        aue_config: AUEConfig | None = None,
        **kwargs,
    ):
        # Force-disable the SAM2 fork's own AUE wiring so we don't end up with
        # two competing setups. RUAC's AUE is wired explicitly in `_setup_aue`
        # below, using the structured `aue_config`.
        for k in _FLAT_AUE_KWARGS_TO_DROP:
            kwargs.pop(k, None)
        super().__init__(*args, use_aue=False, **kwargs)

        # Parent's `_build_sam_heads()` (called from SAM2Base.__init__) hard-codes
        # the vanilla `MaskDecoder` class. Swap it for RUAC's BayesianMaskDecoder
        # right after super() so the rest of the model sees the Bayesian decoder.
        self._install_bayesian_decoder()

        if aue_config is not None and aue_config.enabled:
            self._setup_aue(aue_config)
        else:
            self.aue_config = aue_config

    def _install_bayesian_decoder(self) -> None:
        """Replace the vanilla MaskDecoder built by SAM2Base with BayesianMaskDecoder.

        The freshly-built parent decoder is immediately discarded; the replacement
        carries pixel_bndl (RUAC's Weibull-Bayesian pixel head). Architecture-shared
        modules (transformer, hypernet MLPs, IoU head, ...) get fresh init weights;
        SAM2 pretrained checkpoints can still be loaded because BayesianMaskDecoder
        keeps the parent's parameter names.
        """
        from sam2.modeling.sam.transformer import TwoWayTransformer

        from ruac.adapters.sam2.bayesian_decoder import BayesianMaskDecoder

        extra = self.sam_mask_decoder_extra_args or {}
        self.sam_mask_decoder = BayesianMaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.sam_prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            bndl_factor_z=extra.get("bndl_factor_z", 0.0),
            bndl_factor_w=extra.get("bndl_factor_w", 0.0),
            bndl_force_single_sample=extra.get("bndl_force_single_sample", False),
            bndl_sample_num=extra.get("bndl_sample_num", 20),
        )

    def _setup_aue(self, cfg: AUEConfig) -> None:
        """Wire RUAC's AUE pipeline onto `self`.

        Sets the flat attribute contract that `ruac.core.pipeline` and
        `ruac.core.attackers.*` read via `model.<attr>`.
        """
        self.aue_config = cfg

        self.use_aue = True
        self.use_style_adv = cfg.style.enabled
        self.use_deform_adv = cfg.deform.enabled
        self.use_pgd_adv = False
        self.use_patch_adv = False
        self.use_random_noise_adv = False

        self.adversarial_attack_order = list(cfg.attack_order)
        self.adv_use_multi_object = cfg.use_multi_object
        self.adv_enable_background = cfg.enable_background
        self.aue_use_analytic_uncertainty = cfg.use_analytic_uncertainty

        self.style_adv_mode = cfg.style.mode
        self.style_adv_epsilon = cfg.style.epsilon
        self.style_adv_use_gt_region_style = cfg.style.use_gt_region_style
        self.style_adv_use_gcn = cfg.style.gcn.enabled
        self.style_adv_gcn_edge_threshold = cfg.style.gcn.edge_threshold
        self.style_adv_gcn_use_semantic_edges = cfg.style.gcn.use_semantic_edges
        self.style_adv_gcn_use_background_edges = cfg.style.gcn.use_background_edges
        self.style_adv_gcn_distance_threshold = cfg.style.gcn.distance_threshold
        self.style_adv_gcn_use_boundary_distance = cfg.style.gcn.use_boundary_distance
        self.style_adv_gcn_use_visual_features = cfg.style.gcn.use_visual_features
        self.style_adv_gcn_feature_dim = cfg.style.gcn.feature_dim
        self.style_adv_gcn_feature_sim_threshold = cfg.style.gcn.feature_sim_threshold

        self.deform_adv_epsilon = cfg.deform.epsilon

        self._enable_style_visualization = False
        self._latest_gcn_stats = None

        if cfg.style.enabled:
            self._build_style_attacker(cfg)
        if cfg.deform.enabled:
            self._build_deform_attacker(cfg)

        self._aue_module = AUEModule(self)
        self._aue_module.initialize()

        # SAM2Train captured `_initial_*_epsilon` from default (pre-AUE) values
        # in its own __init__. Re-capture now that the real epsilons are set.
        self._initial_style_epsilon = self.style_adv_epsilon
        self._initial_deform_epsilon = self.deform_adv_epsilon

        logging.info(
            "[SAM2RUACTrain] AUE wired: style=%s deform=%s order=%s",
            cfg.style.enabled,
            cfg.deform.enabled,
            cfg.attack_order,
        )

    def _build_style_attacker(self, cfg: AUEConfig) -> None:
        from ruac.core.attackers import AdversarialAttacker
        from ruac.modeling.style_gcn import AdversarialStyleGCN

        if cfg.style.gcn.enabled:
            if not cfg.use_multi_object:
                raise ValueError("style GCN requires use_multi_object=True")
            if cfg.style.use_global_local_mix:
                raise ValueError("style GCN is incompatible with use_global_local_mix")
            self.style_gcn = AdversarialStyleGCN(
                style_dim=6,
                feature_dim=(cfg.style.gcn.feature_dim if cfg.style.gcn.use_visual_features else 0),
                num_layers=cfg.style.gcn.num_layers,
            )
        else:
            self.style_gcn = None

        self.style_attacker = AdversarialAttacker(
            mode=cfg.style.mode,
            aug_type="style",
            epsilon=cfg.style.epsilon,
            use_multi_object=cfg.use_multi_object,
            use_gcn=cfg.style.gcn.enabled,
            use_gt_region_style=cfg.style.use_gt_region_style,
            enable_background=cfg.enable_background,
            use_global_local_mix=cfg.style.use_global_local_mix,
            global_epsilon=cfg.style.global_epsilon,
            global_weight=cfg.style.global_weight,
            num_objects=cfg.max_num_objects + (1 if cfg.enable_background else 0),
        )

    def _build_deform_attacker(self, cfg: AUEConfig) -> None:
        from sam2.modeling.memory_encoder import CXBlock, Fuser, MaskDownSampler

        from ruac.core.attackers import AdversarialAttacker

        # Memory-encoder building blocks are injected here so ruac.core stays
        # free of any sam2 imports. The deform attacker's mask encoder /
        # projection / fuser are shaped to match SAM2's memory encoder so
        # ``load_memory_encoder_weights`` can copy weights verbatim.
        self.deform_attacker = AdversarialAttacker(
            mode="feature_level",
            aug_type="deformation",
            feature_dim=256,
            epsilon=cfg.deform.epsilon,
            use_soft_composite=cfg.deform.use_soft_composite,
            temperature=cfg.deform.temperature,
            use_multi_object=cfg.use_multi_object,
            use_gcn=cfg.deform.use_gcn,
            gcn_num_layers=cfg.deform.gcn_num_layers,
            num_deform_groups=cfg.deform.num_deform_groups,
            init_from_memory_encoder=cfg.deform.init_from_memory_encoder,
            freeze_encoder_components=cfg.deform.freeze_encoder_components,
            image_size=self.image_size,
            zero_mean_offsets=cfg.deform.zero_mean_offsets,
            local_offset_gain=cfg.deform.local_offset_gain,
            mask_downsampler_cls=MaskDownSampler,
            fuser_cls=Fuser,
            cxblock_cls=CXBlock,
        )

        if cfg.deform.init_from_memory_encoder:
            self.deform_attacker.impl.load_memory_encoder_weights(self.memory_encoder)
