# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
AUE Configuration Dataclasses.

Hierarchical config for Adversarial Uncertainty Estimation. Hydra constructs
these from nested yaml via _target_.

    aue_cfg = AUEConfig(
        enabled=True,
        style=StyleAdvConfig(enabled=True, epsilon=0.3),
        deform=DeformAdvConfig(enabled=True, epsilon=0.15),
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# =============================================================================
# Style Adversarial Configuration
# =============================================================================
@dataclass
class StyleGCNConfig:
    """GCN-based multi-object style refinement configuration."""

    enabled: bool = False
    hidden_dim: int = 64
    num_layers: int = 2
    edge_threshold: float = 0.0
    use_semantic_edges: bool = True
    use_background_edges: bool = True
    distance_threshold: float = 1.0
    use_boundary_distance: bool = True
    use_visual_features: bool = True
    feature_dim: int = 256
    feature_sim_threshold: float = 0.5


@dataclass
class StyleAdvConfig:
    """Style adversarial attack configuration."""

    enabled: bool = False
    mode: Literal["image_level"] = "image_level"
    epsilon: float = 1.1  # Perturbation budget for style
    use_gt_region_style: bool = True

    # Global-Local Mixed Style
    use_global_local_mix: bool = False
    global_epsilon: float = 0.5
    global_weight: float = 0.5

    # GCN sub-config
    gcn: StyleGCNConfig = field(default_factory=StyleGCNConfig)

    # NOTE: Hydra constructs this dataclass from nested yaml directly via _target_.


# =============================================================================
# Deformation Adversarial Configuration
# =============================================================================
@dataclass
class DeformAdvConfig:
    """Deformation adversarial attack configuration."""

    enabled: bool = False
    epsilon: float = 3.0  # Deformation strength in pixels
    use_soft_composite: bool = True
    temperature: float = 1.0

    # GCN coordination
    use_gcn: bool = False
    gcn_num_layers: int = 2
    num_deform_groups: int = 4

    # Feature-based deformation options
    init_from_memory_encoder: bool = True
    freeze_encoder_components: bool = True
    zero_mean_offsets: bool = True
    local_offset_gain: float = 1.3

    # NOTE: Hydra constructs this dataclass from nested yaml directly via _target_.


# =============================================================================
# Master AUE Configuration
# =============================================================================
@dataclass
class AUEConfig:
    """
    Master AUE configuration, aggregates all sub-configs.

    Hydra constructs this directly from nested yaml via _target_:

        aue_config:
          _target_: ruac.modeling.aue.AUEConfig
          enabled: true
          style:
            _target_: ruac.modeling.aue.StyleAdvConfig
            enabled: true
            epsilon: 0.3
          deform:
            _target_: ruac.modeling.aue.DeformAdvConfig
            enabled: true
            epsilon: 0.15
    """

    # Master enable flag
    enabled: bool = False
    use_analytic_uncertainty: bool = True

    # Multi-object control
    use_multi_object: bool = True
    enable_background: bool = True

    # Max objects (matches dataset sampler)
    max_num_objects: int = 11  # 10 objects + 1 background

    # Sub-configs
    style: StyleAdvConfig = field(default_factory=StyleAdvConfig)
    deform: DeformAdvConfig = field(default_factory=DeformAdvConfig)

    # Attack ordering
    attack_order: list[str] = field(default_factory=lambda: ["style", "deform"])

    @property
    def any_adversarial_enabled(self) -> bool:
        return self.style.enabled or self.deform.enabled
