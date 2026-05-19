# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
AUE (Adversarial Uncertainty Estimation) configuration + visualization.

This subpackage holds the configuration dataclasses and the visualizer for
the AUE feature. The pipeline lives in ``ruac.core.pipeline``; the SAM2-side
bridge ``AUEModule`` lives in ``ruac.adapters.sam2``.

Main exports:
    - AUEConfig: Master configuration dataclass
    - StyleAdvConfig: Style adversarial attack configuration
    - DeformAdvConfig: Deformation adversarial attack configuration
    - AUEVisualizer: Visualization utilities
"""

from ruac.modeling.aue.config import (
    AUEConfig,
    DeformAdvConfig,
    StyleAdvConfig,
    StyleGCNConfig,
)
from ruac.modeling.aue.visualization import AUEVisualizationData, AUEVisualizer

__all__ = [
    "AUEConfig",
    "StyleAdvConfig",
    "StyleGCNConfig",
    "DeformAdvConfig",
    "AUEVisualizationData",
    "AUEVisualizer",
]
