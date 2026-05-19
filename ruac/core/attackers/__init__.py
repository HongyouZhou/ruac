# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Adversarial attackers (style + deformation).

Each attacker is a ``nn.Module`` with a uniform interface:

    params = attacker.predict_params(features, gt, **ctx)
    out    = attacker.apply_transform(img_or_features, params, gt, **ctx)

Attackers carry a Gradient Reversal Layer internally so they can be trained
with the SAME backward pass as the model (single-optimizer, multi-LR-group).
The deformation impl takes its memory-encoder building-block classes via DI
so this package stays free of any specific segmentation-model imports.
"""

from ruac.core.attackers.base import AdversarialAttacker
from ruac.core.attackers.deform import FeatureBasedDeformModule, FeatureLevelDeformationImpl, SoftCompositor
from ruac.core.attackers.style import ImageLevelStyleImpl, StyleAdversarialNetwork

__all__ = [
    "AdversarialAttacker",
    "ImageLevelStyleImpl",
    "StyleAdversarialNetwork",
    "FeatureLevelDeformationImpl",
    "FeatureBasedDeformModule",
    "SoftCompositor",
]
