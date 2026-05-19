# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Gradient Reversal Layer.

The GRL implements the standard ``forward → x, backward → -dL/dx`` operation
used in domain-adversarial training (Ganin & Lempitsky, 2015). RUAC applies
it inside the style / deformation attackers so they can be trained with the
SAME backward pass as the model, in a single optimizer step. Attack strength
is controlled by the learning-rate schedule and the per-attacker ``epsilon``,
not by an alpha scale on the reversal itself.
"""

import torch
import torch.nn as nn


class GradientReversalLayer(torch.autograd.Function):
    """Pure gradient reversal without scaling.

    Note: alpha scaling was removed as it's redundant with learning rate.
    Control attack strength via LR scheduler and epsilon decay instead.
    """

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


class GRL(nn.Module):
    """Gradient Reversal Layer - pure negation, no scaling."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return GradientReversalLayer.apply(x)
