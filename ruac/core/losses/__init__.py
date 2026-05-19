# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""RUAC loss functions.

Composes three components used by the paper:

* ``L_cal_sym`` — dual-stopgrad symmetric calibration loss between predicted
  uncertainty and per-pixel error (paper Eq. 5 + symmetric mirror), inside
  ``AUELoss``.
* ``BNDLLoss`` — Weibull → Gamma KL regularizer.
* ``CombinedSAMBNDLLoss`` — orchestrator with the curriculum weight schedule.

All losses return plain ``dict[str, Tensor]`` keyed on ``CORE_LOSS_KEY`` (the
``"core_loss"`` constant defined here). The trainer is free to pick the
aggregation key. This module does not depend on any trainer-side import.
"""

CORE_LOSS_KEY = "core_loss"

from ruac.core.losses.aue import AUELoss  # noqa: E402
from ruac.core.losses.bndl import BNDLLoss  # noqa: E402
from ruac.core.losses.combined import CombinedSAMBNDLLoss  # noqa: E402

__all__ = [
    "CORE_LOSS_KEY",
    "BNDLLoss",
    "AUELoss",
    "CombinedSAMBNDLLoss",
]
