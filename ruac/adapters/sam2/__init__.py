# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""SAM2 adapter — the only adapter in this release.

Targets the SAM2 fork that ships its training scaffold under a top-level
``training/`` package. The adapter:

* installs ``BayesianMaskDecoder`` over SAM2's ``MaskDecoder``,
* attaches style/deform attackers under the attribute names SAM2's
  ``track_step`` already expects,
* exposes the optimizer param-group split used by the paper.
"""

from ruac.adapters.sam2.aue_module import AUEModule
from ruac.adapters.sam2.bayesian_decoder import BayesianMaskDecoder
from ruac.adapters.sam2.trainer import SAM2RUACTrain

__all__ = [
    "SAM2RUACTrain",
    "BayesianMaskDecoder",
    "AUEModule",
]
