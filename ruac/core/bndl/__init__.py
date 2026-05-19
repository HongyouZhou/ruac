# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""BNDL distribution + analytic uncertainty.

The mathematical heart of RUAC's uncertainty estimator: Weibull variational
posterior, KL to a Gamma prior, and the closed-form analytic uncertainty
based on MacKay's probit approximation + Bernoulli entropy.
"""

from ruac.core.bndl.uncertainty import (
    BNDLOutputs,
    pixel_weibull_to_entropy_uncertainty,
)

__all__ = [
    "BNDLOutputs",
    "pixel_weibull_to_entropy_uncertainty",
]
