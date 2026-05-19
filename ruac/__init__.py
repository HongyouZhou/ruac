# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""RUAC: Robust Uncertainty-Accuracy Correlation for SAM2.

Reference implementation for the ICML 2026 paper
"Segment Anything with Robust Uncertainty-Accuracy Correlation".

The package is organised as a SAM2 addon: it imports SAM2 and BNDL as
external dependencies and provides subclasses + loss functions that
introduce the Bayesian mask decoder (UE) and the adversarial uncertainty
estimation (AUE) training pipeline described in the paper.
"""

__version__ = "0.1.0"
