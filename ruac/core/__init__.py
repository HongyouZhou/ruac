# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""RUAC core: model-agnostic math.

Contains the Bayesian uncertainty estimator (BNDL), adversarial attackers,
the AUE training pipeline, and calibration losses. None of these modules
import from any specific segmentation backbone — adapters live in
``ruac.adapters`` and inject the model-specific glue.
"""
