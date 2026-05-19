# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Modeling layer: model-agnostic helpers shared by core and adapters.

Contains style utilities, the multi-object Style GCN, the AUE configuration
dataclasses, and the visualizer. The adversarial pipeline itself lives
under ``ruac.core``; the SAM2 plumbing lives under ``ruac.adapters.sam2``.
"""
