# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Per-backbone adapters that wire ``ruac.core`` into a concrete model.

Each adapter is a thin layer that:

1. Replaces the model's pixel head with a Bayesian version that produces
   Weibull parameters alongside logits.
2. Constructs the attackers and the AUE pipeline, passing the model's
   ``forward_image`` and clean features in as callable / tensor arguments.
3. Returns a list of optimizer parameter groups so attacker LRs can be
   scheduled separately from the main model.

Adapters may import from ``ruac.core``; ``ruac.core`` must never import
from ``ruac.adapters``.
"""
