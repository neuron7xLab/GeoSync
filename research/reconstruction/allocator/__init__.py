# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Country-to-bank marginal allocator (X-10R-1, GH issue #638).

Splits BIS country-aggregate marginals into bank-level marginals via
a prior over per-bank shares. The prior is dependency-injected; this
package ships the protocol + the UniformPrior baseline + a synthetic
positive-control substrate. Concrete priors (ECB MFI, EBA
transparency, BankFocus) land in subsequent epic PRs.
"""

from research.reconstruction.allocator.allocator import CountryToBankAllocator
from research.reconstruction.allocator.certificate import (
    BankLevelMarginalsCertificate,
    compute_cert_id,
)
from research.reconstruction.allocator.prior import AllocatorPrior, UniformPrior
from research.reconstruction.allocator.registry import registry_to_bank_country_map
from research.reconstruction.allocator.size_weighted_prior import (
    SizeWeightedPrior,
)
from research.reconstruction.allocator.synthetic import (
    ShareDistribution,
    bank_level_recovery_l1,
    synthetic_country_aggregates,
)

__all__ = [
    "AllocatorPrior",
    "BankLevelMarginalsCertificate",
    "CountryToBankAllocator",
    "ShareDistribution",
    "SizeWeightedPrior",
    "UniformPrior",
    "bank_level_recovery_l1",
    "compute_cert_id",
    "registry_to_bank_country_map",
    "synthetic_country_aggregates",
]
