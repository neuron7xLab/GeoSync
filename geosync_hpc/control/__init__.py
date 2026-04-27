# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Deterministic control acceptors for GeoSync HPC runtime.

Engineering analogue only. No biological-equivalence claim is made for any
naming exported from this package; all such names are shorthand for the
bounded mathematical mechanism described in
``action_result_acceptor.accept_action_result``.
"""

from __future__ import annotations

from geosync_hpc.control.action_result_acceptor import (
    ActionResultStatus,
    ActionResultWitness,
    ExpectedResultModel,
    ObservedActionResult,
    accept_action_result,
)

__all__ = [
    "ActionResultStatus",
    "ActionResultWitness",
    "ExpectedResultModel",
    "ObservedActionResult",
    "accept_action_result",
]
