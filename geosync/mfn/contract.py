# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""MFN command and artifact contracts.

The MFN surface intentionally uses only the Python standard library so the
packaging, help, and smoke-artifact path remain available before heavyweight
scientific dependencies are installed.
"""

from __future__ import annotations

from dataclasses import dataclass

MFN_COMMANDS: tuple[str, ...] = (
    "simulate",
    "extract",
    "detect",
    "forecast",
    "compare",
    "report",
    "run",
    "validate",
)


@dataclass(frozen=True, slots=True)
class MFNContract:
    """Stable fields emitted by every MFN stage artifact."""

    schema_version: str = "mfn.integration.v1"
    seed: int = 1337
    input_window_sec: int = 30
    claim_tier: str = "INSTRUMENTED"
