# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Cross-version compatibility helpers for core runtime modules."""

from __future__ import annotations

from datetime import datetime, timezone

UTC = timezone.utc


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(UTC)

