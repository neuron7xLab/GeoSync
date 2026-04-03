# SPDX-License-Identifier: MIT
"""Public API for the Kuramoto simulation subsystem.

Exports a minimal stable surface:
- :class:`KuramotoConfig` for validated inputs
- :class:`KuramotoEngine` and :func:`run_simulation` for execution
- :class:`KuramotoResult` for typed outputs
"""

from __future__ import annotations

from .config import KuramotoConfig
from .engine import KuramotoEngine, KuramotoResult, run_simulation

__all__ = ["KuramotoConfig", "KuramotoEngine", "KuramotoResult", "run_simulation"]
