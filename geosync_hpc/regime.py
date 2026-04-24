# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Simple regime classification based on realized volatility."""

from __future__ import annotations

import numpy as np


class RegimeModel:
    def __init__(self, bins: tuple[float, ...] = (0.0, 0.5, 1.5, 5.0)) -> None:
        self.bins = bins
        self.state = 0

    def update(self, feats: dict[str, float]) -> dict[str, int | list[float]]:
        rv = feats.get("rv", np.nan)
        if np.isnan(rv):
            return {"regime": self.state, "probs": [1.0, 0.0, 0.0]}
        g = int(np.digitize([rv], self.bins)[0] - 1)
        g = max(0, min(g, 2))
        self.state = g
        probs = [0.0, 0.0, 0.0]
        probs[g] = 1.0
        return {"regime": g, "probs": probs}

    def reset(self) -> None:
        """Reset to the neutral starting regime for a new run."""
        self.state = 0

    def get_state(self) -> dict[str, int]:
        """Return serializable regime state."""
        return {"state": int(self.state)}

    def set_state(self, state: dict[str, int]) -> None:
        """Restore state captured by ``get_state``."""
        self.state = int(state["state"])
