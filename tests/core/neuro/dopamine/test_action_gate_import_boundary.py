# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path


ACTION_GATE_PATH = Path("src/geosync/core/neuro/dopamine/action_gate.py")


def test_action_gate_module_does_not_import_concrete_dopamine_controller() -> None:
    source = ACTION_GATE_PATH.read_text(encoding="utf-8")

    assert "dopamine_controller" not in source
    assert "DopamineController" not in source
    assert "TemperatureBoundsProvider" in source
