from __future__ import annotations

import importlib
from typing import Any


def test_evaluation_protocol_exports_names() -> None:
    module: Any = importlib.import_module("scripts.evaluate_dopamine_rpe_extension")
    assert module.VARIANTS
    assert module.NULL_MODELS
