from __future__ import annotations

from scripts.evaluate_dopamine_rpe_extension import NULL_MODELS, VARIANTS


def test_evaluation_protocol_exports_names() -> None:
    assert VARIANTS
    assert NULL_MODELS
