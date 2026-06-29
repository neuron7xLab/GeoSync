from __future__ import annotations

from scripts.evaluate_dopamine_rpe_extension import NULL_MODELS, VARIANTS


def test_evaluation_protocol_enumerates_variants_and_nulls() -> None:
    assert "canonical_td0" in VARIANTS
    assert "combined_surface" in VARIANTS
    assert "shuffled_returns" in NULL_MODELS
    assert len(NULL_MODELS) == 8
