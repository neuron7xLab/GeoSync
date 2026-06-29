from __future__ import annotations

from geosync.core.neuro.dopamine.telemetry import METRIC_NAMES, metric_catalog


def test_metric_catalog_contains_required_sli_names() -> None:
    catalog = metric_catalog()
    assert len(METRIC_NAMES) == 16
    assert "dopamine.rpe_finite_rate" in catalog
    assert "dopamine.p99_step_latency_ms" in catalog
    assert "dopamine.cognitive_value_score" in catalog
    assert all(spec["owner"] == "geosync.dopamine" for spec in catalog.values())
