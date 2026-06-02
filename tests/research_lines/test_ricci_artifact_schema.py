from __future__ import annotations

import json
from pathlib import Path

from tools.research.validate_ricci_artifact_schema import validate_artifact


def test_example_ricci_artifact_satisfies_schema() -> None:
    # INV-RC1 / INV-DET1: canonical Ricci evidence artifacts must be schema-valid.
    errors = validate_artifact(
        schema_path=Path("schemas/research/ricci_microstructure_artifact.schema.json"),
        artifact_path=Path("artifacts/runs/ricci_microstructure_v1/example_artifact.json"),
    )

    assert errors == []


def test_schema_rejects_synthetic_promotion_without_required_fields(tmp_path) -> None:
    # INV-NFD1 / INV-DET1: malformed artifacts cannot promote claim state.
    artifact = tmp_path / "bad.json"
    artifact.write_text(
        json.dumps(
            {
                "run_id": "RFC-2026-MARKET-RICCI-V1",
                "git_sha": "not-a-sha",
                "git_dirty": False,
                "data_sha256": "0" * 64,
                "config_sha256": "1" * 64,
                "seed": 1337,
                "timestamp_utc": "2026-05-31T20:27:00Z",
                "input_window_sec": 30,
                "score": 0.742,
                "uncertainty": 0.031,
                "decision": "TRADE",
                "claim_tier": "MEASURED_SINGLE",
                "falsification_status": "PASS",
            }
        ),
        encoding="utf-8",
    )

    errors = validate_artifact(
        schema_path=Path("schemas/research/ricci_microstructure_artifact.schema.json"),
        artifact_path=artifact,
    )

    assert any("git_sha" in error for error in errors)
    assert any("decision" in error for error in errors)
