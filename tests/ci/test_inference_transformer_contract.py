from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.geosync.research.transformer.contracts import (
    ClaimTier,
    GeometryState,
    RegimeCertificate,
    ResearchInferenceArtifact,
    sha256_json,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/ci/check_inference_transformer_contract.py"


def test_inference_transformer_contract_script_passes() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "PASS"
    assert payload["claim_tier"] == "HYPOTHESIS"


def _geometry_state() -> GeometryState:
    return GeometryState(
        method_version="inference-transformer.v1",
        domain="bar_proxy_research",
        ricci_summary={"method": "forman_proxy", "status": "instrumented"},
        kuramoto_summary={"order_parameter": 0.0, "usage": "regime_feature_only"},
        topology_summary={"status": "not_implemented"},
        uncertainty={"confidence_floor": 0.0},
        downgrade_reason="not_measured",
    )


def test_regime_certificate_round_trip_dict_is_hypothesis_abstain() -> None:
    certificate = RegimeCertificate(
        run_id="run-001",
        research_line="ricci_microstructure_v1",
        regime_label="ABSTAIN",
        claim_tier=ClaimTier.HYPOTHESIS,
        confidence=0.0,
        domain="bar_proxy_research",
        geometry_state=_geometry_state(),
        downgrade_reason="not_measured",
        assumptions=("no order-book evidence", "no rank change"),
    )
    payload = certificate.to_dict()
    assert payload["claim_tier"] == "HYPOTHESIS"
    assert payload["status"] == "ABSTAIN"
    assert payload["confidence"] == 0.0
    assert payload["geometry_state"]["downgrade_reason"] == "not_measured"


def test_hypothesis_certificate_rejects_non_abstain_status() -> None:
    with pytest.raises(ValueError, match="HYPOTHESIS certificates"):
        RegimeCertificate(
            run_id="run-002",
            research_line="ricci_microstructure_v1",
            regime_label="UNSUPPORTED",
            claim_tier="HYPOTHESIS",
            confidence=0.2,
            domain="bar_proxy_research",
            geometry_state=_geometry_state(),
            downgrade_reason="not_measured",
            status="CANDIDATE",
        )


def test_research_artifact_adds_stable_hash() -> None:
    certificate = RegimeCertificate(
        run_id="run-003",
        research_line="ricci_microstructure_v1",
        regime_label="ABSTAIN",
        claim_tier="HYPOTHESIS",
        confidence=0.0,
        domain="bar_proxy_research",
        geometry_state=_geometry_state(),
        downgrade_reason="not_measured",
    )
    artifact = ResearchInferenceArtifact(
        run_id="run-003",
        git_sha="0" * 40,
        data_sha256="1" * 64,
        config_sha256="2" * 64,
        replay_command="python scripts/ci/check_inference_transformer_contract.py",
        certificate=certificate,
    )
    payload = artifact.to_dict()
    expected = sha256_json({key: value for key, value in payload.items() if key != "output_sha256"})
    assert payload["output_sha256"] == expected


def test_confidence_range_is_checked() -> None:
    with pytest.raises(ValueError, match="confidence"):
        RegimeCertificate(
            run_id="run-004",
            research_line="ricci_microstructure_v1",
            regime_label="ABSTAIN",
            claim_tier="HYPOTHESIS",
            confidence=1.5,
            domain="bar_proxy_research",
            geometry_state=_geometry_state(),
            downgrade_reason="not_measured",
        )
