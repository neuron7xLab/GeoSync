from pathlib import Path
from typing import Any

from pydantic import ValidationError

from BBB_NVU_Cognitive_Noise_Gate_2026.src.deterministic_engine import load_yaml
from BBB_NVU_Cognitive_Noise_Gate_2026.src.operational_kernel import OperationalKernel, verify_operational_envelope
from BBB_NVU_Cognitive_Noise_Gate_2026.tests.requirements import requirement

ROOT = Path(__file__).resolve().parents[1]
RULES = load_yaml(ROOT / "config" / "risk_rules.yaml")
TIME = "2026-06-03T00:00:00Z"
ENGINE_HASH = "engine-hash-for-operational-kernel-tests"
STABLE: dict[str, Any] = {"subject_id": "S-STABLE", "critical_data_invalid": False, "confidence": 0.95, "domain_indices": {"BSI": 20, "NRI": 20, "VML": 20, "GRS": 80, "CNI": 20}, "degradations": []}
RED: dict[str, Any] = {"subject_id": "S-RED", "critical_data_invalid": False, "confidence": 0.90, "domain_indices": {"BSI": 82, "NRI": 20, "VML": 20, "GRS": 80, "CNI": 20}, "degradations": []}
BAD: dict[str, Any] = {"subject_id": "S-BAD", "critical_data_invalid": True, "confidence": 1.0, "domain_indices": {"BSI": 20, "NRI": 20, "VML": 20, "GRS": 80, "CNI": 20}, "degradations": []}


def k() -> OperationalKernel:
    return OperationalKernel(RULES, engine_hash=ENGINE_HASH)


def req(doc: dict[str, Any], source: str, **extra: Any) -> dict[str, Any]:
    return {"input_doc": doc, "source_id": source, **extra}


@requirement("R003")
def test_operational_envelope_is_deterministic_and_replay_verified():
    a = k().execute([req(STABLE, "s.json"), req(RED, "r.json")], created_at=TIME)
    b = k().execute([req(STABLE, "s.json"), req(RED, "r.json")], created_at=TIME)
    assert a.envelope_hash == b.envelope_hash
    assert a.manifest["replay_verified"] == [True, True]


@requirement("R004")
def test_operational_envelope_contains_metrics_audit_and_incidents():
    e = k().execute([req(STABLE, "s.json"), req(RED, "r.json")], created_at=TIME)
    assert e.request_count == 2
    assert e.metrics_snapshot["incident_candidates"] == 1
    assert e.incidents[0]["severity"] == "CRITICAL"


@requirement("R002")
def test_operational_kernel_preserves_fail_closed_invalid_behavior():
    e = k().execute([req(BAD, "bad.json")], created_at=TIME)
    assert e.outputs[0]["risk"]["risk_state"] == "BLACK_INVALID"
    assert e.outputs[0]["risk"]["confidence"] == 0.0


@requirement("R002")
def test_operational_kernel_rejects_boundary_request_extra_fields():
    try:
        k().execute([req(STABLE, "s.json", extra="x")], created_at=TIME)
    except ValidationError as error:
        assert error.errors()[0]["type"] == "extra_forbidden"
    else:
        raise AssertionError("extra field accepted")


@requirement("R003")
def test_operational_envelope_verifier_accepts_fresh_envelope():
    assert verify_operational_envelope(k().execute([req(STABLE, "s.json")], created_at=TIME))


@requirement("R003")
def test_operational_envelope_verifier_rejects_hash_tampering():
    e = k().execute([req(STABLE, "s.json")], created_at=TIME).model_dump()
    e["outputs"][0]["risk"]["confidence"] = 0.1
    assert not verify_operational_envelope(e)


@requirement("R003")
def test_operational_envelope_verifier_rejects_manifest_tampering():
    e = k().execute([req(STABLE, "s.json")], created_at=TIME).model_dump()
    e["manifest"]["replay_verified"] = [False]
    assert not verify_operational_envelope(e)


@requirement("R002")
def test_operational_envelope_verifier_fails_closed_on_malformed_payload():
    assert not verify_operational_envelope({"envelope_hash": "bad", "outputs": None})
