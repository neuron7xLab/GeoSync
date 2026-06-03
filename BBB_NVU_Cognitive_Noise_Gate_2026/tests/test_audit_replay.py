import json
from pathlib import Path

from BBB_NVU_Cognitive_Noise_Gate_2026.src.audit import (
    AuditEvent,
    build_replay_bundle,
    replay_bundle,
    verify_replay_bundle,
    write_audit_event,
)
from BBB_NVU_Cognitive_Noise_Gate_2026.src.deterministic_engine import DeterministicInferenceEngine, load_yaml
from BBB_NVU_Cognitive_Noise_Gate_2026.tests.requirements import requirement

ROOT = Path(__file__).resolve().parents[1]
RULES = load_yaml(ROOT / "config" / "risk_rules.yaml")
SAMPLE_INPUT = json.loads((ROOT / "examples" / "sample_run_input.json").read_text(encoding="utf-8"))
FIXED_TIME = "2026-06-03T00:00:00Z"
FIXED_ENGINE_HASH = "engine-hash-for-audit-replay-tests"


def build_output():
    engine = DeterministicInferenceEngine.from_rules(RULES, engine_hash=FIXED_ENGINE_HASH)
    return engine.build_output(SAMPLE_INPUT, source_id="sample_run_input.json", created_at=FIXED_TIME)


@requirement("R003")
def test_inference_output_contains_hash_material_for_replay():
    output = build_output()
    assert output["input_hash"]
    assert output["rules_hash"]
    assert output["engine_hash"] == FIXED_ENGINE_HASH
    assert output["run_hash"]


@requirement("R003")
def test_audit_event_is_deterministic_jsonl(tmp_path):
    output = build_output()
    event = AuditEvent.from_output(output)
    line = event.to_jsonl()
    parsed = json.loads(line)

    assert parsed["event_type"] == "inference_run"
    assert parsed["run_hash"] == output["run_hash"]
    assert parsed["risk_state"] == "YELLOW_WATCH"
    assert parsed["action_ids"] == [action["action_id"] for action in output["actions"]]

    audit_path = tmp_path / "audit.jsonl"
    write_audit_event(audit_path, event)
    assert json.loads(audit_path.read_text(encoding="utf-8"))["run_hash"] == output["run_hash"]


@requirement("R003")
def test_replay_bundle_reconstructs_expected_run_hash():
    output = build_output()
    bundle = build_replay_bundle(output, SAMPLE_INPUT, RULES)
    replayed = replay_bundle(bundle)

    assert verify_replay_bundle(bundle)
    assert replayed["run_hash"] == output["run_hash"]
    assert replayed["risk"] == output["risk"]
    assert bundle["bundle_hash"]


@requirement("R003")
def test_replay_bundle_detects_input_tampering():
    output = build_output()
    bundle = build_replay_bundle(output, SAMPLE_INPUT, RULES)
    bundle["input_doc"] = dict(bundle["input_doc"])
    bundle["input_doc"]["confidence"] = 0.73

    assert not verify_replay_bundle(bundle)
