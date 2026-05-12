#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import jsonschema

ROOT = Path(__file__).resolve().parent
ARTIFACT_SCHEMA_PATH = ROOT / "schema" / "artifact.schema.json"
SPEC_SCHEMA_PATH = ROOT / "schema" / "spec.schema.json"
SPEC_PATH = ROOT / "spec.json"

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
RISK_REQUIRED_TESTS = {
    "low": {"boundary_check", "replay_test"},
    "medium": {"boundary_check", "null_model_comparison", "replay_test"},
    "high": {
        "adversarial_test",
        "boundary_check",
        "null_model_comparison",
        "counterexample_search",
        "replay_test",
    },
}
RISK_REQUIRED_NULL_MODELS = {
    "low": {"replay_consistency_null"},
    "medium": {"baseline_random_or_stationary", "replay_consistency_null"},
    "high": {
        "baseline_random_or_stationary",
        "boundary_shuffle",
        "counterexample_search_null",
        "replay_consistency_null",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inference Extrapolation Validator v3.0.0")
    sub = parser.add_subparsers(dest="mode", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--context", required=True)
    g.add_argument("--hypothesis", required=True)
    g.add_argument("--model-id", required=True)
    g.add_argument("--model-version", required=True)
    g.add_argument("--seed", required=True, type=int)
    g.add_argument("--prompt-hash", required=True)
    g.add_argument("--requirements-lock-sha256", required=True)
    g.add_argument("--risk", choices=["low", "medium", "high"], required=True)
    g.add_argument("--result", choices=["survived", "killed_with_counterexample"], required=True)
    g.add_argument("--test-passed", choices=["true", "false"], required=True)
    g.add_argument("--tests-run", required=True)
    g.add_argument("--falsifiers-run", required=True)
    g.add_argument("--failure-modes-checked", required=True)
    g.add_argument("--null-models-run", required=True)
    g.add_argument(
        "--null-model-results",
        required=True,
        help='JSON object keyed by null model id, e.g. {"baseline_random_or_stationary":0.4}',
    )
    g.add_argument("--hypothesis-score", required=True, type=float)
    g.add_argument("--reality-probe-id", required=True)
    g.add_argument("--reality-substrate", required=True)
    g.add_argument("--observation-hash", required=True)
    g.add_argument("--drift-score", required=True, type=float)
    g.add_argument(
        "--witness-status", choices=["approved", "rejected", "not_required"], required=True
    )
    g.add_argument("--witness-reviewer-id", default=None)
    g.add_argument("--witness-review-timestamp-utc", default=None)
    g.add_argument("--witness-review-hash", default=None)
    g.add_argument("--witness-notes-hash", default=None)
    g.add_argument("--generated-at-utc", default=None)
    g.add_argument("--owner", default="inference_extrapolation_working_group")
    g.add_argument("--purpose-id", required=True)
    g.add_argument("--purpose-statement", required=True)
    g.add_argument("--purpose-alignment-score", required=True, type=float)
    g.add_argument("--purpose-drift-check", required=True, type=float)
    g.add_argument("--api-version", default="1.0")
    g.add_argument("--stochastic-tolerance", type=float, default=0.0)
    g.add_argument("--noise-seed", type=int, default=0)
    g.add_argument("--out", required=True)

    v = sub.add_parser("verify")
    v.add_argument("--artifact", required=True)
    return parser


def parse_csv_list(raw: str) -> list[str]:
    values = [x.strip() for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("comma-separated list cannot be empty")
    return values


def parse_utc_timestamp(raw: str | None) -> str:
    value = raw or datetime.now(timezone.utc).isoformat()
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("timestamp must include timezone")
    return dt.astimezone(timezone.utc).isoformat()


def canonical_json_bytes(obj: dict[str, Any]) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def compute_artifact_sha256(artifact_without_sha: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(artifact_without_sha)).hexdigest()


def verify_artifact_sha256(artifact: dict[str, Any]) -> bool:
    if "sha256" not in artifact:
        return False
    copy = dict(artifact)
    observed = copy.pop("sha256")
    if not isinstance(observed, str):
        return False
    expected = compute_artifact_sha256(copy)
    return observed == expected


def _load_json(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _validate_hex64(name: str, value: str) -> None:
    if not HEX64_RE.match(value):
        raise ValueError(f"{name} must be 64-char lowercase hex")


def _validate_witness(risk: str, witness: dict[str, Any]) -> None:
    status = witness["status"]
    if risk == "high":
        if status != "approved":
            raise ValueError("high risk requires witness approved")
        for f in ["reviewer_id", "review_timestamp_utc", "review_hash"]:
            if witness.get(f) in (None, ""):
                raise ValueError(f"high risk requires witness.{f}")
    if status == "rejected" and witness.get("declared_result") == "survived":
        raise ValueError("rejected witness cannot produce survived claim")


def compute_acc_conflict_score(artifact: dict[str, Any]) -> tuple[float, dict[str, float]]:
    witness_status = artifact["witness"]["status"]
    witness_uncertainty = (
        1.0 if witness_status == "rejected" else (0.6 if witness_status == "not_required" else 0.0)
    )

    hypothesis_score = artifact["metrics"]["hypothesis_score"]
    max_null = max(artifact["null_model_results"].values())
    null_contradiction = max(0.0, max_null - hypothesis_score)

    falsifier_disagreement = 0.0
    if "counterexample_search" not in artifact["falsifiers_run"]:
        falsifier_disagreement = 0.9

    purpose_drift = artifact["purpose"]["drift_check"]
    external_drift = artifact["external_falsification"]["drift_score"]

    schema_claim_inconsistency = 0.0
    if artifact["result"] == "survived" and artifact["claim_status"] != "EVIDENCE":
        schema_claim_inconsistency = 1.0

    boundary_violation_risk = 0.0
    if artifact["claim_boundary"] != "VERIFIED_EVIDENCE" and artifact["claim_status"] == "EVIDENCE":
        boundary_violation_risk = 1.0

    vector = {
        "falsifier_disagreement": min(1.0, falsifier_disagreement),
        "null_model_contradiction": min(1.0, null_contradiction),
        "witness_uncertainty": min(1.0, witness_uncertainty),
        "purpose_drift": min(1.0, purpose_drift),
        "external_falsification_drift": min(1.0, external_drift),
        "schema_claim_inconsistency": min(1.0, schema_claim_inconsistency),
        "evidence_boundary_violation_risk": min(1.0, boundary_violation_risk),
    }
    score = max(vector.values())
    return score, vector


def stn_hyperdirect_gate(artifact: dict[str, Any], threshold: float = 0.5) -> None:
    score, vector = compute_acc_conflict_score(artifact)
    stochastic = artifact.get("stochastic_falsification", {"tolerance": 0.0, "noise_seed": 0})
    rng = random.Random(stochastic.get("noise_seed", 0))
    adaptive_noise = rng.uniform(0.0, max(0.0, stochastic.get("tolerance", 0.0)))
    adaptive_threshold = min(1.0, threshold + adaptive_noise)
    artifact["stn_hyperdirect_gate"] = {
        "enabled": True,
        "acc_conflict_score": score,
        "threshold": threshold,
        "adaptive_threshold": adaptive_threshold,
        "adaptive_noise": adaptive_noise,
        "action": "exit_2_fail_closed",
        "conflict_vector": vector,
    }
    if score > 0.8 or score > adaptive_threshold:
        artifact["result"] = "killed_with_counterexample"
        artifact["claim_status"] = "KILLED"
        artifact["claim_boundary"] = "FALSIFIED_OR_REJECTED"
        raise ValueError("hyperdirect fail-closed triggered (computed conflict above threshold)")


def validate_contract(artifact: dict[str, Any]) -> None:
    risk = artifact["risk_class"]
    result = artifact["result"]
    tests_run = set(artifact["tests_run"])
    required_tests = RISK_REQUIRED_TESTS[risk]
    if not required_tests.issubset(tests_run):
        raise ValueError("missing required tests")

    null_models_run = set(artifact["null_models_run"])
    required_nulls = RISK_REQUIRED_NULL_MODELS[risk]
    if artifact["claim_status"] == "EVIDENCE" and not required_nulls.issubset(null_models_run):
        raise ValueError("missing required null models")

    if artifact["test_passed"] is False and result == "survived":
        raise ValueError("survived cannot have test_passed=false")

    if result == "survived":
        if (
            artifact["claim_status"] != "EVIDENCE"
            or artifact["claim_boundary"] != "VERIFIED_EVIDENCE"
        ):
            raise ValueError("invalid survived mapping")
    else:
        if (
            artifact["claim_status"] != "KILLED"
            or artifact["claim_boundary"] != "FALSIFIED_OR_REJECTED"
        ):
            raise ValueError("invalid killed mapping")

    _validate_witness(risk, artifact["witness"])
    _validate_hex64("prompt_hash", artifact["reproducibility"]["prompt_hash"])
    if not artifact.get("purpose", {}).get("id") or not artifact.get("purpose", {}).get(
        "statement"
    ):
        raise ValueError("purpose metadata required")
    if artifact["purpose"]["alignment_score"] < 0.7:
        raise ValueError("purpose alignment score below threshold")
    if artifact["purpose"]["drift_check"] > 0.3:
        raise ValueError("purpose drift too high")

    if artifact["claim_status"] == "EVIDENCE":
        rp = artifact["external_falsification"]
        if rp["drift_score"] > 0.5:
            raise ValueError("external falsification drift too high for evidence")
        if not rp["probe_id"] or not rp["substrate"]:
            raise ValueError("external falsification metadata incomplete")

    stn_hyperdirect_gate(artifact, threshold=0.5)

    # witness freshness (max 30 days)
    ts = artifact["witness"].get("review_timestamp_utc")
    if ts:
        wdt = datetime.fromisoformat(ts)
        if datetime.now(timezone.utc) - wdt > timedelta(days=30):
            raise ValueError("witness review timestamp too old")

    # null model result consistency
    nm_run = set(artifact["null_models_run"])
    nm_keys = set(artifact["null_model_results"].keys())
    if nm_run != nm_keys:
        raise ValueError("null_model_results keys must match null_models_run")
    for nm_key, nm_value in artifact["null_model_results"].items():
        if isinstance(nm_value, bool) or not isinstance(nm_value, (int, float)):
            raise ValueError(
                f"null_model_results[{nm_key!r}] must be numeric (int or float), got "
                f"{type(nm_value).__name__}"
            )

    # evidence must beat null models
    if artifact["claim_status"] == "EVIDENCE":
        if artifact["metrics"]["hypothesis_score"] <= max(artifact["null_model_results"].values()):
            raise ValueError("hypothesis_score must exceed all null model results")

    if artifact.get("api_version") != "1.0":
        raise ValueError("unsupported api_version")


def build_artifact(args: argparse.Namespace) -> dict[str, Any]:
    _validate_hex64("prompt_hash", args.prompt_hash)
    _validate_hex64("requirements_lock_sha256", args.requirements_lock_sha256)
    _validate_hex64("observation_hash", args.observation_hash)
    generated_at_utc = parse_utc_timestamp(args.generated_at_utc)
    witness_ts = (
        parse_utc_timestamp(args.witness_review_timestamp_utc)
        if args.witness_review_timestamp_utc
        else None
    )
    witness = {
        "status": args.witness_status,
        "declared_result": args.result,
        "reviewer_id": args.witness_reviewer_id,
        "review_timestamp_utc": witness_ts,
        "review_hash": args.witness_review_hash,
        "notes_hash": args.witness_notes_hash,
    }
    if witness["review_hash"] is not None:
        _validate_hex64("witness.review_hash", witness["review_hash"])
    if witness["notes_hash"] is not None:
        _validate_hex64("witness.notes_hash", witness["notes_hash"])

    test_passed = args.test_passed == "true"
    if args.result == "survived":
        claim_status = "EVIDENCE"
        claim_boundary = "VERIFIED_EVIDENCE"
    else:
        claim_status = "KILLED"
        claim_boundary = "FALSIFIED_OR_REJECTED"

    null_model_results = json.loads(args.null_model_results)
    if not isinstance(null_model_results, dict) or not null_model_results:
        raise ValueError("null_model_results must be non-empty JSON object")
    for nm_key, nm_value in null_model_results.items():
        if isinstance(nm_value, bool) or not isinstance(nm_value, (int, float)):
            raise ValueError(
                f"null_model_results[{nm_key!r}] must be numeric (int or float), got "
                f"{type(nm_value).__name__}"
            )

    artifact = {
        "module": "inference_extrapolation_validator",
        "api_version": args.api_version,
        "stochastic_falsification": {
            "tolerance": args.stochastic_tolerance,
            "noise_seed": args.noise_seed,
        },
        "version": "3.0.0",
        "source_context_hash": hashlib.sha256(args.context.encode("utf-8")).hexdigest(),
        "hypothesis": args.hypothesis,
        "tests_run": sorted(set(parse_csv_list(args.tests_run))),
        "falsifiers_run": sorted(set(parse_csv_list(args.falsifiers_run))),
        "failure_modes_checked": sorted(set(parse_csv_list(args.failure_modes_checked))),
        "null_models_run": sorted(set(parse_csv_list(args.null_models_run))),
        "null_model_results": null_model_results,
        "test_passed": test_passed,
        "result": args.result,
        "claim_status": claim_status,
        "claim_boundary": claim_boundary,
        "risk_class": args.risk,
        "metrics": {
            "hypothesis_score": args.hypothesis_score,
        },
        "witness": witness,
        "purpose": {
            "id": args.purpose_id,
            "statement": args.purpose_statement,
            "framework": "DIKWP",
            "alignment_score": args.purpose_alignment_score,
            "drift_check": args.purpose_drift_check,
        },
        "external_falsification": {
            "probe_id": args.reality_probe_id,
            "substrate": args.reality_substrate,
            "observation_hash": args.observation_hash,
            "drift_score": args.drift_score,
        },
        "reproducibility": {
            "model_id": args.model_id,
            "model_version": args.model_version,
            "seed": args.seed,
            "prompt_hash": args.prompt_hash,
            "decoding": {"temperature": 0.0, "top_p": 1.0},
            "environment": {
                "python": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "requirements_lock_sha256": args.requirements_lock_sha256,
            },
            "code": {"git_head": subprocess.getoutput("git rev-parse HEAD 2>/dev/null") or None},
            "command_line": " ".join(sys.argv),
            "command_line_sha256": hashlib.sha256(" ".join(sys.argv).encode("utf-8")).hexdigest(),
        },
        "generated_at_utc": generated_at_utc,
        "owner": args.owner,
        "rollback": {"mode": "force_kill", "command": "set result=killed_with_counterexample"},
        "decision_trace": [
            "schema",
            "tests",
            "null_models",
            "witness",
            "purpose",
            "stn_gate",
            "sha256",
        ],
        "epistemic_unit": "closed_validation_cycle",
        "truth_claim": "bounded_contractual_evidence_only",
        "no_overclaim": True,
    }
    validate_contract(artifact)
    artifact["sha256"] = compute_artifact_sha256(artifact)
    return artifact


def _validate_artifact_schema(artifact: dict[str, Any]) -> None:
    schema = _load_json(ARTIFACT_SCHEMA_PATH)
    jsonschema.validate(instance=artifact, schema=schema)


def run(args: argparse.Namespace) -> int:
    if args.mode == "generate":
        artifact = build_artifact(args)
        _validate_artifact_schema(artifact)
        out = Path(args.out)
        out.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"event": "artifact_generated", "path": str(out), "status": "ok"}))
        return 0

    artifact_path = Path(args.artifact)
    try:
        artifact = _load_json(artifact_path)
    except Exception as exc:
        print(f"parse_failure: {exc}", file=sys.stderr)
        return 3
    try:
        _validate_artifact_schema(artifact)
        validate_contract(artifact)
        if not verify_artifact_sha256(artifact):
            raise ValueError("sha256 verification failed")
    except Exception as exc:
        print(f"contract_violation: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps({"event": "artifact_verify", "status": "pass", "artifact": str(artifact_path)})
    )
    return 0


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        code = run(args)
    except ValueError as exc:
        print(f"contract_violation: {exc}", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
