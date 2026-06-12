#!/usr/bin/env python3
"""Validate a deterministic JSON artifact contract.

The validator is intentionally stdlib-only. It checks structure and evidence
semantics; it does not execute repository tests and does not assign scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REQUIRED_TOP_LEVEL = (
    "contract_type",
    "objective",
    "inputs",
    "execution_plan",
    "python_runner_contract",
    "validation_tests",
    "failure_register",
    "evidence_status",
    "output_constraints",
    "next_deterministic_action",
    "final_verdict",
)

VALID_STATUSES = {
    "EXTERNAL_VERIFIED",
    "CI_VERIFIED",
    "TOOL_VERIFIED",
    "HUMAN_REVIEWED",
    "CANDIDATE_NOT_VALIDATED",
    "BLOCKED",
    "REJECTED",
}

VALID_EVIDENCE_SOURCES = {
    "unavailable",
    "python_runner",
    "tool_log",
    "ci_log",
    "parser_output",
    "benchmark_output",
    "human_review",
}

VALIDATED_STATUSES = {
    "EXTERNAL_VERIFIED",
    "CI_VERIFIED",
    "TOOL_VERIFIED",
    "HUMAN_REVIEWED",
}


def _load(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, [f"invalid_json:{exc.lineno}:{exc.colno}"]
    except OSError as exc:
        return None, [f"read_error:{exc}"]
    if not isinstance(data, dict):
        return None, ["root_must_be_object"]
    return data, []


def validate_contract(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    for key in REQUIRED_TOP_LEVEL:
        if key not in data:
            errors.append(f"missing_top_level:{key}")

    evidence = data.get("evidence_status")
    verdict = data.get("final_verdict")

    if not isinstance(evidence, dict):
        errors.append("evidence_status_must_be_object")
        evidence = {}
    if not isinstance(verdict, dict):
        errors.append("final_verdict_must_be_object")
        verdict = {}

    external = evidence.get("external_evidence_available")
    if not isinstance(external, bool):
        errors.append("external_evidence_available_must_be_bool")
        external = False

    source = evidence.get("evidence_source")
    if source not in VALID_EVIDENCE_SOURCES:
        errors.append("invalid_evidence_source")

    score = evidence.get("score")
    confidence = evidence.get("confidence")
    evidence_status = evidence.get("status")
    final_status = verdict.get("status")
    tests_executed = evidence.get("tests_executed")
    tests_not_executed = evidence.get("tests_not_executed")

    if evidence_status not in VALID_STATUSES:
        errors.append("invalid_evidence_status")
    if final_status not in VALID_STATUSES:
        errors.append("invalid_final_status")

    if not isinstance(tests_executed, list):
        errors.append("tests_executed_must_be_list")
    if not isinstance(tests_not_executed, list):
        errors.append("tests_not_executed_must_be_list")

    if score is not None and (not isinstance(score, (int, float)) or isinstance(score, bool)):
        errors.append("score_must_be_number_or_null")

    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        errors.append("confidence_must_be_number")
    elif not 0 <= float(confidence) <= 1:
        errors.append("confidence_out_of_range")

    if not external:
        if score is not None:
            errors.append("score_requires_external_evidence")
        if confidence is not None and isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            if float(confidence) > 0.65:
                errors.append("confidence_too_high_without_evidence")
        if evidence_status != "CANDIDATE_NOT_VALIDATED":
            errors.append("evidence_status_must_be_candidate_without_evidence")
        if final_status != "CANDIDATE_NOT_VALIDATED":
            errors.append("final_status_must_be_candidate_without_evidence")

    if external:
        if source == "unavailable":
            errors.append("external_evidence_source_required")
        if final_status in VALIDATED_STATUSES and not tests_executed:
            errors.append("validated_status_requires_tests_executed")

    constraints = data.get("output_constraints")
    if not isinstance(constraints, dict):
        errors.append("output_constraints_must_be_object")
    else:
        for key in (
            "json_only",
            "no_markdown",
            "no_prose",
            "no_self_scoring",
            "no_simulated_tests",
            "no_validation_without_evidence",
        ):
            if constraints.get(key) is not True:
                errors.append(f"constraint_not_true:{key}")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a JSON artifact contract")
    parser.add_argument("path", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    data, errors = _load(args.path)
    if data is not None:
        errors.extend(validate_contract(data))

    result = {
        "schema_version": "json_artifact_contract_validator.v1",
        "input": str(args.path),
        "status": "OK" if not errors else "ERROR",
        "errors": errors,
    }

    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
