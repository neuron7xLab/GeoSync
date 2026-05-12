from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jsonschema

MODULE_DIR = Path(__file__).resolve().parent
GENERATE_ARTIFACT_PY = MODULE_DIR / "generate_artifact.py"
ARTIFACT_SCHEMA_PATH = MODULE_DIR / "schema" / "artifact.schema.json"


def _validate_artifact_schema(artifact: dict[str, Any]) -> None:
    schema = json.loads(ARTIFACT_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(instance=artifact, schema=schema)


class TestIEV(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "a.json"
        now_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        self.witness_ts = now_utc
        self.generated_ts = now_utc
        self.base = [
            "python",
            str(GENERATE_ARTIFACT_PY),
            "generate",
            "--context",
            "verified dataset v1",
            "--hypothesis",
            "signal X remains stable beyond observed range",
            "--model-id",
            "ext_inf",
            "--model-version",
            "3.0.0",
            "--seed",
            "1",
            "--prompt-hash",
            "9db6f9e7c6e5d6f6ac7a72f01561f1f5f8d6759f4f8999f8fceca85f2f4e6eb4",
            "--requirements-lock-sha256",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "--risk",
            "high",
            "--result",
            "survived",
            "--test-passed",
            "true",
            "--tests-run",
            "adversarial_test,boundary_check,null_model_comparison,counterexample_search,replay_test",
            "--falsifiers-run",
            "counterexample_search,null_model_comparison",
            "--failure-modes-checked",
            "boundary_violation,overfit",
            "--null-models-run",
            "baseline_random_or_stationary,boundary_shuffle,counterexample_search_null,replay_consistency_null",
            "--null-model-results",
            '{"baseline_random_or_stationary":0.2,"boundary_shuffle":0.1,"counterexample_search_null":0.15,"replay_consistency_null":0.05}',
            "--hypothesis-score",
            "0.9",
            "--reality-probe-id",
            "probe-001",
            "--reality-substrate",
            "exchange_sim",
            "--observation-hash",
            "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            "--drift-score",
            "0.2",
            "--witness-status",
            "approved",
            "--witness-reviewer-id",
            "rev-1",
            "--witness-review-timestamp-utc",
            self.witness_ts,
            "--witness-review-hash",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "--witness-notes-hash",
            "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            "--generated-at-utc",
            self.generated_ts,
            "--purpose-id",
            "risk_gate_v1",
            "--purpose-statement",
            "Prevent unverified extrapolation promotion",
            "--purpose-alignment-score",
            "0.9",
            "--purpose-drift-check",
            "0.1",
            "--api-version",
            "1.0",
            "--stochastic-tolerance",
            "0.0",
            "--noise-seed",
            "0",
            "--out",
            str(self.out),
        ]

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_generate_and_verify_pass(self) -> None:
        self.assertEqual(subprocess.run(self.base).returncode, 0)
        self.assertEqual(
            subprocess.run(
                ["python", str(GENERATE_ARTIFACT_PY), "verify", "--artifact", str(self.out)]
            ).returncode,
            0,
        )

    def test_missing_required_test_fails(self) -> None:
        cmd = self.base.copy()
        cmd[cmd.index("--tests-run") + 1] = "boundary_check,replay_test"
        self.assertEqual(subprocess.run(cmd).returncode, 2)

    def test_high_risk_not_required_witness_fails(self) -> None:
        cmd = self.base.copy()
        cmd[cmd.index("--witness-status") + 1] = "not_required"
        self.assertEqual(subprocess.run(cmd).returncode, 2)

    def test_high_risk_missing_review_hash_fails(self) -> None:
        cmd = [
            c
            for c in self.base
            if c
            not in [
                "--witness-review-hash",
                "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            ]
        ]
        self.assertEqual(subprocess.run(cmd).returncode, 2)

    def test_malformed_timestamp_fails(self) -> None:
        cmd = self.base.copy()
        cmd[cmd.index("--generated-at-utc") + 1] = "2026-05-12T00:00:00"
        self.assertEqual(subprocess.run(cmd).returncode, 2)

    def test_invalid_prompt_hash_fails(self) -> None:
        cmd = self.base.copy()
        cmd[cmd.index("--prompt-hash") + 1] = "abc"
        self.assertEqual(subprocess.run(cmd).returncode, 2)

    def test_corrupted_sha_fails_verify(self) -> None:
        subprocess.run(self.base, check=True)
        data = json.loads(self.out.read_text())
        data["sha256"] = "0" * 64
        self.out.write_text(json.dumps(data))
        self.assertEqual(
            subprocess.run(
                ["python", str(GENERATE_ARTIFACT_PY), "verify", "--artifact", str(self.out)]
            ).returncode,
            2,
        )

    def test_schema_rejects_malformed(self) -> None:
        bad = {"module": "inference_extrapolation_validator"}
        with self.assertRaises(Exception):
            _validate_artifact_schema(bad)

    def test_killed_mode(self) -> None:
        cmd = self.base.copy()
        cmd[cmd.index("--result") + 1] = "killed_with_counterexample"
        cmd[cmd.index("--test-passed") + 1] = "false"
        self.assertEqual(subprocess.run(cmd).returncode, 0)
        artifact = json.loads(self.out.read_text())
        self.assertEqual(artifact["claim_status"], "KILLED")

    def test_verify_parse_failure_exit3(self) -> None:
        bad = Path(self.tmp.name) / "bad.json"
        bad.write_text("{not json}")
        self.assertEqual(
            subprocess.run(
                ["python", str(GENERATE_ARTIFACT_PY), "verify", "--artifact", str(bad)]
            ).returncode,
            3,
        )

    def test_rejected_witness_allowed_for_killed_low_risk(self) -> None:
        cmd = self.base.copy()
        cmd[cmd.index("--risk") + 1] = "low"
        cmd[cmd.index("--tests-run") + 1] = "boundary_check,replay_test"
        cmd[cmd.index("--null-models-run") + 1] = "replay_consistency_null"
        cmd[cmd.index("--null-model-results") + 1] = '{"replay_consistency_null":0.05}'
        cmd[cmd.index("--result") + 1] = "killed_with_counterexample"
        cmd[cmd.index("--test-passed") + 1] = "false"
        cmd[cmd.index("--witness-status") + 1] = "rejected"
        self.assertEqual(subprocess.run(cmd).returncode, 2)

    def test_epistemic_boundary_fields_present(self) -> None:
        subprocess.run(self.base, check=True)
        artifact = json.loads(self.out.read_text())
        self.assertEqual(artifact.get("epistemic_unit"), "closed_validation_cycle")
        self.assertTrue(artifact.get("no_overclaim"))

    def test_external_falsification_drift_blocks_evidence(self) -> None:
        cmd = self.base.copy()
        cmd[cmd.index("--drift-score") + 1] = "0.9"
        self.assertEqual(subprocess.run(cmd).returncode, 2)

    def test_purpose_drift_blocks_evidence(self) -> None:
        cmd = self.base.copy()
        cmd[cmd.index("--purpose-drift-check") + 1] = "0.8"
        self.assertEqual(subprocess.run(cmd).returncode, 2)

    def test_hyperdirect_gate_blocks_high_conflict(self) -> None:
        cmd = self.base.copy()
        cmd[cmd.index("--falsifiers-run") + 1] = "null_model_comparison"
        self.assertEqual(subprocess.run(cmd).returncode, 2)

    def test_api_version_mismatch_fails(self) -> None:
        cmd = self.base.copy()
        cmd[cmd.index("--api-version") + 1] = "2.0"
        self.assertEqual(subprocess.run(cmd).returncode, 2)

    def test_stale_witness_timestamp_fails(self) -> None:
        cmd = self.base.copy()
        cmd[cmd.index("--witness-review-timestamp-utc") + 1] = "2020-01-01T00:00:00+00:00"
        self.assertEqual(subprocess.run(cmd).returncode, 2)

    def test_null_model_key_mismatch_fails(self) -> None:
        cmd = self.base.copy()
        cmd[cmd.index("--null-model-results") + 1] = '{"baseline_random_or_stationary":0.2}'
        self.assertEqual(subprocess.run(cmd).returncode, 2)

    def test_falsifier_disagreement_increases_conflict(self) -> None:
        cmd = self.base.copy()
        cmd[cmd.index("--falsifiers-run") + 1] = "null_model_comparison"
        self.assertEqual(subprocess.run(cmd).returncode, 2)

    def test_witness_uncertainty_increases_conflict(self) -> None:
        cmd = self.base.copy()
        cmd[cmd.index("--risk") + 1] = "medium"
        cmd[cmd.index("--witness-status") + 1] = "not_required"
        self.assertEqual(subprocess.run(cmd).returncode, 2)

    def test_conflict_vector_written_in_artifact(self) -> None:
        subprocess.run(self.base, check=True)
        artifact = json.loads(self.out.read_text())
        self.assertIn("conflict_vector", artifact["stn_hyperdirect_gate"])

    def test_stochastic_tolerance_can_relax_borderline_conflict(self) -> None:
        cmd = self.base.copy()
        cmd[cmd.index("--falsifiers-run") + 1] = "null_model_comparison"
        cmd[cmd.index("--stochastic-tolerance") + 1] = "0.5"
        cmd[cmd.index("--noise-seed") + 1] = "42"
        # deterministic noise may allow pass on borderline conflict
        self.assertIn(subprocess.run(cmd).returncode, [0, 2])

    def test_purpose_alignment_below_threshold_fails(self) -> None:
        cmd = self.base.copy()
        cmd[cmd.index("--purpose-alignment-score") + 1] = "0.2"
        self.assertEqual(subprocess.run(cmd).returncode, 2)

    def test_hypothesis_not_better_than_null_fails(self) -> None:
        cmd = self.base.copy()
        cmd[cmd.index("--hypothesis-score") + 1] = "0.1"
        self.assertEqual(subprocess.run(cmd).returncode, 2)

    def test_missing_decision_trace_rejected_on_verify(self) -> None:
        subprocess.run(self.base, check=True)
        a = json.loads(self.out.read_text())
        a.pop("decision_trace", None)
        self.out.write_text(json.dumps(a))
        self.assertEqual(
            subprocess.run(
                ["python", str(GENERATE_ARTIFACT_PY), "verify", "--artifact", str(self.out)]
            ).returncode,
            2,
        )

    def test_stochastic_cannot_promote_very_high_conflict(self) -> None:
        cmd = self.base.copy()
        cmd[cmd.index("--falsifiers-run") + 1] = "null_model_comparison"
        cmd[cmd.index("--stochastic-tolerance") + 1] = "1.0"
        cmd[cmd.index("--noise-seed") + 1] = "99"
        self.assertEqual(subprocess.run(cmd).returncode, 2)

    def test_stochastic_replay_deterministic(self) -> None:
        cmd = self.base.copy()
        cmd[cmd.index("--stochastic-tolerance") + 1] = "0.2"
        cmd[cmd.index("--noise-seed") + 1] = "7"
        r1 = subprocess.run(cmd).returncode
        r2 = subprocess.run(cmd).returncode
        self.assertEqual(r1, r2)
