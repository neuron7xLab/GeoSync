from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "run_rust_accel_contract.py"
spec = importlib.util.spec_from_file_location("run_rust_accel_contract", MODULE_PATH)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


class RunRustAccelContractTests(unittest.TestCase):
    def test_dry_run_emits_structured_plan_for_all_acceptance_criteria(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = runner.main(["--dry-run", "--format", "json"])

        self.assertEqual(exit_code, 0)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["status"], "DRY_RUN")
        self.assertTrue(report["dry_run"])
        self.assertEqual(report["artifact_id"], "rust-geosync-accel-ci-contract")
        self.assertGreaterEqual(len(report["criteria"]), 4)
        self.assertTrue(all(item["status"] == "DRY_RUN" for item in report["criteria"]))
        self.assertTrue(all(item["exit_code"] == 0 for item in report["criteria"]))
        self.assertIn("tool_versions", report)
        self.assertTrue(all("stdout_sha256" in item for item in report["criteria"]))

    def test_only_unknown_criterion_fails_closed(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            exit_code = runner.main(["--only", "not_a_real_gate", "--format", "json"])

        self.assertEqual(exit_code, 2)
        self.assertIn("unknown acceptance criterion id", stderr.getvalue())

    def test_contract_rest_state_can_execute_as_single_criterion(self) -> None:
        report = runner.build_report(
            runner.DEFAULT_CONTRACT,
            ["contract_rest_state"],
            dry_run=False,
        )

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(len(report["criteria"]), 1)
        criterion = report["criteria"][0]
        self.assertEqual(criterion["id"], "contract_rest_state")
        self.assertEqual(criterion["exit_code"], 0)
        self.assertIn("state=REST action_potential=0", criterion["stdout"])


if __name__ == "__main__":
    unittest.main()
