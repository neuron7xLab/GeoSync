#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def _build_base() -> list[str]:
    now_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return [
        "python",
        "artifacts/inference_extrapolation_validator/generate_artifact.py",
        "generate",
        "--context",
        "verified dataset v1",
        "--hypothesis",
        "h",
        "--model-id",
        "m",
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
        "x,y",
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
        now_utc,
        "--witness-review-hash",
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "--generated-at-utc",
        now_utc,
        "--purpose-id",
        "risk_gate_v1",
        "--purpose-statement",
        "Prevent unverified extrapolation promotion",
        "--purpose-alignment-score",
        "0.9",
        "--purpose-drift-check",
        "0.1",
    ]


def run_expect(cmd: list[str], code: int) -> subprocess.CompletedProcess[str]:
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == code, (r.returncode, r.stderr)
    return r


def main() -> None:
    base = _build_base()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.json"
        run_expect(base + ["--out", str(p)], 0)
        run_expect(
            [
                "python",
                "artifacts/inference_extrapolation_validator/generate_artifact.py",
                "verify",
                "--artifact",
                str(p),
            ],
            0,
        )
        cmd = base.copy()
        cmd[cmd.index("--tests-run") + 1] = "boundary_check,replay_test"
        run_expect(cmd + ["--out", str(p)], 2)
        cmd = base.copy()
        cmd.remove("--witness-review-hash")
        cmd.remove("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
        run_expect(cmd + ["--out", str(p)], 2)
        data = json.loads(p.read_text())
        data["sha256"] = "0" * 64
        p.write_text(json.dumps(data))
        run_expect(
            [
                "python",
                "artifacts/inference_extrapolation_validator/generate_artifact.py",
                "verify",
                "--artifact",
                str(p),
            ],
            2,
        )
        cmd = base.copy()
        cmd[cmd.index("--test-passed") + 1] = "false"
        run_expect(cmd + ["--out", str(p)], 2)
        cmd = base.copy()
        cmd[cmd.index("--null-models-run") + 1] = "replay_consistency_null"
        run_expect(cmd + ["--out", str(p)], 2)
        cmd = base.copy()
        cmd[cmd.index("--result") + 1] = "killed_with_counterexample"
        cmd[cmd.index("--test-passed") + 1] = "false"
        run_expect(cmd + ["--out", str(p)], 0)
    print("FALSIFIER PASS")


if __name__ == "__main__":
    main()
