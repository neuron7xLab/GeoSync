#!/usr/bin/env python3
"""One-command verification harness for the BBB-NVU artifact."""

from __future__ import annotations

import json
import py_compile
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PYTHON_FILES = [
    ROOT / "src" / "deterministic_engine.py",
    ROOT / "src" / "runtime_boundary.py",
    ROOT / "src" / "audit.py",
    ROOT / "src" / "observability.py",
    ROOT / "src" / "operational_kernel.py",
    ROOT / "scripts" / "adversarial_auditor.py",
    ROOT / "tools" / "traceability.py",
    ROOT / "tools" / "compile_invariants.py",
]

TEST_FILES = [
    ROOT / "tests" / "test_deterministic_engine.py",
    ROOT / "tests" / "test_invariants.py",
    ROOT / "tests" / "test_adversarial_auditor.py",
    ROOT / "tests" / "test_traceability.py",
    ROOT / "tests" / "test_l1_data_quality_gate.py",
    ROOT / "tests" / "test_runtime_boundary.py",
    ROOT / "tests" / "test_audit_replay.py",
    ROOT / "tests" / "test_observability_incidents.py",
    ROOT / "tests" / "test_operational_kernel.py",
]


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO, check=True)


def parse_json_artifacts() -> None:
    for path in [
        *sorted((ROOT / "schemas").glob("*.json")),
        *sorted((ROOT / "examples").glob("*.json")),
        ROOT / "tests" / "adversarial_golden_vectors.json",
    ]:
        json.loads(path.read_text(encoding="utf-8"))


def compile_python() -> None:
    for path in PYTHON_FILES:
        py_compile.compile(str(path), doraise=True)


def operational_smoke() -> None:
    from BBB_NVU_Cognitive_Noise_Gate_2026.src.deterministic_engine import load_yaml
    from BBB_NVU_Cognitive_Noise_Gate_2026.src.operational_kernel import OperationalKernel, verify_operational_envelope

    rules = load_yaml(ROOT / "config" / "risk_rules.yaml")
    request = {
        "input_doc": {
            "subject_id": "S-SMOKE",
            "critical_data_invalid": False,
            "confidence": 0.74,
            "domain_indices": {"BSI": 42, "NRI": 28, "VML": 62, "GRS": 48, "CNI": 31},
            "degradations": ["sleep_proxy_low_specificity"],
        },
        "source_id": "smoke.json",
    }
    envelope = OperationalKernel(rules, engine_hash="verification-smoke").execute(
        [request], created_at="2026-06-03T00:00:00Z"
    )
    assert envelope.outputs[0]["risk"]["risk_state"] == "YELLOW_WATCH"
    assert envelope.manifest["replay_verified"] == [True]
    assert verify_operational_envelope(envelope)
    assert envelope.envelope_hash


def main() -> int:
    run([sys.executable, str(ROOT / "tools" / "traceability.py")])
    run([sys.executable, "-m", "pytest", "-q", "-W", "ignore::DeprecationWarning", *map(str, TEST_FILES)])
    compile_python()
    parse_json_artifacts()
    operational_smoke()
    run(["git", "diff", "--check", "--", str(ROOT.relative_to(REPO))])
    print("verification ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
