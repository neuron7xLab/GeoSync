from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def test_claim_gate_requires_terminal_pass() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/ci/check_dopamine_claim_promotion.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "PASS"


def test_claim_evaluator_blocks_unsupported_dopamine_promotion() -> None:
    module: Any = importlib.import_module("scripts.ci.check_dopamine_claim_promotion")
    reasons = module.evaluate_text(
        "Dopamine produces market edge and biological dopamine replication.",
        eval_exists=False,
        manifest_exists=False,
    )
    assert any("market edge" in reason for reason in reasons)
    assert any("biological dopamine replication" in reason for reason in reasons)


def test_claim_evaluator_ignores_unrelated_governance_words() -> None:
    module: Any = importlib.import_module("scripts.ci.check_dopamine_claim_promotion")
    text = "General claim is anchored and tested.\n\nDopamine remains structural and research-gated."
    reasons = module.evaluate_text(text, eval_exists=False, manifest_exists=False)
    assert reasons == []


def test_claim_evaluator_allows_structural_research_boundary() -> None:
    module: Any = importlib.import_module("scripts.ci.check_dopamine_claim_promotion")
    reasons = module.evaluate_text(
        "Dopamine remains structural and research-gated with bounded telemetry.",
        eval_exists=False,
        manifest_exists=False,
    )
    assert reasons == []
