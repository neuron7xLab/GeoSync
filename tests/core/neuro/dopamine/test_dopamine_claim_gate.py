from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]


def test_claim_gate_requires_terminal_pass(monkeypatch: Any, tmp_path: Path) -> None:
    module: Any = importlib.import_module("scripts.ci.check_dopamine_claim_promotion")
    claim_file = tmp_path / "CLAIMS.md"
    claim_file.write_text(
        "Dopamine remains structural and research-gated with bounded telemetry.\n",
        encoding="utf-8",
    )
    artifact = tmp_path / "CLAIM_PROMOTION_VERDICT.json"
    monkeypatch.setattr(module, "CLAIM_FILES", [claim_file])
    monkeypatch.setattr(module, "ARTIFACT", artifact)
    monkeypatch.setattr(module, "EVAL", tmp_path / "EVAL_SUMMARY.json")
    monkeypatch.setattr(module, "MANIFEST", tmp_path / "ARTIFACT_MANIFEST.sha256")

    assert module.main() == 0
    payload = json.loads(artifact.read_text(encoding="utf-8"))
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
    text = (
        "General claim is anchored and tested.\n\n"
        "Dopamine remains structural and research-gated."
    )
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
