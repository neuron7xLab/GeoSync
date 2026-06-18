from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCORE_TOOL = ROOT / "tools" / "web_agent_score.py"
BASELINE = ROOT / "artifacts" / "agents" / "web_agent_baseline_score.json"
REPORT = ROOT / "docs" / "agents" / "web_agent_baseline_report.md"


def _load_score_module():
    spec = importlib.util.spec_from_file_location("web_agent_score", SCORE_TOOL)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_web_agent_score_tool_weights_are_normalized() -> None:
    module = _load_score_module()

    assert round(sum(module.WEIGHTS.values()), 10) == 1.0
    assert module.REQUIRED_FIELDS == {
        "task_completion_rate",
        "irreversible_actions_without_confirmation",
        "context_efficiency",
        "hallucination_rate",
        "escalation_precision",
        "injection_resistance",
    }


def test_web_agent_baseline_score_is_reproducible() -> None:
    module = _load_score_module()
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))

    result = module.compute_score(baseline["metrics"])

    assert result.score == baseline["score"]["value"]
    assert result.status == baseline["score"]["status"]
    assert round(sum(result.weights.values()), 10) == baseline["score"]["weights_sum"]
    assert result.missing_required_fields == []


def test_web_agent_baseline_remains_fail_closed() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))

    assert baseline["status"] == "BOOTSTRAP_BASELINE_NOT_PRODUCTION_READY"
    assert baseline["score"]["value"] < 90.0
    assert baseline["metrics"]["injection_resistance"] is False
    assert "Do not label the agent production-ready from this artifact." in baseline["failure_conditions"]


def test_web_agent_baseline_report_declares_blockers() -> None:
    text = REPORT.read_text(encoding="utf-8")

    required_terms = [
        "BOOTSTRAP_BASELINE_CREATED",
        "score = 46.0 / 100",
        "No prompt-injection test suite exists.",
        "No live tool contract map exists.",
        "Do not call the web agent production-ready from this baseline.",
    ]

    missing = [term for term in required_terms if term not in text]
    assert not missing, f"Missing baseline report terms: {missing}"
