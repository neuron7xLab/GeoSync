from __future__ import annotations

import json
from pathlib import Path

from tools.check_json_contract_evidence_policy import check_policy, main


def _candidate() -> dict[str, object]:
    return {
        "contract_type": "repository_repair_contract",
        "inputs": {"missing": []},
        "evidence_status": {
            "external_evidence_available": False,
            "evidence_source": "unavailable",
            "tests_executed": [],
            "score": None,
            "status": "CANDIDATE_NOT_VALIDATED",
        },
        "final_verdict": {"status": "CANDIDATE_NOT_VALIDATED"},
    }


def test_candidate_policy_is_ok() -> None:
    assert check_policy(_candidate()) == []


def test_status_alignment_is_required() -> None:
    payload = _candidate()
    verdict = payload["final_verdict"]
    assert isinstance(verdict, dict)
    verdict["status"] = "BLOCKED"

    assert "status_mismatch" in check_policy(payload)


def test_cli_writes_result(tmp_path: Path) -> None:
    src = tmp_path / "contract.json"
    out = tmp_path / "result.json"
    src.write_text(json.dumps(_candidate()), encoding="utf-8")

    assert main([str(src), "--out", str(out)]) == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["status"] == "OK"
