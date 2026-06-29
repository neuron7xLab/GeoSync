from __future__ import annotations

import copy
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _contract() -> dict[str, Any]:
    return json.loads((ROOT / "contracts/dopamine_contract.v1.json").read_text(encoding="utf-8"))


def test_contract_checker_requires_terminal_pass() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_dopamine_contract.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "PASS"


def test_contract_validator_rejects_claim_promotion_mutation() -> None:
    module: Any = importlib.import_module("scripts.check_dopamine_contract")
    mutated = copy.deepcopy(_contract())
    mutated["semantics"]["RAW_TD_RPE"]["market_claim_allowed"] = True
    reasons = module.validate_contract(mutated, ROOT)
    assert any("promotion boundary" in reason for reason in reasons)


def test_contract_validator_rejects_missing_owner_and_empty_invariants() -> None:
    module: Any = importlib.import_module("scripts.check_dopamine_contract")
    mutated = copy.deepcopy(_contract())
    mutated["semantics"]["RAW_TD_RPE"]["owner"] = "missing/dopamine_surface.py"
    mutated["semantics"]["RAW_TD_RPE"]["required_invariants"] = []
    reasons = module.validate_contract(mutated, ROOT)
    assert any("missing owner" in reason for reason in reasons)
    assert any("missing invariants" in reason for reason in reasons)
