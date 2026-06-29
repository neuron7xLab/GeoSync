from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_dopamine_contract_shape() -> None:
    contract = json.loads((ROOT / "contracts/dopamine_contract.v1.json").read_text(encoding="utf-8"))
    assert contract["component"] == "geosync.dopamine"
    assert "RAW_TD_RPE" in contract["semantics"]
    assert "BOUNDED_EXECUTION_RPE" in contract["semantics"]
    assert contract["semantics"]["RAW_TD_RPE"]["market_claim_allowed"] is False


def test_contract_checker_runs() -> None:
    result = subprocess.run([sys.executable, "scripts/check_dopamine_contract.py"], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode in {0, 1}
    assert "status" in result.stdout
