from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_claim_gate_runs_and_emits_json() -> None:
    result = subprocess.run([sys.executable, "scripts/ci/check_dopamine_claim_promotion.py"], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode in {0, 1}
    assert "status" in result.stdout
