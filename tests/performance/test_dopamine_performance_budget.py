from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "artifacts/dopamine_performance/PERFORMANCE_REPORT.json"


def test_timing_report_has_measured_step() -> None:
    result = subprocess.run(
        [sys.executable, "benchmarks/dopamine_benchmark.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert 0.0 < payload["step_ms"] <= payload["budget_ms"]
    assert payload["iteration_count"] >= 1000
