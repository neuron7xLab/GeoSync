from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_dopamine_benchmark_emits_report() -> None:
    result = subprocess.run([sys.executable, "benchmarks/dopamine_benchmark.py"], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "PASS" in result.stdout
    assert (ROOT / "artifacts/dopamine_performance/PERFORMANCE_REPORT.json").exists()
