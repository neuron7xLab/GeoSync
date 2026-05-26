from __future__ import annotations

import subprocess
import sys


def test_zero_latency_interrupter_runs() -> None:
    proc = subprocess.run([sys.executable, 'scripts/guards/zero_latency_interrupter.py'], check=False)
    assert proc.returncode == 0
