# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Regression guard for the five-minute D-H-T-F-V proof command."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from geosync.proof import run as proof


REQUIRED_KEYS = {
    "mechanism",
    "input_dataset",
    "hypothesis",
    "metric",
    "threshold",
    "result",
    "verdict",
    "failure_condition",
    "repro_command",
    "timestamp_utc",
    "code_version",
}


def test_fixture_generates_reject_verdict() -> None:
    verdict = proof.build_verdict()
    assert set(verdict) == REQUIRED_KEYS
    assert verdict["mechanism"] == "lag1_directional_persistence_fixture_test"
    assert verdict["verdict"] == "REJECT"
    result = verdict["result"]
    assert isinstance(result, dict)
    assert result["training_rule"] == "continuation"
    assert result["training_hit_rate"] > 0.60
    assert result["holdout_hit_rate"] == 0.0
    assert result["samples"] >= 3


def test_proof_command_writes_verdict_json(tmp_path: Path) -> None:
    artifact = Path("artifacts/geosync_proof/verdict.json")
    if artifact.exists():
        artifact.unlink()
    env = dict(os.environ)
    env["SOURCE_DATE_EPOCH"] = "1700000000"
    env["GEOSYNC_PROOF_CODE_VERSION"] = "test-version"
    proc = subprocess.run(
        [sys.executable, "-m", "geosync.proof.run"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert "GEOSYNC_PROOF verdict=REJECT" in proc.stdout
    assert artifact.exists()
    data = json.loads(artifact.read_text(encoding="utf-8"))
    assert data["verdict"] == "REJECT"
    assert data["repro_command"] == "python -m geosync.proof.run"
    assert data["timestamp_utc"] == "2023-11-14T22:13:20Z"
    assert data["code_version"] == "test-version"


def test_degenerate_fixture_is_inconclusive(tmp_path: Path) -> None:
    path = tmp_path / "short.csv"
    path.write_text("timestamp,close\n2026-01-01T00:00:00Z,100\n2026-01-02T00:00:00Z,101\n", encoding="utf-8")
    try:
        proof.build_verdict(path)
    except ValueError as exc:
        assert "enough bars" in str(exc)
    else:
        raise AssertionError("short fixture must fail closed")
