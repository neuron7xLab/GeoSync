# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Contract tests for coverage-ascent calibration automation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import tools.coverage.calibrate_coverage_ascent as planner


def _summary(*, valid: bool = True, release: float = 85.03) -> dict[str, Any]:
    return {
        "verdict": "MACHINE_ASSISTED",
        "evidence_valid": valid,
        "release_line_coverage": release,
        "risk_weighted_score": 88.1,
        "untested_file_count": 14,
        "surfaces": {
            "backtest": {
                "line_rate": 68.55,
                "branch_rate": 42.0,
                "statements": 1000,
                "target_final": 98.0,
                "claim_risk": "critical",
            },
            "execution": {
                "line_rate": 87.74,
                "branch_rate": 80.0,
                "statements": 800,
                "target_final": 95.0,
                "claim_risk": "critical",
            },
            "complete": {
                "line_rate": 100.0,
                "branch_rate": 100.0,
                "statements": 50,
                "target_final": 90.0,
                "claim_risk": "low",
            },
        },
    }


def test_plan_selects_next_band_and_ranks_weighted_gaps() -> None:
    plan = planner.build_plan(_summary(), (90.0, 92.0, 95.0, 97.0), max_surfaces=6)

    assert plan.next_global_band == 90.0
    assert plan.global_gap_to_next_band == 4.97
    assert [item.surface for item in plan.recommended_surfaces] == ["backtest", "execution"]
    assert plan.recommended_surfaces[0].weighted_deficit == 1180.0


def test_invalid_evidence_fails_closed_by_default(tmp_path: Path) -> None:
    summary_path = tmp_path / "coverage_summary.json"
    summary_path.write_text(json.dumps(_summary(valid=False)), encoding="utf-8")

    rc = planner.main(["--summary", str(summary_path), "--out", str(tmp_path / "out")])

    assert rc == planner.EXIT_EVIDENCE_INVALID


def test_invalid_evidence_can_emit_review_plan_when_explicitly_requested(tmp_path: Path) -> None:
    summary_path = tmp_path / "coverage_summary.json"
    out_dir = tmp_path / "out"
    summary_path.write_text(json.dumps(_summary(valid=False)), encoding="utf-8")

    rc = planner.main(
        [
            "--summary",
            str(summary_path),
            "--out",
            str(out_dir),
            "--no-enforce-evidence",
        ]
    )

    assert rc == planner.EXIT_OK
    payload = json.loads((out_dir / "coverage_calibration_plan.json").read_text(encoding="utf-8"))
    assert payload["evidence_valid"] is False
    assert (out_dir / "coverage_calibration_plan.md").exists()


def test_malformed_summary_does_not_create_outputs(tmp_path: Path) -> None:
    summary_path = tmp_path / "coverage_summary.json"
    summary_path.write_text(chr(123), encoding="utf-8")

    rc = planner.main(["--summary", str(summary_path), "--out", str(tmp_path / "out")])

    assert rc == planner.EXIT_EVIDENCE_INVALID
    assert not (tmp_path / "out" / "coverage_calibration_plan.json").exists()
