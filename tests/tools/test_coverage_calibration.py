# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Falsification suite for the coverage-ascent calibration planner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.coverage import calibrate_coverage_ascent as cca


def _summary(*, evidence_valid: bool = True, release: float = 85.03) -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "verdict": "MACHINE_ASSISTED",
        "evidence_valid": evidence_valid,
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
                "meets_final": False,
            },
            "execution": {
                "line_rate": 87.74,
                "branch_rate": 80.0,
                "statements": 800,
                "target_final": 95.0,
                "claim_risk": "critical",
                "meets_final": False,
            },
            "analytics": {
                "line_rate": 75.99,
                "branch_rate": 70.0,
                "statements": 200,
                "target_final": 90.0,
                "claim_risk": "medium",
                "meets_final": False,
            },
            "docs_surface_noise": {
                "line_rate": 100.0,
                "branch_rate": 100.0,
                "statements": 50,
                "target_final": 90.0,
                "claim_risk": "low",
                "meets_final": True,
            },
        },
    }


def test_next_band_moves_from_current_release_to_next_goal() -> None:
    plan = cca.build_plan(
        _summary(release=85.03),
        (90.0, 92.0, 95.0, 97.0),
        max_surfaces=6,
    )

    assert plan.next_global_band == 90.0
    assert plan.global_gap_to_next_band == 4.97


def test_surface_ranking_uses_risk_weighted_missing_statements() -> None:
    plan = cca.build_plan(_summary(), (90.0, 92.0, 95.0, 97.0), max_surfaces=6)

    assert [item.surface for item in plan.recommended_surfaces[:2]] == ["backtest", "execution"]
    backtest = plan.recommended_surfaces[0]
    assert backtest.claim_risk == "critical"
    assert backtest.missing_statements_to_effective_target == 215
    assert backtest.missing_statements_to_final_target == 295
    assert backtest.weighted_deficit == 1180.0
    assert "untested production files" in backtest.recommended_protocol


def test_complete_surfaces_are_not_recommended() -> None:
    plan = cca.build_plan(_summary(), (90.0, 92.0, 95.0, 97.0), max_surfaces=6)

    assert "docs_surface_noise" not in {item.surface for item in plan.recommended_surfaces}


def test_evidence_invalid_exits_fail_closed(tmp_path: Path) -> None:
    summary_path = tmp_path / "coverage_summary.json"
    summary_path.write_text(json.dumps(_summary(evidence_valid=False)), encoding="utf-8")

    rc = cca.main(["--summary", str(summary_path), "--out", str(tmp_path / "out")])

    assert rc == cca.EXIT_EVIDENCE_INVALID


def test_no_enforce_evidence_still_writes_plan_for_human_review(tmp_path: Path) -> None:
    summary_path = tmp_path / "coverage_summary.json"
    out_dir = tmp_path / "out"
    summary_path.write_text(json.dumps(_summary(evidence_valid=False)), encoding="utf-8")

    rc = cca.main(
        [
            "--summary",
            str(summary_path),
            "--out",
            str(out_dir),
            "--no-enforce-evidence",
        ]
    )

    assert rc == cca.EXIT_OK
    plan = json.loads((out_dir / "coverage_calibration_plan.json").read_text(encoding="utf-8"))
    assert plan["evidence_valid"] is False
    assert (out_dir / "coverage_calibration_plan.md").exists()


def test_malformed_summary_fails_closed(tmp_path: Path) -> None:
    summary_path = tmp_path / "coverage_summary.json"
    summary_path.write_text("not-json", encoding="utf-8")

    rc = cca.main(["--summary", str(summary_path), "--out", str(tmp_path / "out")])

    assert rc == cca.EXIT_EVIDENCE_INVALID
    assert not (tmp_path / "out" / "coverage_calibration_plan.json").exists()


def test_written_markdown_contains_acceptance_discipline(tmp_path: Path) -> None:
    summary_path = tmp_path / "coverage_summary.json"
    out_dir = tmp_path / "out"
    summary_path.write_text(json.dumps(_summary()), encoding="utf-8")

    rc = cca.main(["--summary", str(summary_path), "--out", str(out_dir)])

    assert rc == cca.EXIT_OK
    md = (out_dir / "coverage_calibration_plan.md").read_text(encoding="utf-8")
    assert "Do not lower coverage thresholds" in md
    assert "weighted deficit" in md
