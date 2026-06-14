# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest

from tools.coverage import coverage_control_plane as ccp


def test_build_phase_commands_composes_authority_then_calibration(tmp_path: Path) -> None:
    commands = ccp.build_phase_commands(
        coverage=Path("reports/coverage/coverage.xml"),
        junit=Path("reports/coverage/junit.xml"),
        targets=Path("configs/quality/coverage_targets.toml"),
        critical=Path("configs/quality/critical_surface.toml"),
        claims=Path("docs/CLAIMS.yaml"),
        out_dir=tmp_path / "intelligence",
        calibration_dir=tmp_path / "calibration",
        stage="final",
        limit=3,
        run_intelligence=True,
    )

    assert len(commands) == 2
    assert commands[0][1:3] == ["-m", "tools.coverage.geosync_coverage_intelligence"]
    assert commands[1][1:3] == ["-m", "tools.coverage.coverage_calibration_loop"]
    assert commands[1][-4:] == ["--stage", "final", "--limit", "3"]


def test_build_phase_commands_can_reuse_existing_intelligence_summary(tmp_path: Path) -> None:
    commands = ccp.build_phase_commands(
        coverage=Path("coverage.xml"),
        junit=Path("junit.xml"),
        targets=Path("targets.toml"),
        critical=Path("critical.toml"),
        claims=Path("CLAIMS.yaml"),
        out_dir=tmp_path / "intelligence",
        calibration_dir=tmp_path / "calibration",
        stage="mid_term",
        limit=8,
        run_intelligence=False,
    )

    assert len(commands) == 1
    assert commands[0][1:3] == ["-m", "tools.coverage.coverage_calibration_loop"]


def test_run_control_plane_stops_after_failed_authority_phase(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], *, cwd: Path) -> ccp.PhaseResult:
        calls.append(command)
        return ccp.PhaseResult(
            name="authority",
            command=command,
            returncode=2,
            stdout="",
            stderr="coverage missing",
        )

    monkeypatch.setattr(ccp, "_run_command", fake_run)

    manifest = ccp.run_control_plane(
        root=tmp_path,
        coverage=Path("reports/coverage/coverage.xml"),
        junit=Path("reports/coverage/junit.xml"),
        targets=Path("configs/quality/coverage_targets.toml"),
        critical=Path("configs/quality/critical_surface.toml"),
        claims=Path("docs/CLAIMS.yaml"),
        out_dir=tmp_path / "reports/coverage/intelligence",
        calibration_dir=tmp_path / "reports/coverage/calibration",
        stage="mid_term",
        limit=5,
    )

    assert manifest.verdict == "FAIL"
    assert len(calls) == 1
    assert manifest.stages[0].returncode == 2
    assert all(not artifact.exists for artifact in manifest.artifacts)


def test_run_control_plane_requires_all_integrated_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run(command: list[str], *, cwd: Path) -> ccp.PhaseResult:
        return ccp.PhaseResult(
            name="phase",
            command=command,
            returncode=0,
            stdout="ok",
            stderr="",
        )

    monkeypatch.setattr(ccp, "_run_command", fake_run)
    intelligence = tmp_path / "reports/coverage/intelligence"
    calibration = tmp_path / "reports/coverage/calibration"
    intelligence.mkdir(parents=True)
    calibration.mkdir(parents=True)
    (intelligence / "coverage_summary.json").write_text("{}\n", encoding="utf-8")
    (intelligence / "coverage_intelligence.md").write_text("# ok\n", encoding="utf-8")
    (calibration / "calibration_plan.json").write_text("{}\n", encoding="utf-8")

    manifest = ccp.run_control_plane(
        root=tmp_path,
        coverage=Path("coverage.xml"),
        junit=Path("junit.xml"),
        targets=Path("targets.toml"),
        critical=Path("critical.toml"),
        claims=Path("CLAIMS.yaml"),
        out_dir=intelligence,
        calibration_dir=calibration,
        stage="mid_term",
        limit=5,
        run_intelligence=False,
    )

    assert manifest.verdict == "FAIL"
    assert any(
        artifact.path.endswith("calibration_plan.md") and artifact.bytes == 0
        for artifact in manifest.artifacts
    )


def test_run_control_plane_passes_with_complete_integrated_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run(command: list[str], *, cwd: Path) -> ccp.PhaseResult:
        return ccp.PhaseResult(
            name="phase",
            command=command,
            returncode=0,
            stdout="ok",
            stderr="",
        )

    monkeypatch.setattr(ccp, "_run_command", fake_run)
    intelligence = tmp_path / "reports/coverage/intelligence"
    calibration = tmp_path / "reports/coverage/calibration"
    intelligence.mkdir(parents=True)
    calibration.mkdir(parents=True)
    for path in [
        intelligence / "coverage_summary.json",
        intelligence / "coverage_intelligence.md",
        calibration / "calibration_plan.json",
        calibration / "calibration_plan.md",
    ]:
        path.write_text("ok\n", encoding="utf-8")

    manifest = ccp.run_control_plane(
        root=tmp_path,
        coverage=Path("coverage.xml"),
        junit=Path("junit.xml"),
        targets=Path("targets.toml"),
        critical=Path("critical.toml"),
        claims=Path("CLAIMS.yaml"),
        out_dir=intelligence,
        calibration_dir=calibration,
        stage="short_term",
        limit=2,
        run_intelligence=False,
    )

    assert manifest.verdict == "PASS"
    assert all(artifact.exists and artifact.bytes > 0 for artifact in manifest.artifacts)


def test_control_plane_resolves_relative_artifacts_against_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run(command: list[str], *, cwd: Path) -> ccp.PhaseResult:
        return ccp.PhaseResult(
            name="phase",
            command=command,
            returncode=0,
            stdout="ok",
            stderr="",
        )

    monkeypatch.setattr(ccp, "_run_command", fake_run)
    for path in [
        tmp_path / "reports/coverage/intelligence/coverage_summary.json",
        tmp_path / "reports/coverage/intelligence/coverage_intelligence.md",
        tmp_path / "reports/coverage/calibration/calibration_plan.json",
        tmp_path / "reports/coverage/calibration/calibration_plan.md",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok\n", encoding="utf-8")

    manifest = ccp.run_control_plane(
        root=tmp_path,
        coverage=Path("reports/coverage/coverage.xml"),
        junit=Path("reports/coverage/junit.xml"),
        targets=Path("configs/quality/coverage_targets.toml"),
        critical=Path("configs/quality/critical_surface.toml"),
        claims=Path("docs/CLAIMS.yaml"),
        out_dir=Path("reports/coverage/intelligence"),
        calibration_dir=Path("reports/coverage/calibration"),
        stage="mid_term",
        limit=8,
        run_intelligence=False,
    )

    assert manifest.verdict == "PASS"
    assert {artifact.path for artifact in manifest.artifacts} == {
        "reports/coverage/intelligence/coverage_summary.json",
        "reports/coverage/intelligence/coverage_intelligence.md",
        "reports/coverage/calibration/calibration_plan.json",
        "reports/coverage/calibration/calibration_plan.md",
    }
