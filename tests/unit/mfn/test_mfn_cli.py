# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Contract tests for the dependency-light MFN integration CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geosync.mfn.cli import api_main, main, validate_main
from geosync.mfn.contract import MFN_COMMANDS
from geosync.mfn.pipeline import read_json, validate_bundle


def test_mfn_run_creates_bundle_contract(tmp_path: Path, capsys) -> None:
    bundle = tmp_path / "bundle"

    exit_code = main(["--out", str(bundle), "--seed", "7", "--points", "8", "run"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert f"first_file_to_open={bundle / 'manifest.json'}" in captured.out
    assert (bundle / "manifest.json").exists()
    assert (bundle / "SHA256SUMS").exists()
    assert (bundle / "runbook.md").exists()
    assert validate_bundle(bundle) == []

    manifest = read_json(bundle / "manifest.json")
    files = {entry["path"] for entry in manifest["files"]}
    assert {
        "simulate.json",
        "extract.json",
        "detect.json",
        "forecast.json",
        "compare.json",
        "report.json",
        "runbook.md",
    } <= files


def test_mfn_rejects_too_few_points(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--points", "3", "run"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "points must be >= 4" in captured.err


def test_mfn_validate_rejects_missing_bundle(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "missing"

    exit_code = validate_main(["--bundle", str(missing)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "missing required artifact" in captured.err


def test_mfn_bundle_is_byte_reproducible(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
    first = tmp_path / "first"
    second = tmp_path / "second"

    assert main(["--out", str(first), "--seed", "7", "--points", "8", "run"]) == 0
    assert main(["--out", str(second), "--seed", "7", "--points", "8", "run"]) == 0

    for name in ("SHA256SUMS", "manifest.json", "runbook.md"):
        assert (first / name).read_text(encoding="utf-8") == (second / name).read_text(
            encoding="utf-8"
        )


def test_mfn_api_json_lists_required_commands(capsys) -> None:
    exit_code = api_main(["--format", "json"])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["status"] == "INSTRUMENTED"
    assert payload["min_points"] == 4
    assert set(MFN_COMMANDS) <= set(payload["commands"])
