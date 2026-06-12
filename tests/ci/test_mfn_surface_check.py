# SPDX-License-Identifier: MIT
"""Tests for the local MFN surface checker."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools" / "mfn_surface_check.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mfn_surface_check", MODULE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_surface_check_reports_expected_statuses(tmp_path: Path) -> None:
    module = _load_module()
    _write(
        tmp_path,
        "Makefile",
        "test-coverage:\n\ttest\n\ntest-all:\n\ttest\n\ntest-ci-full:\n\ttest\n",
    )
    _write(tmp_path, "configs/quality/release_90.coveragerc", "source = geosync\n")
    _write(
        tmp_path,
        ".github/workflows/mfn-release-gate.yml",
        "python -m pip install -e .\npython -m pytest tests/unit/mfn -q\n",
    )
    _write(
        tmp_path,
        "geosync/mfn/pipeline.py",
        "MANIFEST_SCHEMA_VERSION\n_finite_float\n_validate_manifest_entry\n"
        "_validate_stage_document\n_validate_runbook\n",
    )
    _write(
        tmp_path,
        "geosync/mfn/cli.py",
        "_dispatch\nFileNotFoundError\nValueError\nKeyError\nTypeError\nmfn:\n",
    )
    _write(
        tmp_path,
        "geosync/mfn/contract.py",
        "__post_init__\nschema_version\nseed\ninput_window_sec\nclaim_tier\n",
    )

    result = module.payload(module.collect(tmp_path))

    assert result["status"] == module.RED
    rows = {row["name"]: row for row in result["rows"]}
    assert rows["makefile_cov"]["status"] == module.RED
    assert rows["release_cov"]["status"] == module.OK
    assert rows["workflow_import_path"]["status"] == module.OK


def test_surface_check_can_reach_ok_state(tmp_path: Path) -> None:
    module = _load_module()
    _write(
        tmp_path,
        "Makefile",
        "test-coverage:\n\t--cov=geosync\n\ntest-all:\n\t--cov=geosync\n"
        "\ntest-ci-full:\n\t--cov=geosync\n",
    )
    _write(tmp_path, "configs/quality/release_90.coveragerc", "geosync\n")
    _write(
        tmp_path,
        ".github/workflows/mfn-release-gate.yml",
        "python -m pip install -e .\npython -m pytest tests/unit/mfn -q\n",
    )
    _write(
        tmp_path,
        "geosync/mfn/pipeline.py",
        "MANIFEST_SCHEMA_VERSION\n_finite_float\n_validate_manifest_entry\n"
        "_validate_stage_document\n_validate_runbook\n",
    )
    _write(
        tmp_path,
        "geosync/mfn/cli.py",
        "_dispatch\nFileNotFoundError\nValueError\nKeyError\nTypeError\nmfn:\n",
    )
    _write(
        tmp_path,
        "geosync/mfn/contract.py",
        "__post_init__\nschema_version\nseed\ninput_window_sec\nclaim_tier\n",
    )

    result = module.payload(module.collect(tmp_path))

    assert result["status"] == module.OK
    assert result["counts"] == {module.OK: 6, module.RED: 0, module.UNKNOWN: 0}
