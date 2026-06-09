# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Tests for the dependency-free build backend used by MFN clean installs."""

from __future__ import annotations

import zipfile
from pathlib import Path

from tools import build_backend


def test_build_backend_declares_no_external_build_requirements() -> None:
    assert build_backend.get_requires_for_build_wheel() == []
    assert build_backend.get_requires_for_build_editable() == []


def test_build_editable_wheel_contains_entrypoints_and_repo_pth(tmp_path: Path) -> None:
    wheel_name = build_backend.build_editable(str(tmp_path))
    wheel_path = tmp_path / wheel_name

    assert wheel_path.exists()
    with zipfile.ZipFile(wheel_path) as archive:
        names = set(archive.namelist())
        assert "geosync_editable.pth" in names
        entry_points_name = next(name for name in names if name.endswith("/entry_points.txt"))
        entry_points = archive.read(entry_points_name).decode("utf-8")

    assert "mfn = geosync.mfn.cli:main" in entry_points
    assert "mfn-api = geosync.mfn.cli:api_main" in entry_points
    assert "mfn-validate = geosync.mfn.cli:validate_main" in entry_points
