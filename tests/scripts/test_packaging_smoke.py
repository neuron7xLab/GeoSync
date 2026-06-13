# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Packaging smoke contract for repository script entrypoints.

The repository exposes console_scripts that target modules under ``scripts``.
This test keeps the build contract honest: the modules must be packaged by
setuptools and package directories used by entrypoints must carry
``__init__.py`` markers so clean wheel/editable installs do not depend on an
accidental repository-root ``PYTHONPATH``.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"


def _pyproject() -> dict[str, Any]:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _entrypoint_module(target: str) -> str:
    module, _sep, _callable = target.partition(":")
    assert module, f"empty console-script module in {target!r}"
    return module


def test_scripts_console_entrypoints_are_packaged() -> None:
    payload = _pyproject()
    scripts = payload["project"]["scripts"]
    include = set(
        payload["tool"]["setuptools"]["packages"]["find"]["include"]
    )

    script_modules = {
        _entrypoint_module(target)
        for target in scripts.values()
        if _entrypoint_module(target).startswith("scripts.")
    }

    assert script_modules, "expected at least one scripts.* console-script target"
    assert {"scripts", "scripts.*"} <= include

    for module in sorted(script_modules):
        module_path = ROOT.joinpath(*module.split("."))
        assert (
            module_path.with_suffix(".py").exists() or (module_path / "__init__.py").exists()
        ), f"console-script module target is missing: {module}"

        package_parts = module.split(".")[:-1]
        for depth in range(1, len(package_parts) + 1):
            package_init = ROOT.joinpath(*package_parts[:depth], "__init__.py")
            assert package_init.exists(), f"missing package marker for {module}: {package_init}"


def test_ci_entrypoints_are_explicitly_registered() -> None:
    payload = _pyproject()
    scripts = payload["project"]["scripts"]

    assert scripts["geosync-release-gate"] == "scripts.ci.release_gate:main"
    assert scripts["geosync-import-architecture"] == "scripts.ci.check_import_architecture:main"
    assert scripts["geosync-pai"] == "scripts.ci.compute_pai:main"
    assert scripts["geosync-scripts"] == "scripts.cli:main"
