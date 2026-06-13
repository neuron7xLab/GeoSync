# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Regression tests for the shell compatibility wrapper.

The fail-closed PR preflight logic lives in ``tools/ci/pr_preflight.py``. The
legacy shell command remains as the user-facing door and must not grow a second
copy of critical check logic.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import textwrap
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "test-pr-locally.sh"

FORBIDDEN_SHELL_PATTERNS = (
    "|| true",
    "--exit-zero",
    "2>/dev/null",
    "&>/dev/null",
    "ruff check",
    "black --check",
    "mypy --no-error-summary",
    "pytest tests/",
    "detect-secrets scan",
    "pip install",
)


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def test_shell_wrapper_delegates_to_python_runner_without_critical_check_logic() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "tools/ci/pr_preflight.py" in text
    assert "--report-dir" in text
    assert "exit \"${exit_code}\"" in text

    offenders = [pattern for pattern in FORBIDDEN_SHELL_PATTERNS if pattern in text]
    assert offenders == []


def test_shell_wrapper_reports_first_file_to_open() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "FIRST_FILE_TO_OPEN=${REPORT_DIR}/preflight_report.json" in text


def test_shell_wrapper_forwards_python_runner_exit_code(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "scripts").mkdir()
    (repo / "tools" / "ci").mkdir(parents=True)
    shutil.copy2(SCRIPT, repo / "scripts" / "test-pr-locally.sh")
    (repo / "tools" / "ci" / "pr_preflight.py").write_text("raise SystemExit(0)\n", encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "python3",
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            echo "stub python runner invoked: $*"
            exit 23
            """
        ),
    )

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    result = subprocess.run(
        ["bash", "scripts/test-pr-locally.sh", "--skip-install"],
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )

    assert result.returncode == 23
    assert "stub python runner invoked:" in result.stdout
    assert "--skip-install" in result.stdout
    assert "FIRST_FILE_TO_OPEN=" in result.stdout
