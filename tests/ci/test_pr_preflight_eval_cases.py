# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Eval-style behavioral cases for the PR preflight runner."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import textwrap
from typing import Any

ENGINE = Path(__file__).resolve().parents[2] / "tools" / "ci" / "pr_preflight.py"
CASES = Path(__file__).with_name("pr_preflight_eval_cases.jsonl")


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _tool_stub(tool: str, behavior: object) -> str:
    if behavior == "timeout":
        return "#!/bin/sh\n/bin/sleep 3\n"
    if behavior == 7:
        return f'#!/bin/sh\necho "{tool} stdout fail"\necho "{tool} stderr fail" >&2\nexit 7\n'
    return f'#!/bin/sh\necho "{tool} stdout pass"\necho "{tool} stderr pass" >&2\nexit 0\n'


def _pip_stub(behavior: object) -> str:
    if behavior == "project_fail":
        return textwrap.dedent(
            """\
            #!/bin/sh
            case " $* " in
              *" requirements-dev.txt "*|*" constraints/security.txt "*)
                echo "pip stdout project fail"
                echo "pip stderr project fail" >&2
                exit 7
                ;;
            esac
            echo "pip stdout pass"
            echo "pip stderr pass" >&2
            exit 0
            """
        )
    return _tool_stub("pip", behavior)


def _load_cases() -> list[dict[str, Any]]:
    return [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]


def _make_repo(tmp_path: Path, case: dict[str, Any]) -> tuple[Path, Path, dict[str, str]]:
    root = tmp_path / case["case_id"] / "repo"
    root.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'eval-repo'\n", encoding="utf-8")
    (root / "tests").mkdir()
    for dirname in ("core", "backtest", "execution", "src", "application"):
        (root / dirname).mkdir()
    repo_files = case["repo_files"]
    if repo_files.get("requirements.txt"):
        (root / "requirements.txt").write_text("# runtime\n", encoding="utf-8")
    if repo_files.get("requirements-dev.txt"):
        (root / "requirements-dev.txt").write_text("# dev\n", encoding="utf-8")
    if repo_files.get("constraints/security.txt"):
        (root / "constraints").mkdir(exist_ok=True)
        (root / "constraints" / "security.txt").write_text("# constraints\n", encoding="utf-8")
    if repo_files.get("coverage.xml"):
        (root / "coverage.xml").write_text("<coverage />\n", encoding="utf-8")

    bin_dir = tmp_path / case["case_id"] / "bin"
    bin_dir.mkdir()
    for tool, behavior in case["stub_tools"].items():
        if behavior == "missing":
            continue
        body = _pip_stub(behavior) if tool == "pip" else _tool_stub(tool, behavior)
        _write_executable(bin_dir / tool, body)
    env = os.environ.copy()
    env["PATH"] = str(bin_dir)
    return root, tmp_path / case["case_id"] / "report", env


def _check(report: dict[str, Any], check_id: str) -> dict[str, Any]:
    for item in report["checks"]:
        if item["id"] == check_id:
            return item
    raise AssertionError(f"missing check {check_id}")


def test_eval_cases_execute_runner(tmp_path: Path) -> None:
    for case in _load_cases():
        root, report_dir, env = _make_repo(tmp_path, case)
        result = subprocess.run(
            [
                sys.executable,
                str(ENGINE),
                "--root",
                str(root),
                "--report-dir",
                str(report_dir),
                *case["args"],
            ],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
            check=False,
        )
        report = json.loads((report_dir / "preflight_report.json").read_text(encoding="utf-8"))

        assert result.returncode == case["expected_exit_code"], case["case_id"]
        assert report["status"] == case["expected_final_status"], case["case_id"]
        for check_id, status in case["expected_check_statuses"].items():
            assert _check(report, check_id)["status"] == status, case["case_id"]
