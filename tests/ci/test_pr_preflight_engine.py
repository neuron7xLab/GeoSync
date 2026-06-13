# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Behavioral tests for the structured PR preflight engine."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import textwrap

ENGINE = Path(__file__).resolve().parents[2] / "tools" / "ci" / "pr_preflight.py"


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _stub_body(tool: str, fail_tool: str | None) -> str:
    fail_line = "exit 0"
    if fail_tool == tool:
        fail_line = f'echo "{tool} stderr forced failure" >&2; echo "{tool} stdout forced failure"; exit 7'
    return textwrap.dedent(
        f"""\
        #!/bin/sh
        set -eu
        if [ "{fail_tool or ''}" = "{tool}" ]; then
            {fail_line}
        fi
        echo "{tool} stdout ok"
        echo "{tool} stderr ok" >&2
        exit 0
        """
    )


def _make_repo(tmp_path: Path, *, fail_tool: str | None = None, include_detect: bool = True) -> tuple[Path, Path, dict[str, str]]:
    root = tmp_path / "repo"
    root.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'stub-geosync'\n", encoding="utf-8")
    (root / "requirements.txt").write_text("# runtime\n", encoding="utf-8")
    (root / "requirements-dev.txt").write_text("# dev\n", encoding="utf-8")
    (root / "constraints").mkdir()
    (root / "constraints" / "security.txt").write_text("# constraints\n", encoding="utf-8")
    (root / "tests").mkdir()
    for dirname in ("core", "backtest", "execution", "src", "application"):
        (root / dirname).mkdir()

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pip_fail = "pip" if fail_tool == "pip" else None
    if fail_tool == "pip-project":
        _write_executable(
            bin_dir / "pip",
            textwrap.dedent(
                """\
                #!/bin/sh
                set -eu
                case " $* " in
                  *" requirements-dev.txt "*|*" constraints/security.txt "*)
                    echo "pip stderr forced project dependency failure" >&2
                    echo "pip stdout forced project dependency failure"
                    exit 7
                    ;;
                esac
                echo "pip stdout ok"
                echo "pip stderr ok" >&2
                exit 0
                """
            ),
        )
    else:
        _write_executable(bin_dir / "pip", _stub_body("pip", pip_fail))
    for tool in ("ruff", "black", "mypy", "pytest"):
        _write_executable(bin_dir / tool, _stub_body(tool, fail_tool))
    if include_detect:
        _write_executable(bin_dir / "detect-secrets", _stub_body("detect-secrets", fail_tool))

    env = os.environ.copy()
    env["PATH"] = str(bin_dir)
    report_dir = tmp_path / "report"
    return root, report_dir, env


def _run_engine(root: Path, report_dir: Path, env: dict[str, str], *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ENGINE), "--root", str(root), "--report-dir", str(report_dir), *extra],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
        check=False,
    )


def _report(report_dir: Path) -> dict[str, object]:
    return json.loads((report_dir / "preflight_report.json").read_text(encoding="utf-8"))


def _check(report: dict[str, object], check_id: str) -> dict[str, object]:
    checks = report["checks"]
    assert isinstance(checks, list)
    for item in checks:
        assert isinstance(item, dict)
        if item["id"] == check_id:
            return item
    raise AssertionError(f"missing check {check_id}")


def test_engine_passes_when_all_critical_stubs_pass(tmp_path: Path) -> None:
    root, report_dir, env = _make_repo(tmp_path)
    result = _run_engine(root, report_dir, env)
    report = _report(report_dir)

    assert result.returncode == 0
    assert report["status"] == "PASS"
    assert report["failure_count"] == 0
    assert _check(report, "ruff")["status"] == "PASS"
    assert _check(report, "coverage_artifact")["status"] == "SKIPPED_OPTIONAL"


def test_engine_fails_when_ruff_fails(tmp_path: Path) -> None:
    root, report_dir, env = _make_repo(tmp_path, fail_tool="ruff")
    result = _run_engine(root, report_dir, env)
    report = _report(report_dir)

    assert result.returncode == 1
    assert report["status"] == "FAIL"
    assert _check(report, "ruff")["status"] == "FAIL"
    assert _check(report, "ruff")["exit_code"] == 7


def test_engine_fails_when_mypy_fails(tmp_path: Path) -> None:
    root, report_dir, env = _make_repo(tmp_path, fail_tool="mypy")
    result = _run_engine(root, report_dir, env)
    report = _report(report_dir)

    assert result.returncode == 1
    assert report["status"] == "FAIL"
    assert _check(report, "mypy")["status"] == "FAIL"


def test_engine_fails_when_pytest_fails(tmp_path: Path) -> None:
    root, report_dir, env = _make_repo(tmp_path, fail_tool="pytest")
    result = _run_engine(root, report_dir, env)
    report = _report(report_dir)

    assert result.returncode == 1
    assert report["status"] == "FAIL"
    assert _check(report, "quick_tests")["status"] == "FAIL"


def test_engine_fails_when_project_dependency_install_fails(tmp_path: Path) -> None:
    root, report_dir, env = _make_repo(tmp_path, fail_tool="pip-project")
    result = _run_engine(root, report_dir, env)
    report = _report(report_dir)

    assert result.returncode == 1
    assert report["status"] == "FAIL"
    assert _check(report, "pip_bootstrap")["status"] == "PASS"
    assert _check(report, "project_dependencies")["status"] == "FAIL"


def test_engine_skips_detect_secrets_when_missing(tmp_path: Path) -> None:
    root, report_dir, env = _make_repo(tmp_path, include_detect=False)
    result = _run_engine(root, report_dir, env)
    report = _report(report_dir)
    detect = _check(report, "detect_secrets")

    assert result.returncode == 0
    assert detect["status"] == "SKIPPED_OPTIONAL"
    assert detect["critical"] is False
    assert "executable not found" in str(detect["failure_reason"])


def test_engine_fails_detect_secrets_when_present_and_failing(tmp_path: Path) -> None:
    root, report_dir, env = _make_repo(tmp_path, fail_tool="detect-secrets", include_detect=True)
    result = _run_engine(root, report_dir, env)
    report = _report(report_dir)
    detect = _check(report, "detect_secrets")

    assert result.returncode == 1
    assert report["status"] == "FAIL"
    assert detect["critical"] is True
    assert detect["status"] == "FAIL"


def test_engine_writes_report_json_on_failure(tmp_path: Path) -> None:
    root, report_dir, env = _make_repo(tmp_path, fail_tool="ruff")
    result = _run_engine(root, report_dir, env, "--json")
    report_path = report_dir / "preflight_report.json"
    report = _report(report_dir)

    assert result.returncode == 1
    assert report_path.is_file()
    assert report["first_file_to_open"] == str(report_path.resolve())
    assert '"status": "FAIL"' in result.stdout


def test_engine_preserves_stdout_and_stderr_logs(tmp_path: Path) -> None:
    root, report_dir, env = _make_repo(tmp_path, fail_tool="ruff")
    result = _run_engine(root, report_dir, env)
    report = _report(report_dir)
    ruff = _check(report, "ruff")

    assert result.returncode == 1
    stdout_log = Path(str(ruff["stdout_log"]))
    stderr_log = Path(str(ruff["stderr_log"]))
    assert stdout_log.read_text(encoding="utf-8") == "ruff stdout forced failure\n"
    assert stderr_log.read_text(encoding="utf-8") == "ruff stderr forced failure\n"


def test_shell_wrapper_forwards_python_runner_exit_code(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[2] / "scripts" / "test-pr-locally.sh"
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "scripts").mkdir()
    (repo / "tools" / "ci").mkdir(parents=True)
    (repo / "tools" / "ci" / "pr_preflight.py").write_text("raise SystemExit(19)\n", encoding="utf-8")
    target = repo / "scripts" / "test-pr-locally.sh"
    target.write_text(script.read_text(encoding="utf-8"), encoding="utf-8")
    target.chmod(target.stat().st_mode | stat.S_IXUSR)

    result = subprocess.run(
        ["bash", "scripts/test-pr-locally.sh"],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )

    assert result.returncode == 19
    assert "FIRST_FILE_TO_OPEN=" in result.stdout


def test_engine_blocks_invalid_root_and_writes_report(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    missing_root = tmp_path / "missing-root"
    result = _run_engine(missing_root, report_dir, os.environ.copy())
    report = _report(report_dir)

    assert result.returncode == 1
    assert report["status"] == "BLOCKED"
    assert report["failure_count"] == 1
    invalid = _check(report, "invalid_root")
    assert invalid["status"] == "BLOCKED"
    assert "invalid root" in str(invalid["failure_reason"])


def test_engine_blocks_when_report_cannot_be_written(tmp_path: Path) -> None:
    root, _report_dir, env = _make_repo(tmp_path)
    report_path_is_file = tmp_path / "report-file"
    report_path_is_file.write_text("not a directory\n", encoding="utf-8")
    result = _run_engine(root, report_path_is_file, env)

    assert result.returncode == 1
    assert "PR_PREFLIGHT_STATUS=BLOCKED" in result.stderr
    assert "failed to write preflight evidence" in result.stderr
    assert "PASS" not in result.stdout
    assert "FIRST_FILE_TO_OPEN" not in result.stdout


def test_tool_resolution_policy_matrix(tmp_path: Path) -> None:
    critical_tool_to_check = {
        "pip": "pip_bootstrap",
        "ruff": "ruff",
        "black": "black",
        "mypy": "mypy",
        "pytest": "quick_tests",
    }
    for tool, check_id in critical_tool_to_check.items():
        case_root = tmp_path / tool
        root, report_dir, env = _make_repo(case_root)
        (Path(env["PATH"]) / tool).unlink()
        result = _run_engine(root, report_dir, env)
        report = _report(report_dir)
        check = _check(report, check_id)

        assert result.returncode == 1, tool
        assert report["status"] == "BLOCKED", tool
        assert check["status"] == "BLOCKED", tool
        assert check["critical"] is True, tool
        assert "executable not found" in str(check["failure_reason"]), tool

    root, report_dir, env = _make_repo(tmp_path / "detect-missing", include_detect=False)
    result = _run_engine(root, report_dir, env)
    detect = _check(_report(report_dir), "detect_secrets")
    assert result.returncode == 0
    assert detect["status"] == "SKIPPED_OPTIONAL"
    assert detect["critical"] is False

    root, report_dir, env = _make_repo(tmp_path / "detect-failing", fail_tool="detect-secrets", include_detect=True)
    result = _run_engine(root, report_dir, env)
    detect = _check(_report(report_dir), "detect_secrets")
    assert result.returncode == 1
    assert detect["status"] == "FAIL"
    assert detect["critical"] is True


def test_engine_writes_review_bundle_readme(tmp_path: Path) -> None:
    root, report_dir, env = _make_repo(tmp_path, fail_tool="ruff")
    result = _run_engine(root, report_dir, env)
    report = _report(report_dir)
    readme = report_dir / "README_FOR_REVIEW.md"

    assert result.returncode == 1
    assert readme.is_file()
    assert str(report_dir / "preflight_report.json") in readme.read_text(encoding="utf-8")
    assert report["first_failing_log"] == _check(report, "ruff")["stderr_log"]
    assert report["first_failing_log"] in readme.read_text(encoding="utf-8")
    for check in report["checks"]:
        assert Path(str(check["stdout_log"])).is_file()
        assert Path(str(check["stderr_log"])).is_file()


def test_exit_code_semantics_match_final_status(tmp_path: Path) -> None:
    pass_root, pass_report_dir, pass_env = _make_repo(tmp_path / "pass")
    pass_result = _run_engine(pass_root, pass_report_dir, pass_env)
    pass_report = _report(pass_report_dir)
    assert pass_report["status"] == "PASS"
    assert pass_report["failure_count"] == 0
    assert pass_result.returncode == 0

    fail_root, fail_report_dir, fail_env = _make_repo(tmp_path / "fail", fail_tool="ruff")
    fail_result = _run_engine(fail_root, fail_report_dir, fail_env)
    fail_report = _report(fail_report_dir)
    assert fail_report["status"] == "FAIL"
    assert fail_report["failure_count"] > 0
    assert fail_result.returncode == 1

    blocked_report_dir = tmp_path / "blocked-report"
    blocked_result = _run_engine(tmp_path / "missing-root", blocked_report_dir, os.environ.copy())
    blocked_report = _report(blocked_report_dir)
    assert blocked_report["status"] == "BLOCKED"
    assert blocked_report["failure_count"] > 0
    assert blocked_result.returncode == 1
