import json
import os
import subprocess
import sys
from pathlib import Path

from tools.ci import pr_preflight


def _write_executable(path: Path, body: str) -> Path:
    path.write_text(f"#!{sys.executable} -S\n{body}", encoding="utf-8")
    path.chmod(0o755)
    return path


def _write_tool_stub(bin_dir: Path, name: str, env_prefix: str) -> Path:
    body = f"""
import os
import sys

sys.stdout.write(os.environ.get("{env_prefix}_STDOUT", "{name}-stdout\\n"))
sys.stderr.write(os.environ.get("{env_prefix}_STDERR", "{name}-stderr\\n"))
sys.exit(int(os.environ.get("{env_prefix}_EXIT", "0")))
"""
    return _write_executable(bin_dir / name, body)


def _write_python_stub(bin_dir: Path) -> Path:
    body = """
import os
import sys

sys.stdout.write("python-stdout\\n")
sys.stderr.write("python-stderr\\n")
if "-m" in sys.argv and "pip" in sys.argv and "--upgrade" in sys.argv:
    sys.exit(int(os.environ.get("STUB_PIP_BOOTSTRAP_EXIT", "0")))
sys.exit(int(os.environ.get("STUB_PROJECT_INSTALL_EXIT", "0")))
"""
    return _write_executable(bin_dir / "python", body)


def _prepare_repo(tmp_path: Path, monkeypatch, detect_secrets: bool = False) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'stub'\n", encoding="utf-8")
    (root / "tests").mkdir()

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_stub = _write_python_stub(bin_dir)
    _write_tool_stub(bin_dir, "ruff", "STUB_RUFF")
    _write_tool_stub(bin_dir, "black", "STUB_BLACK")
    _write_tool_stub(bin_dir, "mypy", "STUB_MYPY")
    _write_tool_stub(bin_dir, "pytest", "STUB_PYTEST")
    if detect_secrets:
        _write_tool_stub(bin_dir, "detect-secrets", "STUB_DETECT_SECRETS")

    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("PR_PREFLIGHT_PYTHON", str(python_stub))
    return root, root / "artifacts" / "pr_preflight"


def _run_engine(root: Path, report_dir: Path) -> tuple[int, dict]:
    exit_code = pr_preflight.main(["--root", str(root), "--report-dir", str(report_dir)])
    report = json.loads((report_dir / "preflight_report.json").read_text(encoding="utf-8"))
    return exit_code, report


def _check(report: dict, check_id: str) -> dict:
    return next(check for check in report["checks"] if check["id"] == check_id)


def test_engine_passes_when_all_critical_stubs_pass(tmp_path, monkeypatch):
    root, report_dir = _prepare_repo(tmp_path, monkeypatch)

    exit_code, report = _run_engine(root, report_dir)

    assert exit_code == 0
    assert report["status"] == "PASS"
    assert report["failure_count"] == 0
    assert _check(report, "detect_secrets")["status"] == "SKIPPED_OPTIONAL"


def test_engine_fails_when_ruff_fails(tmp_path, monkeypatch):
    root, report_dir = _prepare_repo(tmp_path, monkeypatch)
    monkeypatch.setenv("STUB_RUFF_EXIT", "2")

    exit_code, report = _run_engine(root, report_dir)

    assert exit_code == 1
    assert report["status"] == "FAIL"
    assert _check(report, "ruff")["status"] == "FAIL"


def test_engine_fails_when_mypy_fails(tmp_path, monkeypatch):
    root, report_dir = _prepare_repo(tmp_path, monkeypatch)
    monkeypatch.setenv("STUB_MYPY_EXIT", "2")

    exit_code, report = _run_engine(root, report_dir)

    assert exit_code == 1
    assert report["status"] == "FAIL"
    assert _check(report, "mypy")["status"] == "FAIL"


def test_engine_fails_when_pytest_fails(tmp_path, monkeypatch):
    root, report_dir = _prepare_repo(tmp_path, monkeypatch)
    monkeypatch.setenv("STUB_PYTEST_EXIT", "2")

    exit_code, report = _run_engine(root, report_dir)

    assert exit_code == 1
    assert report["status"] == "FAIL"
    assert _check(report, "quick_tests")["status"] == "FAIL"


def test_engine_fails_when_project_dependency_install_fails(tmp_path, monkeypatch):
    root, report_dir = _prepare_repo(tmp_path, monkeypatch)
    monkeypatch.setenv("STUB_PROJECT_INSTALL_EXIT", "19")

    exit_code, report = _run_engine(root, report_dir)

    assert exit_code == 1
    assert report["status"] == "FAIL"
    assert _check(report, "pip_bootstrap")["status"] == "PASS"
    assert _check(report, "project_dependencies")["status"] == "FAIL"


def test_engine_skips_detect_secrets_when_missing(tmp_path, monkeypatch):
    root, report_dir = _prepare_repo(tmp_path, monkeypatch, detect_secrets=False)

    exit_code, report = _run_engine(root, report_dir)

    detect = _check(report, "detect_secrets")
    assert exit_code == 0
    assert detect["status"] == "SKIPPED_OPTIONAL"
    assert detect["critical"] is False
    assert detect["tool_available"] is False


def test_engine_fails_detect_secrets_when_present_and_failing(tmp_path, monkeypatch):
    root, report_dir = _prepare_repo(tmp_path, monkeypatch, detect_secrets=True)
    monkeypatch.setenv("STUB_DETECT_SECRETS_EXIT", "7")

    exit_code, report = _run_engine(root, report_dir)

    assert exit_code == 1
    assert report["status"] == "FAIL"
    detect = _check(report, "detect_secrets")
    assert detect["critical"] is True
    assert detect["status"] == "FAIL"


def test_engine_writes_report_json_on_failure(tmp_path, monkeypatch):
    root, report_dir = _prepare_repo(tmp_path, monkeypatch)
    monkeypatch.setenv("STUB_PYTEST_EXIT", "4")

    exit_code, report = _run_engine(root, report_dir)

    assert exit_code == 1
    assert (report_dir / "preflight_report.json").exists()
    assert report["failure_count"] >= 1
    assert report["first_file_to_open"] == "artifacts/pr_preflight/preflight_report.json"


def test_engine_preserves_stdout_and_stderr_logs(tmp_path, monkeypatch):
    root, report_dir = _prepare_repo(tmp_path, monkeypatch)
    monkeypatch.setenv("STUB_RUFF_STDOUT", "ruff-out\n")
    monkeypatch.setenv("STUB_RUFF_STDERR", "ruff-err\n")

    exit_code, report = _run_engine(root, report_dir)

    assert exit_code == 0
    ruff = _check(report, "ruff")
    assert (root / ruff["stdout_log"]).read_text(encoding="utf-8") == "ruff-out\n"
    assert (root / ruff["stderr_log"]).read_text(encoding="utf-8") == "ruff-err\n"


def test_shell_wrapper_forwards_python_runner_exit_code(tmp_path):
    repo = tmp_path / "repo"
    scripts_dir = repo / "scripts"
    bin_dir = tmp_path / "bin"
    scripts_dir.mkdir(parents=True)
    bin_dir.mkdir()
    source_wrapper = Path(__file__).resolve().parents[2] / "scripts" / "test-pr-locally.sh"
    target_wrapper = scripts_dir / "test-pr-locally.sh"
    target_wrapper.write_text(source_wrapper.read_text(encoding="utf-8"), encoding="utf-8")
    target_wrapper.chmod(0o755)
    _write_executable(bin_dir / "python3", "import sys\nsys.exit(37)\n")
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"}

    result = subprocess.run(
        ["bash", str(target_wrapper)],
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 37
    assert "FIRST_FILE_TO_OPEN=" in result.stdout
