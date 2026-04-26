"""Tests for the dependency-truth unifier.

Three contracts:

1. The shipping repository, with PR #445 already applied, produces NO
   drift for `torch` (F01) and NO drift for `strawberry-graphql` (F03).
   The validator's silence on those packages is the load-bearing
   regression test for both.

2. Each drift class (D1–D6) is mechanically detectable on a synthetic
   tree planted with the exact pattern.

3. The deterministic JSON output is stable when the input is stable.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from textwrap import dedent
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = REPO_ROOT / "tools" / "deps" / "validate_dependency_truth.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("vdt", VALIDATOR_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["vdt"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def vdt() -> ModuleType:
    return _load()


def _seed_minimal_repo(
    root: Path,
    pyproject: str = "",
    requirements_txt: str = "",
    requirements_scan_txt: str = "",
    requirements_lock: str = "",
    requirements_dev_lock: str = "",
    requirements_scan_lock: str = "",
    constraints: str = "",
    dockerfile: str | None = None,
    workflow: str | None = None,
) -> None:
    (root / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    (root / "requirements.txt").write_text(requirements_txt, encoding="utf-8")
    (root / "requirements-scan.txt").write_text(requirements_scan_txt, encoding="utf-8")
    (root / "requirements.lock").write_text(requirements_lock, encoding="utf-8")
    (root / "requirements-dev.lock").write_text(requirements_dev_lock, encoding="utf-8")
    (root / "requirements-scan.lock").write_text(requirements_scan_lock, encoding="utf-8")
    (root / "constraints").mkdir(exist_ok=True)
    (root / "constraints" / "security.txt").write_text(constraints, encoding="utf-8")
    if dockerfile:
        (root / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    if workflow:
        wf_dir = root / ".github" / "workflows"
        wf_dir.mkdir(parents=True, exist_ok=True)
        (wf_dir / "ci.yml").write_text(workflow, encoding="utf-8")


# ---------------------------------------------------------------------------
# Contract 1 — F01 + F03 closure regression tests
# ---------------------------------------------------------------------------


def test_repo_has_no_torch_drift(vdt: ModuleType) -> None:
    """If F01 ever regresses, this test fails."""
    report = vdt.collect(REPO_ROOT)
    torch_drifts = [d for d in report.drifts if d.package == "torch"]
    assert not torch_drifts, "torch drift detected — F01 has regressed:\n  " + "\n  ".join(
        f"{d.drift_class}: {d.detail}" for d in torch_drifts
    )


def test_repo_has_no_strawberry_drift(vdt: ModuleType) -> None:
    """If F03 ever regresses (manifest-side), this test fails."""
    report = vdt.collect(REPO_ROOT)
    sb_drifts = [d for d in report.drifts if d.package.startswith("strawberry-graphql")]
    assert (
        not sb_drifts
    ), "strawberry-graphql drift detected — F03 has regressed:\n  " + "\n  ".join(
        f"{d.drift_class}: {d.detail}" for d in sb_drifts
    )


def test_validator_main_exits_zero_in_report_only_mode(vdt: ModuleType) -> None:
    rc = vdt.main(["--repo-root", str(REPO_ROOT)])
    assert rc == 0


def test_validator_exits_nonzero_when_actionable_drift_present(
    vdt: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A planted F01-class drift on a NEW package must trigger non-zero exit."""
    _seed_minimal_repo(
        tmp_path,
        pyproject=dedent("""
            [project]
            name = "fake"
            version = "0.0.0"
            dependencies = [
                "newpkg>=2.0.0",
            ]
            """),
        requirements_txt="newpkg>=1.0.0\n",
    )
    rc = vdt.main(["--repo-root", str(tmp_path), "--exit-on-drift"])
    assert rc == 1


# ---------------------------------------------------------------------------
# Contract 2 — drift-class mechanical detection
# ---------------------------------------------------------------------------


def test_d1_detects_pyproject_above_requirements(vdt: ModuleType, tmp_path: Path) -> None:
    _seed_minimal_repo(
        tmp_path,
        pyproject=dedent("""
            [project]
            name = "fake"
            version = "0.0.0"
            dependencies = ["foo>=2.0.0"]
            """),
        requirements_txt="foo>=1.0.0\n",
    )
    report = vdt.collect(tmp_path)
    d1 = [d for d in report.drifts if d.drift_class == "D1" and d.package == "foo"]
    assert d1, report.drifts


def test_d2_detects_lock_below_floor(vdt: ModuleType, tmp_path: Path) -> None:
    _seed_minimal_repo(
        tmp_path,
        pyproject=dedent("""
            [project]
            name = "fake"
            version = "0.0.0"
            dependencies = ["bar>=2.0.0"]
            """),
        requirements_txt="bar>=2.0.0\n",
        requirements_lock="bar==1.5.0\n",
    )
    report = vdt.collect(tmp_path)
    d2 = [d for d in report.drifts if d.drift_class == "D2" and d.package == "bar"]
    assert d2, report.drifts


def test_d3_detects_scan_runtime_lock_divergence(vdt: ModuleType, tmp_path: Path) -> None:
    _seed_minimal_repo(
        tmp_path,
        pyproject=dedent("""
            [project]
            name = "fake"
            version = "0.0.0"
            dependencies = ["baz>=1.0.0"]
            """),
        requirements_txt="baz>=1.0.0\n",
        requirements_lock="baz==1.0.0\n",
        requirements_scan_lock="baz==1.5.0\n",
    )
    report = vdt.collect(tmp_path)
    d3 = [d for d in report.drifts if d.drift_class == "D3" and d.package == "baz"]
    assert d3, report.drifts


def test_d4_detects_dockerfile_unscanned_manifest(vdt: ModuleType, tmp_path: Path) -> None:
    _seed_minimal_repo(
        tmp_path,
        pyproject=dedent("""
            [project]
            name = "fake"
            version = "0.0.0"
            dependencies = []
            """),
        dockerfile="FROM python:3.12\nRUN pip install -r requirements.txt\n",
        workflow=dedent("""
            name: CI
            on: [push]
            jobs:
              build:
                runs-on: ubuntu-latest
                steps:
                  - run: pip install -r requirements.txt
            """),
    )
    # The workflow above only `pip install`s — does NOT pip-audit. The
    # detector treats this as "not scanned".
    report = vdt.collect(tmp_path)
    d4 = [d for d in report.drifts if d.drift_class == "D4"]
    assert d4, report.drifts


def test_d4_does_not_flag_when_workflow_runs_pip_audit(vdt: ModuleType, tmp_path: Path) -> None:
    _seed_minimal_repo(
        tmp_path,
        pyproject=dedent("""
            [project]
            name = "fake"
            version = "0.0.0"
            dependencies = []
            """),
        dockerfile="FROM python:3.12\nRUN pip install -r requirements.txt\n",
        workflow=dedent("""
            name: CI
            on: [push]
            jobs:
              audit:
                runs-on: ubuntu-latest
                steps:
                  - run: pip-audit -r requirements.txt
            """),
    )
    report = vdt.collect(tmp_path)
    d4 = [d for d in report.drifts if d.drift_class == "D4"]
    assert not d4, report.drifts


def test_d5_detects_constraints_below_manifest(vdt: ModuleType, tmp_path: Path) -> None:
    _seed_minimal_repo(
        tmp_path,
        pyproject=dedent("""
            [project]
            name = "fake"
            version = "0.0.0"
            dependencies = ["qux>=2.0.0"]
            """),
        constraints="qux==1.5.0\n",
    )
    report = vdt.collect(tmp_path)
    d5 = [d for d in report.drifts if d.drift_class == "D5" and d.package == "qux"]
    assert d5, report.drifts


def test_d6_pointer_is_synthetic_and_low_priority(vdt: ModuleType, tmp_path: Path) -> None:
    _seed_minimal_repo(
        tmp_path,
        pyproject=dedent("""
            [project]
            name = "fake"
            version = "0.0.0"
            dependencies = []
            """),
    )
    report = vdt.collect(tmp_path)
    d6 = [d for d in report.drifts if d.drift_class == "D6"]
    assert len(d6) == 1
    assert d6[0].priority == "LOW"
    assert "deptry" in d6[0].detail.lower()


# ---------------------------------------------------------------------------
# Contract 3 — deterministic output
# ---------------------------------------------------------------------------


def test_collect_is_deterministic(vdt: ModuleType, tmp_path: Path) -> None:
    _seed_minimal_repo(
        tmp_path,
        pyproject=dedent("""
            [project]
            name = "fake"
            version = "0.0.0"
            dependencies = ["foo>=2.0.0", "bar>=3.0.0"]
            """),
        requirements_txt="foo>=1.0.0\nbar>=2.0.0\n",
    )
    a = vdt.collect(tmp_path).to_dict()
    b = vdt.collect(tmp_path).to_dict()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
