# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Repository-surface contract tests."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def test_canonical_surface_files_exist() -> None:
    expected = [
        "README.md",
        "PRODUCT_CATEGORY.md",
        "CLAIMS.md",
        "FORBIDDEN_CLAIMS.md",
        "docs/REPOSITORY_SYSTEM.md",
        "AGENTS.md",
    ]
    missing = [rel for rel in expected if not (ROOT / rel).is_file()]
    assert missing == []


def test_readme_links_repository_governance_files() -> None:
    readme = _read("README.md")
    for target in [
        "PRODUCT_CATEGORY.md",
        "CLAIMS.md",
        "FORBIDDEN_CLAIMS.md",
        "docs/REPOSITORY_SYSTEM.md",
        "AGENTS.md",
    ]:
        assert f"]({target})" in readme


def test_repository_system_has_review_and_completion_contracts() -> None:
    doc = _read("docs/REPOSITORY_SYSTEM.md")
    for phrase in [
        "Reviewer Protocol",
        "Definition of Repository Completion",
        "PRODUCT_CATEGORY.md",
        "CLAIMS.md",
        "FORBIDDEN_CLAIMS.md",
        "AGENTS.md",
        "replay command succeeds",
        "retired or failed states are preserved",
    ]:
        assert phrase in doc


def test_agents_contract_has_required_sections() -> None:
    doc = _read("AGENTS.md")
    for phrase in [
        "Prime Directive",
        "Canonical Files",
        "Required Behavior",
        "Forbidden Behavior",
        "Evidence-Bearing Artifact Minimum",
        "Documentation Change Rule",
        "Code Change Rule",
        "Release Rule",
        "Preferred Final Response From Agents",
        "Changed files:",
        "Commit SHA:",
        "Verification command:",
        "Claim tier impact:",
        "Remaining blocker:",
    ]:
        assert phrase in doc


def test_public_research_boundary_language_remains_visible() -> None:
    combined = "\n".join(_read(path) for path in ["README.md", "docs/REPOSITORY_SYSTEM.md"])
    for phrase in [
        "HYPOTHESIS",
        "INSTRUMENTED",
        "NOT EVIDENCE-BEARING",
        "Synthetic data may",
        "Schema validity proves",
    ]:
        assert phrase in combined


def test_repo_integrity_workflow_runs_on_public_surface_changes() -> None:
    workflow = _read(".github/workflows/repo-integrity-gate.yml")
    for target in [
        "README.md",
        "PRODUCT_CATEGORY.md",
        "CLAIMS.md",
        "FORBIDDEN_CLAIMS.md",
        "AGENTS.md",
        "docs/**/*.md",
        "tools/mfn_surface_check.py",
        "tools/patch_makefile_geosync_coverage.py",
        "tools/validate_json_artifact_contract.py",
        "tools/check_json_contract_evidence_policy.py",
        "tests/ci/**",
    ]:
        assert target in workflow


def test_repo_integrity_script_runs_contract_test_directories() -> None:
    script = _read("scripts/ci/prove_repo_integrity.sh")
    assert "tools/mfn_surface_check.py --strict" in script
    assert "tools/validate_json_artifact_contract.py" in script
    assert "tools/check_json_contract_evidence_policy.py" in script
    assert "json_artifact_contract.result.json" in script
    assert "json_artifact_policy.result.json" in script
    assert "tests/research_lines" in script
    assert "tests/ci" in script
    assert "pytest" in script


def test_testing_guide_matches_python_contract() -> None:
    guide = _read("docs/operations/TESTING.md")
    assert ">=3.11,<3.13" in guide
    assert "3.11–3.13" not in guide


def test_testing_guide_keeps_geosync_in_local_coverage_commands() -> None:
    guide = _read("docs/operations/TESTING.md")
    assert "--cov=geosync" in guide
    assert "`geosync/`" in guide


def test_testing_guide_uses_existing_make_shortcut_names() -> None:
    guide = _read("docs/operations/TESTING.md")
    assert "make test-fast" in guide
    assert "make test-all" in guide
    assert "make test-heavy" in guide
    assert "make test:fast" not in guide
    assert "make test:all" not in guide
    assert "make test:heavy" not in guide


def test_mfn_release_workflow_uses_installed_package_path() -> None:
    workflow = _read(".github/workflows/mfn-release-gate.yml")
    hidden_path_token = "PYTHON" + "PATH=."
    assert hidden_path_token not in workflow
    assert "python -m pip install -e ." in workflow
    assert "tests/unit/mfn" in workflow


def test_mfn_release_workflow_runs_surface_checker() -> None:
    workflow = _read(".github/workflows/mfn-release-gate.yml")
    tool_path = "tools/" + "mfn_surface_check.py"
    assert tool_path in workflow
    assert tool_path + " --" + "strict" in workflow
    assert "tests/ci/test_mfn_surface_check.py" in workflow
    assert "tests/ci/test_mfn_surface_payload_contract.py" in workflow


def test_mfn_release_workflow_checks_reproducible_bundle_files() -> None:
    workflow = _read(".github/workflows/mfn-release-gate.yml")
    for artifact_name in ["SHA256SUMS", "manifest.json", "runbook.md"]:
        assert artifact_name in workflow
