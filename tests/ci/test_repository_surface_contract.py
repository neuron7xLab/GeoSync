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
