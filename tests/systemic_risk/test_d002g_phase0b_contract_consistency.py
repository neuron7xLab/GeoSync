# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""P1-3 Codex review fix — Phase 0b BCa↔percentile contract consistency.

Attack
------
The original Phase 0b implementation advertised a BCa (bias-corrected
accelerated) bootstrap CI in its docstring + adversarial audit narrative,
but actually used a simple percentile quantile of bootstrap means. That
mismatch is itself a contract bug: downstream consumers reading the
Phase 0b verdict under the "BCa" promise would make decisions under a
stronger assumption than the code provides.

User decision (Path 2): downgrade the contract. Phase 0b uses
percentile bootstrap CI. True BCa is future hardening.

This test enforces the contract consistency programmatically:
  1. ``d002g_phase0_verification.py`` source contains NO "BCa" /
     "bias-corrected" / "accelerated" tokens outside the explicit
     P1-3 limitation paragraph.
  2. ``D002G_P1_IMPLEMENTATION_REPORT.md`` contains the verbatim
     limitation fragment "percentile bootstrap CI, not BCa".
  3. ``D002G_P1_DESIGN_ADVERSARIAL_AUDIT.md`` marks Strike-R3 as
     DOWNGRADED with the limitation.

A test-suite drift in either direction (re-introducing the BCa claim
without an implementation, OR implementing BCa without updating the
contract) immediately surfaces as a test failure.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE0_SRC = REPO_ROOT / "research" / "systemic_risk" / "d002g_phase0_verification.py"
REPORT = REPO_ROOT / "docs" / "governance" / "D002G_P1_IMPLEMENTATION_REPORT.md"
AUDIT = REPO_ROOT / "docs" / "governance" / "D002G_P1_DESIGN_ADVERSARIAL_AUDIT.md"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_phase0_source_only_mentions_BCa_inside_P1_3_limitation_paragraph() -> None:
    """The source file may mention BCa ONLY inside the explicit limitation."""
    src = _read(PHASE0_SRC)
    # Every BCa / bias-corrected / accelerated mention must live in the
    # limitation paragraph that explicitly references the Codex P1-3 fix.
    pat = re.compile(r"(BCa|bias-corrected|accelerated)", re.IGNORECASE)
    for m in pat.finditer(src):
        # Window: ±600 chars around the match so the limitation paragraph
        # context (which carries the P1-3 marker) is fully captured even
        # when the matched token appears later in the paragraph than the
        # opening "P1-3" marker.
        start = max(0, m.start() - 600)
        end = min(len(src), m.end() + 600)
        window = src[start:end]
        has_p13 = "P1-3" in window or "Codex review fix" in window
        has_downgrade = "percentile" in window.lower()
        msg = (
            f"P1-3 VIOLATED: Phase 0b source mentions {m.group()!r} at offset "
            f"{m.start()} OUTSIDE the explicit P1-3 limitation paragraph. "
            f"Window:\n---\n{window}\n---"
        )
        assert has_p13 and has_downgrade, msg


def test_report_contains_verbatim_percentile_bootstrap_limitation() -> None:
    """Report must contain the verbatim limitation fragment."""
    body = _read(REPORT)
    fragment = "percentile bootstrap CI, not BCa"
    assert fragment in body, (
        f"P1-3 VIOLATED: D002G_P1_IMPLEMENTATION_REPORT.md missing the "
        f"verbatim limitation fragment {fragment!r}. The P1-3 contract "
        f"downgrade is not visible in the report."
    )


def test_report_pins_p1_3_section_title() -> None:
    """The report must carry the P1-3 section heading."""
    body = _read(REPORT)
    pinned = "## P1-3 percentile bootstrap CI"
    assert pinned in body, f"P1-3 VIOLATED: report missing pinned section heading {pinned!r}."


def test_audit_marks_R3_as_downgraded() -> None:
    """Adversarial audit doc must mark Strike-R3 as DOWNGRADED."""
    body = _read(AUDIT)
    has_downgraded = "DOWNGRADED" in body
    has_percentile = "percentile bootstrap CI" in body or "percentile bootstrap" in body
    has_r3 = "R3" in body or "Strike-R3" in body
    assert has_downgraded and has_percentile and has_r3, (
        "P1-3 VIOLATED: D002G_P1_DESIGN_ADVERSARIAL_AUDIT.md must mark "
        "Strike-R3 as DOWNGRADED and reference 'percentile bootstrap CI'. "
        f"has_downgraded={has_downgraded} has_percentile={has_percentile} "
        f"has_r3={has_r3}"
    )


def test_locked_files_never_mention_bca_for_phase0b() -> None:
    """Sanity: the LOCKED files must NOT have ever mentioned BCa for Phase 0b.

    The contract downgrade is in the implementation report + audit (both
    modifiable). The locked governance files (prereg / acceptance /
    design / commit-acceptor / D-002C ledger) MUST NOT carry a BCa
    claim for Phase 0b — otherwise this PR would need to touch them,
    which is FORBIDDEN. If any locked file mentions BCa in the Phase 0b
    context, ABORT — the contract downgrade is incomplete.
    """
    locked_files = [
        REPO_ROOT / "docs" / "governance" / "D002G_PREREGISTRATION.yaml",
        REPO_ROOT / "docs" / "governance" / "D002G_NONDEGENERATE_NULL_DESIGN.md",
        REPO_ROOT / "docs" / "governance" / "D002G_ACCEPTANCE_RULES.md",
    ]
    for fp in locked_files:
        if not fp.exists():
            continue
        body = _read(fp)
        # The locked design doc references BCa CI for the CANONICAL
        # sweep R1 (not Phase 0b). The contract downgrade scope is
        # Phase 0b only. Reject only on the conjunction "Phase 0b" +
        # any BCa token in the same locked file.
        if "phase 0b" in body.lower() or "phase_0b" in body.lower():
            has_bca = bool(re.search(r"(BCa|bias-corrected|accelerated)", body))
            assert not has_bca, (
                f"P1-3 LOCKED-FILE LEAK: {fp} mentions BOTH Phase 0b and "
                f"BCa/bias-corrected/accelerated. Locked files must not "
                f"carry a Phase-0b BCa claim — this PR is forbidden from "
                f"modifying them. Report and ABORT."
            )
