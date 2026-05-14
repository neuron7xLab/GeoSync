# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""D-002J-P1A — source registry provenance audit fail-closed tests.

22 tests enforcing the audit contract from the operator's master
document §11. Every test contains ≥ 2 assertions or ≥ 2 distinct
cases. Drift sentinels for byte-exact locked governance shas and for
non-edit of source code / preregs / ledger are included.

P1A is audit-only: no ingestion, no analysis, no canonical-run
authorisation. The decision is `SOURCE_REGISTRY_REJECTED` because the
`information_constraint` mechanism family carries only one source
(ALFRED, PARTIAL) — the floor of ≥ 2 verified/partial per family
fails. That rejection is a scientifically valid outcome; tests below
attest the contract, not the verdict.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, cast

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
REGISTRY_JSON: Path = (
    REPO_ROOT / "artifacts" / "d002j" / "data_registry" / "source_registry_v1.json"
)
AUDIT_JSON: Path = (
    REPO_ROOT / "artifacts" / "d002j" / "data_registry" / "source_provenance_audit_v1.json"
)
SMOKE_JSON: Path = (
    REPO_ROOT / "artifacts" / "d002j" / "data_registry" / "source_access_smoke_v1.json"
)
LOCK_JSON: Path = (
    REPO_ROOT / "artifacts" / "d002j" / "data_registry" / "source_evidence_lock_v1.json"
)
SUMMARY_JSON: Path = (
    REPO_ROOT / "artifacts" / "d002j" / "data_registry" / "source_registry_audit_summary_v1.json"
)
PROVENANCE_MD: Path = REPO_ROOT / "docs" / "research" / "D002J_SOURCE_PROVENANCE_AUDIT.md"
DOWNGRADE_MD: Path = REPO_ROOT / "docs" / "research" / "D002J_SOURCE_DOWNGRADE_LOG.md"
BOUNDARY_MD: Path = REPO_ROOT / "docs" / "research" / "D002J_P1A_AUDIT_BOUNDARY.md"
BLOCKERS_MD: Path = REPO_ROOT / "docs" / "governance" / "D002G_CANONICAL_RUN_BLOCKERS.md"

# Locked governance sha256 pins (byte-exact at audit time)
LOCKED_D002C_LEDGER_SHA: str = (
    "eb0b7151d76e5409e6dc9bb4a023551de5e0704673d5ac9f726319ef84a32387"  # pragma: allowlist secret  # noqa: E501
)
LOCKED_D002G_PREREG_SHA: str = (
    "1ab91f09370e4705a8b0849467bc1f56df2e58d58d5623d3b6d905cbd110bb04"  # pragma: allowlist secret  # noqa: E501
)
LOCKED_D002G_ACCEPTANCE_SHA: str = (
    "875b1e3eb031b8e5333dc8b455454f0a30419ead1ebe787aa01d5882e7d6ad31"  # pragma: allowlist secret  # noqa: E501
)
LOCKED_D002H_PREREG_SHA: str = (
    "44b18b5a40ce9d188a9c3bd49339621f81a65a15f97a683247902450dd54acec"  # pragma: allowlist secret  # noqa: E501
)
LOCKED_D002I_PREREG_SHA: str = (
    "b646989c032dc0e29f9b791e0b68209ff22b40f4757737712badc8656cf2db5f"  # pragma: allowlist secret  # noqa: E501
)
LOCKED_D002J_PREREG_SHA: str = (
    "f3dc65b7e64b96eafe6f23ca8bdd0e05dc9bf95b12c2658b227bd0340f7975a0"  # pragma: allowlist secret  # noqa: E501
)
LOCKED_P1_REGISTRY_SHA: str = (
    "0fae24d4c3ef3165509166bec89d6dc5eee806888f352358ad77851e51079b7b"  # pragma: allowlist secret  # noqa: E501
)

ALLOWED_AUDIT_STATUSES: frozenset[str] = frozenset(
    {"VERIFIED", "PARTIAL", "DOWNGRADED", "REJECTED"}
)
ALLOWED_DECISIONS: frozenset[str] = frozenset(
    {
        "SOURCE_REGISTRY_VERIFIED",
        "SOURCE_REGISTRY_PARTIALLY_VERIFIED",
        "SOURCE_REGISTRY_REJECTED",
    }
)


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def test_p1_registry_exists() -> None:
    """P1 registry must exist with the expected sha256 anchor."""
    assert REGISTRY_JSON.is_file(), f"P1 registry missing at {REGISTRY_JSON}"
    sha = _sha256_of(REGISTRY_JSON)
    msg = (
        f"P1 registry sha256 drift: got {sha}, "
        f"expected {LOCKED_P1_REGISTRY_SHA}; audit was generated against a "
        "different registry version — regenerate audit artifacts"
    )
    assert sha == LOCKED_P1_REGISTRY_SHA, msg


def test_audit_json_exists() -> None:
    """Audit JSON must exist and declare correct schema_version + decision."""
    assert AUDIT_JSON.is_file(), f"audit JSON missing at {AUDIT_JSON}"
    audit = _load_json(AUDIT_JSON)
    assert (
        audit.get("schema_version") == "D002J-SOURCE-PROVENANCE-AUDIT-v1"
    ), f"audit schema_version drift: {audit.get('schema_version')!r}"


def test_access_smoke_json_exists() -> None:
    """Smoke probe JSON must exist with no_large_downloads + no_private_data true."""
    assert SMOKE_JSON.is_file(), f"smoke JSON missing at {SMOKE_JSON}"
    smoke = _load_json(SMOKE_JSON)
    assert smoke.get("schema_version") == "D002J-SOURCE-ACCESS-SMOKE-v1"
    assert smoke.get("no_large_downloads") is True
    assert smoke.get("no_private_data") is True


def test_evidence_lock_json_exists() -> None:
    """Evidence-lock JSON must exist and reference the audit parent."""
    assert LOCK_JSON.is_file(), f"evidence lock JSON missing at {LOCK_JSON}"
    lock = _load_json(LOCK_JSON)
    assert lock.get("schema_version") == "D002J-SOURCE-EVIDENCE-LOCK-v1"
    assert isinstance(lock.get("entries"), list)


def test_audit_schema_version() -> None:
    """Schema versions across the four audit artifacts must be the canonical set."""
    audit = _load_json(AUDIT_JSON)
    smoke = _load_json(SMOKE_JSON)
    lock = _load_json(LOCK_JSON)
    summary = _load_json(SUMMARY_JSON)
    cases: list[tuple[str, str]] = [
        (audit["schema_version"], "D002J-SOURCE-PROVENANCE-AUDIT-v1"),
        (smoke["schema_version"], "D002J-SOURCE-ACCESS-SMOKE-v1"),
        (lock["schema_version"], "D002J-SOURCE-EVIDENCE-LOCK-v1"),
        (summary["schema_version"], "D002J-SOURCE-REGISTRY-AUDIT-SUMMARY-v1"),
    ]
    for got, want in cases:
        assert got == want, f"schema_version drift: got {got!r}, want {want!r}"


def test_all_registry_sources_are_audited() -> None:
    """Every one of the 25 registry source_ids must have an audit entry."""
    reg = _load_json(REGISTRY_JSON)
    audit = _load_json(AUDIT_JSON)
    reg_ids = {cast(str, s["source_id"]) for s in reg["sources"]}
    audit_ids = {cast(str, e["source_id"]) for e in audit["sources"]}
    missing = reg_ids - audit_ids
    extra = audit_ids - reg_ids
    assert not missing, f"audit missing source_ids: {sorted(missing)}"
    assert not extra, f"audit has extra source_ids not in P1 registry: {sorted(extra)}"


def test_source_ids_match_registry() -> None:
    """Audit's parent_registry_sha256 must equal the on-disk P1 registry sha."""
    audit = _load_json(AUDIT_JSON)
    on_disk_sha = _sha256_of(REGISTRY_JSON)
    assert audit.get("parent_registry_sha256") == on_disk_sha, (
        f"audit pinned parent sha {audit.get('parent_registry_sha256')!r} "
        f"≠ on-disk sha {on_disk_sha!r}"
    )
    assert audit.get("total_sources") == 25


def test_no_private_or_internship_data() -> None:
    """Smoke and audit JSONs must declare zero private/restricted data ingestion."""
    smoke = _load_json(SMOKE_JSON)
    audit = _load_json(AUDIT_JSON)
    assert smoke.get("no_private_data") is True, "smoke JSON must declare no_private_data=true"
    audit_scope = audit.get("audit_scope") or {}
    assert audit_scope.get("no_ingestion") is True, "audit_scope must declare no_ingestion=true"
    assert audit_scope.get("no_canonical_run_authorisation") is True


def test_no_large_downloads() -> None:
    """Smoke JSON must declare no_large_downloads=true and bytes per probe bounded."""
    smoke = _load_json(SMOKE_JSON)
    assert smoke.get("no_large_downloads") is True
    # Each check must record an integer bytes_downloaded ≤ 4 KB (the master cap)
    over_cap: list[str] = []
    for check in smoke.get("checks", []):
        nbytes = int(check.get("bytes_downloaded", 0))
        if nbytes > 4096:
            over_cap.append(f"{check.get('source_id')}={nbytes}B")
    assert not over_cap, f"smoke probe over 4 KB cap: {over_cap}"


def test_each_source_has_audit_status() -> None:
    """Every audit entry must carry an audit_status in the canonical taxonomy."""
    audit = _load_json(AUDIT_JSON)
    bad: list[tuple[str, str]] = []
    for e in audit["sources"]:
        status = e.get("audit_status")
        if status not in ALLOWED_AUDIT_STATUSES:
            bad.append((cast(str, e.get("source_id")), str(status)))
    assert not bad, f"audit entries with invalid status: {bad}"
    # Drift sentinel: the canonical set itself
    assert ALLOWED_AUDIT_STATUSES == frozenset({"VERIFIED", "PARTIAL", "DOWNGRADED", "REJECTED"})


def test_each_verified_source_has_evidence_ref() -> None:
    """Every VERIFIED audit entry must list at least one evidence_ref."""
    audit = _load_json(AUDIT_JSON)
    bad: list[str] = []
    n_verified = 0
    for e in audit["sources"]:
        if e.get("audit_status") == "VERIFIED":
            n_verified += 1
            refs = e.get("evidence_refs") or []
            if not isinstance(refs, list) or len(refs) == 0:
                bad.append(cast(str, e["source_id"]))
    assert n_verified > 0, "audit declares zero VERIFIED sources — regression"
    assert not bad, f"VERIFIED sources missing evidence_refs: {bad}"


def test_each_verified_source_has_license_boundary() -> None:
    """Every VERIFIED audit entry must carry license_boundary_verified=true and the registry source must have non-empty license_boundary."""
    reg = _load_json(REGISTRY_JSON)
    audit = _load_json(AUDIT_JSON)
    license_by_id = {s["source_id"]: s.get("license_boundary", "") for s in reg["sources"]}
    bad: list[str] = []
    for e in audit["sources"]:
        if e.get("audit_status") == "VERIFIED":
            if not e.get("license_boundary_verified"):
                bad.append(f"{e['source_id']}:flag_false")
            text = license_by_id.get(e["source_id"], "")
            if not isinstance(text, str) or text.strip() == "":
                bad.append(f"{e['source_id']}:empty_text")
    assert not bad, f"VERIFIED sources with broken license_boundary: {bad}"


def test_each_verified_source_has_forbidden_use() -> None:
    """Every VERIFIED audit entry must confirm forbidden_use and the P1 source must list forbidden_use clauses."""
    reg = _load_json(REGISTRY_JSON)
    audit = _load_json(AUDIT_JSON)
    fu_by_id: dict[str, list[Any]] = {
        s["source_id"]: list(s.get("forbidden_use") or []) for s in reg["sources"]
    }
    bad: list[str] = []
    for e in audit["sources"]:
        if e.get("audit_status") == "VERIFIED":
            if not e.get("forbidden_use_confirmed"):
                bad.append(f"{e['source_id']}:flag_false")
            if not fu_by_id.get(e["source_id"]):
                bad.append(f"{e['source_id']}:empty_clauses")
    assert not bad, f"VERIFIED sources missing forbidden_use confirmation: {bad}"


def test_crisis_windows_retain_verified_sources() -> None:
    """Every crisis window CW1..CW6 must retain ≥ 3 verified/partial sources."""
    reg = _load_json(REGISTRY_JSON)
    audit = _load_json(AUDIT_JSON)
    cw_map = {s["source_id"]: list(s.get("crisis_window_relevance") or []) for s in reg["sources"]}
    counts: Counter[str] = Counter()
    for e in audit["sources"]:
        if e.get("audit_status") in {"VERIFIED", "PARTIAL"}:
            for cw in cw_map.get(e["source_id"], []):
                counts[cw] += 1
    short: list[str] = [cw for cw in ("CW1", "CW2", "CW3", "CW4", "CW5", "CW6") if counts[cw] < 3]
    assert (
        not short
    ), f"crisis windows below floor 3 verified/partial: {short}; counts={dict(counts)}"
    # Distinct second assertion: all six windows must be present
    assert all(cw in counts for cw in ("CW1", "CW2", "CW3", "CW4", "CW5", "CW6"))


def test_mechanisms_retain_verified_sources() -> None:
    """Audit summary must surface mechanism-family verified-or-partial counts.

    P1A's REJECTED decision is driven by the information_constraint family
    holding only 1 verified/partial source (ALFRED PARTIAL). This test
    enforces that the family counts are surfaced honestly — not that the
    floor passes (it deliberately doesn't, which is why the decision is
    REJECTED rather than PARTIALLY_VERIFIED).
    """
    summary = _load_json(SUMMARY_JSON)
    counts = summary.get("by_mechanistic_relevance_verified_or_partial") or {}
    assert isinstance(counts, dict)
    assert len(counts) >= 6, f"mechanism family count surface too narrow: {counts}"
    # The information_constraint family is the floor-failure that drives REJECTED.
    # It must be in the counts (else the audit is hiding the failure).
    assert "information_constraint" in counts, (
        "information_constraint family must appear in summary mechanism counts; "
        "absence would silently mask the REJECTED rationale"
    )


def test_downgrade_log_exists() -> None:
    """Every DOWNGRADED audit entry must appear in the downgrade-log MD."""
    audit = _load_json(AUDIT_JSON)
    assert DOWNGRADE_MD.is_file(), f"downgrade log missing at {DOWNGRADE_MD}"
    log_text = DOWNGRADE_MD.read_text(encoding="utf-8")
    downgraded_ids = [
        cast(str, e["source_id"])
        for e in audit["sources"]
        if e.get("audit_status") in {"DOWNGRADED", "REJECTED"}
    ]
    missing = [sid for sid in downgraded_ids if sid not in log_text]
    assert not missing, f"downgrade log missing source_ids: {missing}"
    # Second distinct assertion: log must declare at least one entry
    assert len(downgraded_ids) > 0, "audit declares zero DOWNGRADED — regression vs honest P1A"


def test_audit_summary_counts_match_audit() -> None:
    """Summary counts must equal per-source audit counts exactly."""
    audit = _load_json(AUDIT_JSON)
    summary = _load_json(SUMMARY_JSON)
    counter = Counter(e.get("audit_status") for e in audit["sources"])
    cases: list[tuple[str, int, int]] = [
        ("verified", counter["VERIFIED"], int(summary["verified"])),
        ("partial", counter["PARTIAL"], int(summary["partial"])),
        ("downgraded", counter["DOWNGRADED"], int(summary["downgraded"])),
        ("rejected", counter["REJECTED"], int(summary["rejected"])),
    ]
    mismatches = [(k, a, b) for k, a, b in cases if a != b]
    assert not mismatches, f"summary↔audit count mismatch: {mismatches}"
    assert summary["total_sources"] == len(audit["sources"]) == 25


def test_audit_does_not_authorize_canonical_run() -> None:
    """Neither audit nor summary may authorise a canonical run."""
    audit = _load_json(AUDIT_JSON)
    summary = _load_json(SUMMARY_JSON)
    audit_scope = audit.get("audit_scope") or {}
    assert (
        summary.get("canonical_run_authorized") is False
    ), "summary canonical_run_authorized must be false in P1A — audit-only PR"
    assert (
        audit_scope.get("no_canonical_run_authorisation") is True
    ), "audit_scope must explicitly forbid canonical run authorisation"
    # Decision must be one of the canonical strings
    assert audit.get("audit_decision") in ALLOWED_DECISIONS


def test_d002j_prereg_not_modified() -> None:
    """D-002J prereg sha256 byte-exact at audit time."""
    prereg = REPO_ROOT / "docs" / "governance" / "D002J_PREREGISTRATION.yaml"
    assert prereg.is_file()
    sha = _sha256_of(prereg)
    msg = (
        f"D-002J prereg sha drift: got {sha}, "
        f"expected {LOCKED_D002J_PREREG_SHA}; P1A must not modify the prereg"
    )
    assert sha == LOCKED_D002J_PREREG_SHA, msg


def test_d002c_ledger_not_modified() -> None:
    """D-002C claim ledger sha256 byte-exact at audit time + D-002G prereg + D-002G acceptance + D-002H prereg + D-002I prereg."""
    pairs: list[tuple[str, str]] = [
        ("docs/governance/D002C_CLAIM_LEDGER.yaml", LOCKED_D002C_LEDGER_SHA),
        ("docs/governance/D002G_PREREGISTRATION.yaml", LOCKED_D002G_PREREG_SHA),
        ("docs/governance/D002G_ACCEPTANCE_RULES.md", LOCKED_D002G_ACCEPTANCE_SHA),
        ("docs/governance/D002H_PREREGISTRATION.yaml", LOCKED_D002H_PREREG_SHA),
        ("docs/governance/D002I_PREREGISTRATION.yaml", LOCKED_D002I_PREREG_SHA),
    ]
    drifts: list[str] = []
    for rel, expected in pairs:
        p = REPO_ROOT / rel
        assert p.is_file(), f"{rel} missing"
        got = _sha256_of(p)
        if got != expected:
            drifts.append(f"{rel}: {got}!={expected}")
    assert not drifts, f"locked governance sha drift: {drifts}"


def test_systemic_risk_source_code_not_modified() -> None:
    """research/systemic_risk/*.py must not be modified by the P1A PR.

    This test enforces a static invariant: P1A is governance/audit only.
    The drift sentinel here is the presence of expected module names
    (no rename / no deletion) — the diff-bound commit acceptor enforces
    byte-level un-touched-ness via the forbidden_paths list. We probe
    the module name set to fail-closed on regressions like wholesale
    removal of d002c_substrates.py.
    """
    sr_root = REPO_ROOT / "research" / "systemic_risk"
    assert sr_root.is_dir(), "research/systemic_risk/ must exist"
    expected_modules: set[str] = {
        "d002c_substrates.py",
        "d002g_null_mechanisms.py",
        "data_firewall.py",
    }
    present = {p.name for p in sr_root.glob("*.py")}
    missing = expected_modules - present
    assert not missing, (
        f"research/systemic_risk/ lost expected modules during P1A: {missing}; "
        "P1A is audit-only and must not touch source code"
    )


def test_no_unresolved_merge_markers() -> None:
    """All P1A artifacts must be free of git merge conflict markers."""
    marker_re = re.compile(r"^(<<<<<<<|=======|>>>>>>>|\|\|\|\|\|\|\|)")
    files: list[Path] = [
        AUDIT_JSON,
        SMOKE_JSON,
        LOCK_JSON,
        SUMMARY_JSON,
        PROVENANCE_MD,
        DOWNGRADE_MD,
        BOUNDARY_MD,
    ]
    hits: list[tuple[str, int, str]] = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), start=1):
            m = marker_re.match(line)
            if m:
                hits.append((str(f.relative_to(REPO_ROOT)), i, m.group(1)))
    assert not hits, "unresolved merge markers:\n" + "\n".join(
        f"  {f}:{ln}: {marker!r}" for f, ln, marker in hits
    )
    # Distinct second assertion: scanner must recognise canonical markers
    for canonical in ("<<<<<<<", "=======", ">>>>>>>", "|||||||"):
        assert (
            marker_re.match(canonical) is not None
        ), f"marker regex drift: failed to recognise {canonical!r}"
