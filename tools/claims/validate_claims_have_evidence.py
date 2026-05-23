#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EVIDENCE_PATH = Path("reports/test_audit/evidence_index.json")
REQUIRED_FIELDS = {
    "claim_id",
    "claim_text",
    "domain",
    "claim_status",
    "command",
    "artifact_path",
    "verdict",
    "release_decision",
}
ALLOWED_STATUSES = {"proven", "partial", "goal", "remove", "quarantine"}
GOAL_BAD_RELEASE = {
    "achieved",
    "accepted_as_proven",
    "release_evidence",
    "unrestricted_release",
    "allow",
}
PARTIAL_BAD_RELEASE = {"proven", "accepted_as_proven", "unrestricted_release", "allow"}
QUARANTINE_ALLOWED_RELEASE = {
    "blocked",
    "blocked_as_release_evidence",
    "not_admissible",
    "quarantine",
}
REMOVE_ALLOWED_RELEASE = {"remove", "blocked"}


def _iter_artifact_paths(raw: str | None) -> list[Path]:
    if raw is None:
        return []
    parts = [chunk.strip().strip("`") for chunk in str(raw).split(",")]
    return [Path(part) for part in parts if part]


def _resolve_audit_root(evidence_path: Path, audit_root: Path | None) -> Path:
    if audit_root is not None:
        return audit_root
    if evidence_path.resolve() == EVIDENCE_PATH.resolve():
        return Path("reports/test_audit")
    return evidence_path.parent


def _failed_baseline_dirs(audit_root: Path) -> set[Path]:
    failed: set[Path] = set()
    for ex in audit_root.glob("baseline_run_*/exit_code.txt"):
        code = ex.read_text(encoding="utf-8").strip()
        if code and code != "0":
            failed.add(ex.parent.resolve())
    return failed


def _artifact_in_failed_baseline(artifact_path: Path, failed_dirs: set[Path]) -> bool:
    resolved = artifact_path.resolve()
    for failed_dir in failed_dirs:
        if failed_dir in resolved.parents or resolved == failed_dir:
            return True
    return False


def _verify_manifest(audit_root: Path, allowed_dirs: set[Path] | None) -> list[str]:
    if allowed_dirs == set():
        return []

    manifests: list[Path]
    if allowed_dirs is None:
        manifests = sorted(audit_root.glob("baseline_run_*/artifact_manifest.sha256"))
    else:
        manifests = []
        for baseline_dir in sorted(allowed_dirs):
            manifest = baseline_dir / "artifact_manifest.sha256"
            if manifest.exists():
                manifests.append(manifest)

    errors: list[str] = []
    for manifest in manifests:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            entry = line.strip()
            if not entry:
                continue
            chunks = entry.split(maxsplit=1)
            if len(chunks) != 2:
                errors.append(f"malformed manifest entry: {manifest}: {entry}")
                continue

            expected_hash, rel_name = chunks
            target = manifest.parent / rel_name.strip()
            if not target.exists():
                errors.append(f"manifest missing file: {target}")
                continue

            actual_hash = hashlib.sha256(target.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                errors.append(f"manifest hash mismatch: {target}")
    return errors


def _referenced_baseline_dirs(claims: list[dict[str, Any]], audit_root: Path) -> set[Path]:
    root = audit_root.resolve()
    referenced: set[Path] = set()

    for claim in claims:
        for artifact in _iter_artifact_paths(claim.get("artifact_path")):
            artifact_path = artifact if artifact.is_absolute() else (Path.cwd() / artifact)
            try:
                rel = artifact_path.resolve().relative_to(root)
            except ValueError:
                continue

            if rel.parts and rel.parts[0].startswith("baseline_run_"):
                referenced.add((root / rel.parts[0]).resolve())

    return referenced


def validate(
    evidence_path: Path = EVIDENCE_PATH,
    *,
    audit_root: Path | None = None,
    manifest_scope: str = "referenced",
) -> list[str]:
    errors: list[str] = []

    if manifest_scope not in {"referenced", "all", "none"}:
        return [f"invalid manifest_scope={manifest_scope}"]

    if not evidence_path.exists():
        return [f"Evidence index not found: {evidence_path}"]

    doc = json.loads(evidence_path.read_text(encoding="utf-8"))
    if doc.get("schema_version") != "1.1":
        errors.append("schema_version must be 1.1")

    claims = doc.get("claims")
    if not isinstance(claims, list):
        return ["claims must be a list"]

    resolved_audit_root = _resolve_audit_root(evidence_path, audit_root)
    failed_dirs = _failed_baseline_dirs(resolved_audit_root)

    for c in claims:
        missing = [f for f in REQUIRED_FIELDS if f not in c]
        if missing:
            errors.append(f"{c.get('claim_id', '<missing-id>')}: missing fields {missing}")
            continue

        cid = c["claim_id"]
        status = c["claim_status"]
        release_decision = str(c["release_decision"]).strip().lower()

        if status not in ALLOWED_STATUSES:
            errors.append(f"{cid}: invalid status {status}")
            continue

        artifact_paths = _iter_artifact_paths(c.get("artifact_path"))

        if status == "proven":
            for f in ("command", "artifact_path", "last_verified_sha", "timestamp_utc"):
                if not str(c.get(f, "")).strip():
                    errors.append(f"{cid}: proven missing {f}")

            if not artifact_paths:
                errors.append(f"{cid}: proven missing artifact_path")
            for ap in artifact_paths:
                artifact_path = ap if ap.is_absolute() else (Path.cwd() / ap)
                if not artifact_path.exists():
                    errors.append(f"{cid}: artifact missing {artifact_path}")
                    continue
                if (
                    _artifact_in_failed_baseline(artifact_path, failed_dirs)
                    and "fail" not in str(c["claim_text"]).lower()
                ):
                    errors.append(f"{cid}: proven uses failed baseline artifact {artifact_path}")

        if status == "goal" and release_decision in GOAL_BAD_RELEASE:
            errors.append(f"{cid}: goal has invalid release_decision={release_decision}")
        if status == "partial" and release_decision in PARTIAL_BAD_RELEASE:
            errors.append(f"{cid}: partial has invalid release_decision={release_decision}")
        if status == "quarantine" and release_decision not in QUARANTINE_ALLOWED_RELEASE:
            errors.append(f"{cid}: quarantine has invalid release_decision={release_decision}")
        if status == "remove" and release_decision not in REMOVE_ALLOWED_RELEASE:
            errors.append(f"{cid}: remove has invalid release_decision={release_decision}")

    if manifest_scope == "referenced":
        manifest_dirs = _referenced_baseline_dirs(claims, resolved_audit_root)
    elif manifest_scope == "all":
        manifest_dirs = None
    else:  # manifest_scope == "none"
        manifest_dirs = set()

    errors.extend(_verify_manifest(resolved_audit_root, manifest_dirs))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate claims have evidence.")
    parser.add_argument("--evidence-index", type=Path, default=EVIDENCE_PATH)
    parser.add_argument("--audit-root", type=Path, default=None)
    parser.add_argument(
        "--manifest-scope",
        choices=("referenced", "all", "none"),
        default="referenced",
    )
    args = parser.parse_args(argv)

    errors = validate(
        args.evidence_index,
        audit_root=args.audit_root,
        manifest_scope=args.manifest_scope,
    )
    if errors:
        print("CLAIM EVIDENCE VALIDATION FAILED")
        for e in errors:
            print(f" - {e}")
        return 1

    print("Claim evidence validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())