#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EVIDENCE_PATH = Path("reports/test_audit/evidence_index.json")


def _split_artifact_paths(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _resolve_audit_root(evidence_path: Path, audit_root: Path | None) -> Path:
    if audit_root is not None:
        return audit_root
    if evidence_path.resolve() == EVIDENCE_PATH.resolve():
        return Path("reports/test_audit")
    return evidence_path.parent


def _failed_baseline_dirs(audit_root: Path) -> set[Path]:
    failed: set[Path] = set()
    for marker in audit_root.glob("baseline_run_*/exit_code.txt"):
        try:
            value = marker.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value and value != "0":
            failed.add(marker.parent)
    return failed


def _verify_manifest(audit_root: Path, allowed_dirs: set[Path] | None) -> list[str]:
    if allowed_dirs is not None and not allowed_dirs:
        return []

    manifests: list[Path] = []
    if allowed_dirs is None:
        manifests = sorted(audit_root.glob("baseline_run_*/artifact_manifest.sha256"))
    else:
        for baseline_dir in sorted(allowed_dirs):
            manifest = baseline_dir / "artifact_manifest.sha256"
            if manifest.exists():
                manifests.append(manifest)

    errors: list[str] = []
    for manifest in manifests:
        baseline_dir = manifest.parent
        for line in manifest.read_text(encoding="utf-8").splitlines():
            entry = line.strip()
            if not entry:
                continue
            chunks = entry.split(maxsplit=1)
            if len(chunks) != 2:
                errors.append(f"Malformed manifest entry in {manifest}: {entry}")
                continue
            rel_path = chunks[1].strip()
            if not (baseline_dir / rel_path).exists():
                errors.append(f"Manifest references missing artifact: {(baseline_dir / rel_path).as_posix()}")
    return errors


def _referenced_baseline_dirs(claims: list[dict[str, Any]], audit_root: Path) -> set[Path]:
    referenced: set[Path] = set()
    root = audit_root.resolve()
    for claim in claims:
        for artifact in _split_artifact_paths(claim.get("artifact_path")):
            artifact_path = Path(artifact)
            if not artifact_path.is_absolute():
                artifact_path = Path.cwd() / artifact_path
            try:
                rel = artifact_path.resolve().relative_to(root)
            except ValueError:
                continue
            parts = rel.parts
            if parts and parts[0].startswith("baseline_run_"):
                referenced.add(root / parts[0])
    return referenced


def validate(
    evidence_path: Path = EVIDENCE_PATH,
    audit_root: Path | None = None,
    manifest_scope: str = "referenced",
) -> list[str]:
    if not evidence_path.exists():
        return [f"Evidence index not found: {evidence_path}"]

    doc = json.loads(evidence_path.read_text(encoding="utf-8"))
    claims = doc.get("claims", [])
    if not isinstance(claims, list):
        return ["Invalid evidence index: claims must be a list"]

    root = _resolve_audit_root(evidence_path, audit_root)
    failed_dirs = _failed_baseline_dirs(root)

    errors: list[str] = []
    for claim in claims:
        claim_id = claim.get("claim_id", "<unknown>")
        status = claim.get("status")
        artifact_paths = _split_artifact_paths(claim.get("artifact_path"))
        if status == "proven":
            if not artifact_paths:
                errors.append(f"{claim_id}: proven claim missing artifact_path")
                continue
            for artifact in artifact_paths:
                artifact_path = Path(artifact)
                if not artifact_path.exists():
                    errors.append(f"{claim_id}: missing artifact {artifact}")
                    continue
                for baseline_dir in failed_dirs:
                    try:
                        artifact_path.resolve().relative_to(baseline_dir.resolve())
                    except ValueError:
                        continue
                    errors.append(f"{claim_id}: artifact belongs to failed baseline {baseline_dir.as_posix()}")

    if manifest_scope == "referenced":
        manifest_dirs = _referenced_baseline_dirs(claims, root)
    elif manifest_scope == "all":
        manifest_dirs = None
    elif manifest_scope == "none":
        manifest_dirs = set()
    else:
        return [f"Invalid manifest_scope: {manifest_scope}"]

    errors.extend(_verify_manifest(root, manifest_dirs))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate claims have evidence.")
    parser.add_argument("--evidence-index", type=Path, default=EVIDENCE_PATH)
    parser.add_argument("--audit-root", type=Path, default=None)
    parser.add_argument("--manifest-scope", choices=("referenced", "all", "none"), default="referenced")
    args = parser.parse_args(argv)

    errors = validate(args.evidence_index, audit_root=args.audit_root, manifest_scope=args.manifest_scope)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Claims evidence validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
