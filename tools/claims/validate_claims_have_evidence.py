from __future__ import annotations

import hashlib
import json
from pathlib import Path

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


def _failed_baseline_dirs() -> set[Path]:
    failed = set()
    for ex in Path("reports/test_audit").glob("baseline_run_*/exit_code.txt"):
        code = ex.read_text(encoding="utf-8").strip()
        if code and code != "0":
            failed.add(ex.parent.resolve())
    return failed


def _artifact_in_failed_baseline(artifact_path: str, failed_dirs: set[Path]) -> bool:
    ap = Path(artifact_path.split(",")[0].strip().strip("`")).resolve()
    for d in failed_dirs:
        if d in ap.parents or ap == d:
            return True
    return False


def _verify_manifest() -> list[str]:
    errors = []
    for manifest in Path("reports/test_audit").glob("baseline_run_*/artifact_manifest.sha256"):
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, fname = line.split(maxsplit=1)
            target = manifest.parent / fname.strip()
            if not target.exists():
                errors.append(f"manifest missing file: {target}")
                continue
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            if actual != expected:
                errors.append(f"manifest hash mismatch: {target}")
    return errors


def validate(evidence_path: Path = EVIDENCE_PATH) -> list[str]:
    errors: list[str] = []
    doc = json.loads(evidence_path.read_text(encoding="utf-8"))
    if doc.get("schema_version") != "1.1":
        errors.append("schema_version must be 1.1")
    claims = doc.get("claims")
    if not isinstance(claims, list):
        return ["claims must be a list"]

    failed_dirs = _failed_baseline_dirs()

    for c in claims:
        missing = [f for f in REQUIRED_FIELDS if f not in c]
        if missing:
            errors.append(f"{c.get('claim_id','<missing-id>')}: missing fields {missing}")
            continue
        cid = c["claim_id"]
        st = c["claim_status"]
        rd = str(c["release_decision"]).strip().lower()
        if st not in ALLOWED_STATUSES:
            errors.append(f"{cid}: invalid status {st}")
            continue

        if st == "proven":
            for f in ("command", "artifact_path", "last_verified_sha", "timestamp_utc"):
                if not str(c.get(f, "")).strip():
                    errors.append(f"{cid}: proven missing {f}")
            ap = Path(str(c.get("artifact_path", "")).split(",")[0].strip().strip("`"))
            if not ap.exists():
                errors.append(f"{cid}: artifact missing {ap}")
            elif (
                _artifact_in_failed_baseline(str(ap), failed_dirs)
                and "fail" not in c["claim_text"].lower()
            ):
                errors.append(f"{cid}: proven uses failed baseline artifact {ap}")

        if st == "goal" and rd in GOAL_BAD_RELEASE:
            errors.append(f"{cid}: goal has invalid release_decision={rd}")
        if st == "partial" and rd in PARTIAL_BAD_RELEASE:
            errors.append(f"{cid}: partial has invalid release_decision={rd}")
        if st == "quarantine" and rd not in QUARANTINE_ALLOWED_RELEASE:
            errors.append(f"{cid}: quarantine has invalid release_decision={rd}")
        if st == "remove" and rd not in REMOVE_ALLOWED_RELEASE:
            errors.append(f"{cid}: remove has invalid release_decision={rd}")

    errors.extend(_verify_manifest())
    return errors


def main() -> int:
    errors = validate()
    if errors:
        print("CLAIM EVIDENCE VALIDATION FAILED")
        for e in errors:
            print(f" - {e}")
        return 1
    print("Claim evidence validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
