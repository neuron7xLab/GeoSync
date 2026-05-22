from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

METRICS_PATH = Path("docs/METRICS_CONTRACT.md")
EVIDENCE_PATH = Path("reports/test_audit/evidence_index.json")
STATUS_ALLOWED = {"proven", "partial", "goal", "remove", "quarantine"}
ACHIEVEMENT_WORDS = ("verified", "enforced", "guaranteed", "production-grade", "enterprise-grade")


def parse_metrics_claims(metrics_text: str) -> dict[str, dict[str, str]]:
    claims: dict[str, dict[str, str]] = {}
    for line in metrics_text.splitlines():
        if not line.startswith("| ") or line.startswith("|----"):
            continue
        normalized = line.replace(r"\|", "__PIPE__")
        parts = [
            p.strip().replace("__PIPE__", "|") for p in normalized.strip().strip("|").split("|")
        ]
        if len(parts) < 7 or not re.fullmatch(r"[A-Z0-9_]+", parts[0]):
            continue
        claims[parts[0]] = {"claim": parts[2], "status": parts[5], "notes": parts[6]}
    return claims


def _artifact_exists(path_value: str) -> bool:
    cleaned = path_value.strip().strip("`")
    primary = cleaned.split(",")[0].strip()
    if not primary or primary == "N/A":
        return False
    primary = primary.split(":")[0] if ".yml:" in primary or ".md:" in primary else primary
    return Path(primary).exists()


def _check_manifest_if_present() -> list[str]:
    errors: list[str] = []
    for manifest in Path("reports/test_audit").glob("baseline_run_*/artifact_manifest.sha256"):
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, fname = line.split(maxsplit=1)
            target = manifest.parent / fname.strip()
            if not target.exists():
                errors.append(f"Manifest missing file: {target}")
                continue
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            if actual != expected:
                errors.append(f"Manifest hash mismatch: {target}")
    return errors


def validate(metrics_path: Path = METRICS_PATH, evidence_path: Path = EVIDENCE_PATH) -> list[str]:
    errors: list[str] = []
    metrics = parse_metrics_claims(metrics_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    claims = evidence.get("claims", [])
    records = {c.get("claim_id"): c for c in claims}

    for cid, rec in records.items():
        if cid not in metrics:
            errors.append(f"{cid}: evidence claim missing from METRICS_CONTRACT")
            continue
        if not rec.get("claim_status"):
            errors.append(f"{cid}: evidence record missing status")
            continue

        m = metrics[cid]
        e_status = rec.get("claim_status")
        if e_status not in STATUS_ALLOWED:
            errors.append(f"{cid}: invalid evidence status {e_status}")
        if m["status"] != e_status:
            errors.append(f"{cid}: status mismatch metrics={m['status']} evidence={e_status}")

        if e_status == "proven":
            for field in (
                "claim_id",
                "command",
                "artifact_path",
                "last_verified_sha",
                "timestamp_utc",
            ):
                if not str(rec.get(field, "")).strip():
                    errors.append(f"{cid}: proven claim missing {field}")
            if not _artifact_exists(str(rec.get("artifact_path", ""))):
                errors.append(f"{cid}: proven claim artifact path does not exist")

        text_low = (m["claim"] + " " + m["notes"] + " " + str(rec.get("notes", ""))).lower()
        if (
            e_status == "goal"
            and rec.get("verdict") == "accepted"
            and any(w in text_low for w in ACHIEVEMENT_WORDS)
        ):
            errors.append(f"{cid}: goal claim presented as achievement")
        if e_status == "partial" and "proven" in text_low and rec.get("verdict") == "accepted":
            errors.append(f"{cid}: partial claim presented as proven")

    for cid, m in metrics.items():
        if m["status"] == "proven" and cid not in records:
            errors.append(f"{cid}: missing evidence record for proven claim")

    errors.extend(_check_manifest_if_present())
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
