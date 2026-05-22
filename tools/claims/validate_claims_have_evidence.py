from __future__ import annotations

import json
import re
from pathlib import Path

METRICS = Path("docs/METRICS_CONTRACT.md")
EVIDENCE = Path("reports/test_audit/evidence_index.json")
SCAN_DIRS = [Path("README.md"), Path("docs/testing/canonical_commands.md"), Path("reports/test_audit/current_state_20260522.md")]
REQUIRED_FIELDS = {
    "claim_id",
    "command",
    "artifact_path",
    "status",
    "last_verified_sha",
    "timestamp",
}
STRONG_WORDS = ["verified", "production-grade", "enterprise-grade", "robust", "secure"]
ALLOWED_STATUSES = {"proven", "partial", "goal", "remove", "quarantine"}


def _claim_rows(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        if line.startswith("| ") and not line.startswith("|----"):
            normalized = line.replace(r"\|", "__PIPE__")
            parts = [p.strip().replace("__PIPE__", "|") for p in normalized.strip().strip("|").split("|")]
            if len(parts) >= 7 and parts[0] != "id" and re.fullmatch(r"[A-Z0-9_]+", parts[0]):
                rows.append({"id": parts[0], "status": parts[5], "claim": parts[2]})
    return rows


def _iter_scan_files() -> list[Path]:
    files: list[Path] = []
    for p in SCAN_DIRS:
        if p.is_file():
            files.append(p)
        elif p.exists():
            files.extend([x for x in p.rglob("*.md") if "METRICS_CONTRACT.md" not in str(x)])
    return files


def main() -> int:
    metrics_text = METRICS.read_text(encoding="utf-8")
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    index = {r.get("claim_id"): r for r in evidence.get("records", [])}

    errors: list[str] = []
    claim_rows = _claim_rows(metrics_text)

    for row in claim_rows:
        if row["status"] not in ALLOWED_STATUSES:
            errors.append(f"Claim {row['id']} has invalid status {row['status']}")

    for row in claim_rows:
        cid, status = row["id"], row["status"]
        record = index.get(cid)
        if record is None:
            if status in {"proven", "partial", "goal", "quarantine"}:
                errors.append(f"Missing evidence record for claim {cid} ({status})")
            continue

        missing_fields = [f for f in REQUIRED_FIELDS if f not in record]
        if missing_fields:
            errors.append(f"Evidence record {cid} missing fields: {', '.join(missing_fields)}")

    # overclaim scan
    for file in _iter_scan_files():
        text = file.read_text(encoding="utf-8", errors="ignore").lower()
        for word in STRONG_WORDS:
            if word in text and "metrics_contract" not in text and "claim boundary" not in text:
                errors.append(f"Potential strong claim word '{word}' in {file}")
                break

    if errors:
        print("CLAIM EVIDENCE VALIDATION FAILED")
        for e in errors:
            print(f" - {e}")
        return 1

    print("Claim evidence validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
