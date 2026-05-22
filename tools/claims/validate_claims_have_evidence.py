from __future__ import annotations

import json
import re
from pathlib import Path

METRICS = Path("docs/METRICS_CONTRACT.md")
EVIDENCE = Path("reports/test_audit/evidence_index.json")
SCAN_DIRS = [Path("README.md"), Path("docs/testing/canonical_commands.md"), Path("reports/test_audit/current_state_20260522.md")]
ALLOWED_STATUSES = {"proven", "partial", "goal", "remove", "quarantine"}
STRONG_WORDS = ("verified", "production-grade", "enterprise-grade", "robust", "secure")
REQUIRED_FIELDS = (
    "claim_id",
    "command",
    "artifact_path",
    "status",
    "last_verified_sha",
    "timestamp",
)


def _parse_claim_rows(metrics_text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in metrics_text.splitlines():
        if not line.startswith("| ") or line.startswith("|----"):
            continue
        normalized = line.replace(r"\|", "__PIPE__")
        parts = [p.strip().replace("__PIPE__", "|") for p in normalized.strip().strip("|").split("|")]
        if len(parts) < 7 or parts[0] == "id":
            continue
        if not re.fullmatch(r"[A-Z0-9_]+", parts[0]):
            continue
        rows.append({"claim_id": parts[0], "status": parts[5], "claim": parts[2]})
    return rows


def _is_strong_claim(row: dict[str, str]) -> bool:
    return row["status"] == "proven"


def _scan_docs_for_overclaims() -> list[str]:
    violations: list[str] = []
    allowlist = {"docs/METRICS_CONTRACT.md"}
    for base in SCAN_DIRS:
        files = [base] if base.is_file() else list(base.rglob("*.md"))
        for path in files:
            rel = str(path).replace("\\", "/")
            if rel in allowlist:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            for word in STRONG_WORDS:
                if word in text and "metrics contract" not in text and "claim boundary" not in text:
                    violations.append(f"Potential overclaim word '{word}' in {rel}")
                    break
    return violations


def main() -> int:
    metrics_text = METRICS.read_text(encoding="utf-8")
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    records = evidence.get("records", [])
    index = {r.get("claim_id"): r for r in records}

    errors: list[str] = []
    claim_rows = _parse_claim_rows(metrics_text)

    for row in claim_rows:
        if row["status"] not in ALLOWED_STATUSES:
            errors.append(f"Invalid status for {row['claim_id']}: {row['status']}")

        record = index.get(row["claim_id"])
        if record is None:
            errors.append(f"Missing evidence record for {row['claim_id']}")
            continue

        missing_fields = [f for f in REQUIRED_FIELDS if f not in record]
        if missing_fields:
            errors.append(f"Evidence record {row['claim_id']} missing fields: {', '.join(missing_fields)}")

        if _is_strong_claim(row):
            if not str(record.get("last_verified_sha", "")).strip():
                errors.append(f"Strong claim {row['claim_id']} missing last_verified_sha")
            if not str(record.get("timestamp", "")).strip():
                errors.append(f"Strong claim {row['claim_id']} missing timestamp")
            if record.get("status") != "proven":
                errors.append(f"Strong claim {row['claim_id']} has mismatched evidence status {record.get('status')}")

    errors.extend(_scan_docs_for_overclaims())

    if errors:
        print("CLAIM EVIDENCE VALIDATION FAILED")
        for e in errors:
            print(f" - {e}")
        return 1

    print("Claim evidence validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
