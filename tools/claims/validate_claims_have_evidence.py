from __future__ import annotations

import json
from pathlib import Path

METRICS = Path("docs/METRICS_CONTRACT.md")
EVIDENCE = Path("reports/test_audit/evidence_index.json")


def _claim_rows(text: str) -> list[list[str]]:
    rows = []
    for line in text.splitlines():
        if line.startswith("| ") and not line.startswith("|----"):
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            if len(parts) >= 7 and parts[0] != "id":
                rows.append(parts)
    return rows


def main() -> int:
    metrics_text = METRICS.read_text(encoding="utf-8")
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    index = {r["claim_id"]: r for r in evidence.get("records", [])}

    missing: list[str] = []
    for row in _claim_rows(metrics_text):
        claim_id = row[0]
        status = row[5]
        if status == "proven" and claim_id not in index:
            missing.append(claim_id)

    if missing:
        print("Missing evidence records for proven claims:")
        for claim in missing:
            print(f" - {claim}")
        return 1

    print("All proven claims have evidence records.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
