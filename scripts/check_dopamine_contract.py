#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts/dopamine_contract.v1.json"
ARTIFACT = ROOT / "artifacts/dopamine_contract/CONTRACT_VERDICT.json"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_artifact(payload: dict[str, Any]) -> str:
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    payload.setdefault("generated_at_utc", "1970-01-01T00:00:00Z")
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    ARTIFACT.write_bytes(raw)
    digest = sha256(raw)
    ARTIFACT.with_suffix(ARTIFACT.suffix + ".sha256").write_text(
        f"{digest}  {ARTIFACT.name}\n",
        encoding="utf-8",
    )
    return digest


def load_contract() -> dict[str, Any]:
    if not CONTRACT.exists():
        return {"semantics": {}}
    loaded = json.loads(CONTRACT.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {"semantics": {}}


def main() -> int:
    reasons: list[str] = []
    contract = load_contract()
    semantics = contract.get("semantics", {})
    if not isinstance(semantics, dict) or not semantics:
        reasons.append("missing semantics")
        semantics = {}
    for name, spec in semantics.items():
        if not isinstance(spec, dict):
            reasons.append(f"{name}: invalid semantic spec")
            continue
        owner = str(spec.get("owner", ""))
        if not owner or not (ROOT / owner).exists():
            reasons.append(f"{name}: missing owner {owner}")
        if spec.get("market_claim_allowed") is not False:
            reasons.append(f"{name}: promotion boundary must stay false")
        if not spec.get("required_invariants"):
            reasons.append(f"{name}: missing invariants")
    if len(contract.get("forbidden_edges", [])) < 3:
        reasons.append("missing forbidden edges")
    status = "PASS" if not reasons else "FAIL"
    digest = write_artifact(
        {
            "component": contract.get("component", "geosync.dopamine"),
            "gate": "dopamine_contract",
            "status": status,
            "blocking_reasons": reasons,
            "semantic_count": len(semantics),
        }
    )
    print(json.dumps({"status": status, "sha256": digest}, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
