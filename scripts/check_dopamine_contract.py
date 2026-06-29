#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts/dopamine_contract.v1.json"
ARTIFACT = ROOT / "artifacts/dopamine_contract/CONTRACT_VERDICT.json"
VALID_STATUSES = {"PASS", "FAIL", "BLOCK", "BLOCKED", "P2_RESEARCH"}


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


def validate_required_artifacts(
    contract: dict[str, Any], root: Path = ROOT
) -> list[str]:
    reasons: list[str] = []
    required = contract.get("required_artifacts", [])
    if not isinstance(required, list) or not required:
        return ["missing required_artifacts"]
    for item in required:
        if not isinstance(item, str) or not item.strip():
            reasons.append("invalid required artifact entry")
            continue
        path = root / item
        if not path.exists():
            reasons.append(f"missing artifact: {item}")
            continue
        if path.suffix != ".json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            reasons.append(f"invalid artifact json: {item}")
            continue
        if not isinstance(payload, dict) or payload.get("status") not in VALID_STATUSES:
            reasons.append(f"invalid artifact verdict: {item}")
        if not path.with_suffix(path.suffix + ".sha256").exists():
            reasons.append(f"missing artifact digest: {item}")
    return reasons


def validate_contract(
    contract: dict[str, Any],
    root: Path = ROOT,
    *,
    require_artifacts: bool = True,
) -> list[str]:
    reasons: list[str] = []
    if contract.get("component") != "geosync.dopamine":
        reasons.append("invalid component")
    semantics = contract.get("semantics", {})
    if not isinstance(semantics, dict) or not semantics:
        return reasons + ["missing semantics"]
    for name, spec in semantics.items():
        if not isinstance(spec, dict):
            reasons.append(f"{name}: invalid semantic spec")
            continue
        owner = str(spec.get("owner", ""))
        if not owner or not (root / owner).exists():
            reasons.append(f"{name}: missing owner {owner}")
        if spec.get("market_claim_allowed") is not False:
            reasons.append(f"{name}: promotion boundary must stay false")
        invariants = spec.get("required_invariants")
        if not isinstance(invariants, list) or not invariants:
            reasons.append(f"{name}: missing invariants")
        elif any(not isinstance(item, str) or not item.strip() for item in invariants):
            reasons.append(f"{name}: invalid invariant entry")
    forbidden_edges = contract.get("forbidden_edges", [])
    if not isinstance(forbidden_edges, list) or len(forbidden_edges) < 3:
        reasons.append("missing forbidden edges")
    if require_artifacts:
        reasons.extend(validate_required_artifacts(contract, root))
    return reasons


def main() -> int:
    contract = load_contract()
    reasons = validate_contract(contract)
    status = "PASS" if not reasons else "BLOCKED"
    digest = write_artifact(
        {
            "artifact_complete": not reasons,
            "blocking_reasons": reasons,
            "component": contract.get("component", "geosync.dopamine"),
            "gate": "dopamine_contract",
            "semantic_count": len(contract.get("semantics", {})),
            "status": status,
        }
    )
    print(json.dumps({"sha256": digest, "status": status}, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
