#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts/dopamine_claims/CLAIM_PROMOTION_VERDICT.json"
CLAIM_FILES = [
    ROOT / "docs/CLAIMS.yaml",
    ROOT / "physics_contracts/catalog.yaml",
    ROOT / "docs/PHYSICS_CANON.md",
    ROOT / "docs/PHYSICS_VERDICT.md",
]
EVAL = ROOT / "results/dopamine_rpe_extension/EVAL_SUMMARY.json"
MANIFEST = ROOT / "results/dopamine_rpe_extension/ARTIFACT_MANIFEST.sha256"


def write(payload: dict[str, Any]) -> str:
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    payload.setdefault("generated_at_utc", "1970-01-01T00:00:00Z")
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    ARTIFACT.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    ARTIFACT.with_suffix(ARTIFACT.suffix + ".sha256").write_text(
        f"{digest}  {ARTIFACT.name}\n",
        encoding="utf-8",
    )
    return digest


def main() -> int:
    reasons: list[str] = []
    for path in CLAIM_FILES:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        if "dopamine" not in text:
            continue
        if "anchored" in text and ("rollback" not in text or not MANIFEST.exists()):
            reasons.append(f"unsupported anchored wording: {path.relative_to(ROOT)}")
        if "extrapolated" in text and not MANIFEST.exists():
            reasons.append(f"unsupported extrapolated wording: {path.relative_to(ROOT)}")
        if "tested" in text and not EVAL.exists():
            reasons.append(f"unsupported tested wording: {path.relative_to(ROOT)}")
    status = "PASS" if not reasons else "FAIL"
    digest = write(
        {
            "component": "geosync.dopamine",
            "gate": "claim_promotion",
            "status": status,
            "experimental_surface_status": "P2_RESEARCH"
            if not MANIFEST.exists()
            else "TESTED",
            "market_claim_allowed": False,
            "blocking_reasons": reasons,
        }
    )
    print(json.dumps({"status": status, "sha256": digest}, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
