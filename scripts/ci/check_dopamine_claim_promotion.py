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
BLOCKED_TERMS = (
    "market edge",
    "biological dopamine replication",
    "autonomous financial intelligence",
    "anchored alpha",
    "empirical superiority",
)


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


def dopamine_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            if current:
                block = "\n".join(current)
                if "dopamine" in block.lower():
                    blocks.append(block)
                current = []
            continue
        if line[:1] in {"#", "-"} and current:
            block = "\n".join(current)
            if "dopamine" in block.lower():
                blocks.append(block)
            current = [line]
        else:
            current.append(line)
    if current:
        block = "\n".join(current)
        if "dopamine" in block.lower():
            blocks.append(block)
    return blocks


def evaluate_text(text: str, *, eval_exists: bool, manifest_exists: bool) -> list[str]:
    reasons: list[str] = []
    for block in dopamine_blocks(text):
        lowered = block.lower()
        reasons.extend(
            f"blocked claim term: {term}" for term in BLOCKED_TERMS if term in lowered
        )
        if "anchored" in lowered and ("rollback" not in lowered or not manifest_exists):
            reasons.append("unsupported anchored wording")
        if "extrapolated" in lowered and not manifest_exists:
            reasons.append("unsupported extrapolated wording")
        if "tested" in lowered and not eval_exists:
            reasons.append("unsupported tested wording")
    return reasons


def main() -> int:
    reasons: list[str] = []
    for path in CLAIM_FILES:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        scoped_reasons = evaluate_text(
            text,
            eval_exists=EVAL.exists(),
            manifest_exists=MANIFEST.exists(),
        )
        reasons.extend(
            f"{path.relative_to(ROOT)}: {reason}" for reason in scoped_reasons
        )
    status = "PASS" if not reasons else "FAIL"
    digest = write(
        {
            "blocking_reasons": reasons,
            "component": "geosync.dopamine",
            "experimental_surface_status": (
                "TESTED" if MANIFEST.exists() else "P2_RESEARCH"
            ),
            "gate": "claim_promotion",
            "market_claim_allowed": False,
            "status": status,
        }
    )
    print(json.dumps({"sha256": digest, "status": status}, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
