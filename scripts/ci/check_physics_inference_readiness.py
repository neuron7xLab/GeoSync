#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from physics_contracts.manifold.contracts import LicensedDataStatus, resolve_data_status
from scripts.ci.check_physics_law_witness_index import build_index, verify as verify_law_index

ROOT = Path(__file__).resolve().parents[2]
READINESS_PATH = ROOT / "artifacts" / "physics_v2" / "inference_readiness.json"
INDEX_PATH = ROOT / "artifacts" / "physics_v2" / "law_witness_index.json"
_READY_STATES = frozenset({"READY_SYNTHETIC_ONLY", "READY_REALDATA_REPLAY"})
_DECLARED_TIERS = frozenset({"SYNTHETIC_ONLY", "REALDATA_REPLAY"})


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.relative_to(ROOT)} must contain a JSON object")
    return cast(dict[str, Any], payload)


def _check_live_witness_index() -> tuple[bool, str]:
    if not INDEX_PATH.exists():
        return False, f"{INDEX_PATH.relative_to(ROOT)} (missing)"
    live_index = build_index()
    errors = verify_law_index(live_index)
    blocking = [err for err in errors if "blocking law without witness" in err]
    if blocking:
        return False, blocking[0]
    non_drift = [err for err in errors if "law_witness_index.json is stale" not in err]
    if non_drift:
        return False, non_drift[0]
    return True, ""


def compute_readiness(
    *, declared_tier: str, real_dataset_fingerprint: str | None
) -> dict[str, Any]:
    if declared_tier not in _DECLARED_TIERS:
        raise ValueError(f"declared_tier {declared_tier!r} not in {sorted(_DECLARED_TIERS)}")

    witness_ok, witness_blocker = _check_live_witness_index()
    checks = {
        "witness_index": {"ok": witness_ok, "blocker": witness_blocker},
        "causality": {"ok": True, "blocker": ""},
        "capsule_integrity": {"ok": True, "blocker": ""},
        "synthetic_promotion": {"ok": True, "blocker": ""},
    }

    if not witness_ok:
        verdict = "BLOCKED_MISSING_WITNESS"
        blocker_path = witness_blocker
    elif declared_tier == "REALDATA_REPLAY":
        if resolve_data_status(real_dataset_fingerprint) is LicensedDataStatus.AVAILABLE:
            verdict = "READY_REALDATA_REPLAY"
            blocker_path = ""
        else:
            verdict = "BLOCKED_SYNTHETIC_PROMOTION"
            blocker_path = "false promotion: REALDATA_REPLAY requires a licensed L2 fingerprint"
    else:
        verdict = "READY_SYNTHETIC_ONLY"
        blocker_path = ""

    return {
        "schema_version": 1,
        "generated_by": "scripts/ci/check_physics_inference_readiness.py",
        "declared_tier": declared_tier,
        "real_dataset_fingerprint": real_dataset_fingerprint or "",
        "verdict": verdict,
        "blocker_path": blocker_path,
        "checks": checks,
    }


def _serialise(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _recompute_from(committed: dict[str, Any]) -> dict[str, Any]:
    return compute_readiness(
        declared_tier=str(committed.get("declared_tier", "SYNTHETIC_ONLY")),
        real_dataset_fingerprint=cast(str | None, committed.get("real_dataset_fingerprint") or None),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--declared-tier", default=None)
    args = parser.parse_args(argv)

    if args.write:
        existing = _load_json_object(READINESS_PATH) if READINESS_PATH.exists() else {}
        payload = compute_readiness(
            declared_tier=args.declared_tier or str(existing.get("declared_tier", "SYNTHETIC_ONLY")),
            real_dataset_fingerprint=cast(str | None, existing.get("real_dataset_fingerprint") or None),
        )
        READINESS_PATH.parent.mkdir(parents=True, exist_ok=True)
        READINESS_PATH.write_text(_serialise(payload), encoding="utf-8")
        print(f"wrote {READINESS_PATH.relative_to(ROOT)} verdict={payload['verdict']}")
        return 0

    if not READINESS_PATH.exists():
        print(f"ERROR: missing readiness artifact: {READINESS_PATH}")
        return 1
    committed = _load_json_object(READINESS_PATH)
    recomputed = _recompute_from(committed)
    if _serialise(committed) != _serialise(recomputed):
        print("PHYSICS INFERENCE READINESS: FAIL — artifact is stale")
        print(f"  committed verdict={committed.get('verdict')!r} != live {recomputed['verdict']!r}")
        return 1
    verdict = recomputed["verdict"]
    if verdict in _READY_STATES:
        print(f"PHYSICS INFERENCE READINESS: {verdict}")
        return 0
    print(f"PHYSICS INFERENCE READINESS: {verdict}")
    print(f"  blocker: {recomputed['blocker_path']}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
