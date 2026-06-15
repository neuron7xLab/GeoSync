# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Generate the GeoSync measured-maturity evidence bundle, deterministically.

Reads the *real* repository state — the Ricci microstructure preregistration,
its committed placeholder dataset/config, the promotion guard, and the market
approximation-limits contract — and emits a machine-checkable maturity bundle
under ``artifacts/geosync_maturity/``.

It promotes nothing. The claim tiers it reports are exactly what the guard
returns on the committed inputs: with a placeholder (zero-hash, synthetic)
dataset, the Ricci microstructure line is ``HYPOTHESIS`` / ``BLOCKED`` and the
bundle says so. Run it again after real data lands and the same code reports the
honestly-earned tier — no hand-edited JSON.

    python -m tools.research.build_geosync_maturity_report
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.research.market_limits import load_contract
from tools.research.ricci_preregistration_guard import evaluate_promotion

_ROOT = Path(__file__).resolve().parents[2]
_OUT_DIR = _ROOT / "artifacts" / "geosync_maturity"
_PREREG_DATASET = (
    _ROOT
    / "research_lines"
    / "ricci_microstructure_v1"
    / "preregistration"
    / "dataset.example.json"
)
_PREREG_CONFIG = (
    _ROOT / "research_lines" / "ricci_microstructure_v1" / "preregistration" / "config.example.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(name: str, payload: Any) -> Path:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = _OUT_DIR / name
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def build_claim_maturity_report() -> dict[str, Any]:
    dataset = _load_json(_PREREG_DATASET)
    config = _load_json(_PREREG_CONFIG)
    decision = evaluate_promotion(dataset, config, result=None)
    return {
        "schema": "geosync_maturity/claim_maturity@1",
        "claims": [
            {
                "claim_id": "ricci_microstructure_v1",
                "claim_text": (
                    "Ollivier-Ricci curvature over crypto-perp L2 order-book "
                    "microstructure carries falsifiable OOS signal for short-horizon "
                    "mid-price direction beyond order-flow imbalance + persistence nulls."
                ),
                "current_status": decision.claim_tier,
                "decision": decision.decision,
                "falsification_status": decision.falsification_status,
                "promotion_allowed": decision.allowed,
                "promotion_blockers": list(decision.reasons),
                "evidence_present": [
                    "docs/research/ricci_microstructure_v1_PREREGISTRATION.md",
                    "tools/research/ricci_preregistration_guard.py",
                    "schemas/research/ricci_microstructure_v1_config.schema.json",
                    "schemas/research/ricci_microstructure_v1_dataset.schema.json",
                ],
                "missing_evidence": [
                    "real hash-pinned depth-5 L2 session (>= 3600s)",
                    "replayable inference artifact beating all four nulls net-of-cost",
                    "multi_session_replay across >= 2 independent sessions",
                ],
                "safe_next_status": "TESTED_REAL_SINGLE (only after one real, gate-passing session)",
            }
        ],
        "retired_claims_preserved": True,
        "note": (
            "Tiers are computed by evaluate_promotion() on the committed placeholder "
            "inputs. No tier was hand-assigned."
        ),
    }


def build_l2_ricci_gap_report() -> dict[str, Any]:
    dataset = _load_json(_PREREG_DATASET)
    return {
        "schema": "geosync_maturity/l2_ricci_gap@1",
        "status": "INSTRUMENTED_HYPOTHESIS",
        "pipeline_present": [
            "scripts/collect_binance_perp_l2.py",
            "scripts/fetch_dukascopy_l2.py",
            "scripts/run_l2_full_cycle.py",
            "core/indicators/ricci.py",
            "core/indicators/temporal_ricci.py",
            "tools/research/validate_ricci_artifact_schema.py",
        ],
        "evidence_present": [
            "deterministic depth-5 dataset/config schemas",
            "fail-closed promotion guard (15 passing witnesses)",
        ],
        "missing_evidence": [
            "non-zero data_sha256 (committed placeholder is zero-hash)",
            "is_synthetic == false (committed placeholder is synthetic)",
            "passed nulls + positive net-of-cost result",
        ],
        "committed_placeholder": {
            "data_sha256_is_zero": dataset.get("data_sha256", "0" * 64) == "0" * 64,
            "is_synthetic": bool(dataset.get("is_synthetic", True)),
        },
        "next_patch": (
            "Land one real hash-pinned depth-5 crypto-perp session (>= 1h), run "
            "run_l2_full_cycle.py, and re-run this generator to record the earned tier."
        ),
    }


def build_falsifier_matrix() -> dict[str, Any]:
    nulls = ["permutation_null", "lag_sweep_no_future_data", "cost_model", "multi_session_replay"]
    sessions = [
        ("same_symbol_diff_sessions", "multi_session_replay"),
        ("diff_symbols_same_period", "permutation_null"),
        ("shock_vs_calm_session", "multi_session_replay"),
        ("low_vs_high_liquidity", "cost_model"),
        ("missing_data_session", "lag_sweep_no_future_data"),
        ("replayed_cached_session", "multi_session_replay"),
    ]
    return {
        "schema": "geosync_maturity/falsifier_matrix@1",
        "claim_id": "ricci_microstructure_v1",
        "registered_nulls": nulls,
        "matrix": [
            {
                "falsifier_id": f"ricci_micro.{name}",
                "claim_id": "ricci_microstructure_v1",
                "dataset_class": name,
                "null_baseline": null,
                "expected_failure_condition": "null IC >= signal IC, or signal IC <= 0",
                "pass_condition": "signal OOS IC > 0 and beats null net-of-cost across >= 2 sessions",
                "inconclusive_condition": "insufficient depth (< 3600s) or missing session",
                "executable_now": False,
                "blocked_reason": "no real hash-pinned session committed yet",
            }
            for name, null in sessions
        ],
        "negative_evidence_preserved": True,
        "negative_evidence_policy": (
            "Rejected lines archived as sha-pinned negative artifacts "
            "(prereg section 13); not deleted."
        ),
    }


def build_physics_limits_report() -> dict[str, Any]:
    contract = load_contract()
    return {
        "schema": "geosync_maturity/physics_limits@1",
        "contract_path": "physics_contracts/market_limits.yaml",
        "claim_tier_vocabulary": list(contract.claim_tiers),
        "limits": [
            {
                "limit_id": limit.limit_id,
                "mathematical_object": limit.mathematical_object,
                "valid_when": list(limit.valid_when),
                "invalid_when": list(limit.invalid_when),
                "witness_required": limit.witness_required,
                "falsifier_required": limit.falsifier_required,
                "forbidden_claims": list(limit.forbidden_claims),
                "max_claim_tier_violated": limit.max_claim_tier_violated,
            }
            for limit in contract.limits
        ],
    }


def main() -> int:
    written = [
        _write("CLAIM_MATURITY_REPORT.json", build_claim_maturity_report()),
        _write("L2_RICCI_GAP_REPORT.json", build_l2_ricci_gap_report()),
        _write("FALSIFIER_MATRIX.json", build_falsifier_matrix()),
        _write("PHYSICS_LIMITS_REPORT.json", build_physics_limits_report()),
    ]
    for path in written:
        print(f"wrote {path.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
