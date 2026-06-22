#!/usr/bin/env python3
# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""The single physics-inference readiness gate (Physics v2, Task 7).

One CI gate decides whether the repository is ready for physics inference, and
at which honest tier. It does **not** add a new runtime subsystem — it exercises
the contracts and the law-witness index that already exist:

Inputs
------
* ``artifacts/physics_v2/law_witness_index.json`` (Task 1),
* the manifold contracts: ``MarketCausalGraphSnapshot`` (+ its builder),
  ``RicciFlowTrace``, ``SynchronizationManifoldFrame``,
  the negative-control ``ComparisonReport`` adapter, and
  ``PhysicsEvidenceCapsuleShape``.

Verdict states (exactly one is emitted)
---------------------------------------
* ``READY_SYNTHETIC_ONLY``    — the fail-closed synthetic chain is wired and
  honest; no real-data claim is asserted.
* ``READY_REALDATA_REPLAY``   — a licensed dataset fingerprint and a replay
  command are present and the capsule binds at a real-data tier.
* ``BLOCKED_MISSING_WITNESS`` — a blocking law has no witness in the index.
* ``BLOCKED_SYNTHETIC_PROMOTION`` — a real-data tier is declared without real
  data (false promotion).
* ``BLOCKED_INVALID_CAUSALITY`` — the causal-cutoff / look-ahead guard is not
  live.
* ``BLOCKED_BROKEN_CAPSULE``   — the evidence chain does not assemble or its
  fail-closed guards are not live.

The gate PASSES (exit 0) only when the verdict is a ``READY_*`` state that
matches the declared tier, and prints the exact blocker path otherwise.

Usage::

    python scripts/ci/check_physics_inference_readiness.py            # verify
    python scripts/ci/check_physics_inference_readiness.py --write     # regenerate

Exit codes: ``0`` ready at the declared tier; ``1`` blocked or stale.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from physics_contracts.manifold.builders import OrderFlowFrame, build_causal_graph_snapshot
from physics_contracts.manifold.contracts import (
    LicensedDataStatus,
    resolve_data_status,
)
from physics_contracts.manifold.evidence_binding import bind_physics_evidence
from physics_contracts.manifold.fixtures import synthetic_snapshot
from physics_contracts.manifold.negative_controls import assemble_comparison_report
from physics_contracts.manifold.ricci_trace import build_ricci_flow_trace
from physics_contracts.manifold.sync_frame import build_sync_frame
from scripts.ci.check_physics_law_witness_index import verify as verify_law_index

ROOT = Path(__file__).resolve().parents[2]
READINESS_PATH = ROOT / "artifacts" / "physics_v2" / "inference_readiness.json"
INDEX_PATH = ROOT / "artifacts" / "physics_v2" / "law_witness_index.json"

_READY_STATES = frozenset({"READY_SYNTHETIC_ONLY", "READY_REALDATA_REPLAY"})
_DECLARED_TIERS = frozenset({"SYNTHETIC_ONLY", "REALDATA_REPLAY"})
_REAL_FINGERPRINT_PLACEHOLDER = "a" * 64


@dataclass(frozen=True, slots=True)
class _Check:
    name: str
    ok: bool
    blocker: str


def _check_missing_witness() -> _Check:
    """Blocking law without a witness ⇒ not ready (reads the Task 1 index)."""

    if not INDEX_PATH.exists():
        return _Check("witness_index", False, f"{INDEX_PATH.relative_to(ROOT)} (missing)")
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    errors = verify_law_index(index)
    witness_errors = [e for e in errors if "blocking law without witness" in e]
    if witness_errors:
        return _Check("witness_index", False, witness_errors[0])
    if errors:
        # Index is stale or otherwise invalid — surface it, still a blocker.
        return _Check("witness_index", False, errors[0])
    return _Check("witness_index", True, "")


def _check_causality_guard() -> _Check:
    """The look-ahead / causal-cutoff guard must reject a future-event frame."""

    frame = OrderFlowFrame(
        timestamp_start=0.0,
        timestamp_end=1.0,
        nodes=("A", "B"),
        edges=((0, 1),),
        edge_weights=(0.5,),
        latency_floor_seconds=0.1,
        causal_cutoff_seconds=0.25,
        dataset_fingerprint=_REAL_FINGERPRINT_PLACEHOLDER,
        source_event_times=(2.0,),  # after timestamp_end ⇒ look-ahead
    )
    try:
        build_causal_graph_snapshot(frame)
    except ValueError:
        return _Check("causality", True, "")
    return _Check(
        "causality",
        False,
        "physics_contracts/manifold/builders.py::build_causal_graph_snapshot "
        "(look-ahead guard did not fire)",
    )


def _build_synthetic_capsule_digest() -> str:
    """Exercise the full synthetic evidence chain end-to-end; return its digest."""

    snapshot = synthetic_snapshot(seed=7)
    graph = nx.cycle_graph(6)
    trace = build_ricci_flow_trace(
        snapshot_id=snapshot.snapshot_id,
        graph_before=graph,
        graph_after=graph,
        edge_curvatures_before=[0.5, 0.5, 0.5],
        edge_curvatures_after=[0.4, 0.4, 0.4],
        validity_domain="synthetic-fixture; no market claim",
    )
    rng = np.random.default_rng(0)
    theta = rng.uniform(-np.pi, np.pi, size=(200, 256))
    frame = build_sync_frame(
        snapshot_id=snapshot.snapshot_id,
        theta=theta,
        coupling_K=0.1,
        critical_K=2.0,
        n_oscillators=256,
        locked_fraction=0.0,
        lyapunov_regime="neutral",
        validity_domain="synthetic-fixture; no market claim",
    )
    report = assemble_comparison_report(
        candidate_statistic=0.5,
        control_statistics={
            "time_shuffle": 0.2,
            "iaaft_surrogate": 0.25,
            "degree_preserving_rewire": 0.22,
            "causal_cutoff_violation": 0.3,
        },
        n_null_draws=200,
        validity_domain="synthetic-fixture; no market claim",
        dataset_fingerprint=snapshot.dataset_fingerprint,
    )
    capsule = bind_physics_evidence(
        run_id="readiness-synthetic-probe",
        dataset_fingerprint=snapshot.dataset_fingerprint,
        code_sha="readiness-probe",
        config_hash="readiness-probe",
        snapshot_id=snapshot.snapshot_id,
        ricci_trace_digest=trace.trace_digest,
        sync_frame_digest=frame.frame_digest,
        comparison_report_digest=report.report_digest,
        laws_exercised=("manifold.metric_snapshot_schema",),
        falsifiers_passed=("synthetic_no_lookahead",),
        falsifiers_failed=(),
        claim_maturity="MEASURED_SYNTHETIC",
        verdict="not_distinguished",
        replay_command="python scripts/ci/check_physics_inference_readiness.py --write",
        nulls_survived=report.survived_all_controls,
    )
    return capsule.capsule_digest


def _check_capsule_integrity() -> _Check:
    """The synthetic evidence chain must assemble into a deterministic capsule."""

    try:
        first = _build_synthetic_capsule_digest()
        second = _build_synthetic_capsule_digest()
    except (ValueError, TypeError, KeyError) as exc:
        return _Check(
            "capsule_integrity",
            False,
            f"physics_contracts/manifold/evidence_binding.py (chain raised: {exc})",
        )
    if first != second:
        return _Check(
            "capsule_integrity",
            False,
            "physics_contracts/manifold/evidence_binding.py (capsule digest non-deterministic)",
        )
    return _Check("capsule_integrity", True, "")


def _check_synthetic_promotion_guard() -> _Check:
    """A synthetic dataset must not be bindable at a real-data maturity tier."""

    try:
        bind_physics_evidence(
            run_id="promotion-probe",
            dataset_fingerprint="synthetic:00000001",
            code_sha="probe",
            config_hash="probe",
            snapshot_id="probe",
            ricci_trace_digest="probe",
            sync_frame_digest="probe",
            comparison_report_digest="probe",
            laws_exercised=("manifold.metric_snapshot_schema",),
            falsifiers_passed=("probe",),
            falsifiers_failed=(),
            claim_maturity="REAL_DATA_SINGLE_SESSION",
            verdict="pass",
            replay_command="probe",
            nulls_survived=True,
        )
    except ValueError:
        return _Check("synthetic_promotion", True, "")
    return _Check(
        "synthetic_promotion",
        False,
        "physics_contracts/manifold/evidence_binding.py "
        "(synthetic data accepted at a real-data tier)",
    )


def compute_readiness(
    *, declared_tier: str, real_dataset_fingerprint: str | None
) -> dict[str, Any]:
    """Compute the inference-readiness verdict from the live contracts + index."""

    if declared_tier not in _DECLARED_TIERS:
        raise ValueError(f"declared_tier {declared_tier!r} not in {sorted(_DECLARED_TIERS)}")

    checks = [
        _check_missing_witness(),
        _check_causality_guard(),
        _check_capsule_integrity(),
        _check_synthetic_promotion_guard(),
    ]
    # Blocker precedence: witness → causality → capsule → promotion-guard liveness.
    blocker_for = {
        "witness_index": "BLOCKED_MISSING_WITNESS",
        "causality": "BLOCKED_INVALID_CAUSALITY",
        "capsule_integrity": "BLOCKED_BROKEN_CAPSULE",
        "synthetic_promotion": "BLOCKED_BROKEN_CAPSULE",
    }
    verdict = ""
    blocker_path = ""
    for check in checks:
        if not check.ok:
            verdict = blocker_for[check.name]
            blocker_path = check.blocker
            break

    if not verdict:
        # All guards live. Now decide the honest tier.
        if declared_tier == "REALDATA_REPLAY":
            status = resolve_data_status(real_dataset_fingerprint)
            if status is LicensedDataStatus.AVAILABLE:
                verdict = "READY_REALDATA_REPLAY"
            else:
                verdict = "BLOCKED_SYNTHETIC_PROMOTION"
                blocker_path = (
                    "declared_tier=REALDATA_REPLAY but real_dataset_fingerprint is "
                    f"{real_dataset_fingerprint!r} (resolves to DATA_UNAVAILABLE); "
                    "no licensed L2 dataset present — refusing false promotion"
                )
        else:
            verdict = "READY_SYNTHETIC_ONLY"

    return {
        "schema_version": 1,
        "generated_by": "scripts/ci/check_physics_inference_readiness.py",
        "declared_tier": declared_tier,
        "real_dataset_fingerprint": real_dataset_fingerprint or "",
        "verdict": verdict,
        "blocker_path": blocker_path,
        "checks": {c.name: {"ok": c.ok, "blocker": c.blocker} for c in checks},
    }


def _serialise(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _recompute_from(committed: dict[str, Any]) -> dict[str, Any]:
    declared = committed.get("declared_tier", "SYNTHETIC_ONLY")
    fingerprint = committed.get("real_dataset_fingerprint") or None
    return compute_readiness(declared_tier=declared, real_dataset_fingerprint=fingerprint)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="regenerate the readiness artifact")
    parser.add_argument(
        "--declared-tier",
        default=None,
        help="override the declared tier when writing (SYNTHETIC_ONLY|REALDATA_REPLAY)",
    )
    args = parser.parse_args(argv)

    if args.write:
        if READINESS_PATH.exists():
            existing = json.loads(READINESS_PATH.read_text(encoding="utf-8"))
        else:
            existing = {}
        declared = args.declared_tier or existing.get("declared_tier", "SYNTHETIC_ONLY")
        fingerprint = existing.get("real_dataset_fingerprint") or None
        payload = compute_readiness(declared_tier=declared, real_dataset_fingerprint=fingerprint)
        READINESS_PATH.parent.mkdir(parents=True, exist_ok=True)
        READINESS_PATH.write_text(_serialise(payload), encoding="utf-8")
        print(f"wrote {READINESS_PATH.relative_to(ROOT)} verdict={payload['verdict']}")
        return 0

    if not READINESS_PATH.exists():
        print(f"ERROR: missing readiness artifact: {READINESS_PATH}")
        return 1
    committed = json.loads(READINESS_PATH.read_text(encoding="utf-8"))
    recomputed = _recompute_from(committed)

    if _serialise(committed) != _serialise(recomputed):
        print("PHYSICS INFERENCE READINESS: FAIL — artifact is stale")
        print(f"  committed verdict={committed.get('verdict')!r} != live {recomputed['verdict']!r}")
        print("  regenerate with `python scripts/ci/check_physics_inference_readiness.py --write`")
        return 1

    verdict = recomputed["verdict"]
    if verdict in _READY_STATES:
        print(f"PHYSICS INFERENCE READINESS: {verdict} (declared {recomputed['declared_tier']})")
        return 0
    print(f"PHYSICS INFERENCE READINESS: {verdict}")
    print(f"  blocker: {recomputed['blocker_path']}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
