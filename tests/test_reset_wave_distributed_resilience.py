# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Adversarial async-failure tests for the distributed reset-wave adapter.

Each test injects one async failure mode named in
``geosync/neuroeconomics/reset_wave_distributed.py`` and asserts that
the adapter either fires the safety lock with the correct
``fail_reason`` OR (for benign cases) lets the base solver run.

There are five failure modes covered, plus an end-to-end happy-path
test that confirms the guards do not over-trigger when inputs are
clean.
"""

from __future__ import annotations

from geosync.neuroeconomics.reset_wave_distributed import (
    ConcurrencyGuard,
    DiscontinuityMonitor,
    DistributedResetWaveConfig,
    JitterEnvelope,
    PartialFailureDetector,
    StalenessGate,
    run_reset_wave_distributed,
)
from geosync.neuroeconomics.reset_wave_engine import ResetWaveConfig


def _clean_inputs() -> dict[str, object]:
    """Inputs that pass every guard so the base solver actually runs."""
    return {
        "node_phases": [0.05, -0.05, 0.04],
        "baseline_phases": [0.0, 0.0, 0.0],
        "timestamps_ns": [1_000_000_000, 1_000_000_500, 1_000_001_000],
        "node_seqs": [10, 11, 12],
        "last_seqs": {0: 9, 1: 10, 2: 11},
        "active_indices": [0, 1, 2],
        "prev_phases": [0.06, -0.06, 0.05],
        "now_ns": 1_000_002_000,
        "cfg": DistributedResetWaveConfig(
            base=ResetWaveConfig(coupling_gain=0.8, dt=0.05),
            jitter=JitterEnvelope(max_jitter_ns=10_000_000),
            staleness=StalenessGate(max_age_ns=50_000_000),
            discontinuity=DiscontinuityMonitor(tolerance=1e-3),
        ),
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_distributed_clean_inputs_run_base_solver() -> None:
    out = run_reset_wave_distributed(**_clean_inputs())  # type: ignore[arg-type]
    assert out.fail_reason is None
    assert not out.safety_lock_distributed
    assert out.base.final_potential <= out.base.initial_potential


# ---------------------------------------------------------------------------
# Failure mode 1 — clock-jitter envelope
# ---------------------------------------------------------------------------


def test_distributed_jitter_envelope_violation_locks() -> None:
    inputs = _clean_inputs()
    # Last node arrives 100 ms after the first — exceeds 10 ms envelope.
    inputs["timestamps_ns"] = [1_000_000_000, 1_000_001_000, 1_100_000_000]
    out = run_reset_wave_distributed(**inputs)  # type: ignore[arg-type]
    assert out.fail_reason == "jitter_envelope"
    assert out.jitter_violation
    assert out.safety_lock_distributed
    assert out.base.final_potential == out.base.initial_potential


# ---------------------------------------------------------------------------
# Failure mode 2 — staleness gate
# ---------------------------------------------------------------------------


def test_distributed_staleness_gate_locks_on_old_update() -> None:
    inputs = _clean_inputs()
    # ``now_ns`` is 100 ms after the youngest update — exceeds 50 ms staleness.
    inputs["now_ns"] = 1_100_000_000
    out = run_reset_wave_distributed(**inputs)  # type: ignore[arg-type]
    assert out.fail_reason == "staleness"
    assert out.stale_indices == (0, 1, 2)
    assert out.safety_lock_distributed


# ---------------------------------------------------------------------------
# Failure mode 3 — concurrency / out-of-order sequences
# ---------------------------------------------------------------------------


def test_distributed_concurrency_guard_rejects_replayed_seq() -> None:
    inputs = _clean_inputs()
    # Node 1's new seq (5) is below the last accepted (10) — replay attack.
    inputs["node_seqs"] = [10, 5, 12]
    inputs["last_seqs"] = {0: 9, 1: 10, 2: 11}
    out = run_reset_wave_distributed(**inputs)  # type: ignore[arg-type]
    assert out.fail_reason == "concurrency"
    assert 1 in out.concurrency_violations
    assert out.safety_lock_distributed


def test_concurrency_guard_pure_function_accepts_clean() -> None:
    accepts, rejected = ConcurrencyGuard.accepts([10, 11, 12], {0: 9, 1: 10, 2: 11})
    assert accepts
    assert rejected == ()


def test_concurrency_guard_pure_function_rejects_equal_seq() -> None:
    accepts, rejected = ConcurrencyGuard.accepts([10, 10, 12], {0: 9, 1: 10, 2: 11})
    assert not accepts
    assert 1 in rejected


# ---------------------------------------------------------------------------
# Failure mode 4 — partial node failure
# ---------------------------------------------------------------------------


def test_distributed_partial_failure_detected() -> None:
    inputs = _clean_inputs()
    # Only node 0 and node 2 delivered — node 1 missing.
    inputs["active_indices"] = [0, 2]
    out = run_reset_wave_distributed(**inputs)  # type: ignore[arg-type]
    assert out.fail_reason == "partial_failure"
    assert 1 in out.missing_indices
    assert out.safety_lock_distributed


def test_partial_failure_detector_pure_function() -> None:
    assert PartialFailureDetector.missing([0, 2], 3) == (1,)
    assert PartialFailureDetector.missing([0, 1, 2], 3) == ()
    assert PartialFailureDetector.missing([], 3) == (0, 1, 2)


# ---------------------------------------------------------------------------
# Failure mode 5 — discontinuity (re-entry / adversarial sequence)
# ---------------------------------------------------------------------------


def test_distributed_discontinuity_monitor_flags_re_entry() -> None:
    inputs = _clean_inputs()
    # Adversarial: node phases jumped *away* from baseline since prev_phases.
    inputs["prev_phases"] = [0.01, -0.01, 0.01]  # smaller V than current
    out = run_reset_wave_distributed(**inputs)  # type: ignore[arg-type]
    assert out.fail_reason == "discontinuity"
    assert out.discontinuity_flagged
    assert out.safety_lock_distributed
    # Critical: under lock, V must be exactly preserved.
    assert out.base.final_potential == out.base.initial_potential


def test_discontinuity_monitor_pure_function_clean() -> None:
    monitor = DiscontinuityMonitor()
    # New phases closer to baseline than previous → not a discontinuity.
    flagged = monitor.discontinuity(
        prev_phases=[0.2, -0.2],
        new_phases=[0.05, -0.05],
        baseline_phases=[0.0, 0.0],
    )
    assert not flagged


def test_discontinuity_monitor_pure_function_adversarial() -> None:
    monitor = DiscontinuityMonitor()
    # New phases farther from baseline than previous → discontinuity.
    flagged = monitor.discontinuity(
        prev_phases=[0.05, -0.05],
        new_phases=[0.5, -0.5],
        baseline_phases=[0.0, 0.0],
    )
    assert flagged


# ---------------------------------------------------------------------------
# Determinism — same async input ⇒ same output
# ---------------------------------------------------------------------------


def test_distributed_adapter_is_deterministic() -> None:
    a = run_reset_wave_distributed(**_clean_inputs())  # type: ignore[arg-type]
    b = run_reset_wave_distributed(**_clean_inputs())  # type: ignore[arg-type]
    assert a == b


# ---------------------------------------------------------------------------
# Contract validation — input shape
# ---------------------------------------------------------------------------


def test_distributed_adapter_rejects_mismatched_lengths() -> None:
    inputs = _clean_inputs()
    inputs["timestamps_ns"] = [1_000_000_000]  # too short
    try:
        run_reset_wave_distributed(**inputs)  # type: ignore[arg-type]
    except ValueError as exc:
        assert "timestamps_ns" in str(exc)
    else:
        raise AssertionError("expected ValueError on mismatched timestamps_ns")


def test_distributed_adapter_rejects_mismatched_seqs() -> None:
    inputs = _clean_inputs()
    inputs["node_seqs"] = [10, 11]  # too short
    try:
        run_reset_wave_distributed(**inputs)  # type: ignore[arg-type]
    except ValueError as exc:
        assert "node_seqs" in str(exc)
    else:
        raise AssertionError("expected ValueError on mismatched node_seqs")


def test_distributed_adapter_rejects_mismatched_prev_phases() -> None:
    inputs = _clean_inputs()
    inputs["prev_phases"] = [0.01]  # too short
    try:
        run_reset_wave_distributed(**inputs)  # type: ignore[arg-type]
    except ValueError as exc:
        assert "prev_phases" in str(exc)
    else:
        raise AssertionError("expected ValueError on mismatched prev_phases")
