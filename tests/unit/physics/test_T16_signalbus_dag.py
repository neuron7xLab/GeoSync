# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""T16 — NeuroSignalBus acyclic-fanout witness for INV-SB1.

INV-SB1 states that, within a single scheduler tick, the signal
producer→consumer graph on ``core.neuro.signal_bus.NeuroSignalBus`` is
a DAG: every subscriber fires exactly once per publish call, and the
cross-channel callback graph is free of cycles.

The witness builds a three-layer DAG of subscribers:

    dopamine ──[callback_A]──▶ gaba ──[callback_B]──▶ kuramoto ──[callback_C]

Publishing on the ``dopamine`` channel must fan out down the DAG so
that each of the three callbacks (A on dopamine, B on gaba, C on
kuramoto) fires exactly once. No callback may fire zero or twice. The
final snapshot must reflect every downstream publish the DAG produced.

A hidden cycle in the fanout (e.g. a callback that re-publishes to its
own channel) would either silently skip notifications or trigger
unbounded recursion — both are INV-SB1 violations. The witness's
"exactly once" check catches the silent case; Python's recursion limit
catches the unbounded case automatically via RecursionError which
would surface as a test failure.
"""

from __future__ import annotations

from core.neuro.signal_bus import NeuroSignalBus, NeuroSignals


def test_signalbus_dag_fanout_fires_each_subscriber_exactly_once() -> None:
    """INV-SB1: three-layer DAG fanout fires each subscriber once per publish.

    Builds a linear DAG dopamine → gaba → kuramoto via subscribers that
    publish to the next channel. Asserts every callback fires exactly
    once when the top of the DAG is triggered, and that the final bus
    snapshot reflects the full downstream propagation.
    """
    bus = NeuroSignalBus()

    # Published values are DAG inputs, not fitted thresholds — latch
    # tolerance is zero because the bus is a pure latch.
    payload_root_rpe: float = 0.15  # epsilon: exact latch value for dopamine
    payload_gaba: float = 0.42  # epsilon: exact latch value for gaba
    payload_kuramoto: float = 0.55  # epsilon: exact latch value for kuramoto

    call_counts: dict[str, int] = {
        "dopamine_callback": 0,
        "gaba_callback": 0,
        "kuramoto_callback": 0,
    }

    # Layer 1: dopamine subscriber publishes to gaba. This adds a
    # dopamine → gaba edge to the fanout DAG.
    def on_dopamine(signals: NeuroSignals) -> None:
        call_counts["dopamine_callback"] += 1
        bus.publish_gaba(inhibition=payload_gaba)

    # Layer 2: gaba subscriber publishes to kuramoto. Adds gaba → kuramoto.
    def on_gaba(signals: NeuroSignals) -> None:
        call_counts["gaba_callback"] += 1
        bus.publish_kuramoto(R=payload_kuramoto)

    # Layer 3: kuramoto subscriber is a leaf — no further publishes,
    # so the DAG terminates here.
    def on_kuramoto(signals: NeuroSignals) -> None:
        call_counts["kuramoto_callback"] += 1

    bus.subscribe("dopamine", on_dopamine)
    bus.subscribe("gaba", on_gaba)
    bus.subscribe("kuramoto", on_kuramoto)

    # Trigger the DAG from the root.
    bus.publish_dopamine(rpe=payload_root_rpe)

    # Each of the three callbacks must have fired exactly once — the
    # direct INV-SB1 "each node per tick" guarantee.
    expected_calls = 1
    for callback_name in (
        "dopamine_callback",
        "gaba_callback",
        "kuramoto_callback",
    ):
        observed = call_counts[callback_name]
        assert observed == expected_calls, (
            f"INV-SB1 VIOLATED: {callback_name} fired {observed} times, "
            f"expected {expected_calls}. "
            f"Expected every subscriber to fire exactly once per DAG "
            f"traversal from a single root publish. "
            f"Observed at N=3 DAG layers, seed=none (deterministic bus). "
            f"Physical reasoning: acyclic fanout means each node is visited "
            f"exactly once — zero firings mean a missing edge, two or more "
            f"mean a cycle or duplicated subscription."
        )

    # The downstream publishes must have landed on the bus snapshot —
    # if the DAG were broken, gaba and kuramoto would still be at their
    # defaults (0.0 each).
    snapshot = bus.snapshot()
    assert snapshot.dopamine_rpe == payload_root_rpe, (
        f"INV-SB1 VIOLATED: snapshot dopamine_rpe={snapshot.dopamine_rpe} "
        f"≠ published {payload_root_rpe}. "
        f"Expected publish_dopamine to latch the exact payload. "
        f"Observed at N=3 DAG layers, seed=none. "
        f"Physical reasoning: the root publish must reach the snapshot "
        f"latch even without any subscribers."
    )
    assert snapshot.gaba_inhibition == payload_gaba, (
        f"INV-SB1 VIOLATED: snapshot gaba_inhibition="
        f"{snapshot.gaba_inhibition} ≠ expected {payload_gaba}. "
        f"Expected the layer-1 callback to publish the gaba payload. "
        f"Observed at N=3 DAG layers, seed=none. "
        f"Physical reasoning: if the dopamine → gaba edge fires, the "
        f"bus snapshot must reflect the downstream latched value."
    )
    assert snapshot.kuramoto_R == payload_kuramoto, (
        f"INV-SB1 VIOLATED: snapshot kuramoto_R={snapshot.kuramoto_R} "
        f"≠ expected {payload_kuramoto}. "
        f"Expected the layer-2 callback to publish at the DAG leaf. "
        f"Observed at N=3 DAG layers, seed=none. "
        f"Physical reasoning: a full three-layer traversal must land "
        f"the leaf publish on the snapshot."
    )

    # Sanity check on DAG shape — the total number of callback invocations
    # equals the number of DAG nodes (3), not 3² or 3^k.
    total_fanout_invocations = sum(call_counts.values())
    expected_total = 3
    assert total_fanout_invocations == expected_total, (
        f"INV-SB1 VIOLATED: total fanout invocations={total_fanout_invocations} "
        f"≠ N={expected_total} DAG nodes. "
        f"Expected a linear DAG to produce one invocation per node. "
        f"Observed at N=3 DAG layers, seed=none. "
        f"Physical reasoning: a properly acyclic fanout fires each "
        f"subscriber exactly once, so total invocations = number of nodes."
    )
