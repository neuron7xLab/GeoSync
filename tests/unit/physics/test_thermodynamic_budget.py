# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""PNCC-A — Thermodynamic Budget tests.

Verifies INV-LANDAUER-PROXY (P0, universal):
    For any irreversible action, its total proxy cost is ≥ the cost
    of its hypothetical reversible alternative with identical
    token / latency / entropy components — strict whenever
    irreversibility_score > 0.

Also covers INV-HPC1 (determinism) and INV-HPC2 (finite-in / finite-out
+ fail-closed on bad inputs).
"""

from __future__ import annotations

import math
import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from core.physics.thermodynamic_budget import (
    LANDAUER_LN2,
    BudgetEntry,
    BudgetLedger,
    EntropyCost,
    IrreversibleActionCost,
    LatencyCost,
    ThermodynamicBudgetConfig,
    TokenCost,
    aggregate_entry,
    compute_entropy_cost,
    compute_irreversibility_cost,
    compute_latency_cost,
    compute_token_cost,
    filter_horizon,
    reversible_alternative_cost,
    total_cost,
)

# --------------------------------------------------------------------------
# Hypothesis strategies — bounded so totals stay finite at float64 precision.
# --------------------------------------------------------------------------

_INT_TOKENS = st.integers(min_value=0, max_value=1_000_000)
_INT_NS = st.integers(min_value=0, max_value=10**15)  # up to ~11.5 days
_FLOAT_BITS = st.floats(
    min_value=0.0,
    max_value=1.0e9,
    allow_nan=False,
    allow_infinity=False,
)
_FLOAT_SCORE = st.floats(
    min_value=0.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
)


def _default_cfg(penalty: float = 1.0) -> ThermodynamicBudgetConfig:
    return ThermodynamicBudgetConfig(irreversibility_penalty=penalty)


def _make_entry(
    action_id: str,
    *,
    n_in: int,
    n_out: int,
    wall_ns: int,
    p99_ns: int,
    bits_consumed: float,
    bits_erased: float,
    is_irreversible: bool,
    score: float,
    timestamp_ns: int,
    cfg: ThermodynamicBudgetConfig,
) -> BudgetEntry:
    """Helper: assemble a full ``BudgetEntry`` from primitive inputs."""
    token = compute_token_cost(n_in, n_out)
    latency = compute_latency_cost(wall_ns, p99_ns)
    entropy = compute_entropy_cost(bits_consumed, bits_erased)
    irr = compute_irreversibility_cost(is_irreversible, score, cfg)
    return aggregate_entry(action_id, timestamp_ns, token, latency, entropy, irr)


# --------------------------------------------------------------------------
# 1. Token cost: finite + non-negative under universal sweep.
# --------------------------------------------------------------------------


@given(n_in=_INT_TOKENS, n_out=_INT_TOKENS)
@settings(max_examples=200, deadline=None)
def test_token_cost_finite_non_negative(n_in: int, n_out: int) -> None:
    """INV-HPC2: finite inputs ⇒ finite, non-negative proxy cost."""
    tc = compute_token_cost(n_in, n_out)
    msg_finite = f"INV-HPC2 VIOLATED: token proxy_cost not finite at n_in={n_in}, n_out={n_out}"
    assert math.isfinite(tc.proxy_cost), msg_finite
    msg_pos = f"Token proxy must be ≥ 0; got {tc.proxy_cost} at n_in={n_in}, n_out={n_out}"
    assert tc.proxy_cost >= 0.0, msg_pos
    # Decoder dominance: same total tokens, more output ⇒ higher cost.
    assert tc.proxy_cost == n_in + 4 * n_out


# --------------------------------------------------------------------------
# 2. Latency cost: monotone in wall-time (qualitative invariant).
# --------------------------------------------------------------------------


def test_latency_cost_monotone_in_walltime() -> None:
    """Wall_ns increases ⇒ latency proxy non-decreasing. Strictly increases on
    distinct positive values."""
    samples = [0, 1_000, 1_000_000, 10_000_000, 1_000_000_000, 10_000_000_000]
    costs = [compute_latency_cost(w, w).proxy_cost for w in samples]
    for i in range(1, len(samples)):
        assert costs[i] >= costs[i - 1], (
            f"INV-LATENCY-MONO VIOLATED: cost decreased from "
            f"{costs[i - 1]} (wall={samples[i - 1]}) to {costs[i]} "
            f"(wall={samples[i]})"
        )
        # Strict for distinct positive values (log1p is strictly increasing on x>0).
        if samples[i] > samples[i - 1] and samples[i] > 0:
            strict_msg = f"Expected strict increase from {costs[i - 1]} to {costs[i]}"
            assert costs[i] > costs[i - 1], strict_msg
    # log1p(0) == 0 anchor.
    assert costs[0] == 0.0


# --------------------------------------------------------------------------
# 3. Entropy cost: finite under universal sweep.
# --------------------------------------------------------------------------


@given(consumed=_FLOAT_BITS, erased=_FLOAT_BITS)
@settings(max_examples=200, deadline=None)
def test_entropy_cost_finite(consumed: float, erased: float) -> None:
    """INV-HPC2: finite bits ⇒ finite entropy proxy. Proxy = erased · ln(2)."""
    ec = compute_entropy_cost(consumed, erased)
    msg = f"INV-HPC2 VIOLATED: entropy proxy not finite at consumed={consumed}, erased={erased}"
    assert math.isfinite(ec.proxy_cost), msg
    assert ec.proxy_cost >= 0.0
    # Algebraic identity at float precision.
    assert abs(ec.proxy_cost - erased * LANDAUER_LN2) < 1e-12


# --------------------------------------------------------------------------
# 4. INV-LANDAUER-PROXY core test: irreversible ≥ reversible (1000 random pairs).
# --------------------------------------------------------------------------


def test_landauer_proxy_irreversible_dominates_reversible() -> None:
    """INV-LANDAUER-PROXY (P0, universal):
    C_irr ≥ C_rev for identical I/O components, strict when score > 0.
    """
    # Seeded RNG → reproducible (INV-HPC1 spirit).
    rng = random.Random(20260425)
    cfg = _default_cfg(penalty=1.0)
    n_pairs = 1000

    for i in range(n_pairs):
        n_in = rng.randint(0, 10_000)
        n_out = rng.randint(0, 10_000)
        wall_ns = rng.randint(0, 10**12)
        p99_ns = wall_ns + rng.randint(0, 10**9)
        bits_consumed = rng.uniform(0.0, 1.0e6)
        bits_erased = rng.uniform(0.0, 1.0e6)
        score = rng.uniform(0.0, 1.0)
        ts = rng.randint(0, 10**12)

        entry_irr = _make_entry(
            action_id=f"act_irr_{i}",
            n_in=n_in,
            n_out=n_out,
            wall_ns=wall_ns,
            p99_ns=p99_ns,
            bits_consumed=bits_consumed,
            bits_erased=bits_erased,
            is_irreversible=True,
            score=score,
            timestamp_ns=ts,
            cfg=cfg,
        )
        c_rev = reversible_alternative_cost(entry_irr, cfg)
        c_irr = entry_irr.total_proxy_cost

        # Non-strict bound: C_irr ≥ C_rev always.
        assert c_irr >= c_rev, (
            f"INV-LANDAUER-PROXY VIOLATED: C_irr={c_irr} < C_rev={c_rev} "
            f"at pair {i}; n_in={n_in}, n_out={n_out}, wall_ns={wall_ns}, "
            f"bits_erased={bits_erased}, score={score}"
        )
        # Strict bound: when score > 0, C_irr > C_rev (penalty=1.0 > 0).
        if score > 0.0:
            assert c_irr > c_rev, (
                f"INV-LANDAUER-PROXY (strict) VIOLATED: C_irr={c_irr} not > "
                f"C_rev={c_rev} at pair {i} with score={score} > 0"
            )
            # Exact algebraic identity: difference equals penalty · score.
            assert abs((c_irr - c_rev) - cfg.irreversibility_penalty * score) < 1e-9, (
                f"Difference {c_irr - c_rev} ≠ penalty·score "
                f"{cfg.irreversibility_penalty * score} at pair {i}"
            )


# --------------------------------------------------------------------------
# 5. score=0 OR is_irreversible=False ⇒ exactly equal to reversible cost.
# --------------------------------------------------------------------------


def test_irreversibility_zero_recovers_reversible_cost() -> None:
    """Algebraic, exact at float precision.

    Two boundary recoveries:
      (a) is_irreversible=False, any score → C_irr == C_rev.
      (b) is_irreversible=True, score=0.0 → C_irr == C_rev.
    """
    cfg = _default_cfg(penalty=2.5)

    # (a) Reversible action: irreversibility component is exactly 0 regardless of score.
    entry_a = _make_entry(
        action_id="rev_with_score",
        n_in=42,
        n_out=7,
        wall_ns=12_345,
        p99_ns=20_000,
        bits_consumed=3.5,
        bits_erased=8.0,
        is_irreversible=False,
        score=0.9,  # ignored because is_irreversible=False
        timestamp_ns=1,
        cfg=cfg,
    )
    assert entry_a.irreversible.proxy_cost == 0.0
    assert entry_a.total_proxy_cost == reversible_alternative_cost(entry_a, cfg)

    # (b) Irreversible but score==0 ⇒ penalty·0 == 0 exactly.
    entry_b = _make_entry(
        action_id="irr_zero_score",
        n_in=42,
        n_out=7,
        wall_ns=12_345,
        p99_ns=20_000,
        bits_consumed=3.5,
        bits_erased=8.0,
        is_irreversible=True,
        score=0.0,
        timestamp_ns=2,
        cfg=cfg,
    )
    assert entry_b.irreversible.proxy_cost == 0.0
    assert entry_b.total_proxy_cost == reversible_alternative_cost(entry_b, cfg)


# --------------------------------------------------------------------------
# 6. total_cost is additive over entries (algebraic).
# --------------------------------------------------------------------------


def test_total_cost_additive_over_entries() -> None:
    """total_cost(ledger) == Σ entry.total_proxy_cost (no rounding tricks)."""
    cfg = _default_cfg(penalty=1.0)
    entries = tuple(
        _make_entry(
            action_id=f"a{i}",
            n_in=10 * i,
            n_out=2 * i,
            wall_ns=1000 * (i + 1),
            p99_ns=1500 * (i + 1),
            bits_consumed=float(i),
            bits_erased=float(i) / 2.0,
            is_irreversible=(i % 2 == 0),
            score=0.25 * (i % 4) / 3.0,
            timestamp_ns=i * 1000,
            cfg=cfg,
        )
        for i in range(1, 11)
    )
    ledger = BudgetLedger(entries=entries, horizon_ns=cfg.max_horizon_ns)
    expected = sum(e.total_proxy_cost for e in entries)
    assert total_cost(ledger) == expected
    # Empty ledger ⇒ 0.0 exactly.
    assert total_cost(BudgetLedger(entries=(), horizon_ns=0)) == 0.0


# --------------------------------------------------------------------------
# 7. Negative cost / invalid inputs raise. Fail-closed (no silent repair).
# --------------------------------------------------------------------------


def test_negative_cost_raises() -> None:
    """All component constructors fail-closed on negative or non-finite inputs."""
    cfg = _default_cfg()

    with pytest.raises(ValueError):
        compute_token_cost(-1, 0)
    with pytest.raises(ValueError):
        compute_token_cost(0, -1)
    with pytest.raises(ValueError):
        compute_latency_cost(-1, 0)
    with pytest.raises(ValueError):
        compute_latency_cost(0, -1)
    with pytest.raises(ValueError):
        compute_entropy_cost(-0.5, 1.0)
    with pytest.raises(ValueError):
        compute_entropy_cost(1.0, -0.5)
    with pytest.raises(ValueError):
        compute_entropy_cost(float("nan"), 1.0)
    with pytest.raises(ValueError):
        compute_entropy_cost(1.0, float("inf"))
    with pytest.raises(ValueError):
        compute_irreversibility_cost(True, -0.1, cfg)
    with pytest.raises(ValueError):
        compute_irreversibility_cost(True, 1.1, cfg)
    with pytest.raises(ValueError):
        compute_irreversibility_cost(True, float("nan"), cfg)
    # Negative penalty: fail-closed.
    bad_cfg = ThermodynamicBudgetConfig(irreversibility_penalty=-1.0)
    with pytest.raises(ValueError):
        compute_irreversibility_cost(True, 0.5, bad_cfg)
    # aggregate_entry: empty action_id, negative timestamp.
    token = compute_token_cost(1, 1)
    latency = compute_latency_cost(1, 1)
    entropy = compute_entropy_cost(0.0, 0.0)
    irr = compute_irreversibility_cost(False, 0.0, cfg)
    with pytest.raises(ValueError):
        aggregate_entry("", 0, token, latency, entropy, irr)
    with pytest.raises(ValueError):
        aggregate_entry("ok", -1, token, latency, entropy, irr)


# --------------------------------------------------------------------------
# 8. Horizon filter excludes old entries (conditional).
# --------------------------------------------------------------------------


def test_horizon_filter_excludes_old_entries() -> None:
    """filter_horizon keeps timestamps in [now_ns - horizon_ns, ∞), drops older."""
    cfg = _default_cfg()
    horizon_ns = 1_000_000_000  # 1 second
    now_ns = 5_000_000_000

    inside_ts = (
        now_ns,
        now_ns - 100_000_000,
        now_ns - horizon_ns,  # exact boundary, kept
    )
    outside_ts = (
        now_ns - horizon_ns - 1,
        now_ns - 2 * horizon_ns,
        0,
    )

    entries = []
    for i, ts in enumerate(inside_ts + outside_ts):
        entries.append(
            _make_entry(
                action_id=f"e{i}",
                n_in=1,
                n_out=1,
                wall_ns=1,
                p99_ns=1,
                bits_consumed=0.0,
                bits_erased=0.0,
                is_irreversible=False,
                score=0.0,
                timestamp_ns=ts,
                cfg=cfg,
            )
        )

    ledger = BudgetLedger(entries=tuple(entries), horizon_ns=horizon_ns)
    filtered = filter_horizon(ledger, now_ns)

    kept_ts = {e.timestamp_ns for e in filtered.entries}
    expected_ts = set(inside_ts)
    horizon_msg = f"Horizon filter incorrect: kept={kept_ts}, expected={expected_ts}"
    assert kept_ts == expected_ts, horizon_msg
    # Pure: original ledger unchanged.
    assert len(ledger.entries) == len(inside_ts) + len(outside_ts)
    # horizon_ns is propagated unchanged.
    assert filtered.horizon_ns == horizon_ns


# --------------------------------------------------------------------------
# 9. Determinism under fixed seed (INV-HPC1).
# --------------------------------------------------------------------------


def test_deterministic_under_fixed_seed() -> None:
    """INV-HPC1: Bit-identical output under identical inputs.

    The module is purely functional; we run the same sequence twice and
    require entry-by-entry equality including the float ``total_proxy_cost``.
    """
    cfg = _default_cfg(penalty=1.0)

    def build_ledger() -> BudgetLedger:
        rng = random.Random(20260425)
        entries: list[BudgetEntry] = []
        for i in range(50):
            entry = _make_entry(
                action_id=f"a{i}",
                n_in=rng.randint(0, 1000),
                n_out=rng.randint(0, 1000),
                wall_ns=rng.randint(0, 10**9),
                p99_ns=rng.randint(0, 10**9),
                bits_consumed=rng.uniform(0.0, 100.0),
                bits_erased=rng.uniform(0.0, 100.0),
                is_irreversible=rng.random() > 0.5,
                score=rng.uniform(0.0, 1.0),
                timestamp_ns=i,
                cfg=cfg,
            )
            entries.append(entry)
        return BudgetLedger(entries=tuple(entries), horizon_ns=cfg.max_horizon_ns)

    ledger_a = build_ledger()
    ledger_b = build_ledger()

    assert len(ledger_a.entries) == len(ledger_b.entries)
    for a, b in zip(ledger_a.entries, ledger_b.entries):
        # Frozen dataclasses with identical fields ⇒ structural equality.
        assert a == b, f"INV-HPC1 VIOLATED: {a} != {b}"
        # Byte-precise equality of the aggregated cost.
        assert a.total_proxy_cost == b.total_proxy_cost
    assert total_cost(ledger_a) == total_cost(ledger_b)


# --------------------------------------------------------------------------
# 10. bits_erased == 0 ⇒ entropy proxy exactly 0 (algebraic).
# --------------------------------------------------------------------------


@given(consumed=_FLOAT_BITS)
@settings(max_examples=100, deadline=None)
def test_bits_erased_zero_implies_zero_entropy_cost(consumed: float) -> None:
    """Algebraic: erased=0 ⇒ proxy_cost = 0 exactly, regardless of consumed."""
    ec = compute_entropy_cost(consumed, 0.0)
    assert ec.proxy_cost == 0.0, (
        f"Expected exact 0.0 entropy proxy when bits_erased=0; "
        f"got {ec.proxy_cost} at consumed={consumed}"
    )


# --------------------------------------------------------------------------
# Bonus: dataclass immutability (frozen) — guard against silent mutation
# that would defeat the audit-ledger assumption.
# --------------------------------------------------------------------------


def test_dataclasses_are_frozen() -> None:
    """All public dataclasses must be frozen — ledger entries are evidence,
    not state. Mutation would invalidate the audit trail."""
    tc = TokenCost(n_input_tokens=1, n_output_tokens=1, proxy_cost=5.0)
    with pytest.raises(AttributeError):
        tc.proxy_cost = 99.0  # type: ignore[misc]

    lc = LatencyCost(wall_time_ns=1, p99_ns=1, proxy_cost=0.0)
    with pytest.raises(AttributeError):
        lc.proxy_cost = 1.0  # type: ignore[misc]

    ec = EntropyCost(bits_consumed=0.0, bits_erased=0.0, proxy_cost=0.0)
    with pytest.raises(AttributeError):
        ec.proxy_cost = 1.0  # type: ignore[misc]

    ic = IrreversibleActionCost(is_irreversible=False, irreversibility_score=0.0, proxy_cost=0.0)
    with pytest.raises(AttributeError):
        ic.proxy_cost = 1.0  # type: ignore[misc]
