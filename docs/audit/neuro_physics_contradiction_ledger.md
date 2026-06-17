# Neuro-Physics Contradiction Ledger

Machine-readable source of truth: [`data/audit/neuro_physics_contradiction_ledger.json`](../../data/audit/neuro_physics_contradiction_ledger.json)
validated against [`schemas/audit/contradiction_ledger.schema.json`](../../schemas/audit/contradiction_ledger.schema.json)
by `tests/audit/test_contradiction_ledger.py`.

## Law

> No neuro or physics claim exists without a source path, a contradiction path,
> a severity, evidence, and an honest resolution state. A claim without evidence
> is `UNSUPPORTED` — never silently passed. Closure requires a `resolution_ref`
> (no fake closure).

`claim_status ∈ {VERIFIED, PLAUSIBLE, UNSUPPORTED, CONTRADICTED}` ·
`severity ∈ {HIGH, MEDIUM, LOW}` ·
`resolution_state ∈ {OPEN, IN_PROGRESS, RESOLVED, WONTFIX}`.

## Ledger

Verified against real source on 2026-06-17. The remediation program anticipated
many contradictions; most are **already bounded** by the repo's own canon. Only
**C-ENERGY-001** is a true HIGH violation (already in progress on #1146) and only
**C-NEURO-003** is a genuinely-actionable MEDIUM. The rest are LOW (naming nits),
`UNSUPPORTED` (mechanism already exists), or bounded at the documentation layer.

| id | severity | status | source | owner lane | resolution |
|----|----------|--------|--------|------------|------------|
| **C-ENERGY-001** | HIGH | CONTRADICTED | `core/energy.py:147` | lane-2-energy-contract | IN_PROGRESS (#1146) |
| **C-NEURO-003** | MEDIUM | CONTRADICTED | `core/validation/neuro_integrity.py:146` | lane-3-neuromodulator-semantics | OPEN |
| **C-NEURO-001** | LOW | CONTRADICTED | `core/neuro/serotonin_ode.py:118` | lane-3-neuromodulator-semantics | OPEN |
| **C-NEURO-002** | LOW | PLAUSIBLE | `core/neuro/gaba_position_gate.py:102` | lane-3-neuromodulator-semantics | OPEN |
| **C-TACL-001** | LOW | UNSUPPORTED | `core/neuro/ecs_regulator.py:380` | lane-4-tacl-delta-f | WONTFIX |
| **C-KURAMOTO-001** | LOW | PLAUSIBLE | `core/kuramoto/capital_weighted.py:7` | lane-5-kuramoto-ricci-structure | OPEN |
| **C-ORCH-001** | LOW | UNSUPPORTED | `core/agent/orchestrator.py` | lane-6-orchestrator-boundary | WONTFIX |

## Entries

### C-ENERGY-001 — free energy fused with operational cost (HIGH)
`system_free_energy()` sums a Boltzmann-scaled thermodynamic term
(`K_BOLTZMANN_EFFECTIVE=1.38e-23`, `_KT_PRODUCT`) with operational/DevOps costs
(`latency_cost`, `incoherence_cost`) and adds entropy as a *penalty* under the
name "free energy", colliding with canon `F = U − T·S` (INV-FE2) where entropy
*lowers* F. **Owner:** Lane 2 (Energy Contract), in progress on #1146.

### C-NEURO-003 — declared correlation invariant is never checked (MEDIUM, the real gap)
`core/validation/neuro_integrity.py:146` declares
`dopamine_serotonin_correlation_min = -0.8` as a "Minimum inverse correlation
expectation", but `validate_state()` (line 208) never references it — a declared
integrity invariant with no witness (dead invariant). **Owner:** Lane 3 (wire a
trajectory-correlation check or drop the field).

### C-NEURO-001 — `[0,1]` clamp comment labelled "biological plausibility" (LOW)
A scalar clamp to `[0,1]` is a normalisation choice, not biophysical evidence.
Comment-level overclaim; the dynamics are bio-inspired bounded heuristics.
**Owner:** Lane 3 (reword comment).

### C-NEURO-002 — "STDP-like" label in one metaphor module (LOW, PLAUSIBLE)
`gaba_position_gate` calls RPE-sign weight adaptation "STDP-like" with no
spike-timing relation. **Real STDP already exists** in
`modules/gaba_inhibition_gate.py` (`stdp_tau_plus_ms=16.8`,
`exp(-delta_t_ms/tau_plus)` at line 325), so the only actionable scope is
renaming the metaphor in the former — not a missing mechanism. **Owner:** Lane 3.

### C-TACL-001 — ΔF gate ALREADY implemented (LOW, UNSUPPORTED → WONTFIX)
The anticipated "monotonic descent without a ΔF gate" is **not supported**:
`ecs_regulator._enforce_strict_monotonic_descent(new_fe, previous_fe)` (line 380,
applied at 611) + `core.energy.delta_free_energy(F_prev, F_now, dt)` + INV-FE1/FE3
already enforce `ΔF ≤ 0`. No remediation PR is warranted.

### C-KURAMOTO-001 — structure-vs-alpha bounded by product canon (LOW, PLAUSIBLE)
The no-alpha boundary is in a module docstring and reinforced by
`PRODUCT_CATEGORY.md` / `CLAUDE.md` ("research verification, not asserted market
edge"). Bounded at the documentation layer; an executable repo-wide
`R_is_structure_not_alpha` invariant is optional hardening, not an open
violation. **Owner:** Lane 5.

### C-ORCH-001 — orchestrator-as-controller claim absent (LOW, UNSUPPORTED → WONTFIX)
A grep of `core/agent/orchestrator.py` and `core/orchestrator/` finds no
"controller" / "basal ganglia" / "neural controller" assertion (only a
`NEUROMODULATION` enum label). The anticipated overclaim is **not evidenced**;
Lane 6 is preventive only.

## Maintenance

This ledger is descriptive and point-in-time. When a lane lands, set the entry's
`resolution_state` to `RESOLVED` **and** add the merged `resolution_ref` — the
schema and `tests/audit/test_contradiction_ledger.py` reject `RESOLVED` without a
ref (no fake closure). The advisory fake-closure scanner
(`tools/governance/check_fake_closure_claims.py`, PR #1145) enforces the same law
on PR bodies.
