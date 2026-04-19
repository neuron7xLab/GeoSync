# PR #333 — PriorAttenuationGate Finalization

## Short description
PriorAttenuationGate is finalized as a fail-closed Layer 2–3 exploration primitive with deterministic activation/terminal contracts, reduced canonical invariants, mandatory indicator integration witnesses, and reproducible validation artifacts.

## Detailed description

### Scope
- Runtime primitive: `runtime/prior_attenuation_gate.py`
- Protocol wrapper/export surface: `tacl/prior_attenuation_protocol.py`, `tacl/__init__.py`
- Canonical invariants: `.claude/physics/INVARIANTS.yaml`, `runtime/INVARIANTS.yaml`
- Tests and validation: `tests/unit/runtime/test_prior_attenuation_gate.py`, `bench/prior_attenuation_perf.py`, `docs/prior_attenuation_ci_validation.md`
- Architecture/docs: `README.md`, `CLAUDE.md`, `docs/NEURO-TO-DIGITAL-ONTOLOGY.md`

### What was completed
1. PriorAttenuation naming and runtime/protocol surface alignment completed.
2. Canonical invariant set reduced to six enforceable invariants:
   - `INV-PA-1`, `INV-PA-3`, `INV-PA-4`, `INV-PA-5`, `INV-PA-6`, `INV-PA-7`.
3. Integration witnesses are mandatory (no runtime skip path):
   - Kuramoto order parameter integration witness.
   - Entropy integration witness as free-energy descent proxy behavior.
4. Physics kernel gate execution is recorded with a real run:
   - `python .claude/physics/validate_tests.py --self-check`.
5. Performance artifact provided:
   - 10k-cycle benchmark script with avg step latency, memory peak, deepcopy count.

### Contract assurances
- Activation is fail-closed and coherence/priors are strictly validated.
- Reintegration and emergency paths remain restore-confirmed before reset.
- Safety preemption remains deterministic (`kill_switch_active`, `stressed_state`).
- Required six invariants each map to direct witness tests.
