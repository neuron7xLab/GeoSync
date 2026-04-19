# PriorAttenuation CI Validation (X7 Target 7)

## Requested command results

- `python .claude/physics/validate_tests.py --self-check`: passed.
- `formal-verification`: command not found in this environment.
- `pr-gate`: command not found in this environment.

## Executed fallback validation

- `rg -n "runtime\.dmt_mode|DMT_PROTOCOL_NAME|INV-DMT-|test_dmt_mode|PriorAttenuationGate Contract \(DMT\)" .`
- `rg -n "INV-PA-2|INV-PA-10|test_inv_pa_2|test_inv_pa_10" .`
- `python .claude/physics/validate_tests.py --self-check`
- `pytest -q tests/unit/runtime/test_prior_attenuation_gate.py`
- `ruff check runtime/prior_attenuation_gate.py tacl/prior_attenuation_protocol.py tacl/__init__.py tests/unit/runtime/test_prior_attenuation_gate.py`
- `mypy runtime/prior_attenuation_gate.py tacl/prior_attenuation_protocol.py tacl/__init__.py tests/unit/runtime/test_prior_attenuation_gate.py`
- `python bench/prior_attenuation_perf.py`

## Observed status

- Physics kernel self-check passed: 66 invariants loaded, checker coverage and cross-reference checks passed.
- Stale-name grep (`runtime.dmt_mode`, `DMT_PROTOCOL_NAME`, `INV-DMT-*`, legacy test names/headers) returned no active matches.
- Non-canonical PA identifiers grep (`INV-PA-2`, `INV-PA-10`, `test_inv_pa_2`, `test_inv_pa_10`) returned no matches.
- Canonical PriorAttenuation invariant set remains exactly 6 (`INV-PA-1/3/4/5/6/7`) per `.claude/physics/INVARIANTS.yaml` and `runtime/INVARIANTS.yaml`.
- Runtime unit tests passed; integration witnesses were skipped in this environment because optional dependency `omegaconf` is not installed.
- Ruff and mypy passed for all modified Python files.
- Benchmark result: avg step time under 0.5 ms (`0.041615 ms` on this machine).

## Invariant witness map (required six)

- `INV-PA-1` → `test_inv_pa_1_activation_denied_when_parent_not_nominal`, `test_inv_pa_1_activation_denied_when_coherence_below_threshold`, `test_inv_pa_1_activation_rejects_non_real_coherence`.
- `INV-PA-3` → `test_inv_pa_3_entropy_ceiling_forces_reintegration_and_blocks_progression`.
- `INV-PA-4` → `test_inv_pa_4_duration_forces_reintegration_at_threshold_and_closes_step`.
- `INV-PA-5` → `test_inv_pa_5_gate_never_becomes_inactive_without_terminal_call`.
- `INV-PA-6` → `test_inv_pa_6_failed_reintegration_restores_exact_backup`.
- `INV-PA-7` → `test_inv_pa_7_safety_preemption_forces_emergency_exit`.

## Supporting proof surface (non-canonical)

- `test_support_single_instance_second_activation_rejected_without_state_corruption` (safety support).
- `test_support_attenuation_scales_values_exactly_and_preserves_keys` (attenuation algebra support).
- Kuramoto/entropy integration tests (integration support).
- `bench/prior_attenuation_perf.py` benchmark (performance support).
