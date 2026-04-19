# PriorAttenuation CI Validation (X7 Target 7)

## Requested command results

- `python .claude/physics/validate_tests.py --self-check`: passed.
- `formal-verification`: command not found in this environment.
- `pr-gate`: command not found in this environment.

## Executed fallback validation

- `python .claude/physics/validate_tests.py --self-check`
- `pytest -q tests/unit/runtime`
- `ruff check runtime/prior_attenuation_gate.py tacl/prior_attenuation_protocol.py tacl/__init__.py tests/unit/runtime/test_prior_attenuation_gate.py bench/prior_attenuation_perf.py`
- `mypy runtime/prior_attenuation_gate.py tacl/prior_attenuation_protocol.py tacl/__init__.py tests/unit/runtime/test_prior_attenuation_gate.py bench/prior_attenuation_perf.py`
- `python bench/prior_attenuation_perf.py`

## Observed status

- Physics kernel self-check passed: 66 invariants loaded, checker coverage and cross-reference checks passed.
- Runtime unit tests passed, including mandatory integration witnesses for Kuramoto/entropy metrics.
- Ruff and mypy passed for all modified Python files.
- Benchmark result: avg step time under 0.5 ms (`0.040628 ms` on this machine).

## Invariant witness map (required six)

- `INV-PA-1` → `test_inv_pa_1_activation_denied_when_parent_not_nominal`, `test_inv_pa_1_activation_denied_when_coherence_below_threshold`, `test_inv_pa_1_activation_rejects_non_real_coherence`.
- `INV-PA-3` → `test_inv_pa_3_entropy_ceiling_forces_reintegration_and_blocks_progression`.
- `INV-PA-4` → `test_inv_pa_4_duration_forces_reintegration_at_threshold_and_closes_step`.
- `INV-PA-5` → `test_inv_pa_5_gate_never_becomes_inactive_without_terminal_call`.
- `INV-PA-6` → `test_inv_pa_6_failed_reintegration_restores_exact_backup`.
- `INV-PA-7` → `test_inv_pa_7_safety_preemption_forces_emergency_exit`.
