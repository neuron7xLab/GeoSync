# PriorAttenuation CI Validation (X7 Target 7)

## Requested command results

- `physics-kernel-gate`: command not found in this environment.
- `formal-verification`: command not found in this environment.
- `pr-gate`: command not found in this environment.

## Executed fallback validation

- `pytest -q tests/unit/runtime`
- `ruff check runtime/prior_attenuation_gate.py tacl/prior_attenuation_protocol.py tacl/__init__.py tests/unit/runtime/test_prior_attenuation_gate.py bench/prior_attenuation_perf.py`
- `mypy runtime/prior_attenuation_gate.py tacl/prior_attenuation_protocol.py tacl/__init__.py tests/unit/runtime/test_prior_attenuation_gate.py bench/prior_attenuation_perf.py`
- `python bench/prior_attenuation_perf.py`

## Observed status

- Runtime unit tests passed with 2 integration tests skipped because optional dependency `omegaconf` is unavailable.
- Ruff and mypy passed for all modified Python files.
- Benchmark result: avg step time under 0.5 ms (`0.057761 ms` on this machine).
