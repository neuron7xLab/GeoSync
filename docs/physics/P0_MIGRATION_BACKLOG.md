# P0 Migration Backlog

Date: 2026-05-29

## Local validator snapshot

After classifier repair, the validator scans 199 physics-test files and 2178 test functions.

| Level | Count | Meaning |
|---|---:|---|
| L1 | 1917 | missing `INV-*` reference |
| L2 | 17 | invalid `INV-*` ID |
| L3 | 63 | test structure does not match invariant type |
| L4 | 119 | missing or weak failure diagnostics |
| L5 | 24 | possible magic thresholds |
| Total | 2140 | remaining validation issues |

Physics grounding: 261 / 2178 tests = 12%.

## Highest-impact files

| Rank | File | Issues | Main failure mode |
|---:|---|---:|---|
| 1 | `tests/core/neuro/serotonin/test_serotonin_controller.py` | 40 | mixed L1/L3/L4/L5 |
| 2 | `tests/unit/test_kuramoto_ricci_composite.py` | 38 | mostly L1 plus one invalid ID / structure / message / threshold |
| 3 | `tests/unit/core/test_kuramoto_modules.py` | 37 | L1 |
| 4 | `tests/core/neuro/serotonin/test_serotonin_runtime_safety.py` | 35 | mixed L1/L3/L4 |
| 5 | `tests/core/neuro/dopamine/test_invariants.py` | 34 | L1 |
| 6 | `tests/research/calibration/test_grid_kuramoto.py` | 33 | L1 |
| 7 | `tests/test_thermo_optimizations.py` | 33 | L1 |
| 8 | `tests/unit/physics/test_simulation_falsification.py` | 32 | L1 plus L3 |
| 9 | `tests/test_gaba_inhibition_gate.py` | 31 | L1 |
| 10 | `tests/core/neuro/dopamine/test_dopamine_precision.py` | 29 | L1 |

## Batch strategy

1. Fix mixed-severity files first. They carry L3/L4/L5, not just missing IDs.
2. Then migrate pure L1 files by adding correct invariant docstrings only when semantics are clear.
3. Do not add `INV-*` IDs to shape, import, or smoke tests unless they really witness an invariant.
4. Move non-physics tests away from physics-keyword paths rather than gaming the validator.
5. Re-run `python .claude/physics/validate_tests.py tests/ --summary` after each batch.

## Current release gate

C1/C2 code audit is now clean and blocking. Remaining P0 risk is test-grounding density, not production clamp silence.
