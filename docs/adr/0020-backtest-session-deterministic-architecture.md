# ADR-0020: Deterministic BacktestSession Architecture

- **Status:** Accepted
- **Date:** 2026-04-23
- **Decision Makers:** Core Architecture Division
- **Supersedes:** None
- **Superseded by:** N/A

## Context
Backtesting required deterministic replay, explicit state boundaries, and fail-fast contracts. Previous flows allowed partial runtime leakage and environment-sensitive behavior for perf/property verification.

## Decision
Adopt `BacktestSession` as canonical orchestrator with explicit snapshot semantics and segment-resume API:
- `run(..., start_idx, end_idx, reset_state)` for deterministic chunk execution.
- `get_state()/set_state()` includes runtime + component states (execution, conformal, feature, regime, guard).
- `ValidationService` remains single contract surface for runtime invariants.
- Perf harness uses deterministic local fallback when heavyweight engine imports are unavailable.

## Why this is the accepted architecture
1. **Determinism first:** state snapshots and resume are first-class, not bolted on.
2. **Operational safety:** contract checks are centralized and fail-fast.
3. **Reproducibility:** lockfiles and pinned CI workflow provide stable execution substrate.
4. **Backward compatibility:** legacy alias remains for migration.

Alternative designs (ad-hoc resets per component, implicit global state, best-effort logging/IO suppression) were rejected because they obscure failure semantics and break replay guarantees.

## Consequences
### Positive
- Mid-run checkpoint/resume parity is testable and reproducible.
- Failure paths are explicit (no silent CSV save failures).
- CI can execute deterministic/property/perf suites from lockfiles.

### Negative
- Session API is slightly broader (`start_idx/end_idx/reset_state`).
- More strict validation can surface historical data-quality issues earlier.

## Rollback
Revert commits that introduced segmented-run and snapshot extensions; keep alias-based compatibility while migrating callers.

## Links
- `geosync_hpc/backtest.py`
- `geosync_hpc/state.py`
- `geosync_hpc/validation.py`
- `tests/geosync_hpc/test_backtester_state_reset.py`
- `.github/workflows/backtest-determinism-gate.yml`
