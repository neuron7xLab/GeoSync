# BacktestSession Engineering Protocol (GeoSync HPC)

## Problem statement
Build a deterministic, resettable backtesting session that guarantees identical outputs on identical inputs and blocks invalid runtime states early.

## Falsifiable hypothesis
If `BacktestSession` correctly isolates mutable state and enforces finite/contract checks, then three repeated runs over the same calibrated data produce identical equity curves and state snapshot roundtrips preserve behavior.

## Runtime contract
- **Input contract:** required market/feature/target columns exist and dataframe length is at least 2 rows.
- **State contract:** `reset()`-driven run isolation for feature store, regime model, guardrails, execution RNG, conformal state, and return history.
- **Step contract:** finite values, non-negative costs, capped exposure, bounded position jumps, spread-envelope fill prices.
- **Snapshot contract:** `get_state()`/`set_state()` restores session runtime fields + component states.

See machine-readable invariants: `geosync_hpc/invariants.yaml`.

## One-command validation
```bash
pytest -q tests/test_conformal.py tests/geosync_hpc/test_runtime_safety_primitives.py tests/geosync_hpc/test_backtester_state_reset.py
```

## Rollback path
```bash
git revert <commit_sha>
```

## Kill switch
Disable conformal online updates in config:
```yaml
conformal:
  online_update: false
```

## Expected guarantees
1. Deterministic repeated run behavior (same config + same data => same equity path).
2. Explicit failures on NaN/inf and malformed inputs.
3. Serializable runtime state for replay/debug/resume workflows.
