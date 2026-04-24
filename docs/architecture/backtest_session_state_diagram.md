# BacktestSession State Diagram (One Page)

```mermaid
stateDiagram-v2
    [*] --> Initialized
    Initialized --> Calibrated: fit_quantiles + calibrate_conformal
    Calibrated --> Running: run(reset_state=True)
    Running --> Checkpointed: get_state()
    Checkpointed --> Restored: set_state(state)
    Restored --> Running: run(reset_state=False)
    Running --> Completed: end_idx reached
    Completed --> Initialized: run(reset_state=True)

    state Running {
      [*] --> Step
      Step --> Step: feature update / regime update / quantile+conformal / policy / execution
      Step --> Step: ValidationService.trade_step
      Step --> [*]: segment complete
    }
```

## Invariants
- No hidden mutable-state leakage between independent runs.
- Snapshot/restore preserves RNG, guardrails, conformal, feature, and regime states.
- Contract checks fail-fast on non-finite/non-physical runtime values.
