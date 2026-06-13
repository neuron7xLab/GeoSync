# Cognitive Dynamics PR Workflow Trigger

Purpose: trigger the pull-request execution surface for the Cognitive Dynamics Lab workflow without adding a new runtime module.

## Decoupled planes

1. Calibration plane: thresholds, sensitivity, precision, bounded state transitions.
2. Optimization plane: objective directions, weights, score, recommended actions.
3. Measurement plane: replay outputs, benchmark outputs, artifact existence checks.
4. Governance plane: acceptor binding, workflow ordering, issue trace, rollback path.

## Aggregated representation

Integrated state = calibration profile + optimization profile + measurement evidence + governance verdict.

The aggregate must stay metadata-only in this PR. Runtime code remains in the already governed runners:

- scripts/cognitive_dynamics_lab/simulation_runner.py
- scripts/cognitive_dynamics_lab/parameter_review_runner.py
- scripts/cognitive_dynamics_lab/benchmark_runner.py

## Calibration to integration rule

A module is considered integrated only if its thresholds are explicit, its objective contribution is measurable, its artifacts are produced by workflow replay, and its changed files are bound by an acceptor.

## Orchestration order

compile -> simulation replay -> parameter review replay -> benchmark replay -> artifact checks -> workflow artifact upload -> repo gates.

## Reversible boundary

No new runtime module may remain in this PR unless governance binding lands first. If a module cannot be bound, it is removed rather than carried as architectural debt.
