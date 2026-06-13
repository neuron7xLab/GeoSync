# Cognitive Dynamics PR Workflow Trigger

Purpose: trigger the pull-request execution surface for the Cognitive Dynamics Lab workflow without adding a new runtime module.

## Decoupled planes

1. Calibration plane: thresholds, sensitivity, precision, bounded state transitions.
2. Optimization plane: objective directions, weights, score, recommended actions.
3. Measurement plane: replay outputs, benchmark outputs, artifact existence checks.
4. Governance plane: acceptor binding, workflow ordering, issue trace, rollback path.
5. Telemetry plane: workflow status counts, artifact digests, gate conclusions, elapsed runtime.
6. Event log plane: PR comments, issue updates, commit SHAs, rerun markers, failure causes.
7. Scaling plane: parallel Python versions, workflow sharding, benchmark throughput, bounded retries.

## Aggregated representation

Integrated state = calibration profile + optimization profile + measurement evidence + governance verdict + telemetry snapshot + event log + scaling posture.

The aggregate must stay metadata-only in this PR. Runtime code remains in the already governed runners:

- scripts/cognitive_dynamics_lab/simulation_runner.py
- scripts/cognitive_dynamics_lab/parameter_review_runner.py
- scripts/cognitive_dynamics_lab/benchmark_runner.py

## Calibration to integration rule

A module is considered integrated only if its thresholds are explicit, its objective contribution is measurable, its artifacts are produced by workflow replay, its changed files are bound by an acceptor, and its telemetry can be read from GitHub Actions without adding an unbound runtime surface.

## Orchestration order

compile -> simulation replay -> parameter review replay -> benchmark replay -> artifact checks -> workflow artifact upload -> repo gates -> telemetry snapshot -> event log.

## Monitoring rule

The live health signal is computed from GitHub workflow conclusions for the PR head: success, failure, in_progress, queued, and artifact digest presence. A failed required gate blocks production launch; a successful lab workflow permits only the research-preview slice.

## Logging rule

Every operational transition must leave at least one durable GitHub trace: commit SHA, PR comment, issue comment, workflow run id, job id, or artifact digest. Chat-only decisions do not count as system events.

## Scaling rule

Scale only bounded surfaces: parallel Python versions, workflow shards, benchmark sample count, artifact throughput, and retry of failed jobs. Do not scale by adding runtime modules unless acceptor binding is available first.

## Reversible boundary

No new runtime module may remain in this PR unless governance binding lands first. If a module cannot be bound, it is removed rather than carried as architectural debt.
