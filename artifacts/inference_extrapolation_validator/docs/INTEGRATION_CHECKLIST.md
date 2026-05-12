# Integration Checklist (Audit-Risk Focus)

## Preconditions
- [ ] Upstream pipeline can provide falsifier outputs and null-model outputs.
- [ ] High-risk claims routed through witness workflow.
- [ ] Artifact storage is immutable/object-locked.

## Runtime gates
- [ ] `generate` runs in CI/CD and at runtime decision boundary.
- [ ] `verify` runs before any production action.
- [ ] Any non-zero verify exit blocks promotion.

## Monitoring
- [ ] Track count of `KILLED` vs `EVIDENCE` artifacts.
- [ ] Alert on sudden rise in drift_score and witness failures.
- [ ] Alert on any schema/sha verification failures.

## Evidence trail
- [ ] Persist artifacts + verify logs for audit retention window.
- [ ] Retain witness metadata mapping in IAM.

## Rollback
- [ ] Force `killed_with_counterexample` mode enabled by ops runbook.
- [ ] `make iev-gate` included in release checklist.
