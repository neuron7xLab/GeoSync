# claim_status_applied Procedure

## Purpose
`claim_status_applied` is a mandatory governance gate that verifies PR claims are backed by executable evidence.

## Required inputs
- PR title/body
- Changed files list
- Test commands + outputs
- Benchmark outputs (if perf claims are present)
- Rollback and kill-switch references

## Decision states
- `applied`: every claim has direct evidence (test, benchmark, static check, or traceable artifact).
- `partially_applied`: at least one claim lacks evidence but mitigation/waiver is documented.
- `not_applied`: claims are unverified or contradicted by evidence.

## Gate checklist
1. Map each claim to at least one evidence artifact.
2. Ensure negative-path tests exist for each fail-safe claim.
3. Verify reproducibility pins (`requirements.lock`, Python version, CI workflow).
4. Verify rollback and kill-switch are documented.
5. Emit final status with rationale.

## Required PR footer
```text
claim_status_applied: applied|partially_applied|not_applied
claim_evidence_ref: <links/paths>
```

## Owner
Core Architecture Division (neuron7xLab)
