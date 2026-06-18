# GeoSync Physics Validation Verdict

Status: GENERATED_BY_tools/physics_score.py
Score type: `static_evidence_oracle_not_runtime_proof`
S_total: `77.82`
Target interval: `88.0-92.0`
Verdict: `FAIL_BELOW_TARGET_BLOCKED_FOR_VALIDATION`

## Evidence added in this iteration

```text
UQ smoke test: tests/physics/test_uq_smoke.py
UQ artifact: artifacts/physics_validation/uq_summary.json
Independent reference solver: reproducibility/reference_solvers/geosync_kuramoto_numpy.py
Replication test: tests/physics/test_replication.py
Replication artifact: artifacts/physics_validation/replication_summary.json
Ricci bridge tests: tests/physics/test_ricci_bridge.py
Ricci bridge artifact: artifacts/physics_validation/ricci_bridge_summary.json
Commit acceptors: .claude/commit_acceptors/nfi-physics-*.yaml
```

## Remaining gates

- CI or local pytest evidence for the new physics tests.
- Full Ricci graph provenance audit before Ricci can leave experimental status.
- Immutable BN-Syn and MFN+ source references before those systems can be scored here.

## Metric table

| metric | weight | score | weighted | status |
|---|---:|---:|---:|---|
| `S_math_object` | 0.12 | 75.0 | 9.0 | PARTIAL_PASS_TRACEABLE |
| `S_dimensional_consistency` | 0.1 | 70.0 | 7.0 | PARTIAL_PASS_NO_TYPED_UNITS |
| `S_numerical_stability` | 0.12 | 82.0 | 9.84 | PARTIAL_PASS_RUNTIME_REQUIRED |
| `S_invariant_preservation` | 0.12 | 80.0 | 9.6 | PARTIAL_PASS_RUNTIME_REQUIRED |
| `S_falsifiability` | 0.14 | 82.0 | 11.48 | PARTIAL_PASS_RUNTIME_REQUIRED |
| `S_baseline_models` | 0.1 | 82.0 | 8.2 | PARTIAL_PASS_RUNTIME_REQUIRED |
| `S_UQ` | 0.08 | 72.0 | 5.76 | PARTIAL_PASS_RUNTIME_REQUIRED |
| `S_reproducibility` | 0.08 | 78.0 | 6.24 | PARTIAL_PASS_CI_LOGS_REQUIRED |
| `S_interface_contracts` | 0.08 | 70.0 | 5.6 | PARTIAL_PASS_RUNTIME_REQUIRED |
| `S_traceability` | 0.06 | 85.0 | 5.1 | PARTIAL_PASS_RUNTIME_REPORT_PENDING |

## Final decision

```text
CONTROL_PLANE: BUILT
UQ_SMOKE: ADDED
INDEPENDENT_REPLICATION: ADDED
RICCI_BRIDGE_SMOKE: ADDED
COMMIT_ACCEPTORS: ADDED
S_TOTAL: 77.82/100
TARGET: 88-92
FINAL_PHYSICS_VALIDATION: NO
MERGE_READY: NO
```
