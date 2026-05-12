# Changelog

## 3.0.0 - 2026-05-12
- Added fail-closed `generate` / `verify` CLI modes with canonical-JSON
  SHA-256 sealing.
- Added schema gates for artifact and spec (`schema/*.json`).
- Added structured witness policy with high-risk approval requirement
  (reviewer_id, review_timestamp_utc, review_hash) and ≤30-day
  freshness window.
- Added risk-tiered required test / null-model coverage (low / medium /
  high) with `hypothesis_score > max(null_model_results)` enforcement.
- Added purpose-drift gate (alignment ≥ 0.7, drift ≤ 0.3) under DIKWP
  framework.
- Added external-falsification gate (`drift_score ≤ 0.5` for EVIDENCE,
  probe metadata required).
- Added STN-hyperdirect ACC conflict vector (7 dims) with hard ceiling
  0.8 and seed-deterministic adaptive threshold.
- Added falsifier battery (`falsifier.py`, 7 scenarios) and contract
  tests (`test_generate_artifact.py`, 27 tests).
- Added CI gate (`.github/workflows/iev-module-gate.yml`) running tests,
  falsifier, and example-artifact verification on touched paths only.
- Documentation pruned to enumerated, code-anchored set: `PURPOSE`,
  `README`, `CLAIM_BOUNDARY`, `INVARIANTS`, `state_diagram`,
  `THREAT_MODEL`, `ROLLBACK_DOCTRINE`. Sales / commercial / handoff
  documents removed as out of scope for the engineering artifact.
