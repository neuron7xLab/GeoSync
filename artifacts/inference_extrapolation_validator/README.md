# Inference Extrapolation Validator (IEV)

Module version: `3.0.0` · API version: `1.0`

A fail-closed certificate format and verification CLI for claims that
extrapolate beyond a verified context. Emits a canonical-JSON,
SHA-256-sealed artifact that maps a caller-supplied test result onto one
of two terminal states:

- `EVIDENCE` / `VERIFIED_EVIDENCE` — survived the contract.
- `KILLED` / `FALSIFIED_OR_REJECTED` — every other outcome.

There is no third state. Promotion to EVIDENCE requires *all* checks
listed under "Contract" to pass; any single violation flips the artifact
to KILLED and exits non-zero.

## What this is

A certificate gate. Given a hypothesis, a risk class, a set of tests that
were run, and a set of null-model results, the IEV verifies:

1. **Schema** — the artifact validates against
   `schema/artifact.schema.json`.
2. **Risk-tier coverage** — `tests_run` ⊇ required set for the risk
   class; `null_models_run` ⊇ required set when claim_status would be
   EVIDENCE.
3. **Internal arithmetic** — `hypothesis_score > max(null_model_results)`
   and `null_model_results.keys() == set(null_models_run)`.
4. **Result/state mapping** —
   `survived ⇒ claim_status=EVIDENCE ∧ claim_boundary=VERIFIED_EVIDENCE`
   and the opposite for `killed_with_counterexample`.
   `test_passed=false` cannot coexist with `survived`.
5. **Witness rules** — high risk requires `witness.status==approved`
   with structured `reviewer_id`, `review_timestamp_utc` (≤30 days old),
   and `review_hash`. A `rejected` witness cannot declare `survived`.
6. **Purpose drift** — `purpose.alignment_score ≥ 0.7` AND
   `purpose.drift_check ≤ 0.3` (DIKWP framework).
7. **External falsification** — for EVIDENCE,
   `external_falsification.drift_score ≤ 0.5` and probe metadata is
   present.
8. **STN-hyperdirect ACC gate** — fail-closed if the max of a 7-dim
   conflict vector (falsifier disagreement, null-model contradiction,
   witness uncertainty, purpose drift, external falsification drift,
   schema/claim inconsistency, evidence-boundary violation risk)
   exceeds 0.8 (hard ceiling) or the seed-deterministic adaptive
   threshold.
9. **API version** — `api_version == "1.0"` (no silent forward-compat).
10. **SHA integrity** — `artifact.sha256` equals SHA-256 of the canonical
    JSON (sorted keys, `(",",":")` separators, UTF-8) with the `sha256`
    field removed.

The full enumerated set lives in `INVARIANTS.yaml` (IEV-001 .. IEV-021).

## What this is *not*

A test runner. The IEV does not execute the caller's tests, recompute
`hypothesis_score`, generate null distributions, or verify the
`prompt_hash` / `requirements_lock_sha256` / `observation_hash` against
their alleged sources. Those numerics are external inputs; the IEV
enforces only structural consistency, internal arithmetic, and SHA
integrity over them.

See `CLAIM_BOUNDARY.md` for what the certificate does and does not
guarantee.

## Usage

```bash
# Generate a sealed artifact from raw inputs (full flag list in --help).
python artifacts/inference_extrapolation_validator/generate_artifact.py generate ... --out path/to/artifact.json

# Verify an existing artifact (re-runs every contract check).
python artifacts/inference_extrapolation_validator/generate_artifact.py verify --artifact artifacts/inference_extrapolation_validator/example_artifact.json

# Run the in-process unit suite (27 tests).
python -m unittest artifacts/inference_extrapolation_validator/test_generate_artifact.py -v

# Run the standalone falsifier battery (7 scenarios).
python artifacts/inference_extrapolation_validator/falsifier.py

# Aggregate gate.
make iev-gate
```

Exit codes: `0` pass, `2` contract violation, `3` parse failure.

## CI

`.github/workflows/iev-module-gate.yml` runs the three checks above on
every push/PR that touches this module or its workflow. The job is
content-scoped and does not gate unrelated changes.

## Operational docs

- `PURPOSE.md` — why this module exists, in one sentence.
- `CLAIM_BOUNDARY.md` — what the certificate proves and what it doesn't.
- `INVARIANTS.yaml` — enumerated invariants IEV-001 .. IEV-021.
- `state_diagram.md` — the two-terminal-state machine.
- `docs/THREAT_MODEL.md` — adversarial assumptions.
- `docs/ROLLBACK_DOCTRINE.md` — how to invalidate an issued certificate.
- `CHANGELOG.md` — version history.
