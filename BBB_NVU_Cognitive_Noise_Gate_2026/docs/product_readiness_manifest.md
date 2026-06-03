# BBB-NVU Cognitive Noise Gate Product Readiness Manifest

## Status

Engineering status: bounded operational prototype.

Clinical status: research-use only. The artifact does not diagnose, treat, prescribe, triage, or authorize real-world care decisions.

## Product boundary

This subsystem is ready to be consumed as a deterministic repository artifact when all listed gates are green:

1. `Commit Acceptor Gate`
2. `PR Gate / python-quality`
3. `PR Gate / secrets-supply-chain`
4. `Readiness Gate`
5. `Physics Invariants`
6. `python BBB_NVU_Cognitive_Noise_Gate_2026/tools/verify_artifact.py`

The release unit is the directory `BBB_NVU_Cognitive_Noise_Gate_2026/` plus its diff-bound acceptors.

## Runtime contract

The stable integration surface is:

- `RuntimeBoundary.evaluate_run(...)`
- `RuntimeBoundary.evaluate_batch(...)`
- `OperationalKernel.execute(...)`
- `verify_operational_envelope(...)`

The CLI is a compatibility adapter. Product consumers should not couple to CLI parsing.

## Evidence contract

The canonical verifier writes:

```text
tmp/bbb_nvu_cng_verify_artifact.json
```

The artifact records Python version, command status, JSON parsing, Python compilation, focused pytest, operational smoke result, hash material, and whitespace integrity.

## Determinism contract

The deterministic identity of one run is bound by:

```text
input_hash + rules_hash + engine_hash -> run_hash
```

An operational envelope is accepted only when output hashes, audit hashes, replay bundle hashes, replay verification, metrics snapshot id, incident ids, and request cardinality are internally consistent.

## Fail-closed contract

Invalid schema, non-finite numeric values, out-of-range domain values, or explicitly invalid input must produce a bounded invalid output path with zero confidence. Missing or degraded data must be visible as degradations, not silently interpreted as clean evidence.

## Non-goals

This PR does not claim:

- biological validation;
- prospective human validation;
- medical-device readiness;
- autonomous decision authority;
- production service hosting;
- real sensor ingestion;
- privacy or consent completeness for live deployment.

## Next promotion gate

Promotion from bounded operational prototype to integration-ready subsystem requires a follow-up PR that moves the package into a canonical import namespace, adds a release manifest with file digests, and binds artifact verification into CI as a named required check.
