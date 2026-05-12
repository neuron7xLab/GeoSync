# Buyer Readme (Integration Contract)

## What you are buying
A fail-closed evidence-promotion gate for extrapolated model claims.

## API surface
- `generate_artifact.py generate` -> emits attested artifact.
- `generate_artifact.py verify --artifact` -> validates schema+contract+sha.

## Integration SLA assumptions
- Caller provides verified context hash inputs and null-model outputs.
- Caller provides structured witness data for high-risk decisions.
- Downstream system must reject claims when verifier exit code != 0.

## Independent verification checklist
1. Corrupt `sha256` and ensure verify returns code 2.
2. Remove high-risk witness metadata and ensure generation returns code 2.
3. Remove required null model and ensure evidence generation returns code 2.
4. Run `falsifier.py` and require `FALSIFIER PASS`.

## Rollback
On suspected drift or compromised evidence path: force `result=killed_with_counterexample` and block promotions.
