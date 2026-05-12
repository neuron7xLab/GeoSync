# Why This Exists

Most AI incidents are not model failures first; they are **promotion failures**:
unverified extrapolations are treated like evidence.

This module exists to stop that specific failure mode.

If a claim cannot survive required tests, null-models, witness policy (for high-risk), external falsification, schema checks, and SHA integrity, it does not get promoted.

Result:
- fewer silent false positives,
- auditable postmortems,
- immediate reduction of model-governance audit risk.
