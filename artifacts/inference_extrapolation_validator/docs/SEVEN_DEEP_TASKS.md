# Seven Deep Hardening Tasks (Implemented)

1. API contract lock (`api_version=1.0`) with fail-closed mismatch rejection.
2. Witness freshness gate (reject stale witness review timestamps).
3. Null model result consistency (keys must exactly match executed null models).
4. Evidence superiority gate (hypothesis score must beat all null model scores).
5. Purpose alignment + drift gates (already enforced).
6. STN hyperdirect conflict gate (already enforced).
7. Decision trace + command-line SHA for audit replayability.
