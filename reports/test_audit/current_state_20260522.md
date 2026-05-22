# GeoSync Test Evidence Audit — 2026-05-22

## Problem statement
Current test architecture has strong breadth but incomplete proof chain for high-strength claims (`COV_98_CI_GATE` remains `goal`, `MUTATION_90_KILL` remains `partial`).

## Canonical command tiers
- fast: `pytest tests/ -m "not slow and not flaky" -q`
- full: `pytest tests/ -m "not flaky" --cov=core --cov=backtest --cov=execution --cov-report=xml --cov-report=term-missing`
- heavy: `pytest tests/nightly tests/performance -m "not flaky" -q`
- property: `pytest tests/property/ --hypothesis-profile=ci`
- fuzz: `pytest tests/fuzz/ -q`
- security: `pytest tests/security/ -q && bandit -r core backtest execution src -ll -q`
- mutation: `mutmut run --use-coverage && python -m tools.mutation.kill_rate_guard --threshold 0.9 --scope critical`
- benchmark: `pytest tests/performance/ --benchmark-enable`

## Evidence policy
Only claims with machine-readable evidence records are allowed to remain high-strength in docs/release statements.

## Required artifacts for a baseline run
- `coverage.xml`
- `junit.xml`
- `pytest.log`
- `environment.json`
- `pip_freeze.txt`
- `git_sha.txt`

## Next gating command
`python tools/claims/validate_claims_have_evidence.py`
