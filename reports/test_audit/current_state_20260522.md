# GeoSync Test Evidence Audit — 2026-05-22

## Problem (1 sentence)
GeoSync has broad test coverage by suite count, but claim-level evidence links were incomplete and non-fail-closed before this Stage-1 grounding pass.

## Falsifiable hypothesis
If the evidence protocol is valid, every `proven` claim in `docs/METRICS_CONTRACT.md` must have a machine-readable evidence record with command/artifact/status/SHA/timestamp; otherwise gate fails.

## Inventory snapshot (factual)
- Python test files under `tests/`: **1380** (`rg --files tests | rg '\\.py$' | wc -l`)
- Baseline run folder created: `reports/test_audit/baseline_run_20260522/`
- Baseline artifacts present: `pytest.log`, `junit.xml`, `coverage.xml`, `coverage_summary.json`, `environment.json`, `pip_freeze.txt`, `git_sha.txt`, `python_version.txt`, `exit_code.txt`

## Canonical command source of truth
See `docs/testing/canonical_commands.md`.

## Baseline command executed
`pytest tests/ -m "not flaky" --cov=core --cov=backtest --cov=execution --cov-report=xml:reports/test_audit/baseline_run_20260522/coverage.xml --cov-report=term-missing --junitxml=reports/test_audit/baseline_run_20260522/junit.xml`

Output log: `reports/test_audit/baseline_run_20260522/pytest.log`

## Acceptance gate
`python tools/claims/validate_claims_have_evidence.py`

Expected behavior:
- fail if `proven` claim has no evidence record;
- fail if evidence record misses required fields;
- fail if strong claim language appears outside `METRICS_CONTRACT` allowlist.
