# GeoSync Test Evidence Audit — 2026-05-22

## Problem (1 sentence)
GeoSync previously allowed strong quality claims without a mandatory machine-readable evidence chain linking claim → command → artifact → verification metadata.

## Falsifiable hypothesis
If Stage‑1 evidence grounding is complete, every claim in `docs/METRICS_CONTRACT.md` must have an evidence record, and every `proven` claim must include non-empty `last_verified_sha` and `timestamp`; otherwise validation fails.

## Invariant (defined before code)
No release-critical narrative can exceed evidence strength recorded in `METRICS_CONTRACT + evidence_index`.

## Inventory snapshot (factual)
- Python test files under `tests/`: **1380** (`rg --files tests | rg '\\.py$' | wc -l`)
- Baseline bundle path: `reports/test_audit/baseline_run_20260522/`
- Present artifacts: `pytest.log`, `junit.xml`, `coverage_summary.json`, `environment.json`, `pip_freeze.txt`, `git_sha.txt`, `python_version.txt`, `exit_code.txt`
- Baseline exit code: `1` (dependency/import failures in current environment; captured as evidence, not hidden)

## Canonical command source of truth
`docs/testing/canonical_commands.md`

## Baseline command executed
`pytest tests/ -m "not flaky" --cov=core --cov=backtest --cov=execution --cov-report=xml:reports/test_audit/baseline_run_20260522/coverage.xml --cov-report=term-missing --junitxml=reports/test_audit/baseline_run_20260522/junit.xml`

## Acceptance gate
`python tools/claims/validate_claims_have_evidence.py`

## Stage‑1 completion verdict
Stage‑1 is considered closed for governance grounding because:
1. claim registry is fully indexed in machine-readable form,
2. strong claims have SHA+timestamp metadata,
3. canonical commands are fixed,
4. baseline run reality (including failures) is archived without overclaiming.
