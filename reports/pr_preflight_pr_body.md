## Summary

This PR replaces the shell-embedded local PR preflight gate with a structured Python evidence runner while keeping `scripts/test-pr-locally.sh` as the compatibility entrypoint.

## Why this is needed

The old local preflight script mixed orchestration, tool execution, failure accounting, and terminal output in shell. That made fail-open behavior easy to introduce through `|| true`, `--exit-zero`, stderr suppression, or unchecked continuation.

A local PR gate should mirror CI semantics: critical failures must fail closed, preserve logs, and produce inspectable evidence.

## What changed

- Added `tools/ci/pr_preflight.py` as the structured preflight runner.
- Converted `scripts/test-pr-locally.sh` into a thin wrapper.
- Added machine-readable `preflight_report.json`.
- Added per-check stdout/stderr logs.
- Added `README_FOR_REVIEW.md` generation in each preflight report directory.
- Added `first_failing_log` to the report contract so reviewers can jump from the report to the first failing stderr log.
- Added report schema under `schemas/`.
- Added eval-style behavioral cases.
- Added invariant tests for report correctness.
- Added shell wrapper regression tests.
- Added critical-tool resolution matrix coverage.
- Added CI parity mapping in `tools/ci/pr_preflight_ci_contract.json`.
- Added `make pr-preflight`.
- Ignored local preflight artifacts in `.gitignore`.

## Production-readiness argument

The core production property is fail-closed behavior:

`critical failure -> FAIL/BLOCKED -> nonzero exit -> logs preserved -> report emitted`

This PR tests that property with controlled stub executables rather than relying on visual inspection of shell strings.

This PR reduces CI/preflight technical debt by replacing a shell-embedded local gate with a structured Python runner. The wrapper remains backward-compatible, while critical behavior moves into testable code. The runner emits a parseable JSON report and per-check logs, which makes local failures inspectable instead of transient terminal output. Behavioral tests use stub tools to prove fail-closed semantics for lint, type checking, tests, dependency installation, secret scanning, missing executables, skip-install mode, coverage observation, and timeout behavior.

The important property is not that the happy path passes. The important property is that controlled failures cannot be converted into green status.

## Failure semantics

- Critical check failure returns nonzero.
- Missing critical executable becomes `BLOCKED`.
- Missing optional `detect-secrets` becomes `SKIPPED_OPTIONAL`.
- Present but failing `detect-secrets` becomes `FAIL`.
- `--skip-install` marks install checks as `SKIPPED_OPTIONAL`, not `PASS`.
- Coverage absence is an optional observation, not a substitute for test success.
- Timeout becomes `TIMEOUT` and returns nonzero when critical.
- Unknown status collapses to `BLOCKED`.
- Invalid repository root becomes `BLOCKED` with a report when the report directory can be written.
- Report-write failure returns nonzero and prints a blocker to stderr without claiming `FIRST_FILE_TO_OPEN`.

## Evidence

Commands run:

```text
exit 0  git branch --show-current && git status --short && git log --oneline -5
exit 0  sed -n '1,220p' scripts/test-pr-locally.sh; sed -n '1,420p' tools/ci/pr_preflight.py; sed -n '1,180p' schemas/pr_preflight_report.schema.json; sed -n '1,260p' tests/ci/test_pr_preflight_gate.py; sed -n '1,300p' tests/ci/test_pr_preflight_engine.py; sed -n '1,220p' tests/ci/test_pr_preflight_eval_cases.py; sed -n '1,220p' tests/ci/test_pr_preflight_invariants.py; sed -n '1,180p' tests/ci/test_pr_preflight_report_schema.py; sed -n '1,160p' tests/ci/test_pr_preflight_requirements.py
exit 0  python -m pytest tests/ci/test_pr_preflight_report_schema.py -q
exit 0  python -m pytest tests/ci/test_pr_preflight_gate.py tests/ci/test_pr_preflight_engine.py tests/ci/test_pr_preflight_eval_cases.py tests/ci/test_pr_preflight_invariants.py tests/ci/test_pr_preflight_report_schema.py tests/ci/test_pr_preflight_requirements.py tests/ci/test_pr_preflight_ci_contract.py -q
exit 0  bash -n scripts/test-pr-locally.sh
exit 1  python tools/ci/pr_preflight.py --root . --report-dir artifacts/pr_preflight_pr_review --skip-install --json
exit 1  python -m tools.ci.pr_preflight --root . --report-dir artifacts/pr_preflight_pr_review_module --skip-install --json
exit 2  make pr-preflight
exit 0  ruff check tools/ci/pr_preflight.py tests/ci/test_pr_preflight_*.py
exit 0  git diff --check
```

First file to open after local run:

```text
artifacts/pr_preflight_pr_review/preflight_report.json
```

Review bundle entrypoint:

```text
artifacts/pr_preflight_pr_review/README_FOR_REVIEW.md
```

First failing stderr log from the review run:

```text
artifacts/pr_preflight_pr_review/logs/ruff.stderr.log
```

Current repository-local preflight reports are intentionally non-green because existing broad checks fail. The structured preflight currently preserves existing broad-check failures instead of hiding them. This is expected evidence behavior, not a fake green claim. This PR is review-ready, not production-validated. It does not assert deployment validation, complete CI equivalence, guaranteed security coverage, or release-level proof.

## Test plan

* `python -m pytest tests/ci/test_pr_preflight_report_schema.py -q`
* `python -m pytest tests/ci/test_pr_preflight_gate.py tests/ci/test_pr_preflight_engine.py tests/ci/test_pr_preflight_eval_cases.py tests/ci/test_pr_preflight_invariants.py tests/ci/test_pr_preflight_report_schema.py tests/ci/test_pr_preflight_requirements.py tests/ci/test_pr_preflight_ci_contract.py -q`
* `bash -n scripts/test-pr-locally.sh`
* `python tools/ci/pr_preflight.py --root . --report-dir artifacts/pr_preflight_pr_review --skip-install --json`
* `python -m tools.ci.pr_preflight --root . --report-dir artifacts/pr_preflight_pr_review_module --skip-install --json`
* `make pr-preflight` if full local dependencies are available and current broad checks are expected to pass

## Risk and rollback

Risk: local developer workflow changes from terminal-only output to report-backed execution.

Mitigation: `scripts/test-pr-locally.sh` remains the public entrypoint and prints `FIRST_FILE_TO_OPEN`.

Risk: local broad checks currently fail in this repository state.

Mitigation: failures are preserved in `preflight_report.json`, `README_FOR_REVIEW.md`, and per-check logs instead of being converted to green status.

Rollback: restore the previous shell script, but this would reintroduce shell-local fail-open risk and remove structured evidence.

## Reviewer checklist

* [ ] `scripts/test-pr-locally.sh` is only a wrapper and does not duplicate critical check logic.
* [ ] `tools/ci/pr_preflight.py` defines checks as data and uses command arrays.
* [ ] Python runner returns nonzero for critical failures.
* [ ] Missing critical tools become `BLOCKED`.
* [ ] Missing optional `detect-secrets` becomes `SKIPPED_OPTIONAL`.
* [ ] JSON report is emitted when the report directory can be written.
* [ ] Report-write failure returns nonzero and prints a blocker.
* [ ] stdout/stderr logs are preserved under the report directory.
* [ ] Eval cases prove fail-closed behavior.
* [ ] Schema/invariant tests match generated PASS, FAIL, and BLOCKED reports.
* [ ] `tools/ci/pr_preflight_ci_contract.json` documents critical local check alignment.
* [ ] `make pr-preflight` delegates to the wrapper and preserves nonzero failure.
