# PR Preflight Runbook

This runbook describes the local PR preflight gate after the shell gate was promoted into a structured Python evidence runner.

## User-facing command

```bash
scripts/test-pr-locally.sh
```

The shell script is only a compatibility wrapper. It locates the repository root, selects `python3` or `python`, calls `tools/ci/pr_preflight.py`, prints `FIRST_FILE_TO_OPEN`, and forwards the Python runner exit code.

## Direct runner commands

```bash
python tools/ci/pr_preflight.py --root . --report-dir artifacts/pr_preflight
python tools/ci/pr_preflight.py --root . --report-dir artifacts/pr_preflight --skip-install
python tools/ci/pr_preflight.py --root . --report-dir artifacts/pr_preflight --json
python -m tools.ci.pr_preflight --root . --report-dir artifacts/pr_preflight --skip-install
```

Use `--skip-install` when validating control flow without dependency installation. Do not claim a full local preflight pass from a skip-install run.

## Evidence outputs

The runner writes deterministic evidence under:

```text
artifacts/pr_preflight/preflight_report.json
artifacts/pr_preflight/logs/<check_id>.stdout.log
artifacts/pr_preflight/logs/<check_id>.stderr.log
```

The JSON report is written even when the preflight fails or blocks. The first file to open is always the report path printed by the wrapper and stored in `first_file_to_open`.

## Status model

Allowed check statuses:

```text
PASS
FAIL
SKIPPED_OPTIONAL
BLOCKED
TIMEOUT
```

Allowed final statuses:

```text
PASS
FAIL
BLOCKED
```

Unknown states must not become pass. Missing critical tools become `BLOCKED`. Nonzero critical checks become `FAIL`. Missing optional `detect-secrets` becomes `SKIPPED_OPTIONAL`.

## Check registry

The runner defines checks as data through `CheckSpec` and `build_check_registry`. Required checks:

```text
pip_bootstrap
project_dependencies
ruff
black
mypy
detect_secrets
quick_tests
coverage_artifact
```

`coverage_artifact` is an observation check. It must not substitute for quick test success.

## Local validation contract

Before opening or updating a PR that touches the runner, execute:

```bash
python -m pytest tests/ci/test_pr_preflight_gate.py tests/ci/test_pr_preflight_engine.py -q
bash -n scripts/test-pr-locally.sh
python tools/ci/pr_preflight.py --root . --report-dir artifacts/pr_preflight --skip-install
python -m tools.ci.pr_preflight --root . --report-dir artifacts/pr_preflight --skip-install
```

If the non-skip preflight is required, run:

```bash
python tools/ci/pr_preflight.py --root . --report-dir artifacts/pr_preflight
```

Do not hide stderr. Do not use `--exit-zero`. Do not add `|| true` around critical checks. Do not make the wrapper always exit zero. Humanity has produced enough decorative green lights already.

## Failure interpretation

Open `artifacts/pr_preflight/preflight_report.json` first. Then inspect the stdout/stderr logs for the first critical check with status `FAIL`, `BLOCKED`, or `TIMEOUT`.

A blocked preflight is a valid fail-closed result. Treat it as an environment or tool availability problem, not as a successful gate.
