# Testing Guide for GeoSync

This guide records the executable local test contract for GeoSync. It must not advertise commands that are absent from `Makefile` or unsupported by the active Python runtime contract.

## Runtime Contract

GeoSync currently supports:

```text
Python >=3.11,<3.13
```

The primary PR lane uses Python 3.11. Repository-integrity and MFN release proof use Python 3.12. Python 3.13 is outside the declared support window until the matrix and packaging contracts are updated together.

## Fast Local Proof

```bash
pytest tests/unit -m "not slow"
make test-fast
```

## Full Local Proof

```bash
pytest tests/
make test-all
```

## Heavy / Nightly Proof

```bash
make test-heavy
```

## Coverage Proof

Coverage must include the production Python surfaces, including `geosync`:

```bash
pytest tests/ \
  --cov=core --cov=backtest --cov=execution --cov=geosync \
  --cov-config=configs/quality/critical_surface.coveragerc \
  --cov-report=term-missing --cov-report=xml

python -m tools.coverage.guardrail \
  --config configs/quality/critical_surface.toml \
  --coverage coverage.xml
```

Convenience targets:

```bash
make test-coverage
make coverage-baseline
make coverage-90
make coverage-next
```

## Static Analysis

```bash
make lint
```

This runs the configured Python, Go, and shell analyzers where the required tools are available.

## Suite Map

```text
tests/unit/          unit-level behavior
tests/integration/   workflow behavior
tests/property/      generated invariant checks
tests/fuzz/          malformed input checks
tests/contracts/     schema and API contracts
tests/data/          data quality gates
tests/security/      logging and exposure guardrails
tests/e2e/           end-to-end smoke paths
tests/unit/mfn/      MFN package, CLI, artifact, and release-surface checks
```

## MFN Release Surface

MFN has a dedicated release gate:

```bash
make release-gate
```

That gate must exercise:

```text
ruff over geosync/mfn and clean-install verifier
mypy strict over geosync/mfn and clean-install verifier
pytest tests/unit/mfn
clean install verifier
artifact reproducibility checks in CI
```

## Command Naming Rule

Use hyphenated Make targets only:

```text
make test-fast
make test-all
make test-heavy
```

Do not document colon-style targets such as `make test:fast`. They are not Makefile targets in this repository.

## Claim Rule

Coverage, quality, or readiness claims must match the enforcing workflow and metrics contract. Documentation is not evidence. Passing commands, generated reports, and CI artifacts are evidence. Humanity continues to rediscover this, apparently with surprise.
