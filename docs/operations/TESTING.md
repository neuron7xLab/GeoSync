# Testing Guide for GeoSync

This document describes the testing strategy, coverage requirements, and instructions for running tests in the GeoSync project.

> **Quick Start**: Run `pytest tests/unit -m "not slow"` for fast feedback during development.
> 
> **See Also**: [Test Architecture Guide](../TEST_ARCHITECTURE.md) for patterns and best practices.
>
> 📊 **Claims Registry**: All coverage and quality claims are tracked in [docs/METRICS_CONTRACT.md](../METRICS_CONTRACT.md).

## Overview

GeoSync employs a comprehensive testing strategy that includes:

- **Unit Tests**: Validate individual functions and classes in isolation.
- **Integration Tests**: Exercise complete workflows and pipeline chains across modules.
- **Property-Based Tests**: Check invariants and properties using Hypothesis-generated data.
- **Fuzz Tests**: Stress the system with malformed and adversarial payloads.
- **End-to-End (E2E) Tests**: Reproduce realistic user journeys and CLI pipelines.
- **Performance Tests**: Test behavior with large datasets (planned).
- **Contract Tests**: Keep DTO JSON Schemas and OpenAPI documents stable for downstream consumers.
- **Data Quality Gates**: Detect duplicates, spikes, and missing values before analytical jobs run.
- **Security Tests**: Prevent sensitive data leakage in audit logs and shared fixtures.
- **UI Smoke & Accessibility**: Exercise the Next.js dashboard via Playwright and aXe rules.

For a capability-to-suite traceability view, consult the
[regression test matrix](../../tests/TEST_PLAN.md) which links major product behaviours to the
automated coverage that protects them.

## Continuous Integration Gates

The primary PR gate currently runs the main Python quality and test lanes on Python 3.11, while repository-integrity proof runs on Python 3.12. This matches the project runtime contract `>=3.11,<3.13`. Python 3.13 is outside the current declared support window.

The fast lane executes the deterministic pytest suite with slow, heavy-math, nightly, and flaky tests excluded. Heavy, formal, frontend, Rust, Go, dependency, and security gates run under their own content-aware jobs.

Scenarios tagged with [`@pytest.mark.slow`](../../pytest.ini) remain available for manual, heavy, or nightly execution without slowing down PR feedback, while the smoke path still guards against regressions in CLI wiring and reporting.

## Test Structure

```
tests/
├── unit/                    # Module-level tests (core.*, backtest.*, execution.*, geosync.*, ...)
├── integration/             # Workflow-level tests (pipelines, backtests, protocol adapters)
├── property/                # Property-based suites powered by Hypothesis
├── fuzz/                    # Fuzz harnesses targeting ingestion and message handling
├── contracts/               # JSON Schema/OpenAPI compatibility suites
├── data/                    # Data quality guardrails for reference datasets
├── security/                # Logging and secret-leak safeguards
├── e2e/                     # Pytest smoke scenarios that mimic end-user flows
└── fixtures/                # Shared fixtures, builders, and reusable data

scripts/
└── smoke_e2e.py             # CLI entrypoint used by the E2E workflow
```

## Coverage Requirements

Coverage and quality claims are governed by [`docs/METRICS_CONTRACT.md`](../METRICS_CONTRACT.md) and the active CI configuration. Do not upgrade a coverage statement in this guide unless the enforcing workflow and metrics contract agree.

Module coverage goals (not enforced, verify with `make test-coverage`):
- `backtest/`: 100% (goal)
- `execution/`: 100% (goal)
- `core/agent/`: ≥95% (goal)
- `core/data/`: ≥95% (goal)
- `core/indicators/`: ≥90% (goal)
- `core/metrics/`: ≥95% (goal)
- `core/phase/`: ≥95% (goal)
- `geosync/`: governed by `pyproject.toml` coverage source and MFN release gate

📊 **Full claims mapping**: [docs/METRICS_CONTRACT.md](../METRICS_CONTRACT.md)

## Running Tests Locally

### Prerequisites

Install dependencies (development file includes runtime stack):
```bash
pip install -r requirements-dev.lock
```

### Running All Tests

Run the complete test suite:
```bash
pytest tests/
```

> **CI Gate**: Pull requests must pass `.venv/bin/python -m pytest -m "not flaky" tests/`, which exercises the full `tests/` tree (unit, integration, property, fuzz, contracts, data, security, e2e, neuro, strategies, admin, scripts, and utilities) with the flaky marker excluded. Align local runs with this command before opening a PR.

Run with coverage guardrail:
```bash
pytest tests/ \
  --cov=core --cov=backtest --cov=execution --cov=geosync \
  --cov-config=configs/quality/critical_surface.coveragerc \
  --cov-report=term-missing --cov-report=xml

python -m tools.coverage.guardrail \
  --config configs/quality/critical_surface.toml \
  --coverage coverage.xml
```

### Makefile Shortcuts

The Makefile exposes convenience targets that wrap the most common pytest invocations. Use them to keep local runs aligned with CI expectations:

| Target | Description | Underlying command |
| --- | --- | --- |
| `make test:fast` | Fast feedback loop that skips heavyweight suites | `pytest tests/ -m "not slow and not heavy_math and not nightly"` |
| `make test:all` | Full coverage-enabled suite matching CI defaults | `pytest tests/ --cov=core --cov=backtest --cov=execution --cov=geosync --cov-config=configs/quality/critical_surface.coveragerc --cov-report=term-missing --cov-report=xml && python -m tools.coverage.guardrail --config configs/quality/critical_surface.toml --coverage coverage.xml` |
| `make test:heavy` | Executes the slow, heavy math, and nightly gates | `pytest tests/ -m "slow or heavy_math or nightly"` |

Generate HTML coverage report:
```bash
pytest tests/ \
  --cov=core --cov=backtest --cov=execution --cov=geosync \
  --cov-config=configs/quality/critical_surface.coveragerc \
  --cov-report=html
# Open htmlcov/index.html in browser
```

### Static Analysis

Static analysis is required before opening a pull request. The consolidated target runs
the Python and Go analyzers configured for the project:

```bash
make lint
```

- `flake8` enforces the Python style guide across `analytics/`, `core/`, `execution/`, `geosync/`, `tests/`, and other runtime packages.
- `mypy` checks type safety with the settings captured in [`mypy.ini`](../../mypy.ini).
- `golangci-lint` validates the Go services defined under [`go/`](../../go/) using the
  shared [`.golangci.yml`](../../.golangci.yml) policy.

### Running Specific Test Categories

**Unit tests only:**
```bash
pytest tests/unit/
```

**Integration tests only:**
```bash
pytest tests/integration/
```

**Property-based tests only:**
```bash
pytest tests/property/
```

**Fuzz tests only:**
```bash
pytest tests/fuzz/
```

**Contract tests (JSON Schema + OpenAPI):**
```bash
pytest tests/contracts/
```

**Data quality guardrails:**
```bash
pytest tests/data/
```
When preparing new sample data files you can also run the standalone analyzer:
```bash
python scripts/data_sanity.py data --pattern "**/*.csv"
```

**Security guardrails:**
```bash
pytest tests/security/
bandit -r tests/utils tests/scripts -ll
```

**E2E smoke tests (pytest harness):**
```bash
pytest tests/e2e/ -m "not slow and not flaky"
```

Include slow scenarios (e.g., `test_full_pipeline.py`) when you need full pipeline
confidence:
```bash
pytest tests/e2e/ -m slow
```

**Cross-architecture indicator parity (CPU/GPU/ARM simulacrum):**
```bash
pytest -m arm tests/performance/test_indicator_portability.py
```

**Performance regression benchmarks (fail on budget overruns):**
```bash
pytest tests/performance/test_indicator_benchmarks.py --benchmark-enable
```

**Memory pressure guardrails for large indicator windows:**
```bash
pytest tests/performance/test_memory_regression.py
```

**Heavy-math validation gate:**
```bash
pytest -m heavy_math tests/unit/config/test_heavy_math_jobs.py
```

**UI smoke & accessibility (Playwright):**
```bash
cd apps/web
npm ci
npm run build
npx playwright test --config=playwright.config.ts
```
The Playwright project automatically runs aXe scans and fails the run if critical or serious accessibility violations are detected.

### Running Specific Test Files or Functions

Run a specific test file:
```bash
pytest tests/unit/test_agents.py
```

Run a specific test function:
```bash
pytest tests/unit/test_agents.py::test_strategy_mutation_changes_numeric_parameters
```

Run tests matching a pattern:
```bash
pytest tests/ -k "backtest"
```

### Running the E2E Pipeline Script

The CLI entrypoint mirrors the CI smoke workflow and is suitable for local regression checks:

```bash
python scripts/smoke_e2e.py
```

This command builds a deterministic sample OHLCV dataset, invokes the indicator/strategy/backtest stack, and verifies that the reporting artifacts are produced under `artifacts/e2e/`.
