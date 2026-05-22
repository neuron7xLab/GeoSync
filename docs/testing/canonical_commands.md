# Canonical Test Commands

Date: 2026-05-22 (UTC)
Baseline artifact root template: `reports/test_audit/baseline_run_YYYYMMDD/`

## fast
```bash
pytest tests/ -m "not slow and not heavy_math and not nightly and not flaky" -q
```

## full
```bash
pytest tests/ -m "not nightly and not flaky" --junitxml=reports/test_audit/baseline_run_YYYYMMDD/junit.xml
```

## coverage
```bash
pytest tests/ -m "not nightly and not flaky" \
  --cov=core \
  --cov=backtest \
  --cov=execution \
  --cov-config=configs/quality/critical_surface.coveragerc \
  --cov-report=xml:reports/test_audit/baseline_run_YYYYMMDD/coverage.xml \
  --cov-report=term-missing \
  --junitxml=reports/test_audit/baseline_run_YYYYMMDD/junit.xml
```

## property
```bash
pytest tests/property/ -q
```

## fuzz
```bash
pytest tests/fuzz/ -q
```

## security
```bash
pytest tests/security/ -q
bandit -r core backtest execution src -ll -q
```

## mutation
```bash
mutmut run --use-coverage
```

## benchmark
```bash
pytest tests/performance/ --benchmark-enable
```
