# Canonical Test Commands (Stage 1 Evidence Grounding)

All commands write artifacts under `reports/test_audit/baseline_run_20260522/`.

## fast
```bash
pytest tests/ -m "not slow and not flaky" -q
```

## full
```bash
pytest tests/ -m "not flaky" --junitxml=reports/test_audit/baseline_run_20260522/junit.xml
```

## coverage baseline
```bash
pytest tests/ -m "not flaky" \
  --cov=core \
  --cov=backtest \
  --cov=execution \
  --cov-report=xml:reports/test_audit/baseline_run_20260522/coverage.xml \
  --cov-report=term-missing \
  --junitxml=reports/test_audit/baseline_run_20260522/junit.xml \
  | tee reports/test_audit/baseline_run_20260522/pytest.log
```

## heavy
```bash
pytest tests/nightly tests/performance -m "not flaky" -q
```

## property
```bash
pytest tests/property/ --hypothesis-profile=ci
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
python -m tools.mutation.kill_rate_guard --threshold 0.9 --scope critical
```

## benchmark
```bash
pytest tests/performance/ --benchmark-enable
```
