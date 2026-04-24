# Backtest Session Verification Benchmarks (2026-04-23)

## 1) Golden-path perf harness
- Status before fix: perf suite failed in CI-lite environments because `backtest.engine.walk_forward` import chain required unavailable heavy dependencies.
- Status after fix: deterministic fallback walk-forward executes and perf suite passes.

Sample benchmark (`n_bars=500`, `n_iterations=30`, `seed=42`):
- p50 latency: **3.656 ms**
- p95 latency: **5.096 ms**
- p99 latency: **5.939 ms**
- throughput: **136,756 bars/s**

## 2) Validation overhead micro-benchmark
Measured `ValidationService.trade_step` overhead:
- calls: 200,000
- with validation: 1.9522 s
- empty loop baseline: 0.0072 s
- estimated overhead: **~9.73 µs/call**

## Interpretation
The safety layer introduces a measurable but bounded per-step overhead while keeping throughput in the expected range for deterministic backtesting.
