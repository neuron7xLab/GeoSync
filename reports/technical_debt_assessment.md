# Technical Debt Assessment

## Test Reliability
- `pytest` fails during import because `core.indicators.multiscale_kuramoto` does not define the symbols exported in `core.indicators.__init__` (`MultiScaleKuramotoFeature`, `TimeFrame`, and `WaveletWindowSelector`). This makes the entire indicator package unusable for the multi-scale workflow and blocks the integration test suite.【F:core/indicators/__init__.py†L15-L74】【F:core/indicators/multiscale_kuramoto.py†L1-L155】【077c7e†L1-L17】

## Indicator Layer
- The multi-scale Kuramoto module only implements procedural helpers and a `MultiScaleKuramoto` class. It never exposes a feature wrapper or the time-frame utilities promised by the public API, so higher-level blocks cannot be composed from it.【F:core/indicators/multiscale_kuramoto.py†L1-L155】
- `MultiScaleKuramoto.analyze` silently skips any timeframe with fewer than `window + 5` samples and returns zero consensus without signalling which scales were dropped, making downstream decisions opaque.【F:core/indicators/multiscale_kuramoto.py†L138-L155】
- The autocorrelation window selector has no safeguards against constant price series beyond returning the minimum window, which can amplify noise on illiquid assets.【F:core/indicators/multiscale_kuramoto.py†L40-L62】
- Temporal Ricci analysis clears history on every call, so cross-window stability metrics cannot accumulate over time and every invocation works on isolated batches.【F:core/indicators/temporal_ricci.py†L161-L255】
- The lightweight graph utilities recompute edges, shortest paths, and connectivity through repeated Python loops, which becomes quadratic for larger level counts and lacks caching or vectorised operations.【F:core/indicators/temporal_ricci.py†L8-L123】
- The Ricci curvature fallback graph ignores edge weights and lacks any distance metric, so curvature estimates degrade severely whenever `networkx` or SciPy are unavailable.【F:core/indicators/ricci.py†L10-L139】
- `phase_flags` hard-codes heuristic thresholds yet imports indicator functions that are never used, signalling dead code and unclear coupling between metrics and decision logic.【F:core/phase/detector.py†L3-L21】

## Agent & Strategy Logic
- `Strategy.simulate_performance` injects random scores instead of running a deterministic backtest, which makes optimisation unreproducible and unsuitable for production learning loops.【F:core/agent/strategy.py†L7-L22】
- `PiAgent.detect_instability` and `evaluate_and_adapt` rely on brittle threshold checks without logging or hysteresis, so small floating-point jitter can flip decisions between enter/hold/exit states.【F:core/agent/strategy.py†L24-L48】
- Strategy mutation/repair blindly zeroes any large or NaN parameter, potentially destroying calibrated hyperparameters without constraint awareness.【F:core/agent/strategy.py†L14-L41】
- The in-memory strategy store never evicts records beyond a decay filter and keeps everything in RAM; it also represents the strategy address as a tuple of raw floats, making persistence and lookups fragile.【F:core/agent/memory.py†L7-L32】

## Data & Backtesting Tooling
- The CSV ingestor assumes every row contains `ts` and `price` columns, lacks validation/error handling, and leaves unused imports (`time`, `json`, `threading`) from a non-existent asynchronous implementation.【F:core/data/ingestion.py†L1-L42】
- The Binance websocket client is returned directly without lifecycle management or reconnection logic, forcing callers to handle threads and cleanup manually.【F:core/data/ingestion.py†L32-L42】
- `walk_forward` does not validate that the signal array matches the price series length, ignores position bounds, and applies fees per unit change instead of per executed notional, leading to mis-priced results.【F:backtest/engine.py†L13-L28】

## Execution & Risk Controls
- Position sizing clamps exposure using two conflicting formulas (`balance * risk / price` and `balance / price`), effectively ignoring the configured risk parameter whenever price is small, which skews leverage estimates.【F:execution/order.py†L5-L14】
- Portfolio heat treats each position as a dictionary with `qty` and `price`, disregards side/direction, and sums absolute notional without currency normalisation or risk weightings.【F:execution/risk.py†L1-L6】

## Operational Hygiene
- Multiple modules expose public APIs that depend on optional heavy libraries but provide only minimal fallbacks; the degraded implementations sacrifice correctness (e.g., Ricci curvature and graph analytics) without logging or warning the operator.【F:core/indicators/ricci.py†L10-L139】【F:core/indicators/temporal_ricci.py†L8-L215】
- Residual imports and unused dependencies (e.g., NumPy, indicator helpers in `phase_flags`) indicate missing linting/static analysis coverage and increase maintenance overhead.【F:core/phase/detector.py†L3-L21】

