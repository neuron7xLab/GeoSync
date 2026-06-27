# Coverage Ascent Work Act 92-97

Status: engineering protocol, not a readiness claim.

## 1. First principles

Coverage is not the goal. Coverage is an instrument for exposing unobserved production behavior. A test is valuable only when it can kill a wrong implementation, detect an unsafe state, or falsify a claim. The campaign therefore optimizes truth-density, not test count.

The repository-tracked baseline is 85.03 percent release-line coverage, below the 90 percent release gate. The first target surface is `backtest`, because it has the largest measured deficit: 68.55 / 98. The next surfaces are `risk`, `execution`, `core`, `ingestion`, and `analytics`.

No threshold may be lowered. No omit rule may be expanded. No module may be skipped to inflate a number. Any claim of 90, 92, 95, or 97 percent requires same-commit machine evidence from the release coverage authority.

## 2. Value function

Each new test must maximize at least one of these values:

- falsification power: it fails for a plausible wrong implementation;
- risk reduction: it protects money, state, data, orders, or scientific claims;
- invariant capture: it pins a law that should hold across many inputs;
- boundary defense: it covers empty, non-finite, stale, duplicated, malformed, or adversarial inputs;
- regression locality: when it fails, the broken responsibility is obvious;
- deterministic replay: it gives the same verdict across machines and time;
- evidence integrity: it improves the measured release surface without hiding code.

A shallow test that only executes a line but cannot distinguish correct from wrong behavior has low value even if it raises coverage.

## 3. Methodology

The campaign proceeds by surface-risk ordering, not by convenience. First raise `backtest`, then harden `risk` and `execution`, then lift `core`, then `ingestion`, then `analytics`.

For every targeted file, derive tests from contracts:

1. identify the production responsibility;
2. list invariants and failure modes;
3. write the smallest deterministic fixtures;
4. assert behavioral properties, not implementation trivia;
5. include negative paths and edge states;
6. run the local test slice;
7. run full release coverage;
8. record exact evidence artifacts.

## 4. Required protocols

### Backtest protocol

Target files: `backtest/walk_forward.py`, `backtest/geosync_equity_curve.py`, `backtest/run_live_backtest.py`.

Required behaviors: split-boundary determinism, no look-ahead leakage, empty input handling, non-finite input handling, duplicate and unsorted timestamp handling, signal/price length mismatch, fee and slippage monotonicity, start equity, final equity, cumulative return, drawdown, constant-price edge cases, single-row edge cases.

### Risk and execution protocol

Target files: `execution/oms.py`, `execution/order_lifecycle.py`, `execution/position_sizer.py`, `execution/compliance.py`, `execution/order_ledger.py`, `execution/resilience/circuit_breaker.py`, `execution/arbitrage/engine.py`, `execution/arbitrage/models.py`, `execution/arbitrage/metrics.py`, `execution/arbitrage/capital.py`.

Required behaviors: invalid order rejection, duplicate order identity, stale transition rejection, idempotent ledger updates, explicit compliance rejection, gross exposure accounting, position-sizing monotonicity, circuit-breaker state transitions, stale quote rejection, deterministic arbitrage ranking, fee-aware profitability.

### Core protocol

Target examples: `core/agent/prompting/manager.py`, `core/agent/prompting/pqf_pscs.py`, `core/agent/prompting/library.py`, `core/agent/prompting/models.py`, `core/utils/secure_errors.py`, `core/neuro/advanced/divergence.py`, `core/altdata/news.py`, `core/cross_asset_kuramoto/signal.py`.

Required behaviors: model validation, deterministic serialization, invalid configuration rejection, normalization boundaries, mathematical invariants, safe error redaction, and local fixtures only.

### Analytics and market protocol

Target examples: `analytics/code_health/analyzers.py`, `analytics/code_health/aggregator.py`, `analytics/regime/src/adapters/local.py`, `analytics/regime/src/core/_sklearn_compat.py`, `markets/orderbook/src/core/lob.py`, `markets/orderbook/src/ingest/state.py`, `markets/orderbook/src/ingest/ingester.py`, `markets/orderbook/src/ingest/consistency.py`.

Required behaviors: empty-tree handling, malformed-file handling, deterministic aggregation, invalid adapter configuration, order-book bid/ask invariants, crossed-book rejection, idempotent depth updates, sequence-gap detection, stale event rejection.

## 5. Quality requirements

Every coverage PR must be test-only unless a test exposes a real bug; bug fixes must be split or explicitly justified. Tests must be deterministic, offline, small, readable, and tied to a production responsibility. Tests must include at least one failure-mode assertion when the module has reject paths. Tests must not depend on external services, live data, wall-clock randomness, or broad monkeypatching that bypasses the unit under test.

## 6. Definition of done

A PR is done only when new tests pass, the full release coverage measurement completes, the coverage intelligence report is valid, the release-line number moves or the zero movement is explained, and the PR body names the exact targeted files and behaviors. Evidence must include coverage summary, gap map, next-tests report, and the test command used.

## 7. Immediate next unit

Open a test-only PR for `backtest/walk_forward.py` and `backtest/geosync_equity_curve.py`. The goal is not cosmetic coverage. The goal is to convert backtest replay logic into a falsifiable, deterministic, failure-resistant surface on the path from 85.03 percent to 92-97 percent.
