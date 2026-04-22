# Cross-asset Kuramoto · Robustness v1 protocol

Canonical input derivations and statistical-test protocols used by the
v1 framework. Every value below is fixed by source-controlled code in
`research/robustness/protocols/`; anything that drifts from this file
is a bug in the code, not a licence to edit the file.

## 1. Input-data derivations

### 1.1 Daily strategy returns

The frozen demo bundle ships the cumulative wealth curve
`results/cross_asset_kuramoto/demo/equity_curve.csv::strategy_cumret`
but does **not** ship a raw `net_ret` series. The framework therefore
derives daily returns as mathematically exact **log returns**:

```
r_t = log(strategy_cumret_t) − log(strategy_cumret_{t-1})
```

Log returns are chosen over simple `pct_change` because:

- they are the honest, time-additive representation of a multiplicative
  wealth trajectory (every daily `r_t` satisfies `exp(Σ r_s) = wealth_t`);
- they are symmetric under sign inversion;
- they preserve independence under permutation and resampling, which is
  the contract assumed by the bootstrap null families.

The derivation is implemented once in
`research.robustness.protocols.kuramoto_contract.KuramotoRobustnessContract.daily_strategy_returns`.

### 1.2 Daily benchmark returns

Identical derivation applied to `benchmark_cumret`.

## 2. Bootstrap null families (single-stream)

The null suite operates on a *realised* return stream only; it has no
access to a raw `position × price` signal. The two families below are
the honest Sharpe-level nulls for that input shape.

### 2.1 iid_bootstrap

Sample indices i.i.d. from `[0, n)` with replacement, compute Sharpe on
the resampled vector, repeat `n_bootstrap` times. Note that *plain
permutation* would be degenerate: Sharpe is order-invariant on a given
vector, so permutation preserves it up to floating-point noise and
yields a trivial p → 1. With-replacement sampling changes the realised
mean and std of every draw and is the proper i.i.d. null for a Sharpe
statistic on a single series.

### 2.2 stationary_bootstrap (Politis & Romano 1994)

Geometric-block resample with mean block length 21 bars. Tests for
information content beyond short-horizon autocorrelation.

Both families share a seeded `np.random.default_rng` and emit a
Davison–Hinkley +1 continuity-corrected upper-tail p-value.

## 3. Decision thresholds

See `ROBUSTNESS_PROTOCOL.md § Statistical thresholds` (populated by
Task 4) for the canonical `alpha`, `pbo_max`, `psr_min`, and
jitter-tolerance values. All thresholds are encoded as module-level
constants in `research/robustness/protocols/*_suite.py` and
`backtest/robustness_gates.py`; the documentation mirrors the constants,
never the other way round.

## 4. Artefacts written

Every runner invocation writes strictly under
`results/cross_asset_kuramoto/robustness_v1/`:

- `verdict.json` — terminal decision and per-axis booleans
- `cpcv_summary.json` — PBO (fold mirror + LOO grid), PSR, Sharpe
- `null_summary.json` — p-values per family
- `jitter_summary.json` — jitter stability + evaluator mode
- `ROBUSTNESS_RESULTS.md` — human-readable one-page report
- `ROBUSTNESS_PROTOCOL.md` — this document

Nothing is written outside this directory. The frozen SOURCE_HASHES.json
contract covers 28 artefacts and remains hash-verified on every load.
