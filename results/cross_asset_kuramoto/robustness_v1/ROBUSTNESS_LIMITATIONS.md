# Cross-asset Kuramoto · Robustness v1 limitations

Honest catalogue of what the v1 framework *does not* measure cleanly.
Nothing below is a bug: every entry is a known statistical or data-
access limitation that a reader MUST account for when interpreting
`verdict.json`, `null_summary.json`, or `ROBUSTNESS_RESULTS.md`.

## 1. PSR has no autocorrelation adjustment

`research.robustness.cpcv.probabilistic_sharpe_ratio` implements the
Lopez de Prado (2018) Eq. 14.1 PSR. The formula corrects for skewness
(γ₃) and kurtosis (γ₄) of the sample distribution but **does not**
correct for serial correlation in the return stream.

Strategy returns that exhibit positive first-order autocorrelation —
which is typical of regime-following strategies — inflate the
effective sample size used in the Sharpe-variance denominator.
Consequences:

- The reported `psr_daily = 1.0000` on the frozen bundle should
  **not** be read as definitive statistical significance.
- Under HAC (heteroscedasticity- and autocorrelation-consistent)
  adjustment (Newey–West, Andrews–Monahan kernel), the effective
  sample size shrinks and the PSR would be materially lower.

Implementing HAC-adjusted PSR is a forward improvement and is
out of scope for v1. The caveat is cross-linked from
`ROBUSTNESS_RESULTS.md` under the CPCV row.

## 2. Jitter evaluator is `PLACEHOLDER_APPROXIMATION`

`kuramoto_jitter_executor.make_placeholder_evaluator` returns a
smooth quadratic in fractional parameter-space distance scaled by the
anchor Sharpe. This exercises the primitive contract but does **not**
rebuild the strategy under perturbed parameters.

- The row in `ROBUSTNESS_RESULTS.md` shows `N/A`, not ✓.
- `fraction_within_tol_pass` is forced to `False` regardless of raw
  fraction — the decision layer treats placeholder evidence as
  abstention, not a pass.
- Replacing the executor requires access to the raw asset panel (not
  in the frozen bundle); pairing that panel with the frozen parameter
  lock yields a live evaluator.

## 3. LOO-grid PBO has low path count

`results/cross_asset_kuramoto/offline_robustness/leave_one_asset_out.csv`
ships 5 folds × 13 perturbations. Bailey et al.'s CPCV PBO achieves
full statistical power at C(N, k) paths with N ≥ 8. With 5 paths the
PBO estimate has wide confidence intervals; the reported 0.20 is a
point estimate, not a CI-backed lower bound.

A higher-power PBO requires either a richer strategy-parameter grid
(non-frozen; out of scope) or importance-sampled CPCV over an expanded
fold geometry.

## 4. Null families do not include benchmark-matched tests

The single-stream null suite compares the realised Sharpe against
bootstrapped resamples of itself. It does **not** test whether the
strategy outperforms a matched-cost, matched-lag benchmark such as
BF1 equal-weight. That measurement lives in the offline packet
(`benchmark_family.csv`) and is cross-referenced by
`SEPARATION_FINDING.md`.

## 5. Contract covers the frozen bundle only

Everything above operates on `SOURCE_HASHES.json` (28 artefacts) +
`leave_one_asset_out.csv` (inline-hash-verified extension). The framework
does **not** re-run the spike or re-simulate the strategy. It is a
*read-only* audit layer.

## Forward improvements

Any of the five items above can be closed without changing the
existing primitives:

1. HAC-PSR adjustment (Newey–West kernel inside
   `probabilistic_sharpe_ratio`).
2. Live jitter evaluator (raw asset panel + frozen parameter lock).
3. Higher-power PBO (expand LOO grid or import full spike parameter
   sweep).
4. Benchmark-matched null families (import `benchmark_family.csv`).
5. Protocol-level contract covering the live-shadow evidence rail
   (not just the demo bundle).

None of these is required for a valid FAIL or PASS verdict on the
current frozen evidence; each would tighten the confidence interval
around that verdict.
