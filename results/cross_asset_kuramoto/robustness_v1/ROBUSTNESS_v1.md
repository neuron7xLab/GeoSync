# Cross-asset Kuramoto · Robustness v1 report

Terminal decision: **FAIL**

## Suite summary

| Suite | Metric | Value | Pass |
|---|---|---:|:-:|
| CPCV | PBO (fold mirror) | 0.0000 | ✓ |
| CPCV | PSR (daily) | 1.0000 | ✓ |
| CPCV | Annualised Sharpe (daily) | 0.5775 | n/a |
| CPCV | PBO (LOO grid, n=13) | 0.2000 | ✓ |
| Null | iid_permutation p-value | 0.0878 | ✗ |
| Null | stationary_bootstrap p-value | 0.5170 | ✗ |
| Jitter | fraction_within_tol | 1.0000 | ✓ |
| Jitter | evaluator_mode | `PLACEHOLDER_APPROXIMATION` | n/a |

## Reasons

- null: one or more families failed

## Notes

- Evidence is derived from the frozen `offline_robustness/SOURCE_HASHES.json` bundle; 28 artifacts hash-verified.
- Null suite uses cumulative-return pct_change as a return proxy; raw `net_ret` is not in the frozen demo bundle, which limits statistical power relative to the published headline Sharpe (`risk_metrics.csv::sharpe = 1.2619`).
- Jitter evaluator is PLACEHOLDER_APPROXIMATION: rebuild under perturbed parameters requires the raw asset panel.
