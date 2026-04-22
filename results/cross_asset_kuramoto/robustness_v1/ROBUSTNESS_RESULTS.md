# Cross-asset Kuramoto · Robustness v1 report

Terminal decision: **FAIL**

## Suite summary

| Suite | Metric | Value | Pass |
|---|---|---:|:-:|
| CPCV | PBO (fold mirror, n=2, *tautological*) | 0.0000 | ✓ |
| CPCV | PSR (daily) | 1.0000 | ✓ |
| CPCV | Annualised Sharpe (daily) | 0.4832 | n/a |
| CPCV | PBO (LOO grid, n=13, *admissible*) | 0.2000 | ✓ |
| Null | iid_bootstrap p-value | 0.5045 | ✗ |
| Null | stationary_bootstrap p-value | 0.5235 | ✗ |
| Jitter | fraction_within_tol | 1.0000 | N/A |
| Jitter | evaluator_mode | `PLACEHOLDER_APPROXIMATION` (not decision-grade; live evaluator required to flip this row to ✓ / ✗) | n/a |

## Reasons

- null: one or more families failed
- jitter: placeholder evaluator — abstains from live ✓/✗

## Notes

- Evidence is derived from the frozen `offline_robustness/SOURCE_HASHES.json` bundle; 28 artifacts hash-verified.
- Null suite uses mathematically exact daily log-returns (`diff(log(strategy_cumret))`) — no approximation. See `ROBUSTNESS_PROTOCOL.md` § 1 for the derivation contract.
- PBO interpretation: fewer than 3 candidates is `tautological`, fewer than 5 is `weak`, 5+ is `admissible`. The fold-mirror PBO is always tautological by construction and is kept only as a sanity baseline; the LOO-grid PBO is the decision-grade one.
- Jitter row shows `N/A` while the evaluator is `PLACEHOLDER_APPROXIMATION`; a live rebuild is required to replace the row with a real ✓ / ✗.
