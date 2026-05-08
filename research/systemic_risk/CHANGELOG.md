# CHANGELOG — research/systemic_risk

## 2026-05-08 — Protocol X-7

Added score-level CSD indicators, naive baselines, extended
metrics, and documentation gates. **No empirical validation claim
is introduced.** The module status remains
`HYPOTHESIS / SCORE-LEVEL INSTRUMENTATION EXTENSION ONLY`;
end-to-end validation remains pending.

* `critical_slowing_down.py` — `CSDConfig` + `compute_csd_indicators`
  with explicit `ConstantPolicy`, `min_periods` / `lag` validation,
  no-lookahead contract, valid-count tracking.
* `baselines.py` — `rolling_volatility_score`,
  `edge_density_score` (directed / undirected, with / without
  self-edges).
* `metrics.py` — `ClassificationMetrics`, `LeadTimeConfig`,
  `LeadTimeMetrics`, `compute_classification_metrics`,
  `compute_lead_time_metrics`. NaN-not-zero policy on every
  undefined denominator.
* Tests (+49): leakage regression on CSD, density formula
  variants, NaN-not-zero on all four classification metrics,
  pre-event lead-time strictness incl. same-day exclusion,
  post-event refusal, first-valid-signal selection.
* Docs: `BASELINES.md`, `METRICS.md`, `NULL_MODELS.md`,
  `FAILURE_MODES.md`, `REPRODUCIBILITY.md`,
  `BOOTSTRAP_PROTOCOL.md`, this CHANGELOG.

## 2026-05-08 — PR #564 governance gates

* `governance.py`: `assert_claim_tier`,
  `build_validation_readiness_report`,
  `run_premerge_science_gate`, `FORBIDDEN_OVERCLAIM_TERMS`.
* `temporal_panel.validate_temporal_exposure_panel` —
  fail-closed boundary contract.
* `falsification.run_score_level_falsification` (alias) +
  `falsification.run_end_to_end_falsification` (NotImplementedError stub).
* `network_fitting.fit_barabasi_albert_validation_from_topology`.
* README + PROTOCOL: explicit score-level scope boundary; status
  string updated to `HYPOTHESIS / SCORE-LEVEL INSTRUMENTATION
  COMPLETE; END-TO-END VALIDATION PENDING`.
* PR #562 title renamed (post-merge) to remove the
  `production`-prefix overclaim from the public-facing record.

## 2026-05-08 — PR #562 v2 rewrite

* Directed exposure adapter (asymmetric by default).
* MLE-fitted BA null with KS goodness-of-fit and AIC vs
  exponential.
* Asymmetric coupling builder with explicit canonical orientation
  invariant.
* Stratified percentile bootstrap CI on the AUC.
* Bonferroni FWER replacing BH FDR.
* Six pre-registered null surrogate generators.
* Replication manifest (`RunManifest`).
* Canonical docs (`PROTOCOL.md`, `VALIDATION.md`, `LIMITATIONS.md`,
  `data_schema.md`).

## 2026-05-08 — PR #557 initial scaffold

* `BankingCrisisLedger` (Laeven-Valencia 2018 + post-2020
  anchors).
* Synthetic rail validation: lower rail returns `HARD_FAIL`,
  upper rail returns `HARD_PASS`. Both verifications are on
  *synthetic* score series — not empirical evidence.
