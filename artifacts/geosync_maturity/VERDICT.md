# GeoSync Measured-Maturity — Verdict

**Status:** `PARTIAL`
**Objective:** upgrade verification-first infrastructure toward measured empirical maturity
**Generated from:** real repo state on branch `feat/ricci-microstructure-preregistration`

## Why PARTIAL (not PASS, not FAILED)

The audit completed against the real repository and the maturity controls were
strengthened, but the headline L2 Ricci line **cannot be measured** until a real,
hash-pinned order-book session lands. This is the protocol's defined PARTIAL case
("audit completed but real L2 data is unavailable; claims registered but not
measured; falsifiers specified but not all executable").

Nothing was promoted. `evaluate_promotion()` on the committed placeholder returns
`HYPOTHESIS` / not-allowed with three named blockers (zero data hash, synthetic
data, insufficient depth). The verdict reports exactly that.

## Readiness rubric (honest, no inflation)

| Dimension | Max | Score | Basis |
| --- | --- | --- | --- |
| claim_discipline | 20 | 19 | guard + FORBIDDEN_CLAIMS + prereg + market limits, all fail-closed |
| L2_Ricci_grounding | 20 | 10 | full pipeline + schemas + guard present; **no real data → INSTRUMENTED only** |
| multi_session_falsifiers | 20 | 12 | 6-class matrix + 4 nulls registered; **not executable without real sessions** |
| physics_contract_limits | 15 | 14 | 8 explicit market approximation limits added (`market_limits.yaml`) |
| market_specific_invariants | 10 | 8 | limits carry valid_when/invalid_when/falsifier + tier ceilings |
| real_data_pipeline_maturity | 10 | 6 | L2 collectors + full-cycle scripts exist; no real session committed |
| cost_and_drift_awareness | 5 | 5 | cost model in config; non-stationarity/drift named in limits |
| **Total** | **100** | **74** | band 0–79 = `NOT_READY` as a *maturity* label |

Two axes, kept distinct:
- **Execution status of this run:** `PARTIAL` (audit → honest patches landed).
- **Empirical maturity of the L2 Ricci claim:** `NOT_READY` / `Instrumented` —
  it has not been measured and this run does not pretend it has.

## What this run added (machine-checkable)

- `physics_contracts/market_limits.yaml` — 8 market approximation limits
  (non-conservative, non-stationary, agent-driven, liquidity-discontinuous,
  regime-shock, open-system order flow, correlation≠causal,
  descriptor≠predictor), each with valid_when/invalid_when, a required witness,
  a required null, forbidden claims, and a tier ceiling.
- `tools/research/market_limits.py` — fail-closed typed loader/validator.
- `tests/physics_contracts/test_market_limits.py` — 16 witnesses (no dangling
  witness reference; every falsifier is a registered null; tiers are canonical).
- `tools/research/build_geosync_maturity_report.py` — deterministic generator;
  re-run after real data lands to record the earned tier with no hand-edits.
- `artifacts/geosync_maturity/*.json` — the evidence bundle below.

## Not claimed

profitability · market prediction certainty · financial advice ·
closed-form market physics · causal structure from correlation graphs.
