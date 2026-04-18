# CHANGELOG — L2 Ricci cross-sectional edge

Chronological record of every PR that shaped the 10-axis + 5-ablation
Ricci research stack on Binance USDT-M perp L2 substrate.

---

## 2026-04-18 · Demo-readiness session

End-to-end integration from 8-axis narrative to canonical, self-verifying,
demo-shippable package.

### Validation layer — 10 orthogonal axes

- **PR #268** · `feat(robustness)` — Politis-Romano block bootstrap 95% CI +
  Lopez-de-Prado deflated Sharpe + Augmented Dickey-Fuller + mutual
  information (4 axes in one module: `research/microstructure/robustness.py`)
- **PR #270** · `feat(cv)` — purged & embargoed K-fold CV (AFML Ch. 7);
  5/5 folds positive, mean IC = 0.122
- **PR #271** · `feat(spectral)` — Welch PSD, redness slope β = 1.80
- **PR #272** · `feat(regime-markov)` — 6-state transition matrix,
  mean diagonal = 0.832
- **PR #273** · `feat(hurst)` — DFA-1 Hurst, H = 1.014, R² = 0.982
- **PR #274** · `feat(te)` — pairwise Transfer Entropy, 45/45 BIDIRECTIONAL
- **PR #276** · `feat(cte)` — Conditional Transfer Entropy (BTC-conditioned),
  33/36 PRIVATE_FLOW — rules out common-factor artifact
- **PR #280** · `feat(walk-forward)` — rolling temporal-stability summary,
  82.1% windows positive, STABLE_POSITIVE verdict

### Execution layer

- **PR #266** · `feat(diurnal-filter)` — sign-aware per-row direction filter
- **PR #269** · `feat(pnl)` — cost sweep + break-even for REGIME_Q75+DIURNAL,
  f* = 0.23167 (canonical gate fixture)

### Synthesis + demo artifacts

- **PR #275** · `docs(findings)` — 8-axis consolidated narrative
- **PR #278** · `feat(demo)` — three canonical figures + manifest + runner
- **PR #279** · `docs(readme)` — L2 microstructure section
- **PR #281** · `feat(visualize)` — fig4_stability walk-forward panel
- **PR #282** · `feat(visualize)` — fig0_cover single-page demo poster
- **PR #297** · `feat(demo)` — self-contained HTML dashboard (7.2 KB)
- **PR #300** · `feat(make)` — pro-max ergonomic Makefile targets

### Ablation / stress layer — 5 axes

- **PR #290** · `feat(ablation)` — hyperparameter (regime-q × window) sweep →
  **SENSITIVE** (f* drifts ±60%, but all 9 cells below production ceiling)
- **PR #293** · `feat(ablation)` — leave-one-symbol-out → **MIXED**
  (BTC removal drops IC 43%; all 10 cells still positive)
- **PR #295** · `feat(ablation)` — hold-time (60–600 s) → **ROBUST**
  (3/5 cells already profitable at f = 0)
- **PR #296** · `feat(stress)` — slippage stress (±bp/side) → **BOUND**
  (max viable +3 bp/side; typical prod +0.5–1.5 bp)
- **PR #298** · `feat(stress)` — fee-tier sensitivity → **RESILIENT**
  (all 4 VIP tiers bracket below 0.50)

### Coherence / integrity gates

- **PR #286** · `test(coherence)` — 7 independent gate suites
  (deterministic replay, doc-data, per-axis invariants, schema registry,
  CLI discoverability, performance budget, E2E demo smoke)
- **PR #288** · `test(property-based)` — Hypothesis coverage for
  DFA Hurst, TE, CTE, walk-forward

### Final state

- **10 validation axes**, all green on Session 1
- **5 ablation / stress axes** with honest verdicts (SENSITIVE / MIXED /
  ROBUST / BOUND / RESILIENT)
- **5 canonical figures** + HTML dashboard
- **1 one-command runner** + SHA-256 manifest (81 s end-to-end)
- **300+ L2 tests** passing
- **Deterministic replay** confirmed bit-exact across two runs

Canonical entry point: `make l2-demo`
Synthesis document: `research/microstructure/FINDINGS.md`
