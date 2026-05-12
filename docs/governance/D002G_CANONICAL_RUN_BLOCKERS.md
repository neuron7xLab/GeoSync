# D-002G — Canonical-run blockers (consolidated)

This document consolidates the open blockers that prevent a canonical D-002G run from being launched on the full pre-registration grid. It is the single fail-loud reference for downstream consumers — if any blocker below is OPEN, the canonical D-002G run is NOT allowed.

Anchor: PR `feat/x10r-d002g-p1-implementation`, P1 implementation report `D002G_P1_IMPLEMENTATION_REPORT.md`.

## Status: CANONICAL D-002G RUN BLOCKED

The canonical D-002G run is BLOCKED until ALL blockers below are CLOSED. P1 ships infrastructure + adversarial-test scaffolding; it does NOT establish D-002G scientific PASS and does NOT launch a canonical run.

## Blockers

### B1 — Substrate eligibility (M1-INELIGIBLE on 2/3 substrates) — OPEN

Two of three stock substrates are seed-deterministic at λ=0 by design:

| Substrate id        | M1 eligibility   |
|---------------------|------------------|
| `ricci_flow`        | M1-ELIGIBLE      |
| `block_structured`  | M1-INELIGIBLE    |
| `temporal_coupling` | M1-INELIGIBLE    |

Mechanism M1 cannot apply to a seed-deterministic substrate at λ=0 — the precursor and null cohort produce bit-identical K, which is exactly the pathology M1 was designed to remove. `BitIdenticalNullError` fires fail-closed at the realisation layer.

**Resolution required**: implementation of mechanism **M2 — topology-preserving shuffle**, per `D002G_PREREGISTRATION.yaml §4 fallback policy`. Downstream PR tag: **D-002G-P2/M2**.

Until B1 is CLOSED, canonical D-002G is BLOCKED for `block_structured` and `temporal_coupling`. The canonical run could in principle proceed on `ricci_flow` alone, but the pre-registration scopes the canonical grid to all three substrates; partial coverage is NOT a canonical run.

### B2 — Phase 0b CI is percentile bootstrap, not BCa — OPEN (limitation)

The Phase 0b verdict-grade CI on the per-seed paired-difference mean is a percentile bootstrap CI (P1-3 Codex review, Path 2 downgrade). True BCa (bias-corrected accelerated) bootstrap CI was advertised in the original implementation docstring + adversarial audit narrative; the implementation always was percentile.

**Resolution path**: this is documented as a LIMITATION rather than a hard block. The Wilcoxon signed-rank gate (non-parametric, in the conjunction) absorbs most of the skew/bounded calibration concern. True BCa is future hardening; it can land in a downstream PR independent of M2.

Canonical D-002G can in principle launch with B2 OPEN provided the limitation is acknowledged in the run report. **For ABSOLUTE certainty on the skew/bounded calibration**, BCa is required.

## Closure preconditions

Before a canonical D-002G run is allowed:

1. **B1 CLOSED** — M2 mechanism implemented, Phase 0a / 0b / 0c verified on all three substrates, eligibility table updated to show 3/3 M1-or-M2 ELIGIBLE.
2. **B2 acknowledged or CLOSED** — either accept the percentile-CI limitation in the run report or replace with BCa.
3. **Locked governance unchanged** — all 8 sha-pinned files in `tests/systemic_risk/test_d002g_locked_governance_untouched.py` still PASS.

When all three preconditions are met, a canonical D-002G run may be launched. The current P1 PR does NOT meet them and does NOT claim D-002G scientific PASS.

## Claim boundary (verbatim)

> The D-002G P1 PR implements infrastructure and adversarial test scaffolding only. It does NOT establish D-002G scientific PASS. Phase 0 test-suite results are INFRASTRUCTURE SMOKE, not canonical Phase 0 verdict. A fresh canonical D-002G run on prereg-scoped substrates is required before any tier-PASS claim, and is BLOCKED on M2 until the substrate-eligibility gap is closed in a downstream D-002G-P2/M2 PR.
