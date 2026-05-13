# D-002G — Canonical-run blockers (consolidated)

This document consolidates the open blockers that prevent a canonical D-002G run from being launched on the full pre-registration grid. It is the single fail-loud reference for downstream consumers — if any blocker below is OPEN, the canonical D-002G run is NOT allowed.

Anchor: PR `feat/x10r-d002g-p1-implementation`, P1 implementation report `D002G_P1_IMPLEMENTATION_REPORT.md`.

## Status: CANONICAL D-002G RUN BLOCKED

The canonical D-002G run is BLOCKED until ALL blockers below are CLOSED. P1 ships infrastructure + adversarial-test scaffolding; it does NOT establish D-002G scientific PASS and does NOT launch a canonical run.

## Blockers

### B1 — Substrate eligibility (M1-INELIGIBLE on 2/3 substrates) — PARTIALLY MITIGATED (M2 edge-weight)

Two of three stock substrates are seed-deterministic at λ=0 by design:

| Substrate id        | M1 eligibility   |
|---------------------|------------------|
| `ricci_flow`        | M1-ELIGIBLE      |
| `block_structured`  | M1-INELIGIBLE    |
| `temporal_coupling` | M1-INELIGIBLE    |

Mechanism M1 cannot apply to a seed-deterministic substrate at λ=0 — the precursor and null cohort produce bit-identical K, which is exactly the pathology M1 was designed to remove. `BitIdenticalNullError` fires fail-closed at the realisation layer.

**Resolution required**: implementation of mechanism **M2 — topology-preserving shuffle**, per `D002G_PREREGISTRATION.yaml §4 fallback policy`. Downstream PR tag: **D-002G-P2/M2**.

Until B1 is CLOSED, canonical D-002G is BLOCKED for `block_structured` and `temporal_coupling`. The canonical run could in principle proceed on `ricci_flow` alone, but the pre-registration scopes the canonical grid to all three substrates; partial coverage is NOT a canonical run.

#### B1.M2 — Mitigation status (PR D-002G-P2/M2, edge-weight shuffle domain)

PR `feat/x10r-d002g-p2-m2-topology-preserving-shuffle` ships the M2 topology-preserving shuffle infrastructure plus its eligibility verifier. The empirical verdicts on the canonical grid (`lambda_value=0.4`, `base_seed=42`, locked salt 211) are:

| Substrate id        | M2 (edge_weight) verdict                       | M1 ∪ M2 admissible? |
|---------------------|------------------------------------------------|---------------------|
| `ricci_flow`        | ELIGIBLE_M2                                    | YES (M1 primary, M2 fallback both available) |
| `block_structured`  | INELIGIBLE_M2_DEGENERATE_SHUFFLE_POOL          | NO — canonical run still BLOCKED on this substrate |
| `temporal_coupling` | INELIGIBLE_M2_DEGENERATE_SHUFFLE_POOL          | NO — canonical run still BLOCKED on this substrate |

The `block_structured` and `temporal_coupling` substrates apply a single constant additive lift across their precursor support; the resulting ΔK carries exactly ONE distinct payload value. The M2 edge-weight shuffle is by construction a permutation over the support payload multiset — every permutation of `{v, v, …, v}` is a no-op, and the verifier refuses such cells with `INELIGIBLE_M2_DEGENERATE_SHUFFLE_POOL` fail-closed rather than emit `K_null == K_p` bit-identically.

**Status.** B1 is PARTIALLY MITIGATED — M1 ∪ M2 covers 1/3 substrates (`ricci_flow`). The two constant-payload substrates remain HARD-BLOCKED for the canonical D-002G run. B1 is **NOT** CLOSED.

**Future work**: investigate M2 node-payload and M2 injection-sequence sub-domains (reserved in `M2EligibilityVerdict.shuffle_domain`), OR pre-register a fresh M3 mechanism. Downstream PR tag candidates: **D-002G-P3/M2-node-payload**, **D-002G-P3/M2-injection-sequence**, **D-002G-P3/M3**.

#### B1.P3 — M2 sub-domain extension status (PR D-002G-P3, node-payload + injection-sequence)

PR `feat/x10r-d002g-p3-constant-payload-null-recovery` ships the M2 node-payload and M2 injection-sequence sub-domain verifiers + realisers, plus a full adjudication protocol. The empirical verdicts on the locked prereg grid (`lambda_value=0.4`, `base_seed=42`, `null_seed=12345`) are:

| Substrate id        | M2_NODE_PAYLOAD                              | M2_INJECTION_SEQUENCE                                          |
|---------------------|----------------------------------------------|----------------------------------------------------------------|
| `ricci_flow`        | INELIGIBLE_M2_NODE_PAYLOAD_TOPOLOGY_COUPLED  | INELIGIBLE_M2_INJECTION_SEQUENCE_DEGENERATE                    |
| `block_structured`  | INELIGIBLE_M2_NODE_PAYLOAD_TOPOLOGY_COUPLED  | INELIGIBLE_M2_INJECTION_SEQUENCE_DEGENERATE                    |
| `temporal_coupling` | INELIGIBLE_M2_NODE_PAYLOAD_DEGENERATE_POOL   | INELIGIBLE_M2_INJECTION_SEQUENCE_CONTRACT_VIOLATION            |

Neither sub-domain admits the two constant-payload substrates. Full machine-readable matrix is at `artifacts/d002g/p3/null_domain_verdicts.json`; long-form at `docs/governance/D002G_P3_ELIGIBILITY_MATRIX.md`.

**Status.** B1 remains OPEN — upgraded from `OPEN_PARTIAL` to `OPEN_REQUIRES_M3`. The M1 / M2 admissibility surface is **exhausted** for `block_structured` and `temporal_coupling`. A fresh M3 mechanism family must be pre-registered; the draft pre-registration is `docs/governance/D002G_P3_M3_PREREGISTRATION.md`. Downstream PR tag: **D-002G-M3 implementation**.

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
