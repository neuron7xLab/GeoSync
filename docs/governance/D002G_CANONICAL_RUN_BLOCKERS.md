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

#### B1.M3 — Topology-conditioned null mitigation status (PR D-002G-M3, topology-conditioned independent realisation)

PR `feat/x10r-d002g-m3-topology-conditioned-null` ships the M3 topology-conditioned independent-realisation infrastructure: locked marginal set (degree sequence + block-label histogram + spectral radius / N + density), matched-density generator with a locked iteration cap (`M3_GENERATOR_MAX_ITERATIONS=100`), 5-criterion eligibility ladder, RNG salt 523, and pre-declared tolerance band (`tol_marginal=0.05`, `tol_non_degenerate=1e-3`, `tol_density=0.02`, `tol_spectral_radius=0.05`, `tol_degree_wasserstein=0.05`).

The empirical verdicts on the locked prereg grid (`lambda_value=0.4`, `base_seed=42`, `null_seed=12345`, `N ∈ {50, 100, 200}`) are:

| Substrate id        | M3 verdict (per N)                            | M1 ∪ M2 ∪ M3 admissible? |
|---------------------|-----------------------------------------------|---------------------------|
| `ricci_flow`        | ELIGIBLE_M3 (all N)                           | YES (M1 / M2 edge / M3 all available) |
| `block_structured`  | INELIGIBLE_M3_NON_PRECURSOR_SPECIFIC (all N)  | NO — canonical run still BLOCKED on this substrate |
| `temporal_coupling` | INELIGIBLE_M3_NON_PRECURSOR_SPECIFIC (all N)  | NO — canonical run still BLOCKED on this substrate |

The `block_structured` and `temporal_coupling` substrates produce a seed-deterministic precursor lift by construction; their M3 marginal sets are seed-invariant, so the verifier's criterion 3 (`Identifiable from precursor`) refuses the cell fail-closed (0 / 99 adjacent-seed precursor pairs distinct against a required ≥ 50 / 99). This is the honest scientific outcome — M3 has no right to exist on these substrates under the locked pre-registration discipline. A forced ELIGIBLE_M3 would have demanded post-hoc relaxation of the precursor-specificity criterion or the locked tolerances; both are explicitly forbidden by M3 pre-reg §9.1.

Full machine-readable matrix is at `artifacts/d002g/m3/m3_null_domain_verdicts.json`; long-form at `docs/governance/D002G_M3_ELIGIBILITY_MATRIX.md`.

**Status.** B1 remains OPEN — upgraded from `OPEN_REQUIRES_M3` to `OPEN_REQUIRES_M4`. The M1 ∪ M2 ∪ M3 admissibility surface is **exhausted** for `block_structured` and `temporal_coupling`. A fresh M4 mechanism family must be pre-registered; this PR explicitly does NOT pre-register M4 (touching the M3 pre-reg in this PR would constitute a fresh M4 pre-reg by protocol, which is forbidden — the M4 PR must own its own pre-registration document). Downstream PR tag: **D-002G-M4 pre-registration**.

**Canonical-run status (verbatim):** even though M3 lands ELIGIBLE on `ricci_flow`, canonical run authorisation requires B1 closure (all three substrates ELIGIBLE) AND B2 closure / acceptance AND an explicit canonical-run authorisation artefact. NONE of these conjuncts are satisfied by this PR. Canonical D-002G remains BLOCKED.

#### B1.closure — Structural closure under current locked grid

> After PR #681 (M3 topology-conditioned, merge `cced6e60`) returned `INELIGIBLE_M3_NON_PRECURSOR_SPECIFIC` for both `block_structured` and `temporal_coupling`, the M1/M2/M3 mechanism family is exhausted on the locked substrate grid. The bottom-turtle code fact (`research/systemic_risk/d002c_substrates.py:401`) confirms the substrates discard seed by construction. B1 closure under D-002G is therefore STRUCTURALLY BLOCKED; canonical D-002G run remains BLOCKED. See `D002G_STRUCTURAL_CLOSURE_REPORT.md` for the full closure artifact. Resolution requires fresh D-002H pre-registration (scope narrowing OR substrate redesign), NOT an M4 mechanism inside the current D-002G prereg.

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

---

## D-002G ↔ D-002H lineage transition

D-002G closed structurally per the artifact above. Future work proceeds under a fresh
pre-registered lineage:

- **D-002H** (this PR opens it): `docs/governance/D002H_PREREGISTRATION.yaml` —
  ricci_flow-only canonical run scope, scoped pre-registration. Locked at the merge
  commit of `docs/x10r-d002h-ricci-flow-scope-prereg`.
- D-002G remains sha-pinned as a negative-result artifact; **NO M4 inside D-002G**.
- D-002H authorisation gates A..G live in `D002H_CANONICAL_RUN_AUTHORIZATION_GATES.md`
  and must all PASS before any canonical D-002H run begins.

---

## D-002H Gate B — ricci_flow M1/M3 eligibility reverification

**Status:** PASS
**Artifact:** `artifacts/d002h/eligibility/d002h_ricci_eligibility.json`
**Schema:** D002H-GATE-B-v1
**Report:** `docs/governance/D002H_GATE_B_REPORT.md`
**Merge anchor:** will pin at this PR's merge sha
**Cells:** 18/18 PASS (3 N × 6 λ; λ=0 cells emit `N/A_M3_REQUIRES_LAMBDA_GT_ZERO`
per the M3 module contract; every λ>0 cell ELIGIBLE_M1 AND ELIGIBLE_M3
with marginal-match report inside locked tolerances)

Per `D002H_CANONICAL_RUN_AUTHORIZATION_GATES.md`, Gate B PASS is necessary
but NOT sufficient for canonical D-002H run authorisation. Gates A, B
closed. Gates C, D, E, F, G remain open. Canonical run remains BLOCKED.

---

## D-002H Gate C — canonical parameter grid declared

**Status:** PASS
**Artifact:** `artifacts/d002h/canonical/d002h_canonical_grid.json`
**Schema:** D002H-CANONICAL-GRID-v1
**Report:** `docs/governance/D002H_GATE_C_CANONICAL_GRID.md`

Per `D002H_CANONICAL_RUN_AUTHORIZATION_GATES.md` §C: locked canonical
parameter grid declared as machine-readable artifact + human-readable
report. Grid matches D-002H prereg `canonical_grid` block byte-exact.
Gates A, B, C closed. Gates D, E, F, G remain open. Canonical run
remains BLOCKED until conjunction A ∧ B ∧ C ∧ D ∧ E ∧ F ∧ G.

---

## D-002H Gate D — forbidden-claim scanner

**Status:** PASS (zero leaks across scanned D-002H surface)
**Artifact:** `artifacts/d002h/scans/d002h_forbidden_claim_scan.json`
**Schema:** D002H-GATE-D-v1
**Report:** `docs/governance/D002H_GATE_D_FORBIDDEN_CLAIM_SCAN.md`

Per `D002H_CANONICAL_RUN_AUTHORIZATION_GATES.md` §D. Gates A-D closed.
Gates E, F, G remain open. Canonical run BLOCKED until A∧B∧C∧D∧E∧F∧G.

---

## D-002H Gate E — locked-ledger verification

**Status:** PASS
**Artifact:** `artifacts/d002h/locks/d002h_locked_file_pins.json`
**Schema:** D002H-GATE-E-v1
**Report:** `docs/governance/D002H_GATE_E_LOCKED_FILE_REPORT.md`
**Pinned files:** 16 (D-002C + D-002G + D-002H + source code)

Per §E gates doc. Gates A-E closed. Gates F, G remain open. Canonical
run BLOCKED until A∧B∧C∧D∧E∧F∧G.

---

## D-002H Gate F — canonical-run authorisation artifact

**Status:** AUTHORISED (intermediate; Gate G required for absolute final)
**Artifact:** `artifacts/d002h/authorization/d002h_canonical_run_authorisation.json`
**Schema:** D002H-CANONICAL-RUN-AUTHORISATION-v1
**Report:** `docs/governance/D002H_GATE_F_AUTHORIZATION_REPORT.md`

Conjunction A ∧ B ∧ C ∧ D ∧ E ∧ F all certified PASS at the
gate-F-pin shas (5 anchors verified ancestors of main). Gate G is the
final CI lock; canonical run remains BLOCKED until A∧B∧C∧D∧E∧F∧G.

---

## D-002H Gate G — FINAL CI lock (canonical-run authorisation COMPLETE)

**Status:** PASS (TERMINAL — 7-gate conjunction A∧B∧C∧D∧E∧F∧G all PASS)
**Artifact:** `artifacts/d002h/authorization/d002h_canonical_run_final_lock.json`
**Schema:** D002H-GATE-G-v1
**Report:** `docs/governance/D002H_GATE_G_CI_LOCK_REPORT.md`
**Canonical run authorisation:** GRANTED (scoped to ricci_flow only)
**Canonical run execution status:** NOT STARTED

The 7-gate authorisation contract is now CLOSED. A separate
canonical-sweep PR may execute the D-002H ricci_flow sweep downstream.
This authorisation does NOT itself produce scientific results; the
sweep PR's R1/R2/R3/R2-B/NULL_AUDIT verdict is the scientific output.

B1 (D-002G substrate eligibility): STRUCTURALLY CLOSED (negative artifact #682)
B1 (D-002H ricci_flow eligibility): CLOSED via 7-gate conjunction
B2 (Phase 0b percentile vs BCa): KNOWN LIMITATION, carried per D-002G prereg §4
canonical_run_authorized_final: TRUE  ← terminal state achieved

---

## D-002H R2-B scope clarification

**Status:** RESOLVED — R2-B inherited-but-inapplicable under D-002H
**Document:** `docs/governance/D002H_R2B_INAPPLICABILITY_NOTE.md`
**Resolution rule:** D-002H canonical-run acceptance is the 4-term conjunction R1 ∧ R2 ∧ R3 ∧ NULL_AUDIT (R2-B unevaluable because M6 ∉ D-002H null_mechanisms_allowed).

D-002G acceptance rules byte-exact locked. D-002H prereg byte-exact
locked. This is scope clarification, not contract change. Canonical
run authorisation (Gate G, terminal) remains valid; cell-verdict
computation is now contractually unambiguous.

A future M3-based R2-B analogue would constitute a fresh D-002J
pre-registration.

---

## D-002H canonical-sweep execution (downstream of Gate G + R2-B note)

**Status:** EXECUTED — scientific verdict recorded as a downstream
artifact of the closed 7-gate authorisation conjunction and the R2-B
inapplicability clarification.
**Run ID:** `d002h_ricci_flow_canonical_v1_2026-05-14`
**Authoritative artifact:** `artifacts/d002h/canonical/d002h_canonical_run_verdict.json`
**Report:** `docs/governance/D002H_CANONICAL_RUN_REPORT.md`
**Anchor main SHA:** `ee12a9e6a08e5916109c99eec84796d1e1375cd0`
**Acceptance conjunction:** R1 ∧ R2 ∧ R3 ∧ NULL_AUDIT
(R2-B INAPPLICABLE per `D002H_R2B_INAPPLICABILITY_NOTE.md`).

The canonical sweep executes on the locked 18-cell `ricci_flow` grid
under `M1_INDEPENDENT_SEED` and `M3_TOPOLOGY_CONDITIONED` null
mechanisms (`M6_PLACEBO_COUPLING` is structurally excluded per D-002H
prereg). Whatever the data verdict is — `SYNTHETIC_GATE6_CERTIFIED_D002H_REDESIGN`,
`MARGINAL_PASS_SYNTHETIC_D002H`, `D002H_REDESIGN_INSUFFICIENT_AT_TESTED_BUDGET`,
or `REFUSED_NULL_AUDIT_FAIL_D002H` — that IS the result. Truthful FAIL
is preserved as negative artifact.

D-002C claim ledger byte-exact UNCHANGED at sha256
`f96ba9b5a2057d2e0bff84afc28578ab316cff73f6dc6673fb0d6d543b8bd6dd`.
Any ledger update consuming this verdict is a SEPARATE downstream PR.
Verdict scope: `ricci_flow` only; does NOT extend to `block_structured`
or `temporal_coupling` (structurally excluded by D-002G closure).

---

## D-002H Lineage CLOSED-AS-REFUSED — ledger entry appended

**Status:** D-002H lineage CLOSED with REFUSED verdict
**Ledger entry:** claim_id=D002H_RICCI_FLOW_SCOPED_REFUSED (new, append-only)
**Canonical-run anchor:** PR #691 sha 250d8069
**Aggregate verdict:** REFUSED via NULL_AUDIT_FAIL (42/54 audited cells FAIL)

D-002C ledger entries 1 and 2 (attempt-1 SUPPORTED, attempt-2 FALSIFIED)
byte-exact preserved. D-002H entry is the 3rd and is NOT an eclipse —
D-002H is a scoped lineage opened after D-002G structural closure (PR #682).

Future legal paths:
- D-002I fresh pre-reg to investigate WHY null audit FAILs on ricci_flow
- D-002H retained as scoped-REFUSED negative artifact

---

## D-002I lineage opened — null-audit failure-mode investigation pre-reg

**Status:** D-002I PRE-REG LOCKED (4 falsifiable hypotheses)
**Parent:** D-002H canonical sweep REFUSED (PR #691, sha `250d8069`)
**Parent null-audit aggregate:** 42 / 54 audited cells FAIL on `ricci_flow` under M1 ∪ M3.
**4 hypotheses:** H_I1 (M1 offset), H_I2 (M3 tolerance), H_I3 (signal magnitude), H_I4 (Bonferroni denominator).

Each hypothesis will be implemented in a separate D-002I-P1/Hn PR.
Each produces a SUPPORTED or REFUTED scoped verdict. NO mechanism
change, NO canonical sweep run. D-002I is investigation, not
validation. D-002H REFUSED remains the truthful canonical verdict;
D-002I does NOT retroactively flip it.

If all 4 hypotheses REFUTED → D-002H REFUSED is structural at the
tested signal magnitude on `ricci_flow` under M1 ∪ M3 (mechanism
families M1+M3 fundamentally insufficient at this signal strength);
next step = fresh D-002J pre-reg or retain D-002H as terminal scoped
negative artifact.

D-002G acceptance rules byte-exact UNCHANGED at sha256
`875b1e3eb031b8e5333dc8b455454f0a30419ead1ebe787aa01d5882e7d6ad31`.
D-002H prereg byte-exact UNCHANGED at sha256
`44b18b5a40ce9d188a9c3bd49339621f81a65a15f97a683247902450dd54acec`.
D-002C claim ledger sha256 was `f96ba9b5...d6d543b8bd6dd` at D-002I
prereg-lock anchor (PR #693, 2e55b73a). The D-002H REFUSED entry
append in PR #692 legitimately rotates it to
`eb0b7151...319ef84a32387`; the D-002I investigation is unchanged
in scope or content by that append-only ledger event.

---

## D-002J lineage opened — financial-mechanistic systemic-risk benchmark pre-reg

**Status:** D-002J PRE-REG LOCKED (5 research questions, 7 workstreams, 6 crisis windows)
**Parent:** D-002H canonical sweep REFUSED (PR #692, sha `669d4458`)
**Parent investigation lineage:** D-002I (PR #693, sha `2e55b73a`)
**5 research questions:** RQ1 (substrate signal/null separation), RQ2 (pre-crisis precursor), RQ3 (null-model power against false signal), RQ4 (minimum power budget), RQ5 (negative-result-equal benchmark).
**7 workstreams:** W1 (Data Source Matrix), W2 (Crisis Window Registry), W3 (Synthetic Positive Controls), W4 (Financial Mechanistic Substrates), W5 (Null Model Hierarchy), W6 (Power-First Canonical Design), W7 (Benchmark Package).
**6 crisis windows:** CW1 (2007-2009 GFC), CW2 (2011-2012 Eurozone Sovereign), CW3 (2019 US Repo Spike), CW4 (2020 COVID Dash-for-Cash), CW5 (2022 UK Gilt LDI), CW6 (2023 Regional Banking Stress).

D-002J opens a **fresh research lineage** on financially motivated
substrates with explicit crisis-window anchors, planted positive
controls, a 9-null adversarial hierarchy, and a hard power-first
gate (`requires_power_first_approval: true`). D-002J does **NOT**
rescue D-002H — D-002H REFUSED remains the truthful canonical
verdict; D-002J does NOT retroactively flip it. D-002J also does
NOT pre-empt the D-002I investigation outcomes (H_I1..H_I4 may
land in parallel under their own pre-committed protocol).

`canonical_run_authorized: false`; `benchmark_only: true`;
`requires_power_first_approval: true`. No D-002J canonical sweep
runs until the W6 power-first protocol has emitted a power report
with `power_target ≥ 0.8` and the W3 positive-control battery
has detected its planted signal under the W5 null hierarchy. Each
of the 7 workstreams is implemented in a separate downstream PR
(D-002J-W1..W7) under this pre-committed protocol.

D-002G acceptance rules byte-exact UNCHANGED at sha256
`875b1e3eb031b8e5333dc8b455454f0a30419ead1ebe787aa01d5882e7d6ad31`.
D-002G prereg byte-exact UNCHANGED at sha256
`1ab91f09370e4705a8b0849467bc1f56df2e58d58d5623d3b6d905cbd110bb04`.
D-002H prereg byte-exact UNCHANGED at sha256
`44b18b5a40ce9d188a9c3bd49339621f81a65a15f97a683247902450dd54acec`.
D-002C claim ledger sha256 at D-002J prereg-lock anchor:
`eb0b7151d76e5409e6dc9bb4a023551de5e0704673d5ac9f726319ef84a32387`
(unchanged from the D-002I lineage-opening append).

---

## D-002J-P1 — data source registry v1 landed

**Status:** D-002J-P1 LANDED (registry assembled; benchmark-only; no canonical run)
**Parent:** D-002J prereg PR `docs/x10r-d002j-financial-benchmark-prereg` (PR #694)
**PR:** `feat/x10r-d002j-p1-data-source-registry-v1`
**Registry artifact:** `artifacts/d002j/data_registry/source_registry_v1.json` (schema `D002J-SOURCE-REGISTRY-v1`)
**Summary artifact:** `artifacts/d002j/data_registry/source_registry_summary_v1.json` (schema `D002J-SOURCE-REGISTRY-SUMMARY-v1`)
**Total sources:** 25 documented (23 USABLE_NOW, 2 CANDIDATE_REQUIRES_LICENSE_REVIEW, 0 REJECTED)
**Source classes:** banking 4, repo 4, macro_financial 5, market_structure 4, crisis_window 5, literature_support 3
**Crisis-window coverage:** CW1 20, CW2 15, CW3 16, CW4 18, CW5 11, CW6 15 (floor 5 each — PASS)
**Registry decision:** `DATA_REGISTRY_READY` (all floors satisfied; rationale recorded in summary `decision_rationale`)

D-002J-P1 ships the **data source registry v1** under W1 scope: a
machine-readable JSON registry, a JSON summary, a crisis-window
coverage matrix (§10 in `D002J_DATA_SOURCE_MATRIX.md`), a
mechanism-coverage matrix (§11), 8 filled source cards
(`D002J_DATA_SOURCE_CARD.md`), and a full §9 selection rationale
(`D002J_SOURCE_SELECTION_RATIONALE.md`) covering all 25 sources.

Hard scope boundary (repeat for safety):

- D-002J-P1 is **registry only**. P1 does **NOT** ingest any data.
- D-002J-P1 does NOT rescue D-002H. D-002H REFUSED remains the
  truthful canonical verdict.
- D-002J-P1 does **NOT** authorise any canonical run.
  `canonical_run_authorized: false`; `benchmark_only: true`.
- D-002J-P1 does **NOT** claim real-bank validation.
- D-002J-P1 does **NOT** pre-empt the D-002I investigation outcomes.
- D-002J-P1 does **NOT** edit any locked governance file:
  D-002G prereg sha256 byte-exact
  `1ab91f09370e4705a8b0849467bc1f56df2e58d58d5623d3b6d905cbd110bb04`,
  D-002G acceptance rules sha256 byte-exact
  `875b1e3eb031b8e5333dc8b455454f0a30419ead1ebe787aa01d5882e7d6ad31`,
  D-002H prereg sha256 byte-exact
  `44b18b5a40ce9d188a9c3bd49339621f81a65a15f97a683247902450dd54acec`,
  D-002C claim ledger sha256 byte-exact
  `eb0b7151d76e5409e6dc9bb4a023551de5e0704673d5ac9f726319ef84a32387`,
  D-002J prereg sha256 byte-exact at the P0 anchor
  `f3dc65b7e64b96eafe6f23ca8bdd0e05dc9bf95b12c2658b227bd0340f7975a0`.

Next legal PR: `feat(x10r,D-002J-P2): implement crisis window registry v1` —
expands W2 (machine-readable crisis-window registry on top of the
narrative `D002J_CRISIS_WINDOW_REGISTRY.md` scaffold).
