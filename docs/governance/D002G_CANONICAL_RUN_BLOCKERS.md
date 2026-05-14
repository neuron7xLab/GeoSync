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

---

## D-002J-P1A — source registry provenance audit landed (REJECTED)

**Status:** D-002J-P1A LANDED (audit-only; decision `SOURCE_REGISTRY_REJECTED`)
**Parent:** D-002J-P1 (PR #695) registry sha256 byte-exact `0fae24d4c3ef3165509166bec89d6dc5eee806888f352358ad77851e51079b7b`
**PR:** `audit/x10r-d002j-p1a-source-provenance-audit-v1`
**Audit artifact:** `artifacts/d002j/data_registry/source_provenance_audit_v1.json` (schema `D002J-SOURCE-PROVENANCE-AUDIT-v1`)
**Smoke artifact:** `artifacts/d002j/data_registry/source_access_smoke_v1.json` (schema `D002J-SOURCE-ACCESS-SMOKE-v1`)
**Evidence-lock artifact:** `artifacts/d002j/data_registry/source_evidence_lock_v1.json` (schema `D002J-SOURCE-EVIDENCE-LOCK-v1`)
**Summary artifact:** `artifacts/d002j/data_registry/source_registry_audit_summary_v1.json` (schema `D002J-SOURCE-REGISTRY-AUDIT-SUMMARY-v1`)
**Audit decision:** `SOURCE_REGISTRY_REJECTED`
**Counts:** 13 VERIFIED, 7 PARTIAL, 5 DOWNGRADED, 0 REJECTED (verified_or_partial=20 ≥ 18 floor PASS)
**verified_usable_now:** 13 (≥ 12 floor PASS)
**Crisis-window retention (≥3 verified/partial each):** CW1 17, CW2 12, CW3 13, CW4 14, CW5 8, CW6 12 — PASS
**Mechanism-family retention (≥2 verified/partial each):** contagion 4, liquidity_funding 7, balance_sheet 3, market_wide_stress 6, official_response 5, **information_constraint 1 — FAIL**
**Downgraded source_ids:** ECB_CBD, ICAP_MOVE, BIS_QR_NETWORK, FED_TIMELINE, BOE_LDI_REVIEW
**Partial source_ids:** FED_Y9C, ECB_MMSR, ALFRED, KCFSI, CBOE_VIX, FDIC_SVB_POSTMORTEM, LIT_REPO_FUNDING

D-002J-P1A is the first verification gate of the D-002J Frontier
Benchmark Program. P1A audits the 25 sources committed by PR #695
against ten verification dimensions (provider, official URL,
documentation, access method, license boundary, coverage, frequency,
variables, crisis-window relevance, mechanistic relevance). The
audit verdict is **`SOURCE_REGISTRY_REJECTED`** because the
`information_constraint` mechanism family has only one source
(ALFRED, PARTIAL) and the floor of ≥2 verified/partial-per-family is
not satisfied. Rejection is a scientifically valid outcome — the
audit system working as designed.

Hard scope boundary (repeat for safety):

- D-002J-P1A is **audit only**. P1A does **NOT** ingest any data.
- D-002J-P1A does **NOT** authorise any canonical run.
  `canonical_run_authorized: false`.
- D-002J-P1A does **NOT** claim real-bank validation.
- D-002J-P1A does **NOT** rescue D-002H. D-002H REFUSED remains the
  truthful canonical verdict.
- D-002J-P1A does **NOT** pre-empt the D-002I investigation outcomes.
- D-002J-P1A does **NOT** add new sources or modify the P1 registry
  (registry stays byte-exact at sha
  `0fae24d4c3ef3165509166bec89d6dc5eee806888f352358ad77851e51079b7b`).
- D-002J-P1A does **NOT** edit any locked governance file:
  D-002G prereg sha256 byte-exact
  `1ab91f09370e4705a8b0849467bc1f56df2e58d58d5623d3b6d905cbd110bb04`,
  D-002G acceptance rules sha256 byte-exact
  `875b1e3eb031b8e5333dc8b455454f0a30419ead1ebe787aa01d5882e7d6ad31`,
  D-002H prereg sha256 byte-exact
  `44b18b5a40ce9d188a9c3bd49339621f81a65a15f97a683247902450dd54acec`,
  D-002I prereg sha256 byte-exact
  `b646989c032dc0e29f9b791e0b68209ff22b40f4757737712badc8656cf2db5f`,
  D-002J prereg sha256 byte-exact
  `f3dc65b7e64b96eafe6f23ca8bdd0e05dc9bf95b12c2658b227bd0340f7975a0`,
  D-002C claim ledger sha256 byte-exact
  `eb0b7151d76e5409e6dc9bb4a023551de5e0704673d5ac9f726319ef84a32387`.
- D-002J-P1A does **NOT** edit any source code under
  `research/systemic_risk/*.py` or any `scripts/x10r_d002*.py`.

Lineage: `D-002G → D-002H REFUSED → D-002I → D-002J prereg #694 → P1 #695 → P1A this PR (REJECTED at mechanism-family floor)`.

Next legal PR: `fix(x10r,D-002J-P1): repair source registry provenance` —
either adds ≥1 new `information_constraint` source (e.g. Philadelphia
Fed Real-Time Data Set, Atlanta Fed GDPNow vintages, BIS long-series
real-time data) OR amends the prereg's mechanism-family floor with
explicit, dated, signed justification AND re-pins the five
DOWNGRADED URL fields. P2 (`feat(x10r,D-002J-P2): implement crisis
window registry v1`) cannot open until the audit decision flips to
`SOURCE_REGISTRY_VERIFIED` or `SOURCE_REGISTRY_PARTIALLY_VERIFIED`.

## D-002J-P1B — source registry provenance repair landed (PARTIALLY_VERIFIED)

**Status:** D-002J-P1B LANDED (registry repair; decision `SOURCE_REGISTRY_PARTIALLY_VERIFIED`)
**Parent:** D-002J-P1A (PR #697) `SOURCE_REGISTRY_REJECTED` at merge sha `4b64faf67f4c1bec48a66d20eeddbdf6931e762d`
**PR:** `fix/x10r-d002j-p1b-source-registry-provenance-repair`
**Registry artifact (P1B):** `artifacts/d002j/data_registry/source_registry_v1.json` (sha256 `570ca2e219a8a398f9e6819516905623d73d08c7c135d2f6048686b46f5dbbf8`)
**Audit artifact:** `artifacts/d002j/data_registry/source_provenance_audit_v1.json` (schema `D002J-SOURCE-PROVENANCE-AUDIT-v1`)
**Smoke artifact:** `artifacts/d002j/data_registry/source_access_smoke_v1.json` (schema `D002J-SOURCE-ACCESS-SMOKE-v1`)
**Evidence-lock artifact:** `artifacts/d002j/data_registry/source_evidence_lock_v1.json` (schema `D002J-SOURCE-EVIDENCE-LOCK-v1`)
**Summary artifact:** `artifacts/d002j/data_registry/source_registry_audit_summary_v1.json` (schema `D002J-SOURCE-REGISTRY-AUDIT-SUMMARY-v1`)
**Audit decision:** `SOURCE_REGISTRY_PARTIALLY_VERIFIED`
**Counts:** 18 VERIFIED, 8 PARTIAL, 0 DOWNGRADED, 0 REJECTED (verified_or_partial=26 ≥ 18 floor PASS)
**verified_usable_now:** 18 (≥ 12 floor PASS)
**Crisis-window retention (≥3 verified/partial each):** CW1 21, CW2 16, CW3 17, CW4 19, CW5 12, CW6 16 — PASS
**Mechanism-family retention (≥2 verified/partial each):** balance_sheet 4, contagion 5, **information_constraint 2 (was 1 — now PASS)**, liquidity_funding 7, market_wide_stress 7, official_response 7 — ALL PASS
**P1B repair outcomes (5 URLs + 1 new source):**
- ECB_CBD: `REPIN_CANONICAL_URL` → `https://data.ecb.europa.eu/data/datasets/CBD2` (audit_status DOWNGRADED → VERIFIED)
- ICAP_MOVE: `REPIN_CANONICAL_URL` → `https://www.theice.com/iba` (audit_status DOWNGRADED → PARTIAL — methodology PDF unrecoverable; license-bound)
- BIS_QR_NETWORK: `REPIN_CANONICAL_URL` → `https://www.bis.org/publ/quarterly.htm` (audit_status DOWNGRADED → VERIFIED)
- FED_TIMELINE: `REPIN_CANONICAL_URL` → `https://www.federalreserve.gov/publications/financial-stability-report.htm` + `https://www.federalreservehistory.org/essays/great-recession-of-200709` (audit_status DOWNGRADED → VERIFIED)
- BOE_LDI_REVIEW: `REPIN_CANONICAL_URL` → `https://www.bankofengland.co.uk/financial-stability-report/2022/december-2022` (audit_status DOWNGRADED → VERIFIED)
- PHILLY_FED_RTDSM: `ADDED_NEW_INFORMATION_CONSTRAINT_SOURCE` (new VERIFIED source, satisfies information_constraint mech-family floor; complement to ALFRED)

D-002J-P1B is the surgical repair gate following P1A. P1B repairs the
structural defects surfaced by P1A's audit (one mechanism-family floor
failure + five broken URL pins) WITHOUT weakening the rules, WITHOUT
amending the D-002J prereg, and WITHOUT folding the taxonomy. Every
URL repair was HEAD-verified at audit time.

Hard scope boundary (repeat for safety):

- D-002J-P1B is **registry repair only**. P1B does **NOT** ingest any data.
- D-002J-P1B does **NOT** authorise any canonical run.
  `canonical_run_authorized: false`.
- D-002J-P1B does **NOT** claim real-bank validation.
- D-002J-P1B does **NOT** rescue D-002H. D-002H REFUSED remains the
  truthful canonical verdict.
- D-002J-P1B does **NOT** pre-empt the D-002I investigation outcomes.
- D-002J-P1B does **NOT** weaken the prereg `information_constraint`
  mechanism-family floor (≥ 2 verified/partial preserved).
- D-002J-P1B does **NOT** collapse the taxonomy (information_constraint
  retained as a separate mechanism family).
- D-002J-P1B does **NOT** edit any locked governance file:
  D-002G prereg sha256 byte-exact
  `1ab91f09370e4705a8b0849467bc1f56df2e58d58d5623d3b6d905cbd110bb04`,
  D-002G acceptance rules sha256 byte-exact
  `875b1e3eb031b8e5333dc8b455454f0a30419ead1ebe787aa01d5882e7d6ad31`,
  D-002H prereg sha256 byte-exact
  `44b18b5a40ce9d188a9c3bd49339621f81a65a15f97a683247902450dd54acec`,
  D-002I prereg sha256 byte-exact
  `b646989c032dc0e29f9b791e0b68209ff22b40f4757737712badc8656cf2db5f`,
  D-002J prereg sha256 byte-exact
  `f3dc65b7e64b96eafe6f23ca8bdd0e05dc9bf95b12c2658b227bd0340f7975a0`,
  D-002C claim ledger sha256 byte-exact
  `eb0b7151d76e5409e6dc9bb4a023551de5e0704673d5ac9f726319ef84a32387`.
- D-002J-P1B does **NOT** edit any source code under
  `research/systemic_risk/*.py` or any `scripts/x10r_d002*.py`.
- D-002J-P1B does **NOT** rewrite the P1A REJECTED audit verdict —
  the P1A section in `docs/research/D002J_SOURCE_DOWNGRADE_LOG.md` is
  preserved verbatim as banked truth.

Lineage: `D-002G → D-002H REFUSED → D-002I → D-002J prereg #694 → P1 #695 → P1A #697 REJECTED → P1B this PR (PARTIALLY_VERIFIED)`.

Next legal PR: `feat(x10r,D-002J-P2): implement crisis window registry v1` —
opens W2 of the D-002J Frontier Benchmark Program now that the
provenance audit decision has flipped to PARTIALLY_VERIFIED.

---

## D-002J-P2 — crisis window registry v1 landed

**Status:** D-002J-P2 LANDED (registry assembled; benchmark-only; no canonical run)
**Parent:** D-002J-P1B (PR #698) PARTIALLY_VERIFIED at merge sha `5102b008ee9fb89545f22ebafc8289aa40c0571c`
**PR:** `feat/x10r-d002j-p2-crisis-window-registry-v1`
**Registry artifact (P2):** `artifacts/d002j/crisis_windows/crisis_window_registry_v1.json` (schema `D002J-CRISIS-WINDOW-REGISTRY-v1`)
**Summary artifact:** `artifacts/d002j/crisis_windows/crisis_window_summary_v1.json` (schema `D002J-CRISIS-WINDOW-SUMMARY-v1`)
**Parent registry sha256:** `f1899b7a882b4b3efbebb54e3dc942c079839f77f981273e2dd09757973b14ec` (P1B source_registry_v1.json)
**Total windows:** 6 (CW1_GFC_2007_2009, CW2_EUROZONE_2011_2012, CW3_US_REPO_SPIKE_2019, CW4_COVID_DASH_FOR_CASH_2020, CW5_UK_GILT_LDI_2022, CW6_REGIONAL_BANKING_2023)
**Total distinct source_ids referenced:** 26 (every P1B-surviving source touched by ≥ 1 window; no DOWNGRADED, no REJECTED)
**Per-window source counts:** CW1 21, CW2 16, CW3 17, CW4 19, CW5 12, CW6 16 — all ≥ floor of 3 (PASS)
**Event-type distribution:** systemic_banking_crisis 1, sovereign_debt_crisis 1, repo_market_dysfunction 1, liquidity_crisis 1, gilt_dysfunction 1, regional_banking_crisis 1
**Primary mechanism family distribution:** liquidity_funding 3, contagion 1, market_wide_stress 1, balance_sheet 1
**Data availability distribution:** strong 4 (CW1, CW2, CW3, CW4), partial 2 (CW5, CW6)
**Registry decision:** `CRISIS_WINDOW_REGISTRY_READY` (all per-window floors and forbidden-claim guards satisfied)

D-002J-P2 ships the **crisis window registry v1** under W2 scope: a
machine-readable JSON registry, a JSON summary, a per-window
narrative (`docs/research/D002J_CRISIS_WINDOW_REGISTRY.md`), and a
selection rationale (`docs/research/D002J_CRISIS_WINDOW_SELECTION_RATIONALE.md`)
covering the 6 chosen windows AND the explicit non-windows
(LTCM 1998, 1987 Black Monday, 1990-91 S&L, 1997-98 Asian, Cyprus 2013,
2016 Brexit, 2008-09-15 single-day, 2008-Q4 single-quarter,
2014-15 oil collapse, 2018-Q4 mild repo, 2021-Q1 SLR-relief,
2023 Credit Suisse as own window, 2023-Q2-Q3 PacWest / WAL).

Hard scope boundary (repeat for safety):

- D-002J-P2 is **registry only**. P2 does **NOT** ingest any data.
- D-002J-P2 does **NOT** rescue D-002H. D-002H REFUSED remains the
  truthful canonical verdict.
- D-002J-P2 does **NOT** authorise any canonical run.
  `canonical_run_authorized: false`; `benchmark_only: true`.
- D-002J-P2 does **NOT** claim crisis prediction at any window.
- D-002J-P2 does **NOT** claim bank-level validation at any window.
- D-002J-P2 does **NOT** claim cross-asset / interbank causal
  inference at any window (Brunetti e-MID literature scope reminder
  enforced via `test_no_cross_asset_interbank_overclaim`).
- D-002J-P2 does **NOT** pre-empt the D-002I investigation outcomes.
- D-002J-P2 does **NOT** modify the P1B registry, audit, smoke, or
  evidence lock JSON artifacts (P1B registry sha256 byte-exact
  `f1899b7a882b4b3efbebb54e3dc942c079839f77f981273e2dd09757973b14ec`).
- D-002J-P2 does **NOT** edit any locked governance file:
  D-002G prereg sha256 byte-exact
  `1ab91f09370e4705a8b0849467bc1f56df2e58d58d5623d3b6d905cbd110bb04`,
  D-002G acceptance rules sha256 byte-exact
  `875b1e3eb031b8e5333dc8b455454f0a30419ead1ebe787aa01d5882e7d6ad31`,
  D-002H prereg sha256 byte-exact
  `44b18b5a40ce9d188a9c3bd49339621f81a65a15f97a683247902450dd54acec`,
  D-002I prereg sha256 byte-exact
  `b646989c032dc0e29f9b791e0b68209ff22b40f4757737712badc8656cf2db5f`,
  D-002J prereg sha256 byte-exact
  `f3dc65b7e64b96eafe6f23ca8bdd0e05dc9bf95b12c2658b227bd0340f7975a0`,
  D-002C claim ledger sha256 byte-exact
  `eb0b7151d76e5409e6dc9bb4a023551de5e0704673d5ac9f726319ef84a32387`.
- D-002J-P2 does **NOT** edit any source code under
  `research/systemic_risk/*.py` or any `scripts/x10r_d002*.py`.
- D-002J-P2 does **NOT** create `artifacts/d002j/ingestion/`
  (that directory is P3 territory; its absence is asserted in
  `test_no_ingestion`).

Lineage: `D-002G → D-002H REFUSED → D-002I → D-002J prereg #694 → P1 #695 → P1A #697 REJECTED → P1B #698 PARTIALLY_VERIFIED → P2 this PR (CRISIS_WINDOW_REGISTRY_READY)`.

Next legal PR: `feat(x10r,D-002J-P3): implement ingestion manifest and point-in-time adapter boundary` —
opens W3/W4 of the D-002J Frontier Benchmark Program (ingestion
manifest, vintage-aware adapters) on top of this v1 crisis-window
registry. P3 may only open after this P2 PR is merged with decision
`CRISIS_WINDOW_REGISTRY_READY`.

---

## D-002J-P2.5 — verdict DAG bootstrap landed

**Status:** D-002J-P2.5 LANDED (verdict DAG self-anchor; governance infrastructure only; no canonical run)
**Parent:** D-002J-P2 (PR #699) CRISIS_WINDOW_REGISTRY_READY at merge sha `055783785571f68a4e0b07206e08c72a8c928e7c`
**PR:** `chore/x10r-d002j-p25-verdict-dag-bootstrap`
**Capsule artifacts (P2.5):** `artifacts/governance/verdicts/d002j_p{0,1,1a,1b,2}_verdict_v1.json` (schema `D002J-VERDICT-CAPSULE-v1`)
**DAG artifact:** `artifacts/governance/verdicts/d002j_verdict_dag_v1.json` (schema `D002J-VERDICT-DAG-v1`)
**Renderer:** `tools/governance/render_lineage.py` → `docs/research/D002J_LINEAGE_MAP.md` (byte-deterministic)
**Tool module:** `tools/governance/verdict_dag.py` (`load_capsule`, `load_dag`, `check_acyclic`, `topological_order`, `detect_orphans`, `emit_dag_verdict`, `check_legal_transition`)
**Decision:** `VERDICT_DAG_BOOTSTRAPPED`
**DAG state:**

- Nodes: 5 (`D002J-P0`, `D002J-P1`, `D002J-P1A`, `D002J-P1B`, `D002J-P2`).
- Topological order: `D002J-P0 → D002J-P1 → D002J-P1A → D002J-P1B → D002J-P2`.
- Acyclic: `true`. Orphans: `0`.
- Retained rejected nodes: `["D002J-P1A"]` (`TERMINAL_REJECTED`,
  `failure_retention` non-empty, repaired by `D002J-P1B`).
- `D002J-P2.parent_nodes == ["D002J-P1B"]` — P2 depends on the
  repaired registry, NOT on P1A directly.
- `canonical_run_authorized_anywhere`: `false` (every capsule
  carries a `claim_boundary` encoding the no-canonical-run
  invariant; the DAG verdict records the conjunction explicitly).
- DAG self-verdict node: `D002J-P2.5` (`VERDICT_DAG_BOOTSTRAPPED`,
  `TERMINAL_PASS`).

D-002J-P2.5 ships the **verdict DAG self-anchor** under governance
infrastructure scope: a frozen capsule schema, a strict-typed
parser/walker, a deterministic markdown renderer, and a 16-test
integrity guard locking the four transition invariants (acyclicity,
P1A retention, P1B repair edge, P2-not-from-P1A-directly).

Hard scope boundary (repeat for safety):

- D-002J-P2.5 is **governance infrastructure only**. P2.5 does
  **NOT** ingest any data.
- D-002J-P2.5 does **NOT** rescue D-002H. D-002H REFUSED remains
  the truthful canonical verdict.
- D-002J-P2.5 does **NOT** authorise any canonical run anywhere.
  `canonical_run_authorized_anywhere: false`.
- D-002J-P2.5 does **NOT** claim crisis prediction.
- D-002J-P2.5 does **NOT** claim bank-level validation.
- D-002J-P2.5 does **NOT** rewrite the P1A REJECTED audit verdict —
  the P1A capsule preserves `TERMINAL_REJECTED` and a non-empty
  `failure_retention` field.
- D-002J-P2.5 does **NOT** edit the P0/P1/P1A/P1B/P2 source
  summary artifacts under `artifacts/d002j/{prereg,data_registry,crisis_windows}/`.
- D-002J-P2.5 does **NOT** edit any locked governance file:
  D-002G prereg sha256 byte-exact
  `1ab91f09370e4705a8b0849467bc1f56df2e58d58d5623d3b6d905cbd110bb04`,
  D-002G acceptance rules sha256 byte-exact
  `875b1e3eb031b8e5333dc8b455454f0a30419ead1ebe787aa01d5882e7d6ad31`,
  D-002H prereg sha256 byte-exact
  `44b18b5a40ce9d188a9c3bd49339621f81a65a15f97a683247902450dd54acec`,
  D-002I prereg sha256 byte-exact
  `b646989c032dc0e29f9b791e0b68209ff22b40f4757737712badc8656cf2db5f`,
  D-002J prereg sha256 byte-exact
  `f3dc65b7e64b96eafe6f23ca8bdd0e05dc9bf95b12c2658b227bd0340f7975a0`,
  D-002C claim ledger sha256 byte-exact
  `eb0b7151d76e5409e6dc9bb4a023551de5e0704673d5ac9f726319ef84a32387`.
- D-002J-P2.5 does **NOT** edit any source code under
  `research/systemic_risk/*.py` or any `scripts/x10r_d002*.py`.
- D-002J-P2.5 does **NOT** create `artifacts/d002j/ingestion/`
  (that directory remains P3 territory).

Lineage: `D-002G → D-002H REFUSED → D-002I → D-002J prereg #694 → P1 #695 → P1A #697 REJECTED → P1B #698 PARTIALLY_VERIFIED → P2 #699 CRISIS_WINDOW_REGISTRY_READY → P2.5 this PR (VERDICT_DAG_BOOTSTRAPPED)`.

Next legal PR: `feat(x10r,D-002J-P3): implement ingestion manifest
and point-in-time adapter boundary` — opens W3/W4 of the D-002J
Frontier Benchmark Program on top of the now-anchored verdict
DAG. P3 may only open after this P2.5 PR is merged with decision
`VERDICT_DAG_BOOTSTRAPPED`.
