<div align="center">

# GeoSync

### Verification-first quantitative research infrastructure for falsifiable market-structure hypotheses.

<br/>

<img src="https://img.shields.io/badge/status-research_kernel-111111?style=for-the-badge" />
<img src="https://img.shields.io/badge/claims-falsifiable-111111?style=for-the-badge" />
<img src="https://img.shields.io/badge/evidence-machine_checkable-111111?style=for-the-badge" />
<img src="https://img.shields.io/badge/ricci-hypothesis_tier-111111?style=for-the-badge" />

<br/><br/>

**GeoSync is not a trading bot.**  
**GeoSync is not an alpha product.**  
**GeoSync is not a proof that markets obey any asserted physical law.**

It is a research platform where claims are permitted only when they are bound to explicit invariants, data contracts, falsifiers, reproducible artifacts, and release evidence.

</div>

---

## ∴ Research Kernel

Most market-research repositories optimize for narrative confidence.

GeoSync optimizes for something less fashionable and more useful:

> **A claim may exist only when it can identify its invariant, data source, method, artifact, falsifier, and replay path.**

No replay.  
No falsifier.  
No evidence.  
No claim.

---

## System Spine

```text
DOCTRINE / FORBIDDEN CLAIMS
        ↓
CLAIM REGISTRY
        ↓
108 MACHINE-CHECKABLE INVARIANTS
        ↓
DATA CONTRACT
        ↓
INFERENCE CONTRACT
        ↓
NULL BASELINES / FALSIFIERS
        ↓
ARTIFACT VALIDATION
        ↓
RELEASE EVIDENCE HARNESS
        ↓
EXTERNAL REPRODUCTION
```

GeoSync treats research as engineered infrastructure, not as a decorative chart factory wearing a lab coat.

---

## Repository Map

| Surface | Purpose |
| --- | --- |
| [`PRODUCT_CATEGORY.md`](PRODUCT_CATEGORY.md) | Canonical product-category boundary. |
| [`CLAIMS.md`](CLAIMS.md) | Claim tier ledger and evidence pointers. |
| [`FORBIDDEN_CLAIMS.md`](FORBIDDEN_CLAIMS.md) | Status-language firewall and promotion invariants. |
| [`docs/REPOSITORY_SYSTEM.md`](docs/REPOSITORY_SYSTEM.md) | Operational map for reviewers, auditors, and implementation agents. |
| [`docs/INFERENCE_READINESS.md`](docs/INFERENCE_READINESS.md) | Runtime-readiness contract for context, agent, artifact, and falsification control. |
| [`docs/MFN_VERIFICATION_ROADMAP.json`](docs/MFN_VERIFICATION_ROADMAP.json) | MFN gateway machine-readable verification roadmap contract. |
| [`AGENTS.md`](AGENTS.md) | Root implementation contract for Claude, Codex, and automation agents. |

The README is the front door. The repository system document is the spine. The claim ledger is the courtroom. Naturally, the courtroom matters more than the wallpaper.

---

## Current Sprint Boundary

The active research line is:

```text
ricci_microstructure_v1
```

Current state:

```text
HYPOTHESIS-tier
PROVENANCE-INSTRUMENTED
NOT EVIDENCE-BEARING YET
```

The gateway stamps dataset, config, and git provenance into a schema-valid artifact, but it does **not** run empirical inference yet. No real order-book session has been committed.

The implemented Ricci computation currently operates on **cross-asset return-correlation graphs** via:

```text
src/geosync/features/ricci.py::RicciCurvatureGraph.fit_transform
```

Executable proof — claim → source → test → command:

```text
claim:   correlation-graph Ricci adapter with fail-closed / neutral / fixed-graph / non-finite contract
source:  src/geosync/features/ricci.py::RicciCurvatureGraph
test:    tests/features/test_ricci_curvature_graph.py
command: pytest -q tests/features/test_ricci_curvature_graph.py
```

The adapter's contract is pinned by dedicated deterministic tests: insufficient-window fail-closed (`ValueError`), zero-edge neutral output (`kappa_min == 0.0`, `edge_kappa == {}`), a fixed two-node graph with `kappa == 0.5`, non-finite input excluded (no phantom edges), and a claim-boundary lock (the result exposes no L2/microstructure/predictive field). This is graph-topology provenance over correlation structure — **not** an evidence-bearing L2 market-microstructure result. L2 order-book microstructure Ricci remains unimplemented and must stay at `HYPOTHESIS` until a real depth-5 session with a real data hash is recorded.

---

## Claim Tier Matrix

| Claim Tier | Evidence | Falsifier | Status |
| --- | --- | --- | --- |
| Invariant Registry | Script/check coverage of registered invariants | Missing invariant count or unlinked witness | Active |
| Systemic-Risk Phase Signal | Protocol and synthetic/controlled checks only | Real-data gate failure | Not Measured |
| Ricci Microstructure Predictor | Provenance-instrumented only; no real session committed | Multi-session failure, cost-model failure, or null superiority | Not Measured |

---

## Promotion Rule

```text
Hypothesis
  → Preregistration
  → Data Contract
  → Inference Contract
  → Null Baselines
  → Empirical Validation
  → Immutable Artifact
  → Claim Tier Signoff
```

Synthetic data may instrument plumbing.

Synthetic data may test schemas.

Synthetic data may prove the machine can walk.

Synthetic data may **not** promote a research claim to measured status.

---

## Layered State Machine

| Layer | Contract | Blocking Condition |
| --- | --- | --- |
| 0 Doctrine / Claims | Falsifiable claim and forbidden-claim registry | Claim lacks falsifier or uses forbidden language |
| 1 Data / Substrate | Immutable dataset manifest with SHA-256 | Data hash missing or mutable source unresolved |
| 2 Inference Kernel | Deterministic transformation contract | Output omits canonical fields or dirty git state is undisclosed |
| 3 Falsification / Nulls | Null baselines and stress tests | Null matches or exceeds claimed signal |
| 4 Artifact / Evidence | JSON artifact validated against schema | Schema validation failure |
