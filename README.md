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
100 MACHINE-CHECKABLE INVARIANTS
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

L2 order-book microstructure Ricci remains unimplemented and must stay at `HYPOTHESIS` until a real depth-5 session with a real data hash is recorded.

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
| 5 Gates / Release | CI checks for integrity and schema contracts | Any required gate fails |
| 6 Product / Interface | Single-command gateway | Run/verify path unavailable |
| 7 External Witness | Independent reproduction capsule | Reproduction exits non-zero or cannot verify hashes |

---

## Research Line Automaton

```text
[IDEA] → [HYPOTHESIS] → [PREREGISTERED] → [INSTRUMENTED] → [TESTED_SYNTHETIC]
                                                            │
[VALIDATED] ← [REPLICATED] ← [MEASURED] ← [TESTED_REAL_MULTI] ← [TESTED_REAL_SINGLE]
```

Terminal archive states are preserved rather than deleted:

```text
REJECTED
BLOCKED_DATA
BLOCKED_REPRO
BLOCKED_NULL
BLOCKED_COST_MODEL
RETIRED
```

Failure is not embarrassment here. Failure is evidence with manners.

---

## Canonical Artifact Contract

Every evidence-bearing inference artifact must validate as JSON and include the deterministic fields enforced by:

```text
schemas/research/research_inference_artifact.schema.json
```

The committed example at:

```text
artifacts/runs/ricci_microstructure_v1/example_artifact.json
```

is an illustrative placeholder, not a measured result.

```json
{
  "run_id": "RFC-2026-MARKET-RICCI-V1",
  "git_sha": "0000000000000000000000000000000000000000",
  "git_dirty": false,
  "data_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
  "config_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
  "seed": 1337,
  "timestamp_utc": "2026-05-31T20:27:00Z",
  "input_window_sec": 30,
  "score": 0.0,
  "uncertainty": 0.0,
  "decision": "OBSERVE",
  "claim_tier": "HYPOTHESIS",
  "falsification_status": "NOT_RUN",
  "artifact_role": "placeholder",
  "score_source": "placeholder"
}
```

Schema validity proves **shape**, not **truth**.

Truth requires lineage:

```text
dataset_hash
  → config_hash
  → git_sha
  → method_version
  → score_source: computed
  → baseline
  → falsification_status
  → decision
  → replay_command
```

A `PASS` without `method_version`, computed score source, baseline, and replay command is rejected by the truth gate.

---

## Golden Path

```bash
geosync-research run \
  --line ricci_microstructure_v1 \
  --config conf/research/ricci_v1.yaml \
  --data data/manifest.json \
  --out artifacts/runs/

geosync-research verify artifacts/runs/<run_id>
```

Until the gateway is fully implemented, the repository-level schema enforcement point is:

```bash
python tools/research/validate_ricci_artifact_schema.py \
  --schema schemas/research/research_inference_artifact.schema.json \
  --artifact artifacts/runs/ricci_microstructure_v1/example_artifact.json
```

---

## MFN Dependency-Light Gateway

`geosync.mfn` is a standard-library-only operation path:

```text
simulate → extract → detect → forecast → compare → report
```

It produces a self-contained, hash-manifested artifact bundle at the `INSTRUMENTED` tier. It does not promote predictive or trading claims.

Run:

```bash
mfn --out artifacts/runs/mfn_integration run
mfn validate --bundle artifacts/runs/mfn_integration
```

Clean install proof:

```bash
python tools/verify_mfn_clean_install.py \
  --python python3.12 \
  --report artifacts/runs/mfn_product_state.json

make release-gate
```

Purpose:

```text
prove packaging
prove console entrypoints
prove dependency-light operation
prove artifact reproducibility
prevent local path leakage from hiding failure
```

---

## Release Evidence Harness

Every release validation must leave machine-readable evidence, raw logs, commands, and exit codes.

```bash
python tools/release_evidence_harness.py
python tools/release_evidence_harness.py --fixed-timestamp-utc 2026-01-01T00:00:00Z
python tools/release_evidence_harness.py --verify-manifest artifacts/evidence_bundle/manifest.json
```

Expected evidence surface:

```text
release_gate.yaml
artifacts/evidence_bundle/manifest.json
raw logs
CycloneDX SBOM
CI status JSON
Prometheus telemetry export
DataDog telemetry export
dependency-free PDF audit report
environment drift checks
optional GPG signature
```

A green terminal is not a release proof.

A release claim without generated evidence files, or with any required RED gate, is not evidence.

---

## Required Gates

Protected-branch policy must require:

```text
research-integrity-gate
physics-kernel-self-check
claim-evidence-gate
rust-accel-gate with cargo clippy -- -D warnings
go-workspace-integrity
dependency-review with fail-on-severity: high
```

---

## Scope Control

Forbidden wording and permitted status language live in:

```text
FORBIDDEN_CLAIMS.md
```

A claim that cannot identify its falsifier is not a claim.

It is backlog text trying to look important.

---

<div align="center">

## Design Principle

**Evidence before status.**  
**Falsification before belief.**  
**Reproduction before authority.**

<br/>

### GeoSync is built for systems that refuse to lie politely.

</div>
