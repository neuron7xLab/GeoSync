<!-- Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab) -->
<!-- SPDX-License-Identifier: MIT -->

# GeoSync — Repository System

> **Operational map for the repository as a verification-first quantitative research system.**

This document is the compact spine for reviewers, collaborators, auditors, and future implementation agents. It explains what the repository is allowed to claim, which surfaces are canonical, how evidence moves through the system, and where a result must stop when proof is incomplete.

GeoSync is organized around one rule:

> **A research statement is admissible only when its invariant, data source, method, artifact, falsifier, and replay path are explicit.**

The repository is not optimized for persuasive language. It is optimized for controlled promotion of claims under machine-checkable boundaries.

---

## 1. Canonical Surfaces

| Surface | Role |
| --- | --- |
| [`README.md`](../README.md) | Public entry point and high-level system narrative. |
| [`PRODUCT_CATEGORY.md`](../PRODUCT_CATEGORY.md) | Canonical product-category boundary. Defines what GeoSync is and is not. |
| [`CLAIMS.md`](../CLAIMS.md) | Single source of truth for claim tiers and evidence pointers. |
| [`FORBIDDEN_CLAIMS.md`](../FORBIDDEN_CLAIMS.md) | Status-language firewall and promotion invariants. |
| [`AGENTS.md`](../AGENTS.md) | Root implementation contract for Claude, Codex, and automation agents. |
| [`docs/MFN_VERIFICATION_ROADMAP.json`](MFN_VERIFICATION_ROADMAP.json) | MFN gateway machine-readable verification roadmap contract. |
| [`BASELINE.md`](../BASELINE.md) | Baseline evidence and comparison reference when present. |
| [`schemas/`](../schemas) | Machine-readable contracts for artifacts and governance records. |
| [`scripts/ci/`](../scripts/ci) | Enforcement scripts for claim boundary, gates, and release discipline. |
| [`artifacts/`](../artifacts) | Generated or committed evidence surfaces. Placeholder artifacts must identify themselves as placeholders. |
| [`artifacts/governance/pre_release_convergence_task_graph.json`](../artifacts/governance/pre_release_convergence_task_graph.json) | Candidate-only pre-release convergence task graph for role-owned evidence closure. |

All other documents must conform to these surfaces. New text that upgrades a claim without updating `CLAIMS.md` is drift.

---

## 2. System Layers

```text
L0  Doctrine / boundary language
L1  Claim registry / evidence tier
L2  Data contract / immutable hashes
L3  Inference method / deterministic transform
L4  Nulls / falsifiers / alternative hypotheses
L5  Artifact schema / validation
L6  CI gate / release evidence
L7  External reproduction capsule
```

A layer may depend only on validated lower layers. A higher layer cannot repair a broken lower layer by sounding confident. Human civilization keeps trying this. GeoSync does not.

---

## 3. Claim Promotion Automaton

```text
IDEA
  ↓
HYPOTHESIS
  ↓
PREREGISTERED
  ↓
INSTRUMENTED
  ↓
TESTED_SYNTHETIC
  ↓
TESTED_REAL_SINGLE
  ↓
TESTED_REAL_MULTI
  ↓
MEASURED
  ↓
REPLICATED
```

Allowed terminal or blocking states:

```text
REJECTED
BLOCKED_DATA
BLOCKED_REPRO
BLOCKED_NULL
BLOCKED_COST_MODEL
RETIRED
```

A failed result is not removed. It is recorded, classified, and preserved. Silent deletion is not epistemology; it is cosmetic damage control.

---

## 4. Evidence-Bearing Artifact Requirements

A result may support promotion only when the artifact records:

```text
run_id
git_sha
git_dirty
data_sha256
config_sha256
seed
timestamp_utc
method_version
score
score_source
uncertainty
baseline
falsification_status
decision
claim_tier
artifact_role
replay_command
```

Minimum rule:

```text
score_source must be computed
artifact_role must not be placeholder
falsification_status must not be NOT_RUN
baseline must be explicit
replay_command must execute the same artifact path
```

Schema validity proves shape. It does not prove truth.

---

## 5. Ricci Microstructure Boundary

The active line is:

```text
ricci_microstructure_v1
```

Current admissible state:

```text
HYPOTHESIS
INSTRUMENTED
NOT EVIDENCE-BEARING
```

The repository may describe the computation path and provenance instrumentation. It must not describe the line as measured, predictive, externally replicated, or deployment-ready until qualifying real-data artifacts pass the declared gates.

Current implementation boundary:

```text
src/geosync/features/ricci.py::RicciCurvatureGraph.fit_transform
```

This supports graph-level structural computation. It does not by itself establish market-mechanism truth.

---

## 6. MFN Gateway Boundary

The MFN path is a dependency-light operational gateway:

```text
simulate → extract → detect → forecast → compare → report
```

Its job is to prove:

```text
packaging
console entrypoints
dependency-light execution
artifact bundle shape
hash-manifest discipline
local reproducibility
```

It may produce `INSTRUMENTED` artifacts. It may not promote a research statement beyond its evidence tier.

---

## 7. Release Evidence Boundary

A release gate is valid only when it leaves machine-readable evidence:

```text
commands
raw_logs
exit_codes
manifest
checksums
schema_validation
CI_status
SBOM_when_available
telemetry_exports_when_available
environment_drift_result
```

A green terminal line is an observation. A release evidence bundle is an artifact.

---

## 8. Reviewer Protocol

A reviewer should inspect the repository in this order:

```text
1. PRODUCT_CATEGORY.md
2. CLAIMS.md
3. FORBIDDEN_CLAIMS.md
4. README.md
5. docs/REPOSITORY_SYSTEM.md
6. AGENTS.md
7. schemas/
8. scripts/ci/
9. artifacts/
```

Then ask five questions:

```text
What exactly is claimed?
What tier is the claim assigned?
Which artifact supports it?
Which falsifier can reject it?
Can the artifact be replayed from the recorded command?
```

If any answer is missing, the claim does not move upward.

---

## 9. Implementation-Agent Rules

Any implementation agent modifying this repository must obey:

```text
Do not promote claim language without updating CLAIMS.md.
Do not create placeholder evidence that pretends to be measured evidence.
Do not hide negative outcomes.
Do not weaken schema requirements to make a run pass.
Do not bypass claim-boundary checks by moving language to another document.
Do not introduce dependency-heavy paths into the MFN gateway.
Do not call a release valid without an evidence bundle.
```

For the full root contract, see:

```text
AGENTS.md
```

Expected modification pattern:

```text
read canonical surface
make minimal coherent change
update schema or claim ledger when required
add or update validation
record evidence path
leave replay command
```

---

## 10. Definition of Repository Completion

GeoSync is complete for a given research line only when:

```text
claim tier is declared
input data is immutable
method version is pinned
artifact validates against schema
nulls and falsifiers are executed
baseline is recorded
replay command succeeds
CI gate passes
release evidence is emitted
CLAIMS.md is updated
retired or failed states are preserved
```

Anything less is not completion. It is a folder with ambition.
