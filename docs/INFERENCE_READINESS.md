<!-- Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab) -->
<!-- SPDX-License-Identifier: MIT -->

# GeoSync — Inference Readiness Contract

> **Operational boundary for when a repository, prompt system, model context, or knowledge graph stops being a file collection and becomes a controlled execution mechanism.**

Inference readiness is not a status label. It is the moment when the system can accept input, reject noise, preserve the semantic core, apply explicit rules, constrain hallucination, and emit a replayable answer or artifact under measurable runtime conditions.

The value of the system is not its description. The value is its ability to produce stable, falsifiable, reproducible results when executed.

---

## 1. First Principle

```text
A system exits to inference only when its context architecture, agent boundaries,
claim rules, evidence contracts, and replay paths are structured enough to make
runtime behavior predictable, cheap, measurable, and falsifiable.
```

A repository that only stores files is an archive.
A repository that controls execution is an inference surface.

GeoSync therefore treats documentation, schemas, tests, artifacts, and CI as one joined control plane. Text that cannot constrain runtime behavior is not operational evidence.

---

## 2. Runtime Mechanism

The inference-ready path is:

```text
input
  ↓
context selection
  ↓
noise rejection
  ↓
semantic-core preservation
  ↓
claim-boundary binding
  ↓
deterministic method execution
  ↓
null baseline / falsifier check
  ↓
schema-valid artifact emission
  ↓
replayable verification record
```

A result is admissible only when the system records enough information to repeat, measure, falsify, or improve it without manual guessing.

---

## 3. Readiness Gates

| Gate | Requirement | Blocking Condition |
| --- | --- | --- |
| Context Architecture | Canonical files are indexed, linked, and front-loaded for agents. | Critical rules are buried, duplicated, or missing. |
| Token Economy | Context payload excludes decorative or stale material. | Runtime prompt budget is spent on non-operative text. |
| Claim Boundary | Every claim maps to tier, evidence, falsifier, and replay path. | Claim language outruns evidence tier. |
| Agent Control | Agent behavior is bounded by `AGENTS.md`, forbidden claims, tests, and schemas. | Agent can invent status, scope, data, or promotion level. |
| Deterministic Execution | Method, seed, config, git state, and data hash are captured. | Output cannot be replayed from committed state. |
| Artifact Validity | Output conforms to machine-readable schema. | Result exists only as prose, screenshot, or terminal memory. |
| Falsification | Nulls, baselines, and failure modes are explicit. | No path exists to disprove the result. |
| Release Evidence | CI or local evidence bundle records commands, logs, exits, checksums. | Green text exists without durable evidence. |

---

## 4. Operational Metrics

The inference surface should be measured by:

```text
context_retention_rate
context_token_waste_ratio
schema_validation_pass_rate
replay_success_rate
claim_drift_incidents
forbidden_claim_violations
null_baseline_superiority_rate
artifact_hash_coverage
agent_boundary_violation_count
```

A system is not mature because it sounds coherent. It is mature when these measurements improve while claim inflation decreases.

---

## 5. Failure Modes

The repository is not inference-ready when:

```text
critical context is not discoverable
agent instructions conflict across files
outputs are not schema-bound
claims cannot be traced to evidence
null baselines are absent
runtime artifacts are not reproducible
manual interpretation is required to decide validity
CI does not preserve executable evidence
```

These failures block promotion. They are not cosmetic defects.

---

## 6. GeoSync Binding

This contract binds the existing GeoSync system spine:

```text
DOCTRINE / FORBIDDEN CLAIMS
        ↓
CLAIM REGISTRY
        ↓
MACHINE-CHECKABLE INVARIANTS
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

The inference-readiness claim is valid only when each lower layer has an explicit executable or machine-checkable witness.

---

## 7. Final Rule

```text
No indexed context, no inference.
No bounded agent, no inference.
No schema-valid artifact, no inference.
No falsifier, no inference.
No replay path, no inference.
```

The transition is from:

```text
we have a system
```

to:

```text
the system performs work predictably, cheaply, measurably, and falsifiably.
```
