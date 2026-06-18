# Neuron7X Operational Verb Map

Status: `OPERATIONS_DOCUMENTATION_ONLY`
Scope: repo-agent execution vocabulary, PR decomposition, evidence-oriented task framing.
Runtime impact: none.
Scientific claim impact: none.

## Purpose

This document converts the Neuron7X system-thinking vocabulary into an executable operations map for repository agents.

The map is not a motivational glossary. Each verb is treated as a task primitive that must produce an observable state transition, artifact, metric, gate, or decision.

Core rule:

```text
Verb -> operational intent -> action -> evidence -> checkpoint -> next action.
```

A repository agent must not use these verbs as decorative language. Every verb must either transform the repository state, validate the current state, or explicitly block unsafe action.

## Global Execution Contract

```text
1. Intention before action.
2. Measurement before optimization.
3. Validation before merge.
4. Falsification before promotion.
5. Integration before deployment.
6. Evidence before status.
7. Reverse audit after forward execution.
8. Traceability before attribution.
9. Contract before orchestration.
10. Rollback before release.
```

## Forward Execution Loop

Use this order when converting an abstract request into a pull-request sequence:

```text
інтенціювати -> дефінувати -> діагностувати -> трасувати -> інтегрувати -> операціоналізувати -> пріоритезувати -> ініціалізувати -> ізолювати -> інструментувати -> агрегувати -> нормалізувати -> канонізувати -> фільтрувати -> класифікувати -> корелювати -> атрибутувати -> валідувати -> верифікувати -> тестувати -> фальсифікувати -> профілювати -> бенчмаркнути -> калібрувати -> адаптувати -> оптимізувати -> стабілізувати -> рефакторити -> декуплювати -> інкапсулювати -> контрактувати -> оркеструвати -> синхронізувати -> логувати -> моніторити -> візуалізувати -> позиціонувати -> онбордити -> деплоїти -> релізувати -> скейлити -> ітерувати
```

## Reverse Audit Loop

Use this order after implementation to prove the result is not self-deception:

```text
ітерувати -> скейлити -> релізувати -> деплоїти -> онбордити -> позиціонувати -> візуалізувати -> моніторити -> логувати -> синхронізувати -> оркеструвати -> контрактувати -> інкапсулювати -> декуплювати -> рефакторити -> стабілізувати -> оптимізувати -> адаптувати -> калібрувати -> бенчмаркнути -> профілювати -> фальсифікувати -> тестувати -> верифікувати -> валідувати -> атрибутувати -> корелювати -> класифікувати -> фільтрувати -> канонізувати -> нормалізувати -> агрегувати -> інструментувати -> ізолювати -> ініціалізувати -> пріоритезувати -> операціоналізувати -> інтегрувати -> трасувати -> діагностувати -> дефінувати -> інтенціювати
```

The forward loop builds the solution. The reverse loop attacks it.

## Verb-to-Task Protocol

| Verb | Operational meaning | Required output |
| --- | --- | --- |
| Інтенціювати | Fix the goal, expected state, intervention boundary, risk, and stop criterion. | PR intent block, non-goals, stop rule. |
| Дефінувати | Define object, concept boundary, inclusion rule, exclusion rule, and operational sense. | Glossary entry, schema field, contract paragraph. |
| Діагностувати | Identify the observed failure, hidden cause candidates, affected layer, and uncertainty boundary. | Diagnostic report, failure class, suspected root-cause map. |
| Трасувати | Follow a claim, data item, code path, artifact, or decision back to its source. | Trace map, file/line path, provenance link. |
| Інтегрувати | Connect symptoms, data, context, constraints, and roles into one decision model. | Decision matrix, dependency map, integration summary. |
| Операціоналізувати | Convert abstract need into actions, owners, resources, metrics, and checkpoints. | Task list, ownership, acceptance gates. |
| Пріоритезувати | Rank tasks by risk, evidence gap, dependency, reversibility, and blast radius. | Priority queue, PR order, blocked/deferred list. |
| Ініціалізувати | Start module/process with explicit initial state. | Initial config, seed, baseline artifact. |
| Ізолювати | Run work separately for safety, reproducibility, or testing. | Branch, worktree, fixture, sandbox, scope boundary. |
| Інструментувати | Add measurement hooks without changing intended behavior. | Metrics, traces, counters, audit events. |
| Агрегувати | Collect related data, functions, or signals into one representation. | Inventory, manifest, evidence table. |
| Нормалізувати | Bring data, naming, format, or scale into one consistent convention. | Normalized schema, canonical path, adapter. |
| Канонізувати | Select one authoritative representation and map aliases to it. | Canonical schema, alias map, migration note. |
| Фільтрувати | Remove invalid, duplicate, stale, unsafe, or irrelevant signals. | Rejection rule, fail-closed test. |
| Класифікувати | Assign items to explicit categories with inclusion and exclusion criteria. | Taxonomy, severity class, status label. |
| Корелювати | Compare signals for co-movement without asserting causality. | Correlation table, caveat, non-causal interpretation. |
| Атрибутувати | Assign cause, ownership, or responsibility only when evidence supports it. | Causal note, owner map, confidence level. |
| Валідувати | Check whether action matches need, context, safety, goal, and use. | Validation matrix, merge/no-merge decision. |
| Верифікувати | Compare claim, data, output, or action against source, spec, protocol, or fact. | Evidence path, source mapping, reproducible check. |
| Тестувати | Simulate cases and execute assertions against behavior. | Unit/contract/integration tests. |
| Фальсифікувати | Try to disprove the hypothesis through nulls, adversarial data, and counterexamples. | Falsifier test, null baseline, demotion rule. |
| Профілювати | Characterize behavior across time, memory, performance, failure modes, and branches. | Profile report, coverage map, risk map. |
| Бенчмаркнути | Compare behavior against known baseline or reference. | Benchmark artifact, baseline delta. |
| Калібрувати | Tune thresholds, tolerances, sensitivity, or scoring against declared evidence. | Calibration table, tolerance rationale. |
| Адаптувати | Adjust model/process to new environment without violating invariants. | Compatibility patch, environment contract. |
| Оптимізувати | Improve objective under measured constraints without breaking correctness. | Metric delta, regression protection. |
| Стабілізувати | Reduce variance, flakiness, drift, or nondeterminism to a declared tolerance. | Stability report, flake fix, deterministic guard. |
| Рефакторити | Improve structure without changing external behavior. | Equivalent behavior tests, smaller interfaces. |
| Декуплювати | Remove accidental dependency between components. | Interface boundary, dependency reduction. |
| Інкапсулювати | Hide internal complexity behind stable contract. | Public API, private implementation boundary. |
| Контрактувати | Encode expectations as explicit inputs, outputs, invariants, and failure modes. | Contract file, schema, interface tests. |
| Оркеструвати | Coordinate sequence of modules, jobs, agents, or PRs. | Execution graph, ordered workflow, dependency chain. |
| Синхронізувати | Align branch state, artifacts, claims, docs, and CI against the same head. | Same-SHA gate, updated branch, sync report. |
| Логувати | Record events and decisions for traceability. | Structured logs, command log, audit trail. |
| Моніторити | Track state over time and detect drift or regression. | Status dashboard, CI watch, drift report. |
| Візуалізувати | Present structure or result so humans can audit it. | Diagram, table, rendered report. |
| Позиціонувати | State what the system is, is not, and who it serves. | Product boundary, claim boundary, README wording. |
| Онбордити | Make usage and reasoning reproducible for a new operator. | Runbook, tutorial, minimal path. |
| Деплоїти | Move verified artifact into target environment. | Deployment plan, release evidence, rollback rule. |
| Релізувати | Publish a versioned artifact only after evidence, provenance, and rollback are ready. | Release notes, manifest, tag, rollback plan. |
| Скейлити | Increase scope or throughput while preserving invariants. | Load/stress test, scaling bound, capacity metric. |
| Ітерувати | Repeat observe -> intervene -> measure -> correct until stable criterion is met. | Iteration log, convergence criterion. |

## Missing-Layer Additions

The original uploaded vocabulary already covered architecture, AI logic, validation, infrastructure, and medical-style operational commands. The repo-execution layer needs additional verbs because repository work fails less from missing ideas than from missing control surfaces. Delightful, apparently software needs a nervous system too.

| Verb | Why it was added | Required guard |
| --- | --- | --- |
| Діагностувати | Prevents acting on symptoms without a failure model. | Diagnostic classification before patch. |
| Трасувати | Prevents claims, artifacts, and fixes from becoming untraceable folklore. | Source/provenance path before status. |
| Пріоритезувати | Prevents large mixed PRs and dependency inversion. | Risk-ranked PR order. |
| Інструментувати | Makes invisible behavior measurable. | Metrics without semantic mutation. |
| Канонізувати | Converts duplicate contracts into one authority. | Canonical path plus alias map. |
| Класифікувати | Converts vague states into decision categories. | Explicit enum or severity matrix. |
| Корелювати | Allows pattern detection without false causality. | Non-causal caveat. |
| Атрибутувати | Assigns cause or owner only with evidence. | Confidence level and trace. |
| Стабілізувати | Targets nondeterminism and flaky gates. | Variance/flakiness criterion. |
| Контрактувати | Converts expectations into enforceable boundaries. | Schema/interface/invariant tests. |
| Синхронізувати | Ensures docs, artifacts, CI, and branch head refer to the same state. | Same-SHA verification. |
| Релізувати | Separates deployable intent from evidence-bearing publication. | Manifest, provenance, rollback. |
| Карантинувати | Isolates unsafe claims, files, or tests without deleting evidence. | Quarantine reason, owner, expiry. |
| Демотувати | Lowers unsupported claims without deleting history. | Claim-tier change with evidence gap. |
| Ескалювати | Moves unresolved risk to a higher control lane. | Escalation reason and blocking condition. |
| Ролбекнути | Restores last known valid state when a change violates a gate. | Rollback command and target SHA. |
| Атестувати | Certifies that artifact provenance and checksums match expected state. | Attestation/provenance record. |
| Архівувати | Preserves obsolete evidence, decisions, or failed attempts for audit. | Archive path and retention note. |

## Repository-Control Verbs

These verbs must be used when operating pull requests and release evidence.

| Verb | Operational meaning | Required output |
| --- | --- | --- |
| Карантинувати | Isolate unsafe or incomplete work from merge paths while preserving evidence. | Quarantine registry entry, expiry, unblock condition. |
| Демотувати | Lower claim status when evidence is absent, weak, stale, or contradicted. | Claim-tier diff, reason, replacement status. |
| Ескалювати | Promote a risk to a blocking lane when local resolution is unsafe. | Blocking issue, owner, dependency link. |
| Ролбекнути | Return to a last known good state after failed validation or unsafe merge. | Revert/rollback SHA, verification run. |
| Атестувати | Bind artifact identity to checksum, provenance, commit, and environment. | Attestation, SBOM/provenance path. |
| Архівувати | Preserve superseded evidence, failed runs, or retired claims without treating them as active. | Archive manifest, retention note. |

## Pull Request Decomposition Rules

A PR must map to one dominant verb and at most two supporting verbs.

Examples:

```text
PR type: validation
Dominant verb: Валідувати
Supporting verbs: Верифікувати, Фальсифікувати
Expected files: tests/, scripts/ci/, artifacts/audit/
Forbidden scope: runtime behavior rewrite without failing test.
```

```text
PR type: documentation/operations
Dominant verb: Операціоналізувати
Supporting verbs: Дефінувати, Інтегрувати
Expected files: docs/operations/, docs/contracts/
Forbidden scope: code semantics, product claim promotion.
```

```text
PR type: core hardening
Dominant verb: Фальсифікувати
Supporting verbs: Тестувати, Верифікувати
Expected files: tests/core/, core/ only if failing test proves bug.
Forbidden scope: cosmetic refactor mixed with numerical behavior change.
```

```text
PR type: canonicalization
Dominant verb: Канонізувати
Supporting verbs: Трасувати, Контрактувати
Expected files: schemas/, docs/contracts/, tests/schemas/
Forbidden scope: duplicate contract creation without alias mapping.
```

```text
PR type: release evidence
Dominant verb: Атестувати
Supporting verbs: Релізувати, Ролбекнути
Expected files: artifacts/evidence_bundle/, release notes, provenance/SBOM paths.
Forbidden scope: release status without manifest verification.
```

## Merge Readiness Rule

A PR is not ready because its story is coherent. A PR is ready only when its dominant verb has evidence:

```text
Валідувати -> validation matrix exists.
Верифікувати -> source/spec/fact mapping exists.
Фальсифікувати -> failing-condition test exists.
Операціоналізувати -> roles/actions/metrics/checkpoints exist.
Інтегрувати -> symptoms/data/context/constraints/roles are connected.
Ітерувати -> next loop and stop criterion are explicit.
Діагностувати -> failure class and suspected root cause exist.
Трасувати -> provenance/source path exists.
Канонізувати -> canonical representation and alias map exist.
Контрактувати -> explicit invariant/schema/interface exists.
Синхронізувати -> same-head evidence exists.
Карантинувати -> unblock condition and expiry exist.
Демотувати -> unsupported claim is lowered with reason.
Ролбекнути -> rollback target and verification command exist.
Атестувати -> provenance/checksum evidence exists.
```

## Agent Final Response Format

Every agent executing this map must return:

```text
VERB_APPLIED:
INTENT:
ACTIONS:
EVIDENCE:
CHECKPOINTS:
BLOCKERS:
REVERSE_AUDIT:
NEXT_VERB:
FINAL_STATUS:
```

For repository-control verbs, also return:

```text
CLAIM_OR_ARTIFACT_IMPACT:
SAME_HEAD_STATUS:
ROLLBACK_PATH:
TRACE_PATH:
```

## Stop Rule

Stop if the action cannot be tied to evidence, test, artifact, or decision. Do not continue by inventing prose. The repository does not need decorative cognition; it needs observable state transitions.
