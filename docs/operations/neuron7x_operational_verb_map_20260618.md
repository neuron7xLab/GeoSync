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
```

## Forward Execution Loop

Use this order when converting an abstract request into a pull-request sequence:

```text
інтенціювати -> дефінувати -> інтегрувати -> операціоналізувати -> ініціалізувати -> ізолювати -> агрегувати -> нормалізувати -> фільтрувати -> валідувати -> верифікувати -> тестувати -> фальсифікувати -> профілювати -> бенчмаркнути -> калібрувати -> адаптувати -> оптимізувати -> рефакторити -> декуплювати -> інкапсулювати -> оркеструвати -> логувати -> моніторити -> візуалізувати -> позиціонувати -> онбордити -> деплоїти -> скейлити -> ітерувати
```

## Reverse Audit Loop

Use this order after implementation to prove the result is not self-deception:

```text
ітерувати -> скейлити -> деплоїти -> онбордити -> позиціонувати -> візуалізувати -> моніторити -> логувати -> оркеструвати -> інкапсулювати -> декуплювати -> рефакторити -> оптимізувати -> адаптувати -> калібрувати -> бенчмаркнути -> профілювати -> фальсифікувати -> тестувати -> верифікувати -> валідувати -> фільтрувати -> нормалізувати -> агрегувати -> ізолювати -> ініціалізувати -> операціоналізувати -> інтегрувати -> дефінувати -> інтенціювати
```

The forward loop builds the solution. The reverse loop attacks it.

## Verb-to-Task Protocol

| Verb | Operational meaning | Required output |
| --- | --- | --- |
| Інтенціювати | Fix the goal, expected state, intervention boundary, risk, and stop criterion. | PR intent block, non-goals, stop rule. |
| Дефінувати | Define object, concept boundary, inclusion rule, exclusion rule, and operational sense. | Glossary entry, schema field, contract paragraph. |
| Інтегрувати | Connect symptoms, data, context, constraints, and roles into one decision model. | Decision matrix, dependency map, integration summary. |
| Операціоналізувати | Convert abstract need into actions, owners, resources, metrics, and checkpoints. | Task list, ownership, acceptance gates. |
| Ініціалізувати | Start module/process with explicit initial state. | Initial config, seed, baseline artifact. |
| Ізолювати | Run work separately for safety, reproducibility, or testing. | Branch, worktree, fixture, sandbox, scope boundary. |
| Агрегувати | Collect related data, functions, or signals into one representation. | Inventory, manifest, evidence table. |
| Нормалізувати | Bring data, naming, format, or scale into one consistent convention. | Normalized schema, canonical path, adapter. |
| Фільтрувати | Remove invalid, duplicate, stale, unsafe, or irrelevant signals. | Rejection rule, fail-closed test. |
| Валідувати | Check whether action matches need, context, safety, goal, and use. | Validation matrix, merge/no-merge decision. |
| Верифікувати | Compare claim, data, output, or action against source, spec, protocol, or fact. | Evidence path, source mapping, reproducible check. |
| Тестувати | Simulate cases and execute assertions against behavior. | Unit/contract/integration tests. |
| Фальсифікувати | Try to disprove the hypothesis through nulls, adversarial data, and counterexamples. | Falsifier test, null baseline, demotion rule. |
| Профілювати | Characterize behavior across time, memory, performance, failure modes, and branches. | Profile report, coverage map, risk map. |
| Бенчмаркнути | Compare behavior against known baseline or reference. | Benchmark artifact, baseline delta. |
| Калібрувати | Tune thresholds, tolerances, sensitivity, or scoring against declared evidence. | Calibration table, tolerance rationale. |
| Адаптувати | Adjust model/process to new environment without violating invariants. | Compatibility patch, environment contract. |
| Оптимізувати | Improve objective under measured constraints without breaking correctness. | Metric delta, regression protection. |
| Рефакторити | Improve structure without changing external behavior. | Equivalent behavior tests, smaller interfaces. |
| Декуплювати | Remove accidental dependency between components. | Interface boundary, dependency reduction. |
| Інкапсулювати | Hide internal complexity behind stable contract. | Public API, private implementation boundary. |
| Оркеструвати | Coordinate sequence of modules, jobs, agents, or PRs. | Execution graph, ordered workflow, dependency chain. |
| Логувати | Record events and decisions for traceability. | Structured logs, command log, audit trail. |
| Моніторити | Track state over time and detect drift or regression. | Status dashboard, CI watch, drift report. |
| Візуалізувати | Present structure or result so humans can audit it. | Diagram, table, rendered report. |
| Позиціонувати | State what the system is, is not, and who it serves. | Product boundary, claim boundary, README wording. |
| Онбордити | Make usage and reasoning reproducible for a new operator. | Runbook, tutorial, minimal path. |
| Деплоїти | Move verified artifact into target environment. | Deployment plan, release evidence, rollback rule. |
| Скейлити | Increase scope or throughput while preserving invariants. | Load/stress test, scaling bound, capacity metric. |
| Ітерувати | Repeat observe -> intervene -> measure -> correct until stable criterion is met. | Iteration log, convergence criterion. |

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

## Merge Readiness Rule

A PR is not ready because its story is coherent. A PR is ready only when its dominant verb has evidence:

```text
Валідувати -> validation matrix exists.
Верифікувати -> source/spec/fact mapping exists.
Фальсифікувати -> failing-condition test exists.
Операціоналізувати -> roles/actions/metrics/checkpoints exist.
Інтегрувати -> symptoms/data/context/constraints/roles are connected.
Ітерувати -> next loop and stop criterion are explicit.
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

## Stop Rule

Stop if the action cannot be tied to evidence, test, artifact, or decision. Do not continue by inventing prose. The repository does not need decorative cognition; it needs observable state transitions.
