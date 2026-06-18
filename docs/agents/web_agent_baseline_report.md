# Web Agent Baseline Report

`WEB-AGENT-ARCH-001 · Baseline 001`

## Status

```text
BOOTSTRAP_BASELINE_REFRESHED
NOT_PRODUCTION_READY
```

This report turns the architecture protocol from a qualitative control plane into a measured baseline fixture.

It does not certify the agent as production-ready. It records the first conservative score and the exact missing evidence gates.

## Baseline artifact

```text
artifacts/agents/web_agent_baseline_score.json
```

## Measurement mode

```text
static_contract_bootstrap
```

Meaning:

- documentation and schemas exist;
- smoke tests exist;
- deterministic score calculator exists;
- adversarial prompt-injection fixture exists;
- prompt-injection evaluator exists;
- static tool contract inventory exists;
- runtime task success has not been evaluated;
- no prompt-injection runtime trace has passed;
- live adapter behavior has not been verified;
- CI/local execution evidence is still required.

## Baseline metrics

| Metric | Current value | Target | Status |
|---|---:|---:|---|
| task_completion_rate | 0.0 | >= 0.95 | BLOCKED_NO_EVAL_SUITE |
| irreversible_actions_without_confirmation | 0 | 0 | PASS_STATIC_POLICY |
| context_efficiency | 2.0 | <= 1.2 | FAIL_BOOTSTRAP_ESTIMATE |
| hallucination_rate | 0.0 | <= 0.02 | PASS_STATIC_SCOPE_ONLY |
| escalation_precision | 0.0 | >= 0.90 | BLOCKED_NO_ESCALATION_DATASET |
| injection_resistance | false | true | FAIL_NO_RUNTIME_TRACE |
| tool_contract_coverage | 1.0 | 1.0 | PASS_STATIC_CONTRACTS_ONLY |
| retry_budget_compliance | 0.0 | >= 0.95 | BLOCKED_NO_RETRY_EVENTS |
| memory_minimization | 0.0 | >= 0.90 | BLOCKED_NO_MEMORY_TRACE |
| explanation_completeness | 0.0 | >= 0.95 | BLOCKED_NO_DECISION_TRACE |

## Computed score

```text
score = 43.0 / 100
status = NOT_PRODUCTION_READY
```

The score is intentionally pessimistic. Missing runtime evidence scores zero. This prevents metric theater, humanity's favorite office sport after pretending roadmaps are execution.

The score is computed by:

```text
tools/web_agent_score.py
```

## Why the score is not zero

- The irreversible-action rule exists and the bootstrap artifact records zero irreversible actions without confirmation.
- A conservative context-efficiency ratio is declared.
- Unsupported factual claims are zero inside the narrow static fixture scope.
- Static tool contract coverage is defined for the declared minimal tool surface.

## Why the score is not production-ready

- No task evaluation dataset exists.
- Adversarial prompt-injection fixture exists, but no runtime agent trace has passed.
- Static tool contract inventory exists; live tool adapter behavior is still not proven.
- No escalation precision dataset exists.
- No memory trace exists.
- No decision trace exists.
- CI evidence has not been bound to this report.

## Next tasks by information gain

| Rank | Task | Output | Acceptance gate |
|---:|---|---|---|
| 1 | Add runtime task evaluation fixture | `artifacts/agents/task_eval_fixture.json` | task_completion_rate becomes measurable |
| 2 | Add safe runtime injection trace fixture | `artifacts/agents/prompt_injection_trace_fixture.json` | injection evaluator can compute PASS/FAIL |
| 3 | Add decision trace fixture | `artifacts/agents/decision_trace_fixture.json` | explanation completeness becomes measurable |
| 4 | Add memory trace fixture | `artifacts/agents/memory_trace_fixture.json` | memory_minimization becomes measurable |
| 5 | Bind tests into CI | workflow or existing CI path | evidence is machine-executed, not aspirational |

## Stop conditions

```text
- Do not call the web agent production-ready from this baseline.
- Do not claim injection resistance until adversarial runtime trace passes.
- Do not claim task completion rate until an evaluation suite exists.
- Do not treat static tool inventory as live adapter proof.
- Do not merge as final release without CI evidence.
```
