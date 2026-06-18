# Web Agent Baseline Report

`WEB-AGENT-ARCH-001 · Baseline 001`

## Status

```text
BOOTSTRAP_BASELINE_CREATED
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
- runtime task success has not been evaluated;
- injection resistance has not been evaluated;
- live tool contracts have not been evaluated;
- CI/local execution evidence is still required.

## Baseline metrics

| Metric | Current value | Target | Status |
|---|---:|---:|---|
| task_completion_rate | 0.0 | >= 0.95 | BLOCKED_NO_EVAL_SUITE |
| irreversible_actions_without_confirmation | 0 | 0 | PASS_STATIC_POLICY |
| context_efficiency | 2.0 | <= 1.2 | FAIL_BOOTSTRAP_ESTIMATE |
| hallucination_rate | 0.0 | <= 0.02 | PASS_STATIC_SCOPE_ONLY |
| escalation_precision | 0.0 | >= 0.90 | BLOCKED_NO_ESCALATION_DATASET |
| injection_resistance | false | true | FAIL_NO_ADVERSARIAL_SUITE |
| tool_contract_coverage | 0.0 | 1.0 | BLOCKED_NO_LIVE_TOOL_MAP |
| retry_budget_compliance | 0.0 | >= 0.95 | BLOCKED_NO_RETRY_EVENTS |
| memory_minimization | 0.0 | >= 0.90 | BLOCKED_NO_MEMORY_TRACE |
| explanation_completeness | 0.0 | >= 0.95 | BLOCKED_NO_DECISION_TRACE |

## Computed score

```text
score = 46.0 / 100
status = NOT_PRODUCTION_READY
```

The score is intentionally pessimistic. Missing runtime evidence scores zero. This prevents metric theater, humanity's favorite office sport after pretending roadmaps are execution.

## Why the score is not zero

- The irreversible-action rule exists and the bootstrap artifact records zero irreversible actions without confirmation.
- A conservative context-efficiency ratio is declared.
- Unsupported factual claims are zero inside the narrow static fixture scope.

## Why the score is not production-ready

- No task evaluation dataset exists.
- No prompt-injection test suite exists.
- No live tool contract map exists.
- No escalation precision dataset exists.
- No memory trace exists.
- No decision trace exists.
- CI evidence has not been bound to this report.

## Next tasks by information gain

| Rank | Task | Output | Acceptance gate |
|---:|---|---|---|
| 1 | Add deterministic score calculator | `tools/web_agent_score.py` | score matches baseline artifact |
| 2 | Add baseline contract test | `tests/agents/test_web_agent_score.py` | pytest validates score and weights |
| 3 | Add injection fixture | `tests/agents/fixtures/prompt_injection_cases.json` | trusted behavior unchanged |
| 4 | Add tool contract inventory | `docs/agents/tool_contract_inventory.md` | every tool has one responsibility and failure mode |
| 5 | Add decision trace fixture | `artifacts/agents/decision_trace_fixture.json` | explanation completeness becomes measurable |

## Stop conditions

```text
- Do not call the web agent production-ready from this baseline.
- Do not claim injection resistance until adversarial cases pass.
- Do not claim task completion rate until an evaluation suite exists.
- Do not merge as final release without CI evidence.
```
