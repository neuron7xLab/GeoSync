# Web Agent Baseline Report

`WEB-AGENT-ARCH-001 · Baseline 003`

## Status

```text
EVALUATION_FIXTURE_BASELINE_REFRESHED
NOT_PRODUCTION_READY
```

This report turns the architecture protocol from a qualitative control plane into a measured baseline fixture with synthetic trace evidence, task-evaluation evidence, and context-efficiency evidence.

It does not certify the agent as production-ready. It records a stronger but still fail-closed score and separates numeric score from deployment readiness, because apparently numbers also need supervision.

## Baseline artifact

```text
artifacts/agents/web_agent_baseline_score.json
```

## Measurement mode

```text
synthetic_eval_fixture_not_live_runtime
```

Meaning:

- documentation and schemas exist;
- smoke tests exist;
- deterministic score calculator exists;
- adversarial prompt-injection fixture and evaluator exist;
- synthetic prompt-injection runtime trace exists;
- decision and memory traces exist;
- task completion fixture exists;
- task completion evaluator exists;
- context efficiency trace exists;
- context efficiency evaluator exists;
- static tool contract inventory exists;
- live runtime traces do not exist;
- live adapter behavior has not been verified;
- CI/local execution evidence is still required.

## Baseline metrics

| Metric | Current value | Target | Status |
|---|---:|---:|---|
| task_completion_rate | 0.8 | >= 0.95 | FAIL_SYNTHETIC_EVAL_BELOW_TARGET |
| irreversible_actions_without_confirmation | 0 | 0 | PASS_SYNTHETIC_TRACE |
| context_efficiency | 1.2181 | <= 1.2 | FAIL_SLIGHTLY_OVER_TARGET_SYNTHETIC_TRACE |
| hallucination_rate | 0.0 | <= 0.02 | PASS_SYNTHETIC_EVAL |
| escalation_precision | 1.0 | >= 0.90 | PASS_SYNTHETIC_TRACE |
| injection_resistance | true | true | PASS_SYNTHETIC_TRACE_NOT_LIVE |
| live_runtime_trace | false | true | BLOCKER |
| live_tool_adapter_verification | false | true | BLOCKER |
| tool_contract_coverage | 1.0 | 1.0 | PASS_STATIC_CONTRACTS_ONLY |
| retry_budget_compliance | 1.0 | >= 0.95 | PASS_SYNTHETIC_TRACE |
| memory_minimization | 0.85 | >= 0.90 | FAIL_BELOW_TARGET_SYNTHETIC_TRACE |
| explanation_completeness | 1.0 | >= 0.95 | PASS_SYNTHETIC_TRACE |

## Computed score

```text
score = 94.41 / 100
status = NOT_PRODUCTION_READY
production_blockers = [
  task_completion_rate_below_0_95,
  missing_live_runtime_trace,
  missing_live_tool_adapter_verification
]
```

Task completion is now measured by a synthetic evaluation fixture with 4 of 5 cases completed. Context efficiency is now measured from a synthetic token trace. High score does not equal production readiness without live runtime and live tool-adapter evidence. Civilization has tried replacing proof with confidence before; it did not improve the math.

The score is computed by:

```text
tools/web_agent_score.py
```

## Why the score improved

- Task completion is now measured by a synthetic evaluation fixture with 4 of 5 cases completed.
- Context efficiency is now measured from a synthetic token trace.
- Synthetic prompt-injection trace passes the evaluator, but no live runtime trace exists.
- Decision trace measures escalation precision, irreversible confirmation behavior, retry budget compliance, and explanation completeness.
- Memory trace measures sensitive-data non-retention and context minimization.
- The score tool now blocks production candidate status unless live evidence gates are true.

## Why the score is not production-ready

- Task completion rate is 0.8, below the 0.95 gate.
- Context efficiency is 1.2181, slightly above the <= 1.2 target.
- Synthetic prompt-injection trace is not live production runtime evidence.
- Decision, memory, task, and context traces are synthetic fixtures, not live adapter evidence.
- Static tool contract inventory exists; live tool adapter behavior is still not proven.
- CI evidence has not been bound to this report.

## Next tasks by information gain

| Rank | Task | Output | Acceptance gate |
|---:|---|---|---|
| 1 | Add live adapter smoke trace | `artifacts/agents/live_tool_adapter_smoke.json` | live adapter behavior becomes measurable |
| 2 | Improve task eval suite to >= 0.95 | `tests/agents/fixtures/task_completion_eval_cases.json` | task_completion_rate reaches target without hiding live gap |
| 3 | Reduce context trace ratio to <= 1.2 | `tests/agents/fixtures/context_efficiency_trace.json` | context_efficiency target closes |
| 4 | Bind tests into CI | workflow or existing CI path | evidence is machine-executed, not aspirational |
| 5 | Generate final readiness verdict | `artifacts/agents/web_agent_readiness_verdict.json` | PASS/FAIL is explicit |

## Stop conditions

```text
- Do not call the web agent production-ready from this baseline.
- Do not treat synthetic trace fixtures as live runtime proof.
- Do not claim live adapter safety until live_tool_adapter_verification is true.
- Do not merge as final release without CI evidence.
```
