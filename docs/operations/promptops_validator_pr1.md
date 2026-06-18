# PromptOps Validator PR1

## Purpose

This PR introduces the first operational PromptOps slice: prompt artifacts become
versioned CI contracts instead of informal text blobs. The validator is
fail-closed: a prompt artifact is accepted only when structure, references,
security patterns, unit-test declarations, and shadow-test declarations pass.

## Scope

- `schemas/promptops/system_prompt_contract.schema.json` defines the public shape.
- `tools/promptops_validate.py` performs deterministic validation without LLM calls.
- `examples/promptops/system_prompt_contract.example.yaml` provides a canonical artifact.
- `tests/promptops/test_promptops_validate.py` locks pass/fail behavior.
- `.github/workflows/promptops-validator.yml` adds a PR gate for PromptOps changes.

## Current boundary

Shadow inference is represented as a required contract section, not as a live LLM
call. That is deliberate. The first integration must verify that every artifact
carries an executable shadow-test intent before later PRs bind the runner to a
model provider behind explicit secrets and cost controls. Apparently even chaos
needs a budget file now.

## Acceptance command

```bash
python -m pip install pytest pyyaml
python tools/promptops_validate.py examples/promptops --report artifacts/promptops/validation_report.json
python -m pytest tests/promptops -q
```

## Next integration

PR2 should add a `tools/promptops_shadow_runner.py` that consumes the validated
`shadow_tests` section, runs either a mocked local evaluator or a provider-backed
judge, and emits regression metrics with explicit thresholds.
