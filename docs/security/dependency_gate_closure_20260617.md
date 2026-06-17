# Dependency Gate Closure 2026-06-17

## Purpose

Convert dependency checking from advisory completion into a blocking verification gate.

## Current facts

- `pyproject.toml` declares the Python dependency surface.
- `requirements*.lock` provide pinned Python lock state.
- `constraints/security.txt` contains security-sensitive dependency constraints.
- `scripts/dependency_health.py` checks manifest and lock alignment.
- `scripts/dependency_audit.py` wraps package auditing and can return non-zero when findings exist.
- `Makefile` dependency audit targets currently allow audit findings to continue without failing the target.

## Defect class

`ADVISORY_DEPENDENCY_GATE`

The repository has dependency checking tools, but the default command path can complete even when the check reports findings.

## Required behavior

Dependency checking must be fail-closed:

- drift checks pass;
- manifest and lock alignment passes;
- package audit returns no active findings;
- reports are generated;
- the target exits non-zero when findings exist.

## Required patch for Code Claude

1. Update `Makefile` dependency audit targets so package-audit findings fail the target.
2. Use `scripts/dependency_audit.py` instead of raw advisory audit calls.
3. Add a test that rejects masked audit failures in dependency targets.
4. Keep the change limited to dependency checking surfaces.
5. Do not alter physics, neuro, trading, or UI logic.

## Acceptance

`DEPENDENCY_GATE_CLOSED` only when:

- dependency audit targets fail on active findings;
- no finding is downgraded into advisory-only output;
- generated reports exist;
- same-SHA CI is green;
- residual findings are zero or explicitly recorded as non-applicable with evidence.
