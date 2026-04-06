# Adversarial Orchestration

Creator -> Critic -> Auditor -> Verifier
Human = central decision node. No agent has autonomous authority.

Physics Kernel (CLAUDE.md + .claude/physics/) = embedded Critic.
57 invariants. Every test references INV-*. Every threshold derived from theory.

## Validation layers

1. **AST validator** — `python .claude/physics/validate_tests.py` — structural
2. **mypy --strict** — type-level soundness
3. **ruff format + check** — style determinism
4. **pytest** — runtime correctness
5. **MANIFEST.sha256** — reproducibility seal
