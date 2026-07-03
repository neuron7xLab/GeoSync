# ⊛ RVG-1 — Repository Verification Gate (GeoSync deployment)

Deterministic, fail-closed audit harness. It produces a **verified evidence
vector** — not a vibe score — answering seven questions with explicit
numerators, denominators, reproduction commands, and hash-bound artifacts.

## What it computes

| Metric | Formula | Raw source |
|---|---|---|
| `test_pass_rate` | `passed / collected × 100` | `artifacts/rvg/junit.xml` |
| `line_coverage` | `covered_lines / num_statements × 100` | `artifacts/rvg/coverage.json` |
| `branch_coverage` | `covered_branches / num_branches × 100` | `artifacts/rvg/coverage.json` |
| `mutation_score` | `killed / (killed + survived) × 100` | `artifacts/rvg/mutation.json` |
| `oracle_gap` | `branch_coverage − mutation_score` | derived |
| `verified_test_strength` | `min(branch_coverage, mutation_score)` | derived |

**Core principle:** coverage ≠ correctness. Line coverage says what executed;
branch coverage says which paths executed; mutation score says whether the tests
*detect wrong behavior*. `oracle_gap` exposes where execution outruns
verification. `verified_test_strength` is the conservative floor and is the only
number allowed to be quoted as "test strength".

## Fail-closed law

`verdict = PASS` **only if every** critical gate passes (see
`audit/thresholds.json`). Missing evidence, zero denominator, unparseable
artifact, or schema-invalid verdict ⇒ `FAIL`. There is no partial-credit path.

Critical gates: tests pass · line ≥ 90 · branch ≥ 85 · mutation ≥ 75 ·
oracle_gap ≤ 15 · ruff clean · mypy clean · 0 high/critical vulns · SBOM present
· hash manifest present.

> RVG thresholds are a **floor** and are intentionally independent of GeoSync's
> repo-wide `coverage.report.fail_under = 98`. RVG never lowers an existing
> gate; it adds an orthogonal, hash-bound verdict layer.

## Reproduce

```bash
make rvg           # clean → coverage → lint → typecheck → security → sbom → mutation → verify
```

Individual stages: `make rvg-coverage rvg-lint rvg-typecheck rvg-security
rvg-sbom rvg-mutation rvg-verify`. Artifacts land under `artifacts/rvg/` (git-ignored); the
verdict is `artifacts/rvg/RVG_VERDICT.json` (+ `.md`), bound by
`artifacts/rvg/audit.hashes`.

## Integrity guarantees

1. **Recompute, never trust.** `rvg_audit.py` derives every percentage from raw
   counts; `rvg_verify_artifacts.py` independently recomputes them from the
   stored raw pairs and fails if any delta > 0.01.
2. **Hash binding.** Every verdict is tied to a sha256 manifest of the source +
   artifacts it was computed from.
3. **Determinism.** No wall-clock / randomness in the tools; the only varying
   fields are `timestamp_utc`, `commit`, `python_version`. Run `make rvg`
   twice and diff — anything else differing is a `FAIL`.

## CI rollout stance (bootstrap → ratchet)

The `rvg` CI job **blocks** on two things: the 21-test self-verification suite
(the instrument must prove itself) and verdict soundness (schema-valid,
hash-bound, recompute-checked, reproducible). The PASS/FAIL **threshold verdict
is reported, not yet enforced repo-wide** — it is uploaded as an artifact and
written to the job summary. Enforcing line/branch/mutation thresholds across the
entire repo (including a real mutation run) is a follow-up ratchet; turning it on
now would red every PR against gates the whole codebase does not yet satisfy,
which is the opposite of an additive gate. The default `make rvg` audit surface
is the RVG harness itself; widen it with `RVG_LINT_PATHS`, `RVG_MYPY_TARGETS`,
and a real `RVG_MUTATION_ARGS` for a full-repo verdict.

## Mutation policy (§6)

`valid_mutants = killed + survived`. Timeout and incompetent/equivalent mutants
are reported separately and never silently folded into `killed`. Timeouts count
as killed only under `--timeout-stable` (fixed, non-hung runtime asserted).

## Forbidden pseudo-metrics

"looks clean", "tests exist so it's covered", "CI is green so it's correct",
"100% line = 100% correct", "one LLM opinion = an audit", "a global quality %
with no open formula". Only metrics with an explicit numerator, denominator,
reproduction command, and artifact are admissible.
