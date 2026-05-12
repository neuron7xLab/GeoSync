# D-002G P1 — Implementation Report (non-degenerate null infrastructure)

## 1. Anchor

- Worktree: `.claude/worktrees/agent-ab16f75f6b0367760`
- Branch: `feat/x10r-d002g-p1-implementation`
- Anchor commit (read-only): `3e92ce5` (D-002G pre-registration lock)
- This PR's HEAD: filled in at commit time.

## 2. Locked-files sha block

All seven anchor files verified `PASS` (sha unchanged versus anchor commit):

| File | sha256 | Verdict |
|------|--------|---------|
| `docs/governance/D002G_PREREGISTRATION.yaml` | `1ab91f09370e4705a8b0849467bc1f56df2e58d58d5623d3b6d905cbd110bb04` | PASS |
| `docs/governance/D002G_NONDEGENERATE_NULL_DESIGN.md` | `9cef2db7f5d1f90eb9ec71524193c079efff024c35de0ea9758e4f6c747bd8bb` | PASS |
| `docs/governance/D002G_ACCEPTANCE_RULES.md` | `875b1e3eb031b8e5333dc8b455454f0a30419ead1ebe787aa01d5882e7d6ad31` | PASS |
| `.claude/commit_acceptors/x10r-d002g-nondegenerate-null-redesign.yaml` | `eaa704722cd113997fac58d52de3ec38ac7197c70d80389e4197d52d8ce93327` | PASS |
| `docs/governance/D002C_PREREGISTRATION.yaml` | `b1561ddde08a60a8eed416f2103655e0f3ee1ecd4e2b2037f4e7193c424a154e` | PASS |
| `docs/governance/D002C_CLAIM_LEDGER.yaml` | `f96ba9b5a2057d2e0bff84afc28578ab316cff73f6dc6673fb0d6d543b8bd6dd` | PASS |
| `docs/governance/D002C_CANONICAL_RUN_REPORT.md` | `f03ed1c6e96f62dc7ff061b48fc44a6dce0679a13ca6bf449e3785f0a4833ed0` | PASS |
| `docs/governance/D002C_ATTEMPT_2_NULL_AUDIT_FALSIFICATION_REPORT.md` | `83164744e223f236a49111c6411630ff54332285ab871896bfc8921fcd4b0b34` | PASS |

These shas are also pinned inside `tests/systemic_risk/test_d002g_locked_governance_untouched.py`; the test fails closed on any drift.

## 3. Modules and LoC

| Module | LoC | Role |
|--------|-----|------|
| `research/systemic_risk/d002g_null_mechanisms.py` | 611 | M1 + M6 realisation primitive, deterministic seed mixing, `NullRealization` content-addressed dataclass |
| `research/systemic_risk/d002g_phase0_capsule.py` | 142 | `phase0_verification_capsule_v1` writer (canonical JSON + sha256, atomic write) |
| `research/systemic_risk/d002g_phase0_verification.py` | 861 | Phase 0a/0b/0c verifier, robust Phase 0b helper (`phase0b_robust`), Phase 0c power calibration (`phase0c_power_calibration`), per-cell aggregate |
| `research/systemic_risk/d002g_r2b_gate.py` | 515 | R2-B aggregator, topology-coupling indicator (Strike-R2), rule correlation matrix (Strike-R7), INDETERMINATE verdict path |
| `research/systemic_risk/d002c_sweep_runner.py` | +47 (modified) | v1/v2 payload schema constants + sha branching on `NullAuditCellPayload` |

## 4. Attack ladder summary

Full attack ladder in [`D002G_P1_DESIGN_ADVERSARIAL_AUDIT.md`](D002G_P1_DESIGN_ADVERSARIAL_AUDIT.md).

| Rung | Verdict (pre-patch) | Verdict (post-patch) | Code-comment tag |
|------|---------------------|----------------------|------------------|
| R1 — Phase 0a only `np.array_equal` | UNTESTED | TESTED (spectral L∞ + KS gate via `test_d002g_strike_R1_spectral_identity.py`) | n/a (test-only enforcement) |
| R2 — M6 frob preservation = conditional informativeness | UNTESTED | PATCHED (`R2B_TOPOLOGY_COUPLING_FLOOR`, `R2B_INDETERMINATE_VERDICT`, `topology_coupling_indicator`) | `# Strike-R2` |
| R3 — Phase 0b `|t|<2` mis-calibrated for skewed/bounded | FAILS | PATCHED (Wilcoxon signed-rank + bootstrap CI in `phase0b_robust`; verdict path switched) | `# Strike-R3` |
| R4 — Phase 0c range check ≠ power | UNTESTED | PATCHED (`phase0c_power_calibration` helper with deterministic detection-rate gate) | `# Strike-R4` |
| R5 — `null_seed = base_seed + 10000` only checked at seed 42 | FAILS (partial) | TESTED (all 50 seeds via `test_d002g_strike_R5_seed_collision.py`) | n/a (coverage test) |
| R6 — Bonferroni-216 conservatism vs dependence | Doc deficit | DOCUMENTED HERE (§7 below) | n/a |
| R7 — Joint distribution of (R1,R2,R3,R2-B) | UNTESTED | PATCHED (`rule_correlation_labels` + `rule_correlation_matrix` in R2-B capsule) | `# Strike-R7` |
| R8 — Test Phase 0 ≠ canonical Phase 0 | UNTESTED | PATCHED (§9 claim boundary verbatim + locked-files sha test) | n/a |
| R9 — M6 placebo-coupling support inequality | SURVIVES | SURVIVES (no patch needed) | n/a |
| R10 — `generated_at` excluded from sha | SURVIVES | SURVIVES (already correct) | n/a |

No rung deferred via `xfail`. R6 is genuinely an annotation rung — the prereg locks 216, the implementation respects 216, and the conservatism is annotated in §7 below.

## 5. Tests

Test directory: `tests/systemic_risk/`.

| File | Test count | Status |
|------|-----------|--------|
| `test_d002g_locked_governance_untouched.py` | 1 | PASS |
| `test_d002g_strike_R1_spectral_identity.py` | 2 | PASS |
| `test_d002g_strike_R2_m6_conditional_informativeness.py` | 3 | PASS |
| `test_d002g_strike_R3_phase0b_robust.py` | 3 | PASS |
| `test_d002g_strike_R4_phase0c_power.py` | 3 | PASS |
| `test_d002g_strike_R5_seed_collision.py` | 4 | PASS |
| `test_d002g_strike_R7_joint_distribution.py` | 1 | PASS |
| `test_d002g_payload_compat_v1_v2.py` | 4 | PASS |
| `test_d002g_phase0_capsule_schema.py` | 5 | PASS |
| `test_d002g_r2b_capsule_schema.py` | 3 | PASS |
| **Total** | **27** | **27 PASS, 0 fail, 0 xfail** |

Runtime: 6.1 s on `i5-12500H` E-cores (well under the 90 s budget).

## 6. v1/v2 payload sha branching specification

`NullAuditCellPayload` (`research/systemic_risk/d002c_sweep_runner.py`) now carries three additional fields with v1-preserving defaults:

```python
payload_schema: str = PAYLOAD_SCHEMA_V1          # "d002c_null_audit_cell_v1"
null_strategy:  str = DEFAULT_NULL_STRATEGY      # "D002C_PAIRED_CRN_LEGACY"
null_seed:      int | None = None
```

**Sha-input branching (deterministic, fail-closed):**

| Schema | Sha-input fields | On-disk shape |
|--------|------------------|---------------|
| `d002c_null_audit_cell_v1` | 12 legacy fields | omits `payload_schema`, `null_strategy`, `null_seed` |
| `d002c_null_audit_cell_v2` | 12 legacy + `payload_schema` + `null_strategy` + `null_seed` | includes all three |

**Round-trip contract:**

- A legacy v1 dict on disk (no schema field) loads with `payload_schema=v1` and reproduces the legacy sha exactly — back-compat for pre-A2 emissions verified by `test_v1_payload_roundtrips_with_legacy_sha`.
- A v2 dict carrying a v2 schema tag loads with the v2 sha including the new fields — verified by `test_v2_payload_roundtrips_with_extended_sha`.
- v1 and v2 shas differ on otherwise identical scientific inputs — verified by `test_v1_and_v2_shas_differ_on_same_scientific_fields`.
- A single mixed-mode loader (`NullAuditCellPayload.from_payload_dict`) handles both — verified by `test_mixed_mode_loader_reads_both_schemas`.

## 7. Bonferroni-216 conservatism annotation (Strike-R6)

The pre-registration locks `bonferroni_n_cells = 216` (= 3 substrates × 2 metrics × 3 N × 6 λ × 2 cohort halves, etc. — see `D002G_PREREGISTRATION.yaml`). The implementation respects this constant.

However: cells at the same `(substrate, metric, N)` with varying `λ` share `K_0` (the baseline at λ=0) and are strongly statistically dependent. Treating all 216 cells as independent inflates the false-negative rate. Effective tests under dependence are closer to ~36.

**Consequence.** The per-cell α-floor emitted by `evaluate_r2b` (`bonferroni_alpha_per_cell = 0.05 / 216 ≈ 2.31e-4`) is the contract value, NOT the effective threshold. The R2-B capsule also emits `rule_correlation_matrix` (Strike-R7) so downstream consumers can quantify the dependence empirically and report effective-α themselves. The implementation does NOT secretly adjust the divisor — the prereg is the contract.

## 8. Quality gates

All commands run from worktree root with `PYTHONPATH=.`.

```
$ ruff format --check research/systemic_risk/d002g_*.py research/systemic_risk/d002c_sweep_runner.py tests/systemic_risk/
17 files already formatted

$ ruff check research/systemic_risk/d002g_*.py research/systemic_risk/d002c_sweep_runner.py tests/systemic_risk/
All checks passed!

$ black --check research/systemic_risk/d002g_*.py research/systemic_risk/d002c_sweep_runner.py tests/systemic_risk/
All done! ✨ 🍰 ✨
17 files would be left unchanged.

$ mypy --strict --follow-imports=silent research/systemic_risk/d002g_*.py research/systemic_risk/d002c_sweep_runner.py
Success: no issues found in 5 source files

$ mypy --strict --follow-imports=silent tests/systemic_risk/
Success: no issues found in 12 source files

$ pytest tests/systemic_risk/ -k "d002g or d002c" -q
...........................                                              [100%]
27 passed in 6.09s
```

`--follow-imports=silent` is used to scope the mypy check to D-002G modules; the pre-existing `core/kuramoto/jax_engine.py` typing drift is unrelated to this PR and is tracked separately.

## 9. CLAIM BOUNDARY (verbatim)

> This PR implements D-002G infrastructure and adversarial test scaffolding only. It does NOT establish D-002G scientific PASS. Phase 0 test-suite results are INFRASTRUCTURE SMOKE, not canonical Phase 0 verdict. A fresh canonical D-002G run on prereg-scoped substrates is required before any tier-PASS claim.

## 10. Out-of-scope

- No sweep run was executed. The canonical D-002G sweep is a separate downstream PR after Phase 0 verification capsule emission on the prereg-scoped grid.
- No `D002C_CLAIM_LEDGER.yaml` or `D002G_*` ledger update.
- No tier promotion. No cross-promotion to D-002C tier strings or to real-data / bank-level claims.
- No threshold edits to the locked pre-registration constants (`null_seed_offset=10000`, `r2_b_random_site_seed=99`, `bonferroni_n_cells=216`, `R2B_CI_ALPHA=0.05`, `R2B_FPR_THRESHOLD=0.05`).

## Substrate-eligibility note (informational)

Two of the three stock substrates (`block_structured`, `temporal_coupling`) are seed-deterministic at λ=0 by design — they deliberately ignore the seed argument so the baseline K is fully determined by `(substrate_id, N)`. Phase 0a / M1 would fail on these substrates and the prereg §4 fallback to M2 (topology-preserving shuffle) applies. The R5 seed-sweep test is scoped to `ricci_flow` (the only seed-sensitive substrate); the test docstring annotates the scope. M2 is OUT of P1 scope; it lands in a follow-on PR if Phase 0 FAILs at the canonical run.
