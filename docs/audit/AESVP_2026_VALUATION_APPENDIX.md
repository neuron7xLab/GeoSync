# AESVP-2026 valuation appendix — GeoSync

This appendix is a static documentation record of one repository valuation run. It is intentionally not a command transcript, generated JSON dump, or product pitch.

## 1. Claim boundary

| Item | Statement |
| --- | --- |
| Supported claim | Estimated engineering cost to rebuild the measured repository state from zero. |
| Explicitly excluded | Product sale price, startup valuation, revenue potential, token/market value, strategic premium, and official US-GAAP capitalization. |
| Measured repository state | Commit `2d2f638f6db1e4ab6d6b4928da3ac3564920a5f3`, branch `work`. |
| Audit timestamp | `2026-06-04T08:05:53Z` UTC. |
| Measurement scope | 100% of files returned by `git ls-files` during the audit run. |
| Environment note | Python `3.14.4`; `cloc`, `tokei`, and `scc` were unavailable. |
| Confidence | Medium. Metrics are internally consistent, but generated/vendor classification and accounting evidence remain unresolved. |

## 2. Core result

| Result | Amount |
| --- | ---: |
| AESVP engineering TCAV, central scenario | $45,220,234 |
| Practical enterprise rebuild cost | $2,789,200 |
| Full tracked-tree reconstruction cost | $6,116,480 |
| Engineering capitalizable candidate, not GAAP | $1,213,302 |

Interpretation: the repository has substantial measured engineering mass, but the valuation is capped by high-complexity debt, low property-test density, missing accounting-stage traceability, and unresolved generated/vendor disclosure.

## 3. Measured repository facts

| Metric | Value |
| --- | ---: |
| Tracked files | 5,360 |
| Total text lines | 1,285,631 |
| Text SLOC | 1,061,865 |
| Source SLOC | 582,058 |
| Source KSLOC | 582.058 |
| Python SLOC | 552,036 |
| Python functions | 32,497 |
| Python test files | 1,484 |
| Property-test files | 150 |
| Assert statements | 29,860 |
| Critical-function heuristic | 3,535 |
| Branch nodes | 40,366 |
| Try nodes | 1,983 |
| Except handlers | 1,916 |
| Raise nodes | 6,118 |
| Eval/exec call sites | 26 |
| Unsafe YAML load files | 0 |
| Pydantic references | 330 |
| SHA-256 references | 2,844 |
| JSONL references | 496 |
| TODO/FIXME markers | 50 |
| Functions with estimated McCabe complexity > 10 | 811 |
| Dependency manifests | 50 |
| CI workflows | 27 |
| Security workflows | 15 |
| Audit/governance artifacts | 378 |
| Claims/evidence artifacts | 205 |
| Test-to-source ratio | 0.374893910916 |
| Property-test density | 0.042432814710 |
| Documentation-to-source ratio | 0.162866930787 |

## 4. Source SLOC by extension

| Extension | SLOC |
| --- | ---: |
| `.py` | 552,036 |
| `.js` | 10,852 |
| `.tf` | 8,937 |
| `.tsx` | 2,150 |
| `.html` | 1,531 |
| `.ts` | 1,298 |
| `.sh` | 1,068 |
| `.go` | 706 |
| `.rs` | 666 |
| `.tla` | 600 |
| `.mmd` | 559 |
| `.sql` | 507 |
| `.css` | 365 |
| `.proto` | 353 |
| `.tpl` | 181 |
| `.j2` | 146 |
| `.avsc` | 103 |

## 5. Engineering valuation model

### 5.1 Replacement cost

```text
V_RC = A × KSLOC_equ^E × EM × H_pm × R_labor
E = 1.01 + 0.01 × ΣSF
```

| Parameter | Value | Class |
| --- | ---: | --- |
| `A` | 2.94 | Declared model assumption |
| `SF` vector | `[1.24, 5.07, 1.41, 1.10, 1.56]` | Declared model assumption |
| `ΣSF` | 10.38 | Derived |
| `E` | 1.1138 | Derived |
| `KSLOC_equ` | 582.058 | Measured |
| `H_pm` | 152 hours/person-month | Declared model assumption |
| `R_labor` | $90 / $120 / $150 per hour | Scenario assumption |
| `EM` | 0.75 / 1.00 / 1.25 | Scenario assumption |

### 5.2 Architectural quality index

```text
D_idx    = 1 - (eval_exec_calls + complexity_over_10_count) / branch_nodes
V_bounds = 1 - (eval_exec_calls + yaml_unsafe_load_files) / (validation_symbols + pydantic_references)
C_inv    = property_test_files / critical_function_heuristic
Ψ_AQI    = 0.40 × D_idx + 0.35 × V_bounds + 0.25 × C_inv
```

| Component | Inputs | Value |
| --- | --- | ---: |
| `D_idx` | `1 - (26 + 811) / 40,366` | 0.979264727741 |
| `V_bounds` | `1 - (26 + 0) / (189 + 330)` | 0.949903660886 |
| `C_inv` | `150 / 3,535` | 0.042432814710 |
| `Ψ_AQI` | Weighted sum | 0.734780376084 |

### 5.3 Technical debt discount

```text
Ω_TDB = V_RC × debt_factor
```

| Debt component | Formula | Factor | Central impact |
| --- | --- | ---: | ---: |
| Complexity | `0.06 × 811 / 3,535` | 0.013765205092 | $886,695 |
| Dynamic execution | `0.03 × 26 / 40,366` | 0.000019323193 | $1,245 |
| Property-test gap | `0.01 × (1 - 150 / 1,484)` | 0.008989218329 | $579,047 |
| TODO/FIXME | `0.02 × 50 / 100` | 0.010000000000 | $644,157 |
| **Total debt factor** |  | **0.032773746614** | **$2,111,143** |

### 5.4 Risk dividend

| Item | Amount | Treatment |
| --- | ---: | --- |
| Verified risk-mitigation dividend | $0 | Used in TCAV. No verified incident probability register is included here. |
| Declared incident-risk scenario | $9,035 | Recorded only as non-central scenario data; excluded from central result. |

## 6. Valuation outputs

| Scenario | Labor rate | EM | `V_RC` | `Ω_TDB` | TCAV with verified RMD = $0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Low | $90/h | 0.75 | $36,233,819 | $1,187,518 | $25,436,381 |
| Central | $120/h | 1.00 | $64,415,679 | $2,111,143 | $45,220,234 |
| High | $150/h | 1.25 | $100,649,498 | $3,298,661 | $70,656,615 |

## 7. Practical rebuild model

| Model | Boundary | Cost |
| --- | --- | ---: |
| Lean rebuild | Useful core only | $646,304 |
| Enterprise rebuild | Core, tests, CI, audit, and docs | $2,789,200 |
| Full tracked-tree reconstruction | All tracked source, tests, docs, configs, and governance surfaces | $6,116,480 |

This model is a human staffing cross-check, not a sale valuation. It remains below the AESVP central engineering TCAV.

## 8. Capitalization filter

Engineering classification only. This is not an official accounting conclusion.

| Surface | Candidate | Allocation | Candidate amount | Blocker |
| --- | --- | ---: | ---: | --- |
| Development / implemented software | Partial | $1,171,464 | $820,025 | Missing owner approval, time records, and accounting-stage evidence. |
| Testing / verification | Partial | $446,272 | $223,136 | Some tests unavailable in environment; property-test density is low. |
| Security / compliance | Partial | $223,136 | $100,411 | Prior audit issues require remediation evidence. |
| Documentation / governance | Partial | $278,920 | $69,730 | Not all governance content maps directly to development-stage assets. |
| Research / hypothesis | No | $502,056 | $0 | Hypothesis/evidence exploration is excluded. |
| Maintenance / technical debt | No | $167,352 | $0 | Debt remediation is excluded. |
| **Total candidate** |  |  | **$1,213,302** | Official capitalization not claimed. |

## 9. Gate status

| Gate | Status | Primary blocker |
| --- | --- | --- |
| G1 Strict input type check | Partial | Strict validation coverage is not universal. |
| G2 Cryptographic integrity | Pass | No blocker for repository-level hash evidence. |
| G3 Side-effect isolation | Partial | High-complexity runtime/API functions remain. |
| G4 Property-based sufficiency | Partial | Property-test density is 0.042432814710. |
| G5 Idempotency verification | Partial | Per-entrypoint idempotency witnesses are incomplete. |
| G6 Tamper detection | Partial | Config and CI hardening evidence is incomplete. |
| G7 Zero unhandled exceptions | Partial | Fail-closed exception taxonomy is not universal. |
| G8 Traceability matrix | Partial | Requirement-to-line mapping is incomplete. |
| G9 Audit provenance | Partial | Signed JSONL provenance is not universal. |
| G10 Technical debt control | Fail | 811 functions exceed complexity 10. |
| G11 Accounting stage traceability | Partial | No full research/development accounting ledger. |

## 10. Use and falsification

Use this appendix as a fixed documentation snapshot of the audit result. Do not use it as proof that the current branch still has the same metrics after later commits.

Invalidate this appendix if any of the following are true:

1. The measured metrics cannot be reproduced from tracked files at the stated commit.
2. Source SLOC includes generated or vendor artifacts without disclosure.
3. CI, test, or security passing is claimed without fresh evidence.
4. The appendix is used as a sale-price, startup-value, revenue, or official capitalization claim.
