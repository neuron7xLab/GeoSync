# GeoSync Claim Contract

**Positioning:** Verification-first quantitative research platform for falsifiable market-structure hypotheses.

This repository is not a live execution platform, not an alpha product, and not a proof that market dynamics obey any asserted physical law. It is a research platform whose claims are allowed only when they are tied to explicit invariants, data contracts, falsifiers, and reproducible artifacts.

## Current Sprint Boundary

The only sprint-level evidence spine is `ricci_microstructure_v1` over crypto perpetual order-book data. Other research lines remain either infrastructure, protocol sketches, or archived evidence surfaces until they pass the same contract stack.

## Claim Tier Matrix

| Claim Tier | Evidence | Falsifier | Status |
| --- | --- | --- | --- |
| Invariant Registry | FACT: script/check coverage of registered invariants | Missing invariant count or unlinked witness | Active |
| Systemic-Risk Phase Signal | HYPOTHESIS: protocol and synthetic/controlled checks only | Real-data gate failure | Not Measured |
| Ricci Microstructure Predictor | LIMITED EMPIRICAL: one crypto-perps L2 session | Multi-session failure, cost-model failure, or null superiority | Not Deployable |

## Promotion Rule

A research line can move forward only through this sequence:

```text
Hypothesis -> Preregistration -> Data Contract -> Inference Contract -> Null Baselines -> Empirical Validation -> Immutable Artifact -> Claim Tier Signoff
```

Synthetic data may instrument or test plumbing, but it cannot promote a claim to measured status. Promotion beyond `INSTRUMENTED` requires a real-data artifact whose dataset hash, configuration hash, git state, seed, and falsification result are recorded.

## Layered State Machine

| Layer | Contract | Blocking Condition |
| --- | --- | --- |
| 0 Doctrine / Claims | Falsifiable claim and forbidden-claim registry | Claim lacks falsifier or uses forbidden language |
| 1 Data / Substrate | Immutable dataset manifest with SHA-256 | Data hash missing or mutable source unresolved |
| 2 Inference Kernel | Deterministic transformation contract | Output omits canonical fields or dirty git state is not disclosed |
| 3 Falsification / Nulls | Null baselines and stress tests | Null matches or exceeds claimed signal |
| 4 Artifact / Evidence | JSON artifact validated against schema | Schema validation failure |
| 5 Gates / Release | CI checks for integrity and schema contracts | Any required gate fails |
| 6 Product / Interface | Single command gateway | Run/verify path unavailable |
| 7 External Witness | Independent reproduction capsule | Reproduction exits non-zero or cannot verify hashes |

## Research Line State Machine

```text
[IDEA] -> [HYPOTHESIS] -> [PREREGISTERED] -> [INSTRUMENTED] -> [TESTED_SYNTHETIC]
                                                               |
[VALIDATED] <- [REPLICATED] <- [MEASURED] <- [TESTED_REAL_MULTI] <- [TESTED_REAL_SINGLE]
```

Terminal archive states are `REJECTED`, `BLOCKED_DATA`, `BLOCKED_REPRO`, `BLOCKED_NULL`, `BLOCKED_COST_MODEL`, and `RETIRED`. Terminal records are preserved rather than deleted.

## Canonical Artifact Contract

Every inference artifact for an evidence-bearing line must validate as JSON and include the deterministic fields enforced by `schemas/research/ricci_microstructure_artifact.schema.json`:

```json
{
  "run_id": "RFC-2026-MARKET-RICCI-V1",
  "git_sha": "0000000000000000000000000000000000000000",
  "git_dirty": false,
  "data_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
  "config_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
  "seed": 1337,
  "timestamp_utc": "2026-05-31T20:27:00Z",
  "input_window_sec": 30,
  "score": 0.742,
  "uncertainty": 0.031,
  "decision": "NO_TRADE",
  "claim_tier": "HYPOTHESIS",
  "falsification_status": "PASS"
}
```

## Golden Path

The dependency-light MFN integration gateway is the canonical machine-smoke path for creating an `INSTRUMENTED` bundle without promoting a predictive or trading claim:

```bash
mfn --out artifacts/runs/mfn_integration run
mfn validate --bundle artifacts/runs/mfn_integration
```

Clean-install verification for this gateway is executable with:

```bash
python tools/verify_mfn_clean_install.py --python python3.12 --report artifacts/runs/mfn_integration/product_state.json
```

The research-line gateway remains the target interface for evidence-bearing Ricci runs:

```bash
geosync-research run --line ricci_microstructure_v1 --config conf/research/ricci_v1.yaml --data data/manifest.json --out artifacts/runs/
geosync-research verify artifacts/runs/<run_id>
```

Until that research-line gateway is fully implemented, the schema gate is the repository-level enforcement point:

```bash
python tools/research/validate_ricci_artifact_schema.py --schema schemas/research/ricci_microstructure_artifact.schema.json --artifact artifacts/runs/ricci_microstructure_v1/example_artifact.json
```

## Required Gates

Protected-branch policy must require these checks before evidence-bearing changes merge:

- `research-integrity-gate`
- `physics-kernel-self-check`
- `claim-evidence-gate`
- `rust-accel-gate` with `cargo clippy -- -D warnings`
- `go-workspace-integrity`
- `dependency-review` with `fail-on-severity: high`

## Scope Control

See `FORBIDDEN_CLAIMS.md` for prohibited wording and permitted status language. A claim that cannot identify its falsifier is not a claim; it is backlog text.
