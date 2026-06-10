# GeoSync Claim Contract

**Positioning:** Verification-first quantitative research platform for falsifiable market-structure hypotheses.

This repository is not a live execution platform, not an alpha product, and not a proof that market dynamics obey any asserted physical law. It is a research platform whose claims are allowed only when they are tied to explicit invariants, data contracts, falsifiers, and reproducible artifacts.

The kernel declares **97 machine-checkable invariants** in `.claude/physics/INVARIANTS.yaml`; `scripts/count_invariants.py` is the canonical witness for `C-INV-COUNT` in `CLAIMS.md`.

## Current Sprint Boundary

The sprint-focus research line is `ricci_microstructure_v1` over crypto perpetual order-book data. It is currently **HYPOTHESIS-tier and provenance-instrumented only**: the gateway (`tools/research/research_cli.py`) stamps dataset/config/git provenance into a schema-valid artifact but runs no empirical inference, and no real order-book session has been committed — so it is **not** an evidence-bearing line yet. The operationalized Ricci computation that does have working code (`src/geosync/features/ricci.py::RicciCurvatureGraph.fit_transform`) runs on **cross-asset return-correlation graphs**, not L2 order-book microstructure; L2-microstructure Ricci remains unimplemented and stays at HYPOTHESIS until a depth-5 session with a real data hash is recorded. Other research lines remain either infrastructure, protocol sketches, or archived evidence surfaces until they pass the same contract stack.

## Claim Tier Matrix

| Claim Tier | Evidence | Falsifier | Status |
| --- | --- | --- | --- |
| Invariant Registry | FACT: script/check coverage of registered invariants | Missing invariant count or unlinked witness | Active |
| Systemic-Risk Phase Signal | HYPOTHESIS: protocol and synthetic/controlled checks only | Real-data gate failure | Not Measured |
| Ricci Microstructure Predictor | HYPOTHESIS: provenance-instrumented only; no real session committed, gateway runs no inference | Multi-session failure, cost-model failure, or null superiority | Not Measured |

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

Every inference artifact for an evidence-bearing line must validate as JSON and include the deterministic fields enforced by `schemas/research/ricci_microstructure_artifact.schema.json`. The committed example (`artifacts/runs/ricci_microstructure_v1/example_artifact.json`) is an **illustrative placeholder**: its data hash is all-zero, `score`/`uncertainty` are inert, and `falsification_status` is `NOT_RUN` — it demonstrates the schema shape, not a measured result:

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
  "score": 0.0,
  "uncertainty": 0.0,
  "decision": "NO_TRADE",
  "claim_tier": "HYPOTHESIS",
  "falsification_status": "NOT_RUN"
}
```

## Golden Path

```bash
geosync-research run --line ricci_microstructure_v1 --config conf/research/ricci_v1.yaml --data data/manifest.json --out artifacts/runs/
geosync-research verify artifacts/runs/<run_id>
```

Until that gateway is fully implemented, the schema gate is the repository-level enforcement point:

```bash
python tools/research/validate_ricci_artifact_schema.py --schema schemas/research/ricci_microstructure_artifact.schema.json --artifact artifacts/runs/ricci_microstructure_v1/example_artifact.json
```

## MFN Dependency-Light Gateway

`geosync.mfn` is a standard-library-only operation path (`simulate → extract → detect → forecast → compare → report`) that produces a self-contained, hash-manifested artifact bundle at the `INSTRUMENTED` tier — no predictive or trading claim is promoted. It exists to keep packaging, console entrypoints, and a one-command smoke path machine-verifiable before any heavyweight scientific dependency is installed.

```bash
mfn --out artifacts/runs/mfn_integration run
mfn validate --bundle artifacts/runs/mfn_integration
```

Bundles are byte-reproducible when `SOURCE_DATE_EPOCH` is pinned. A hermetic clean-venv check installs the editable package with the canonical setuptools backend (no backend swap), proves the gateway is dependency-light (`--no-deps`), and exercises imports/entrypoints/run/validate from outside the repository working directory so accidental path leakage cannot mask a failure:

```bash
python tools/verify_mfn_clean_install.py --python python3.12 --report artifacts/runs/mfn_product_state.json
make release-gate   # lint + type + tests + clean-install proof
```

This is a focused install/entrypoint smoke; full signed release evidence (SBOM, manifest, telemetry) remains the job of the Release Evidence Harness below.

## Release Evidence Harness

Every release validation must leave machine-readable evidence, raw logs, commands, and exit codes. The harness is semantic: clean install is required by default, and non-empty git status/diff output is RED even when git exits 0:

```bash
python tools/release_evidence_harness.py
# deterministic metadata/log timestamps for reproducibility checks:
python tools/release_evidence_harness.py --fixed-timestamp-utc 2026-01-01T00:00:00Z
# verify raw-log integrity after generation or archive transfer:
python tools/release_evidence_harness.py --verify-manifest artifacts/evidence_bundle/manifest.json
```

The harness writes `release_gate.yaml` plus `artifacts/evidence_bundle/manifest.json`, raw logs, a CycloneDX SBOM (`sbom.cdx.json`), CI status JSON, Prometheus/DataDog telemetry exports, and a dependency-free PDF audit report. Gates execute through the asyncio runner unless a caller uses the lower-level `_run_gate` helper directly. For release signing, pass `--gpg-key <KEYID>` and verify with `--verify-manifest artifacts/evidence_bundle/manifest.json --require-gpg-signature`; unsigned manifests are acceptable only for local development smoke runs. Environment drift checks can be calibrated with `--write-environment-baseline baseline.json` and enforced later with `--environment-baseline baseline.json`; `--skip-clean-install` is forbidden in CI and requires the explicit development-only `--allow-tainted-env` opt-in outside CI. These are generated release artifacts, not source files; commit them only in an explicit release-evidence branch or archive. A release claim without those files, or with any required RED gate, is not considered evidence.

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
