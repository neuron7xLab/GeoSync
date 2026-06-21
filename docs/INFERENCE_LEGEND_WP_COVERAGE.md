# Inference-Legend Work-Package Coverage Ledger

Companion to [`INFERENCE_LEGEND_EXIT_WORK_ACT.md`](INFERENCE_LEGEND_EXIT_WORK_ACT.md).

The act lists ten work packages. Most are **not greenfield**: GeoSync already
ships substantial substrate for them. This ledger records, per package, the
existing artifacts that implement it and the gaps that remain, so the act binds
to the repository instead of re-creating a parallel stack.

It makes **no scientific claim**. It asserts structural correspondence between
the act and the tree, machine-checked by
[`tools/inference/check_wp_coverage.py`](../tools/inference/check_wp_coverage.py)
(fail-closed: an `EXISTS`/`PARTIAL` package that cites a vanished artifact, or a
`GAP` package that lists code, fails the gate). The predictive rank of the
project is unchanged by this document — current status remains
**HYPOTHESIS-tier with strong governance and incomplete predictive proof**, per
the act.

## Status

`EXISTS` — substantive implementation present. `PARTIAL` — components present
but the package's binding/contract is incomplete. `GAP` — not implemented.

| WP | Title | Status | Anchor artifacts (non-exhaustive) |
|----|-------|--------|------------------------------------|
| WP-01 | Contextualization frame | PARTIAL | `results/RUN_MANIFEST.json` (output only; no enforced upfront context contract) |
| WP-02 | L2 microstructure contract | EXISTS | `research/microstructure/BASELINE.L2.md`, `…/attribution.py`, `…/conditional_transfer_entropy.py` |
| WP-03 | Manifold substrate | EXISTS | `core/indicators/market_state_contract.py`, `core/indicators/kuramoto_ricci_composite.py` |
| WP-04 | Ricci curvature field | EXISTS | `core/indicators/ricci.py`, `core/indicators/temporal_ricci.py`, `core/physics/forman_ricci.py`, `core/kuramoto/ricci_flow_engine.py` |
| WP-05 | Kuramoto phase field | EXISTS | `core/kuramoto/kuramoto_ricci_engine.py`, `core/kuramoto/metrics.py` |
| WP-06 | Fused Ricci-Kuramoto kernel | EXISTS | `core/indicators/kuramoto_ricci_composite.py`, `scripts/integrate_kuramoto_ricci.py` |
| WP-07 | Recursive falsification loop | PARTIAL | `core/kuramoto/oos_validation.py` (walk-forward + Diebold-Mariano + SPA), `analytics/signals/null_baseline.py`, `geosync_hpc/nulls/dynamic_null_model.py`, `docs/research/D002J_NULL_MODEL_CONTRACTS.md`, `docs/research/D002J_NULL_MODEL_HIERARCHY.md` (components present; no single end-to-end recursive driver) |
| WP-08 | Claim-tier promotion governance | EXISTS | `governance/CLAIM_MATURITY.yaml`, `analytics/signals/claim_maturity.py`, `application/governance/claim_ledger.py`, `scripts/ci/check_claim_maturity.py`, `docs/schemas/governance/claim_ledger.schema.json` |
| WP-09 | Machine-verification bundle | PARTIAL | `manifests/research/release_provenance.v1.json`, `results/RUN_MANIFEST.json` (provenance + per-run manifests exist; no single assembler binding context+data+nulls+baselines+OOS+cost+claim-card per run) |
| WP-10 | External-rank path | GAP | — (no Zenodo DOI / JOSS draft) |

## The real remaining work (not the whole act)

The act's ten packages reduce to **four genuine gaps** plus a binding layer; the
rest is already implemented and should be *wired*, not rebuilt:

1. **WP-01** — a fail-closed, upfront **context contract** (schema + gate) that
   binds problem / universe / window / target / claim-tier / nulls / stop-rule /
   evidence-path *before* a kernel run, not a post-hoc `RUN_MANIFEST.json`.
2. **WP-07** — an end-to-end **recursive driver** chaining the existing null,
   baseline, OOS and claim-ladder components into one
   `output → claim classifier → null challenge → residual → revised context`
   loop.
3. **WP-09** — a **bundle assembler** that emits one inspectable per-run artifact
   binding the already-existing pieces.
4. **WP-10** — the **external-rank** steps (DOI, JOSS, public benchmark table).

WP-08's existing `evaluate_promotion` ladder (UNOBSERVED → MEASURED with a
null + simpler-baseline + replay + provenance + falsifier + negative-evidence
conjunction, no state-skipping, no synthetic→real promotion) is **stricter** than
the act's WP-08 sketch and is the canonical claim-tier authority. New work
adopts it rather than introducing a second ladder.

## Machine-readable coverage (validated)

<!-- WP-COVERAGE-DATA -->
```json
{
  "schema_version": 1,
  "work_packages": [
    {
      "id": "WP-01",
      "title": "Contextualization frame",
      "status": "PARTIAL",
      "artifacts": ["results/RUN_MANIFEST.json"],
      "gap": "no fail-closed upfront context contract binding problem/universe/window/target/tier/nulls/stop-rule before a run"
    },
    {
      "id": "WP-02",
      "title": "L2 microstructure contract",
      "status": "EXISTS",
      "artifacts": [
        "research/microstructure/BASELINE.L2.md",
        "research/microstructure/attribution.py",
        "research/microstructure/conditional_transfer_entropy.py"
      ]
    },
    {
      "id": "WP-03",
      "title": "Manifold substrate",
      "status": "EXISTS",
      "artifacts": [
        "core/indicators/market_state_contract.py",
        "core/indicators/kuramoto_ricci_composite.py"
      ]
    },
    {
      "id": "WP-04",
      "title": "Ricci curvature field",
      "status": "EXISTS",
      "artifacts": [
        "core/indicators/ricci.py",
        "core/indicators/temporal_ricci.py",
        "core/physics/forman_ricci.py",
        "core/kuramoto/ricci_flow_engine.py"
      ]
    },
    {
      "id": "WP-05",
      "title": "Kuramoto phase field",
      "status": "EXISTS",
      "artifacts": [
        "core/kuramoto/kuramoto_ricci_engine.py",
        "core/kuramoto/metrics.py"
      ]
    },
    {
      "id": "WP-06",
      "title": "Fused Ricci-Kuramoto kernel",
      "status": "EXISTS",
      "artifacts": [
        "core/indicators/kuramoto_ricci_composite.py",
        "scripts/integrate_kuramoto_ricci.py"
      ]
    },
    {
      "id": "WP-07",
      "title": "Recursive falsification loop",
      "status": "PARTIAL",
      "artifacts": [
        "core/kuramoto/oos_validation.py",
        "analytics/signals/null_baseline.py",
        "geosync_hpc/nulls/dynamic_null_model.py",
        "docs/research/D002J_NULL_MODEL_CONTRACTS.md",
        "docs/research/D002J_NULL_MODEL_HIERARCHY.md"
      ],
      "gap": "components present; no single end-to-end recursive driver binding them"
    },
    {
      "id": "WP-08",
      "title": "Claim-tier promotion governance",
      "status": "EXISTS",
      "artifacts": [
        "governance/CLAIM_MATURITY.yaml",
        "analytics/signals/claim_maturity.py",
        "application/governance/claim_ledger.py",
        "scripts/ci/check_claim_maturity.py",
        "docs/schemas/governance/claim_ledger.schema.json"
      ]
    },
    {
      "id": "WP-09",
      "title": "Machine-verification bundle",
      "status": "PARTIAL",
      "artifacts": [
        "manifests/research/release_provenance.v1.json",
        "results/RUN_MANIFEST.json"
      ],
      "gap": "no single per-run assembler binding context+data+nulls+baselines+OOS+cost+claim-card"
    },
    {
      "id": "WP-10",
      "title": "External-rank path",
      "status": "GAP",
      "artifacts": [],
      "gap": "no Zenodo DOI release or JOSS-ready draft"
    }
  ]
}
```
