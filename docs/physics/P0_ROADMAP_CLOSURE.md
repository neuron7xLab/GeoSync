# P0 Roadmap Closure Ledger

Date: 2026-05-29

## Closure principle

A P0 item is closed only when it has all four surfaces:

1. test or executable validator;
2. evidence artifact;
3. claim boundary;
4. fail-closed or explicitly non-promoting gate.

## Current state

| Item | Status | Evidence |
|---|---|---|
| 13 historic zero-witness P0 invariants | PARTIAL_CLOSED | `docs/physics/strict_witness_surface.md` records T9-T16 strict witnesses; registry metadata and evidence matrix still need sync |
| C1/C2 audit blocking | OPEN | workflow still keeps audit command report-only with `|| true` |
| Orphan physics tests below 100 | OPEN | requires local validator sweep and migration batch |
| Kelly theory boundary | CLOSED_DOC | `docs/physics/theory/KELLY_THEORY.md` |
| OMS theory boundary | CLOSED_DOC | `docs/physics/theory/OMS_THEORY.md` |
| SignalBus theory boundary | CLOSED_DOC | `docs/physics/theory/SIGNALBUS_THEORY.md` |
| HPC theory boundary | CLOSED_DOC | `docs/physics/theory/HPC_THEORY.md` |
| Criticality FSS battery | PREREGISTERED | `experiments/criticality_fss/README.md` and `schema.json` |
| OOS signed audit artifact | RETIRED_PENDING | existing audit report marks OOS evidence RETRACTED |
| README three-surface split | PARTIAL_CLOSED | `docs/README_SURFACES.md` defines verified kernel / measured experiments / hypothesis sandbox; README still needs a direct hook |

## Metadata drift

`docs/physics/evidence_matrix.md` is generated from `.claude/physics/INVARIANTS.yaml`. If a witness exists in tests but the YAML row lacks `tests:` metadata, the generated matrix will show `—`. The strict witness surface ledger captures this bridge until registry metadata is updated and the matrix is regenerated.

## Next machine step

Run locally:

    python .claude/physics/validate_tests.py tests/ --summary
    python .claude/physics/validate_tests.py core/ --audit-code --summary

Then update this ledger with exact counts. Do not hand-fill counts.

## Promotion rule

No README headline may claim live alpha, verified physical markets, or criticality until the corresponding artifact passes the claims-evidence gate.
