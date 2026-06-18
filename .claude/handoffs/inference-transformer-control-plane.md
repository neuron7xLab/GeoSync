# Inference Transformer Control Plane Handoff

## Scope

Implement only the research inference transformer contract layer for PR #1204.

Do not change the claim tier for `ricci_microstructure_v1`.

## First read order

1. `configs/research/geosync_inference_transformer.v1.yaml`
2. `schemas/research/inference_transformer_contract.schema.json`
3. `data/research/inference_transformer_blocks.v1.json`
4. `scripts/ci/check_inference_transformer_contract.py`
5. `tests/ci/test_inference_transformer_contract.py`
6. `src/geosync/research/transformer/contracts.py`

## Required verification

```bash
python scripts/ci/check_inference_transformer_contract.py
pytest -q tests/ci/test_inference_transformer_contract.py
```

## Acceptance

- Contract validator exits zero on committed seed files.
- Validator exits non-zero on malformed local variants.
- Typed certificate objects preserve `HYPOTHESIS` and `ABSTAIN` semantics.
- No runtime execution surface is added.
- No empirical claim is promoted.

## Next implementation seam

Add a deterministic placeholder pipeline only after validator and typed contracts are green.

⊛
