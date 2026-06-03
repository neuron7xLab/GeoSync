# GeoSync Architectural Connectome Gate

## Purpose

The connectome gate converts architectural intent into a machine-checkable import contract.  It prevents silent cross-domain leakage between GeoSync logical substrates before that leakage reaches runtime, review, or production.  This is not a style rule.  It is a structural safety boundary.

## First-principles model

GeoSync treats architecture as a directed graph of admissible information flow:

```text
source path -> owning domain -> imported module -> contract verdict
```

A Python module belongs to exactly one governed domain when its path is covered by `docs/architecture/connectome.yaml`.  Every import edge discovered inside that module is classified against the same contract.

The gate follows three invariants:

1. **Ownership invariant**: a governed file must resolve to one and only one domain.
2. **Admission invariant**: cross-domain imports are denied unless explicitly listed in `allowed_imports`.
3. **Veto invariant**: `forbidden_imports` always dominates `allowed_imports`.

If an invariant cannot be evaluated, the checker fails closed instead of guessing.  Apparently civilization needs software to say this out loud.

## Neuro-architectural mapping

The domain names are cognitive metaphors, but the enforcement is plain static analysis:

| Domain | Value function | Enforced boundary |
|---|---|---|
| `sensory` | ingest and normalize inputs | cannot depend on execution or admission machinery |
| `hippocampus` | retain context and regime memory | may read sensory substrate, cannot reach motor substrate |
| `prefrontal` | govern claims and policy | may inspect sensory and memory substrate, cannot execute |
| `amygdala` | detect risk and trigger kill conditions | cannot couple directly to physics/features internals |
| `basal_ganglia` | admit or suppress action | may read risk and governance, cannot bypass physics boundaries |
| `motor` | reserved execution substrate | constrained from research and physics internals |

This preserves a one-way cognitive pipeline: perception -> memory/governance/risk -> admission -> execution.  Code that inverts that direction must first change the contract, which makes the architectural decision reviewable.

## What the validator detects

`tools/architecture/check_connectome.py` parses Python AST and detects:

- `import x`
- `from x import y`
- relative imports such as `from ..motor import venue`
- literal dynamic imports through `importlib.import_module(...)`
- alias dynamic imports such as `import importlib as il; il.import_module(...)`
- callable aliases such as `from importlib import import_module as load_module`
- `__import__(...)` when the first argument is a string literal

Non-literal dynamic imports are intentionally not resolved because that would require runtime execution.  They remain outside this static gate and should be reviewed through separate runtime or policy controls.

## Evidence protocol

For every change to the connectome contract or validator, the minimum evidence is:

```bash
python tools/architecture/check_connectome.py
python tools/architecture/check_connectome.py --format json
python -m pytest tests/tools/architecture/test_check_connectome.py -q
```

A valid PR should explain which domain edge was added, removed, or denied, and why the change does not weaken the one-way information-flow model.

## Claim boundary

This gate proves only that governed Python import edges match the declared contract.  It does not prove runtime behavior, causal safety, market correctness, biological fidelity, or absence of reflection-based loading.  Those are separate claims and must not be smuggled into this evidence.  Very tragic, but physics remains annoyingly specific.
