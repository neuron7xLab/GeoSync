# Fast-lane backlog ledger

## Context

`python-fast-shard` (the fast/default pytest lane, ~17.6k tests) had been
passing **vacuously** on `main` and every PR: `pytest.ini` already carries `-q`,
and the shard added a second `-q` to `--collect-only`, flipping node-id output
to a per-file summary so the `grep '::'` matched nothing and every shard logged
`0/0 selected tests — shard passes`. The lane ran **zero tests** for its entire
history.

PR #1153 restored the oracle (real node-id collection + a fail-closed guard:
zero total collection after the test-surface filter is now FATAL). Restoring it
surfaced the latent backlog the vacuum had hidden.

## Measured backlog shape (PR #1153, `--maxfail=20`)

- total collected: **~17,617** fast node ids
- selected per shard: 4494 / 4373 / 4302 / 4448
- failure clusters: **5**

| cluster | files | issue | lane | status |
|---|---|---|---|---|
| `cli_import_shadowing` | 4 | [#1158](https://github.com/neuron7xLab/GeoSync/issues/1158) | ci-import-architecture | quarantined |
| `acceptor_corpus_legacy_mapping` | 1 | [#1170](https://github.com/neuron7xLab/GeoSync/issues/1170) | governance-schema | quarantined |
| `analytics_import_shadowing` | 1 | [#1171](https://github.com/neuron7xLab/GeoSync/issues/1171) | ci-import-architecture | quarantined |
| `adversarial_invariant_count` | 1 | [#1172](https://github.com/neuron7xLab/GeoSync/issues/1172) | governance-registry | quarantined |
| `numerical_policy_unregistered_clamps` | 1 | [#1173](https://github.com/neuron7xLab/GeoSync/issues/1173) | physics-numerical-policy | quarantined |

**Cluster-count correction (was "1").** The original measurement reported a
single cluster because the four `cli_import_shadowing` import errors aborted
collection before the rest of the fast surface was reached. Once those files
were quarantined, the oracle ran further and surfaced four more deterministic
clusters that the import failures (and, before them, the vacuum) had hidden.
They were latent, not absent — exactly the backlog the restored oracle exists
to expose. None is a flaky/env failure; each is tracked and drained per cluster.

### cli_import_shadowing

`cli/` (repo root, holds `amm_cli.py` / `geosync_cli.py`) is not declared in
`[tool.setuptools.packages.find]`, while `scripts` is. The same-named module
`scripts/cli.py` (`scripts.cli`) shadows the `cli` package under the CI editable
install, so top-level `cli` resolves to `scripts/cli.py` and four test modules
fail to import. `--collect-only` resolves `cli` correctly; the explicit-node-id
run step does not — an invocation-dependent sys.path/rootdir difference. Adding
`cli/__init__.py` alone did not fix CI, so the real fix is an import-architecture
change (declare `cli`, or de-collide `scripts/cli.py`) tracked in #1158 as a
separate PR, outside this CI-oracle PR's boundary.

Quarantined files (`tests/ci/fast_quarantine.txt`):
- `tests/unit/cli/test_amm_cli.py`
- `tests/unit/cli/test_cli_golden.py`
- `tests/unit/cli/test_fete_backtest_cli.py`
- `tests/unit/test_geosync_cli.py`

**Expiry:** when `from cli import amm_cli` resolves to `cli/amm_cli.py` in CI,
remove the entries and re-enable the tests (the gate enforces that the
post-subtraction collection stays non-empty and that every quarantine entry
carries an issue ref).

### acceptor_corpus_legacy_mapping

`tests/governance/test_typed_models.py` loads every commit acceptor. Several
`.claude/commit_acceptors/*.yaml` still declare `required_python_symbols` in the
legacy mapping form `[{file, symbol}]` instead of the canonical
`"<module>::<symbol>"` string — the same schema drift fixed in the root
`commit_acceptors/` by #1160, with stragglers left under `.claude/`. Confirmed
offenders: `forman-ricci-margin-onset-honesty.yaml`,
`governance-verification-protocol-binding.yaml`,
`import-arch-geosync-risk-consolidation.yaml` (+`calibration-contract.yaml` in
CI). **Expiry:** all `.claude/commit_acceptors` use the `module::symbol` string
form; drained in #1170.

### analytics_import_shadowing

`tests/unit/analytics/test_descriptor_capsule.py::test_cli_smoke` runs the
descriptor-capsule CLI in a subprocess, which fails with
`ModuleNotFoundError: No module named 'analytics'`. Same import-architecture
class as `cli_import_shadowing`: the `analytics` package is not declared in
`[tool.setuptools.packages.find]`, so it is absent under the CI editable install
in a fresh subprocess. **Expiry:** `analytics` imports in a fresh subprocess
under the CI editable install; drained in #1171.

### adversarial_invariant_count

`tests/adversarial_invariants/test_invariant_fail_closed.py:34` asserts
`len(invariant_ids) == 97`, but `.claude/physics/INVARIANTS.yaml` registers
**106** (canonical per CLAUDE.md). The hardcoded count is stale. The drain must
not merely bump the number: it must confirm all 106 registered invariants carry
a negative vector that fails closed under 50ms (the test's real purpose).
**Expiry:** the count assertion matches the canonical registry size; drained in
#1172.

### numerical_policy_unregistered_clamps

`tests/tools/test_numerical_policy.py::test_physics_tree_has_zero_unregistered_clamps`
reports four clamps in the scoped physics tree not declared in the
numerical-policy registry: `core/kuramoto/engine.py:295` (`saturate_clip`) and
`core/physics/forman_ricci.py:175/269/375`. Some originate from #1150/#1152
Forman-Ricci work that merged under the vacuous oracle. **Expiry:** the physics
tree has zero unregistered clamps (each registered with a declared bound, or
refactored if unjustified); drained in #1173.

## Invariant

The fast gate is now **real-green, not fake-green**: it runs ~17.6k tests,
subtracts only the explicitly-quarantined-with-issue files, fails if collection
collapses to zero, and fails on any new unquarantined failure.
