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
- failure clusters: **1**

| cluster | files | issue | lane | status |
|---|---|---|---|---|
| `cli_import_shadowing` | 4 | [#1158](https://github.com/neuron7xLab/GeoSync/issues/1158) | ci-import-architecture | quarantined |

No other cluster (missing_dependency, stale_fixture, path_contract,
true_logic_regression, flaky_env) appeared: every other fast test passed.

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

## Invariant

The fast gate is now **real-green, not fake-green**: it runs ~17.6k tests,
subtracts only the explicitly-quarantined-with-issue files, fails if collection
collapses to zero, and fails on any new unquarantined failure.
