# Co-located test wiring + guard robustness

This change closes three measurement/correctness holes surfaced while triaging
the red `Release coverage gate (90%)` lane on `main`.

## 1. Co-located test suites never ran under the coverage gate
`make coverage-baseline` ran only `pytest tests/`. Seven test suites that
physically live **under production packages** (so the top-level `tests/` root
never collects them) were therefore never executed during release-coverage
measurement, even though they pass and exercise real release surface:

| suite | tests | surface exercised |
|-------|------:|-------------------|
| `core/neuro/tests` | 202 | core neuromodulators |
| `analytics/tests` | 28 | analytics |
| `analytics/fpma/tests` | 9 | analytics/fpma |
| `analytics/regime/tests` | 42 | analytics/regime |
| `analytics/signals/tests` | 16 | analytics/signals |
| `markets/orderbook/tests` | 10 | markets/orderbook |
| `geosync/neural_controller/tests` | 36 | neural_controller |

They are wired in as a **second, `--cov-append` invocation** (not appended to
the `tests/` argument list): under `--import-mode=importlib` several share a
basename with files under `tests/`, so a single combined collection raises
`import file mismatch`. Two isolated collection roots accumulate into one
`.coverage` file with zero collisions. The canonical `junit.xml` is still
written by the `tests/` run (it carries the gated-claim falsifiers); the
co-located run writes `junit_colocated.xml` to avoid clobbering it.

Measured effect (CI `coverage.xml` ∪ co-located coverage): release line
coverage **77.89% → 84.45%** (+6.56 pp; 4,920 previously-missing in-scope lines
now covered). This does not by itself reach the 90% release gate — the residual
is ~14k missing lines of genuine production logic (not demos/CLIs), a separate
test-authoring campaign — but it removes the measurement-completeness defect
that made the number meaningless.

## 2. `neural_controller` package was un-importable
`core/params.py` is the package's config facade — `__init__`, `bridge`,
`sensory_pipeline`, `adapter` and the test-suite all import every config from
it — but `PredictiveConfig`, `SensoryConfig` and `OBSERVATION_KEYS` physically
live in `core/neuro_params.py` and were never re-exported. `import
geosync.neural_controller` raised `ImportError`, leaving the whole subpackage at
0% coverage and its tests at 4 collection errors. Fixed by re-exporting the
three symbols from the facade (PEP 484 redundant-alias form; no circular import:
`neuro_params` depends only on `dataclasses`).

## 3. `analytics/signals` tests used a pandas alias removed in 2.2
`pd.date_range(..., freq="T")` — the `"T"` minute alias was removed in
pandas ≥ 2.2 in favour of `"min"`. Five tests raised `ValueError: Invalid
frequency: T`. Repointed to `"min"`.

## 4. Namespace-integrity guard false positives on hidden trees
`scripts/check_namespace_integrity.py` rglob-scanned the whole repo, excluding
only `.venv/venv/.tox/node_modules/__pycache__`. It flagged
`.claude/worktrees/agent-*/src/geosync/...` (ephemeral agent worktrees that
carry their own canonical tree) as violations on any developer machine — yet
those paths do not exist on CI's clean checkout, so the bug was invisible where
the guard is enforced. The scan now skips any hidden directory (`startswith
"."`) plus build/dist. Three regression tests pin both directions: hidden trees
are ignored, visible rogue canonical markers are still flagged.

## 5. Root `conftest.py` torch probe aborted the whole fast lane
The torch-availability probe `import torch as _torch_probe` was guarded only by
`except (ImportError, OSError, ValueError)`. Under Python 3.11 torch emits an
import-time `DeprecationWarning`, and pytest's `-W error::DeprecationWarning`
(from `pytest.ini`) re-raises it as an exception that is none of those types, so
it escapes the probe and **aborts conftest collection** — every `python-fast-shard`
job then collects 0 ids and fail-closes ("0 fast tests collected"). This blocks
the fast lane on every PR on the CI runner (it does not reproduce on a 3.12 dev
box whose torch build is quiet). Fixed by suppressing warnings for the probe
import only (`warnings.catch_warnings()` + `simplefilter("ignore")`); test-time
warning strictness is unchanged. (Same minimal fix as the in-flight
`fix/fast-shard-torch-probe-deprecation` branch; included here so this PR's own
fast lane is real-green rather than blocked by an unrelated environmental bug.)

All of the above are integrity/correctness fixes. No market claim, no claim-tier
promotion, no runtime behaviour change, no lint/type suppressions added.
