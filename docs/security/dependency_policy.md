# Dependency Security-Floor Policy

## Source of truth

`constraints/security.txt` is the **single source of truth** for the minimum
accepted floor of every security-critical package.

## Invariant

For every package in `SECURITY_CRITICAL`, the declared lower bound in each
manifest must be **`>=` the `constraints/security.txt` floor**. A lower bound
below the floor re-opens an already closed dependency boundary.

`pip install .` resolves against `pyproject.toml`, while service and CI install
paths may resolve against `requirements.txt` or `requirements-backend.txt`. The
gate exists to make drift across those manifests fail closed.

## Security-critical packages

```
pyjwt  cryptography  aiohttp  starlette  tornado  python-multipart  fastapi
```

## Manifest ownership & enforcement scope

| Manifest | Owner | Gate scope |
| --- | --- | --- |
| `constraints/security.txt` | security policy | source of truth |
| `pyproject.toml` | package/runtime lane | **strict** by default |
| `requirements.txt` | service/CI lane | **strict under `--all-strict`** |
| `requirements-backend.txt` | backend service lane | **strict under `--all-strict`** |

Task 2 closes the previous advisory gap: `--all-strict` is now the release-grade
check for dependency floor consistency across all checked manifests. The default
mode still reports requirements drift as advisory for fast local diagnosis, but
release evidence must use `--all-strict`.

## Gate

```bash
python tools/security/check_dependency_manifest_consistency.py            # strict pyproject
python tools/security/check_dependency_manifest_consistency.py --all-strict # all checked manifests
pytest -q tests/security/test_dependency_manifest_consistency.py
```

The pytest test is collected by the fast PR gate and asserts the live all-strict
invariant, so a stale requirement floor below `constraints/security.txt` fails
before it can be promoted as release evidence.

## Exception policy

No security-floor exception is admitted without:

```
package · tracking id · affected range · is the code path used? · mitigation ·
scanner/gate that enforces the mitigation · expiry/recheck date · owner
```

Record exceptions in `docs/security/vulnerability_exception_ledger.md` (create
on first exception).

## Related existing controls (not duplicated here)

- `scripts/security/check_dependency_drift.py` — validates manifest specifiers
  against the **lockfile** (orthogonal: lock satisfaction, not cross-manifest
  floor consistency).
- `tests/security/test_checkpoint_loading.py` — enforces safe checkpoint loading;
  this is a separate serialization boundary, not a duplicate of this scanner.
