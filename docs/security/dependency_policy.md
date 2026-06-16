# Dependency Security-Floor Policy

## Source of truth

`constraints/security.txt` is the **single source of truth** for the patched
floor of every security-critical package. Each entry pins the lowest version
that is not subject to a known advisory.

## Invariant

For every package in `SECURITY_CRITICAL`, the declared lower bound in each
manifest must be **`>=` the `constraints/security.txt` floor**. A lower bound
below the floor silently re-admits a vulnerable version.

`pip install .` resolves against `pyproject.toml`, so a stale `pyproject` floor
bypasses any floor raised only in `requirements.txt`. The gate exists to make
that drift fail closed.

## Security-critical packages

```
pyjwt  cryptography  aiohttp  starlette  tornado  python-multipart  fastapi
```

## Manifest ownership & enforcement scope

| Manifest | Owner | Gate scope |
| --- | --- | --- |
| `constraints/security.txt` | security policy | source of truth |
| `pyproject.toml` | this gate | **strict** (exit 1 on violation) |
| `requirements.txt` | security-sweep lane (PR #1111) | advisory (reported) |
| `requirements-backend.txt` | security-sweep lane (PR #1111) | advisory (reported) |

`requirements*.txt` floors are raised by the in-flight security-sweep PR #1111.
Promoting them to strict in this gate while #1111 is open would create a
cross-branch write conflict, so they are **advisory** here. Once #1111 merges,
run the gate with `--all-strict` and tighten the live test to assert zero
advisory violations.

## Gate

```bash
python tools/security/check_dependency_manifest_consistency.py            # strict pyproject
python tools/security/check_dependency_manifest_consistency.py --all-strict # + requirements
pytest -q tests/security/test_dependency_manifest_consistency.py
```

The pytest test is collected by the fast PR gate, so the pyproject invariant is
enforced on every PR without a dedicated workflow step.

## Exception policy

No security-floor exception is admitted without:

```
package · advisory id · affected versions · is the vulnerable code path used? ·
mitigation · scanner/gate that enforces the mitigation · expiry/recheck date · owner
```

Record exceptions in `docs/security/vulnerability_exception_ledger.md` (create
on first exception).

## Related existing controls (not duplicated here)

- `scripts/security/check_dependency_drift.py` — validates manifest specifiers
  against the **lockfile** (orthogonal: lock satisfaction, not cross-manifest
  floor consistency).
- `tests/security/test_checkpoint_loading.py` — enforces `torch.load(...,
  weights_only=True)`; the repository contains no `torch.jit.{script,trace,load}`
  usage, so the torch deserialization advisory dismissal is enforced by that
  test rather than a separate scanner.
