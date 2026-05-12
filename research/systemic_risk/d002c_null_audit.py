# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""D-002C C2.4 — Null audit (permutation test) for the Signal Amplification Sweep.

Mission
=======
A precursor cohort produced by the sweep yields a per-seed (precursor, null)
pair (CRN-paired by construction in C2.3). The aggregate signal is

    S = |mean(precursor) − mean(null)| = |signal_mean|

If the precursor truly carries information, swapping the precursor/null
labels at random will DESTROY the signal — most label-permuted shuffles
produce |signal_mean| << S. If swapping doesn't destroy it, the
"signal" is a label artefact, statistical noise, or a metric that
saturates regardless of precursor injection. The audit's job is to
catch that failure BEFORE the sweep's headline number propagates.

Method
======
For ``n_shuffles`` independent permutations, draw a Bernoulli(0.5)
mask of length ``n_seeds`` per shuffle. Inside each pair, swap the
(precursor, null) values where the mask is 1; recompute
|mean_diff|. The empirical p-value is the fraction of shuffles
with |shuffled| ≥ |unshuffled|. PASS iff ``p_value < 0.05``.

This is the standard paired-difference permutation test for the
sign of a paired effect; bit-exact deterministic in ``rng_seed``,
``n_shuffles``, and the input arrays.

Strict scope
============
Permutation-test logic ONLY. NO sweep launch. NO claim layer. NO
metric estimation. NO promotion of a verdict into a tier — the
sweep's headline claim layer reads the verdict and refuses to
publish if FAIL.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Locked defaults
# ---------------------------------------------------------------------------
DEFAULT_N_SHUFFLES: Final[int] = 100
DEFAULT_RNG_SEED: Final[int] = 42
DEFAULT_P_VALUE_THRESHOLD: Final[float] = 0.05
MIN_N_SHUFFLES: Final[int] = 10
MIN_N_SEEDS: Final[int] = 2

logger = logging.getLogger(__name__)


class NullAuditInvalid(RuntimeError):
    """Bad input to the null audit (empty arrays, shape mismatch, ...)."""


@dataclass(frozen=True)
class NullAuditResult:
    """Per-cell null audit verdict.

    ``unshuffled_greater_than_median`` is the load-bearing sanity check:
    the true signal must dominate the median of the label-shuffled null
    distribution. ``p_value_empirical`` is the formal verdict driver.
    """

    n_seeds: int
    n_shuffles: int
    unshuffled_abs_signal: float
    shuffled_abs_signal_median: float
    shuffled_abs_signal_p95: float
    unshuffled_greater_than_median: bool
    p_value_empirical: float
    verdict: str  # "PASS" | "FAIL"
    sha256: str
    rng_seed: int = DEFAULT_RNG_SEED
    p_value_threshold: float = DEFAULT_P_VALUE_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# Atomic capsule writer (inlined to match sibling-module discipline)
# ---------------------------------------------------------------------------


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    """tmp + fsync + os.replace. Cleanup on exception, never mask original."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, sort_keys=True, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Core permutation test
# ---------------------------------------------------------------------------


def run_null_audit(
    precursor_values: NDArray[np.float64],
    null_values: NDArray[np.float64],
    *,
    n_shuffles: int = DEFAULT_N_SHUFFLES,
    rng_seed: int = DEFAULT_RNG_SEED,
    p_value_threshold: float = DEFAULT_P_VALUE_THRESHOLD,
) -> NullAuditResult:
    """Paired-difference permutation test for the signal's reality.

    Parameters
    ----------
    precursor_values, null_values
        1-D arrays of shape ``(n_seeds,)``. Paired by index: entry ``i``
        in both arrays is the metric from seed ``i``'s precursor and
        null run (CRN-paired upstream in C2.3).
    n_shuffles
        Number of random label-permutations. Must be >= ``MIN_N_SHUFFLES``
        (10). 100 is the locked default — gives a discretised p-value
        resolution of 0.01.
    rng_seed
        Seeds numpy's PCG64. Determinism: same arrays + same
        (rng_seed, n_shuffles) → bit-exact identical result.
    p_value_threshold
        PASS iff ``p_value_empirical < threshold``. Default 0.05.

    Returns
    -------
    NullAuditResult
        Frozen dataclass with canonical sha256 over the load-bearing
        fields. See the module docstring for the verdict semantics.

    Raises
    ------
    NullAuditInvalid
        Empty arrays, mismatched shapes, non-1-D inputs, non-finite
        values, n_shuffles below the floor, or non-finite threshold.
    """
    precursor_values = np.asarray(precursor_values, dtype=np.float64)
    null_values = np.asarray(null_values, dtype=np.float64)
    if precursor_values.ndim != 1 or null_values.ndim != 1:
        raise NullAuditInvalid(
            f"inputs must be 1-D; got shapes precursor={precursor_values.shape} "
            f"null={null_values.shape}"
        )
    if precursor_values.shape != null_values.shape:
        raise NullAuditInvalid(
            f"precursor and null arrays must have identical shape; got "
            f"precursor={precursor_values.shape} null={null_values.shape}"
        )
    n_seeds = int(precursor_values.shape[0])
    if n_seeds < MIN_N_SEEDS:
        raise NullAuditInvalid(f"need >= {MIN_N_SEEDS} paired seeds; got n_seeds={n_seeds}")
    if not np.all(np.isfinite(precursor_values)) or not np.all(np.isfinite(null_values)):
        raise NullAuditInvalid("inputs must be finite (no NaN / Inf)")
    if n_shuffles < MIN_N_SHUFFLES:
        raise NullAuditInvalid(f"n_shuffles must be >= {MIN_N_SHUFFLES}; got {n_shuffles}")
    if not np.isfinite(p_value_threshold) or not (0.0 < p_value_threshold < 1.0):
        raise NullAuditInvalid(
            f"p_value_threshold must be finite and in (0, 1); got {p_value_threshold}"
        )

    # True (unshuffled) signal
    unshuffled_abs = float(abs(precursor_values.mean() - null_values.mean()))

    # Permutation distribution: per-pair Bernoulli(0.5) label swap
    rng = np.random.default_rng(rng_seed)
    # signs[i, j] in {0, 1}; if 1, swap pair j in shuffle i
    masks = rng.integers(0, 2, size=(n_shuffles, n_seeds), dtype=np.int64).astype(bool)
    # Under swap, new_precursor = mask*null + (1-mask)*precursor
    # new_null      = mask*precursor + (1-mask)*null
    # mean_diff(shuffled) = mean(new_precursor - new_null)
    #                     = mean((1 - 2*mask) * (precursor - null))
    diff = precursor_values - null_values
    sign = 1.0 - 2.0 * masks.astype(np.float64)  # +1 keep, -1 swap
    shuffled_means = (sign * diff[np.newaxis, :]).mean(axis=1)
    shuffled_abs = np.abs(shuffled_means)
    shuffled_median = float(np.median(shuffled_abs))
    shuffled_p95 = float(np.percentile(shuffled_abs, 95.0))

    # Empirical p-value: fraction with |shuffled| >= |unshuffled|
    n_ge = int(np.sum(shuffled_abs >= unshuffled_abs))
    p_value = n_ge / float(n_shuffles)
    verdict = "PASS" if p_value < p_value_threshold else "FAIL"

    payload: dict[str, Any] = {
        "n_seeds": n_seeds,
        "n_shuffles": int(n_shuffles),
        "unshuffled_abs_signal": unshuffled_abs,
        "shuffled_abs_signal_median": shuffled_median,
        "shuffled_abs_signal_p95": shuffled_p95,
        "unshuffled_greater_than_median": unshuffled_abs > shuffled_median,
        "p_value_empirical": p_value,
        "verdict": verdict,
        "rng_seed": int(rng_seed),
        "p_value_threshold": float(p_value_threshold),
    }
    sha = _sha256(payload)
    return NullAuditResult(
        n_seeds=n_seeds,
        n_shuffles=int(n_shuffles),
        unshuffled_abs_signal=unshuffled_abs,
        shuffled_abs_signal_median=shuffled_median,
        shuffled_abs_signal_p95=shuffled_p95,
        unshuffled_greater_than_median=unshuffled_abs > shuffled_median,
        p_value_empirical=p_value,
        verdict=verdict,
        sha256=sha,
        rng_seed=int(rng_seed),
        p_value_threshold=float(p_value_threshold),
    )


# ---------------------------------------------------------------------------
# Capsule replay path: parse a sweep capsule and run an audit per cell
# ---------------------------------------------------------------------------


def _extract_per_seed_arrays(
    cell: dict[str, Any],
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Try to extract paired (precursor, null) per-seed arrays from a cell.

    Returns ``None`` if the cell does not carry enough per-seed data to
    audit. The keys we accept are intentionally a small union — a stub
    capsule from C2.3 that only carries aggregate variances will be
    skipped cleanly, with a log line.
    """
    candidates_precursor = (
        "precursor_per_seed",
        "per_seed_precursor",
        "precursor_values",
    )
    candidates_null = ("null_per_seed", "per_seed_null", "null_values")
    p_arr: NDArray[np.float64] | None = None
    n_arr: NDArray[np.float64] | None = None
    for k in candidates_precursor:
        if k in cell and cell[k] is not None:
            p_arr = np.asarray(cell[k], dtype=np.float64)
            break
    for k in candidates_null:
        if k in cell and cell[k] is not None:
            n_arr = np.asarray(cell[k], dtype=np.float64)
            break
    if p_arr is None or n_arr is None:
        return None
    if p_arr.shape != n_arr.shape or p_arr.ndim != 1 or p_arr.size < MIN_N_SEEDS:
        return None
    return p_arr, n_arr


def run_null_audit_from_capsule(
    capsule_path: Path,
    *,
    n_shuffles: int = DEFAULT_N_SHUFFLES,
    rng_seed: int = DEFAULT_RNG_SEED,
    p_value_threshold: float = DEFAULT_P_VALUE_THRESHOLD,
) -> tuple[NullAuditResult, ...]:
    """Run a null audit on each per-cell entry in a sweep capsule.

    Stub-friendly: cells that lack per-seed data are skipped with a
    logger.info line citing the cell's substrate/metric id; cells that
    DO carry per-seed arrays are audited and a result is appended.
    A capsule with NO auditable cells returns an empty tuple — the
    caller decides if that is a failure or simply means C2.3-style
    aggregate-only output.

    Parameters
    ----------
    capsule_path
        Path to a JSON capsule written by the sweep runner (C2.4
        session A) or by C2.3 (CRN validator). The shape we accept
        is the same dict-of-lists schema both use.
    n_shuffles, rng_seed, p_value_threshold
        Forwarded to :func:`run_null_audit` for each auditable cell.

    Raises
    ------
    NullAuditInvalid
        If the capsule path does not exist or does not decode as JSON.
    """
    capsule_path = Path(capsule_path)
    if not capsule_path.exists():
        raise NullAuditInvalid(f"capsule does not exist: {capsule_path}")
    try:
        raw = capsule_path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise NullAuditInvalid(f"could not parse capsule {capsule_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise NullAuditInvalid(f"capsule root must be a JSON object; got {type(data).__name__}")
    results_list = data.get("results")
    if not isinstance(results_list, list):
        logger.info(
            "null_audit: capsule %s has no 'results' list — nothing to audit",
            capsule_path,
        )
        return ()
    out: list[NullAuditResult] = []
    for cell in results_list:
        if not isinstance(cell, dict):
            continue
        arrays = _extract_per_seed_arrays(cell)
        cell_id = (
            f"{cell.get('substrate_id', '?')}×{cell.get('metric_id', '?')}"
            f"@N={cell.get('N', '?')},λ={cell.get('lambda_', '?')}"
        )
        if arrays is None:
            logger.info("null_audit: cell %s lacks per-seed data — skipping", cell_id)
            continue
        p_arr, n_arr = arrays
        res = run_null_audit(
            p_arr,
            n_arr,
            n_shuffles=n_shuffles,
            rng_seed=rng_seed,
            p_value_threshold=p_value_threshold,
        )
        out.append(res)
    return tuple(out)


__all__ = [
    "DEFAULT_N_SHUFFLES",
    "DEFAULT_RNG_SEED",
    "DEFAULT_P_VALUE_THRESHOLD",
    "MIN_N_SHUFFLES",
    "MIN_N_SEEDS",
    "NullAuditInvalid",
    "NullAuditResult",
    "run_null_audit",
    "run_null_audit_from_capsule",
]
