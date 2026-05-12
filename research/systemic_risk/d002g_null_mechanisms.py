# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""D-002G — Non-degenerate null mechanisms (M1 + M6).

Rationale
=========
D-002C attempt-2 (RUN_ID ``d002c_canonical_attempt_2_20260512T160318Z``)
emitted ``tier=D002C_REDESIGN_INSUFFICIENT_AT_TESTED_BUDGET`` because
at λ=0 the locked paired-CRN protocol produces
``K_precursor == K_baseline`` bit-identically. The permutation null
audit then collapses to ``p=1.0`` for those 9 λ=0 cells.

D-002G fixes this with two pre-committed mechanisms locked in
``docs/governance/D002G_PREREGISTRATION.yaml``:

* **M1 (primary)** — independent-seed null cohort. For seed ``s`` the
  precursor cohort uses ``s``; the null cohort uses
  ``s + null_seed_offset`` (offset=10000 per
  ``reproducibility.null_seed_offset``). At λ=0 the two K matrices are
  drawn independently — bit-identity is broken while H0 ("no precursor
  effect at λ=0") is preserved.

* **M6 (supplementary)** — placebo coupling. The precursor injection
  is applied to a RANDOM subset of off-diagonal edges with the SAME
  Frobenius-norm shift as the locked precursor (top-10% κ edges for
  ricci, inter-block edges for block_structured, locked sites for
  temporal_coupling). The metric SHOULD NOT detect this fake injection
  — if it does, the metric is a false-positive prone detector that
  responds to any energy injection rather than to substrate topology.

Strict scope
============
Mechanism implementation ONLY. NO sweep execution. NO claim layer.
NO threshold edits. NO substrate API surgery — this module CONSUMES
the existing ``Substrate`` protocol from :mod:`d002c_substrates`
without modification.

Data contract
=============
A :class:`NullRealization` carries the K_baseline matrix produced by
the chosen mechanism plus full content-addressed provenance:
``payload_sha256`` is sha256 over canonical JSON of
``{strategy, base_seed, null_seed, lambda_value, substrate_id, N,
metadata, K_flat_sha}`` where ``K_flat_sha`` is sha256 over
``K_baseline.tobytes()`` with shape+dtype tag. Same inputs → same
sha across machines and processes.

The K_baseline shape is ``(N, N)``. The substrate API produces a
``(T_HORIZON, N, N)`` trajectory; this module returns the static
slice at ``t=PRE_EVENT_START_QUARTER`` (well-defined because at λ=0
the substrate's ``K_baseline`` is broadcast-identical across the
time axis; for M6 we synthesise a single-slice placebo K).

Phase 0 verification (Phase 0a / 0b / 0c) is implemented in
:mod:`d002g_phase0_verification`. R2-B aggregation is implemented in
:mod:`d002g_r2b_gate`. This module is purely the realisation layer.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Literal

import numpy as np
from numpy.typing import NDArray

from .d002c_substrates import (
    PRECURSOR_INJECTION_WINDOW,
    Substrate,
    SubstrateRealization,
)

# ---------------------------------------------------------------------------
# Locked constants (frozen in docs/governance/D002G_PREREGISTRATION.yaml)
# ---------------------------------------------------------------------------

#: ``reproducibility.null_seed_offset`` in the pre-registration. The M1
#: null cohort draws an independent realisation at the offset seed.
NULL_SEED_OFFSET: Final[int] = 10000

#: ``reproducibility.r2_b_random_site_seed`` in the pre-registration.
#: Combined with ``base_seed`` via :func:`deterministic_mix` to seed the
#: M6 placebo edge-selection RNG.
R2_B_RANDOM_SITE_SEED: Final[int] = 99

#: Strategy string literals carried in the payload + downstream null
#: audit. The legacy value ``D002C_PAIRED_CRN_LEGACY`` is used by the
#: ``d002c_sweep_runner`` extension in Phase 7 for backward
#: compatibility with pre-A2 emissions; it is NOT a D-002G mechanism
#: and is intentionally not part of this Literal.
NullStrategy = Literal["M1_INDEPENDENT_SEED", "M6_PLACEBO_COUPLING"]

#: Locked salt mixed with ``base_seed`` to produce the M6 RNG.
M6_PLACEBO_SALT: Final[int] = R2_B_RANDOM_SITE_SEED

_VALID_STRATEGIES: Final[frozenset[str]] = frozenset({"M1_INDEPENDENT_SEED", "M6_PLACEBO_COUPLING"})


class BitIdenticalNullError(RuntimeError):
    """M1 produced ``K_null == K_precursor`` bit-identically.

    Raised when the chosen substrate happens to be insensitive to the
    seed argument at the requested λ. The caller MUST tag such
    (substrate, N) cells as M1-INELIGIBLE and either fall back to the
    M2 topology-preserving shuffle (pre-registration fallback policy)
    or escalate. Silently accepting a bit-identical M1 null would
    reintroduce the exact pathology M1 was designed to remove.
    """


class M6InsufficientCandidatePool(RuntimeError):
    """M6 placebo-coupling candidate edge pool is smaller than support.

    Raised when the off-support pool of upper-triangle edges (i.e.
    ``upper_triangle ∖ original_support``) has fewer entries than the
    privileged ΔK support size. Sampling cannot proceed without either
    (a) reusing privileged sites (forbidden — re-introduces support
    leakage, the very pathology this fix removes) or (b) sampling with
    replacement (forbidden — breaks M6 distinct-edge contract).

    Fail-closed semantics: the cell is REFUSED at the realisation
    layer; callers must tag it ``M6-INELIGIBLE_CANDIDATE_POOL_TOO_SMALL``
    and either widen N or escalate to the M2 fallback per the locked
    pre-registration §4 fallback policy.
    """


class D002GNullInvalid(ValueError):
    """Bad input to :func:`realize_null` / :func:`deterministic_mix`."""


# ---------------------------------------------------------------------------
# Deterministic seed mixing
# ---------------------------------------------------------------------------


def deterministic_mix(base_seed: int, salt: int) -> int:
    """Deterministic uint63 hash of (base_seed, salt) for RNG seeding.

    The M6 placebo mechanism needs an RNG seed that is:

      * **Deterministic** — same (base_seed, salt) → same seed across
        machines / processes / runs.
      * **Distinct across base_seeds** — two cohort seeds that share
        the same salt MUST produce different M6 RNG seeds so the
        per-seed placebo realisations are independent.
      * **Distinct across salts** — domain-separation tag.

    Implementation: pack ``(base_seed, salt)`` as two int64 big-endian
    words, sha256, take the low 63 bits as the seed (numpy's
    ``default_rng`` accepts non-negative int up to ``2**63 - 1``).

    Python's built-in ``hash()`` is forbidden — it is salted per-process
    by PYTHONHASHSEED and would break determinism across processes.

    Raises
    ------
    D002GNullInvalid
        ``base_seed`` or ``salt`` outside the signed-int64 range.
    """
    if not (-(2**63) <= int(base_seed) < 2**63):
        raise D002GNullInvalid(f"base_seed must fit in int64; got {base_seed!r}")
    if not (-(2**63) <= int(salt) < 2**63):
        raise D002GNullInvalid(f"salt must fit in int64; got {salt!r}")
    packed = struct.pack(">qq", int(base_seed), int(salt))
    digest = hashlib.sha256(packed).digest()
    head = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return int(head & ((1 << 63) - 1))


# ---------------------------------------------------------------------------
# Canonical JSON + sha helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _canonical_json(payload: dict[str, Any]) -> str:
    """Stable canonical JSON (sort_keys, tight separators, NaN sentinel)."""

    def _walk(x: Any) -> Any:
        if isinstance(x, float):
            if math.isnan(x):
                return "NaN"
            if math.isinf(x):
                return "Infinity" if x > 0 else "-Infinity"
            return x
        if isinstance(x, (list, tuple)):
            return [_walk(v) for v in x]
        if isinstance(x, dict):
            return {str(k): _walk(v) for k, v in x.items()}
        return x

    return json.dumps(_walk(payload), sort_keys=True, separators=(",", ":"))


def _K_flat_sha(K: NDArray[np.float64]) -> str:
    """sha256 over K_baseline.tobytes() with shape+dtype tag.

    The tag prefix prevents ``(N, N)`` and ``(N*N,)`` arrays with the
    same byte content from colliding. float64 is mandatory; any other
    dtype is refused by :func:`realize_null`.
    """
    if K.dtype != np.float64:
        raise D002GNullInvalid(f"K_baseline.dtype must be float64; got {K.dtype}")
    tag = f"shape={tuple(K.shape)};dtype=float64".encode("utf-8")
    h = hashlib.sha256()
    h.update(tag)
    h.update(K.tobytes(order="C"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# NullRealization data contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NullRealization:
    """Frozen output of one M1 / M6 null realisation.

    Fields
    ------
    K_baseline
        The N×N float64 matrix the downstream Kuramoto integrator must
        consume as the null cohort's K. The existing substrate API
        emits a ``(T_HORIZON, N, N)`` trajectory; this dataclass
        carries the static slice. Callers that integrate the
        trajectory should broadcast back to ``(T_HORIZON, N, N)``.
    strategy
        Which mechanism produced this realisation.
    base_seed
        The precursor cohort seed this null is paired to.
    null_seed
        The seed actually used to draw the null. For M1 this is
        ``base_seed + NULL_SEED_OFFSET``; for M6 it is
        :func:`deterministic_mix` of ``(base_seed, M6_PLACEBO_SALT)``.
    lambda_value
        The cell coordinate. M1 is locked to λ=0 in the canonical
        sweep but the realisation function accepts any λ ≥ 0 so
        Phase 0b's H0-preservation check can probe arbitrary cells.
        M6 requires lambda_value > 0.
    substrate_id
        From :attr:`Substrate.id`.
    N
        Cohort size. Inherited from the canonical sweep N_grid.
    metadata
        Strategy-specific provenance:
          * M1: ``{}`` (no extra state).
          * M6: ``{"placebo_edges_count": int,
                   "placebo_frobenius_norm": float,
                   "precursor_frobenius_norm": float}``
        Merged with any ``metadata_extra`` from the caller.
    generated_at
        ISO-8601 UTC wallclock at construction. Excluded from the sha.
    payload_sha256
        Content-addressed sha (see :func:`_realization_sha`).
    """

    K_baseline: NDArray[np.float64]
    strategy: NullStrategy
    base_seed: int
    null_seed: int
    lambda_value: float
    substrate_id: str
    N: int
    metadata: dict[str, Any] = field(default_factory=dict)
    generated_at: str = ""
    payload_sha256: str = ""


def _realization_sha(
    *,
    strategy: NullStrategy,
    base_seed: int,
    null_seed: int,
    lambda_value: float,
    substrate_id: str,
    N: int,
    metadata: Mapping[str, Any],
    K_baseline: NDArray[np.float64],
) -> str:
    """sha256 over the canonical-JSON payload of the load-bearing fields.

    Excludes ``generated_at`` and ``payload_sha256`` so the sha is
    invariant under wallclock differences.
    """
    payload: dict[str, Any] = {
        "strategy": str(strategy),
        "base_seed": int(base_seed),
        "null_seed": int(null_seed),
        "lambda_value": float(lambda_value),
        "substrate_id": str(substrate_id),
        "N": int(N),
        "metadata": dict(metadata),
        "K_flat_sha": _K_flat_sha(K_baseline),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Substrate-API adapter
# ---------------------------------------------------------------------------


def _baseline_slice(realisation: SubstrateRealization, *, t: int = 0) -> NDArray[np.float64]:
    """Static K_baseline slice (N, N) from a (T_HORIZON, N, N) realisation.

    At λ=0 the existing substrate API broadcasts ``K_baseline[0]``
    across time. Phase 0a / 0b only need the static slice.
    """
    K = np.asarray(realisation.K_baseline[t], dtype=np.float64)
    _refuse_if_non_square(K)
    return K


def _refuse_if_non_square(K: NDArray[np.float64]) -> None:
    if K.ndim != 2:
        raise D002GNullInvalid(f"K must be 2-D; got ndim={K.ndim}")
    if K.shape[0] != K.shape[1]:
        raise D002GNullInvalid(f"K must be square; got shape {K.shape}")


# ---------------------------------------------------------------------------
# M1 — independent-seed null
# ---------------------------------------------------------------------------


def _realize_m1(
    substrate: Substrate,
    *,
    base_seed: int,
    null_seed: int,
    lambda_value: float,
    N: int,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Draw the M1 independent-seed null K_baseline.

    The null cohort uses the SAME substrate API at the SAME λ as the
    precursor cohort. Bit-identity is broken by the seed offset; H0
    is preserved because the substrate draws ω + θ_0 + integrator
    stream from independent randomness while keeping the same baseline
    distribution.

    Raises
    ------
    BitIdenticalNullError
        If the precursor and null K_baseline slices are
        ``np.array_equal`` identical (substrate insensitive to seed
        at this (N, λ)).
    """
    precursor_real = substrate.realize(N=N, lambda_=lambda_value, seed=base_seed)
    null_real = substrate.realize(N=N, lambda_=lambda_value, seed=null_seed)
    K_p = _baseline_slice(precursor_real)
    K_n = _baseline_slice(null_real)
    if np.array_equal(K_p, K_n):
        raise BitIdenticalNullError(
            f"M1 produced bit-identical K_baseline for substrate "
            f"{substrate.id!r} at N={N}, lambda_={lambda_value}, "
            f"base_seed={base_seed}, null_seed={null_seed}. The "
            "substrate is insensitive to the seed argument at this "
            "(N, lambda_); tag the cell M1-INELIGIBLE and fall back "
            "to mechanism M2 (topology-preserving shuffle) per the "
            "D-002G pre-registration §4 fallback policy."
        )
    return K_n, {}


# ---------------------------------------------------------------------------
# M6 — placebo coupling
# ---------------------------------------------------------------------------


def _upper_tri_indices_count(N: int) -> int:
    return (N * (N - 1)) // 2


def _realize_m6(
    substrate: Substrate,
    *,
    base_seed: int,
    null_seed: int,
    lambda_value: float,
    N: int,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Draw the M6 placebo-coupling K_baseline.

    Procedure (matches ``D002G_NONDEGENERATE_NULL_DESIGN.md §5`` and
    ``D002G_PREREGISTRATION.yaml § supplementary_null``):

    1. Realise the precursor at ``(N, lambda_value, base_seed)`` and
       extract ``K_p = K_precursor[t_inj]`` where ``t_inj`` is inside
       the locked ``PRECURSOR_INJECTION_WINDOW``.
    2. Realise the baseline at the same seed with ``lambda_=0`` and
       extract ``K_0 = K_baseline[t_inj]``.
    3. ΔK = K_p − K_0. The support of ΔK is the substrate's
       privileged sites (top-10% κ for Ricci, inter-block for
       block_structured, sin-modulated baseline for temporal).
    4. Permute the support: select ``|support(ΔK)|`` distinct
       upper-triangle indices uniformly at random (no privileged
       sites) using an RNG seeded by :func:`deterministic_mix`. The
       magnitude distribution of ΔK is applied to those random edges
       — Frobenius norm of ΔK preserved exactly (to within rounding).
    5. K_placebo = K_0 + symmetric ΔK_placebo.

    Returns
    -------
    (K_placebo, metadata)
        ``K_placebo`` is N×N float64 symmetric. ``metadata`` carries
        ``placebo_edges_count``, ``placebo_frobenius_norm``, and
        ``precursor_frobenius_norm``.
    """
    precursor_real = substrate.realize(N=N, lambda_=lambda_value, seed=base_seed)
    baseline_real = substrate.realize(N=N, lambda_=0.0, seed=base_seed)
    inject_t = int(next(iter(PRECURSOR_INJECTION_WINDOW)))
    K_p = np.asarray(precursor_real.K_precursor[inject_t], dtype=np.float64)
    K_0 = np.asarray(baseline_real.K_baseline[inject_t], dtype=np.float64)
    _refuse_if_non_square(K_p)
    _refuse_if_non_square(K_0)
    if not (np.all(np.isfinite(K_p)) and np.all(np.isfinite(K_0))):
        raise D002GNullInvalid("M6: K_precursor / K_baseline contains non-finite")

    delta = K_p - K_0
    iu_r, iu_c = np.triu_indices(N, k=1)
    delta_upper = delta[iu_r, iu_c]
    support_mask = np.abs(delta_upper) > 1e-12
    n_support = int(np.count_nonzero(support_mask))

    precursor_frobenius = float(np.linalg.norm(delta))

    if n_support == 0 or precursor_frobenius == 0.0:
        # No precursor delta at this (N, lambda) — placebo is the
        # baseline itself (no fake injection to add). Clean corner case.
        # P0-1: emit the same support-exclusion metadata schema as the
        # non-degenerate branch so downstream audits read a uniform
        # contract.
        return K_0.copy(), {
            "placebo_edges_count": 0,
            "placebo_frobenius_norm": 0.0,
            "precursor_frobenius_norm": 0.0,
            "null_strategy": "M6_PLACEBO_COUPLING",
            "original_support_count": 0,
            "placebo_support_count": 0,
            "placebo_overlap_count": 0,
            "placebo_overlap_forbidden": True,
            "candidate_pool_size": int(_upper_tri_indices_count(N)),
        }

    n_total_upper = _upper_tri_indices_count(N)
    if n_support > n_total_upper:
        raise D002GNullInvalid(
            f"M6: support size {n_support} exceeds upper-triangle count {n_total_upper}"
        )

    # P0-1 Codex review fix: privileged ΔK support MUST be excluded from
    # the placebo sampling pool. Sampling from the full upper triangle
    # would let placebo edges overlap original support, retaining true
    # precursor topology in the "placebo" cohort and biasing R2-B toward
    # optimism. Build the candidate pool as upper_triangle ∖ support and
    # sample only from the off-support indices. Fail-closed if the pool
    # is smaller than the support size — the cell is REFUSED rather
    # than silently allowing overlap.
    candidate_indices = np.flatnonzero(~support_mask)
    original_support_indices = np.flatnonzero(support_mask)
    candidate_pool_size = int(candidate_indices.size)
    if candidate_pool_size < n_support:
        raise M6InsufficientCandidatePool(
            f"M6: candidate edge pool too small for support exclusion: "
            f"|upper_triangle ∖ support|={candidate_pool_size} < "
            f"|support|={n_support} at N={N}. Cell is M6-INELIGIBLE; "
            f"escalate to M2 topology-preserving shuffle per the "
            f"D-002G pre-registration §4 fallback policy."
        )

    rng = np.random.default_rng(int(null_seed))
    chosen = rng.choice(candidate_indices, size=n_support, replace=False)
    magnitudes = delta_upper[support_mask].copy()
    rng.shuffle(magnitudes)

    # P0-1 audit: verify zero overlap between chosen placebo sites and
    # the privileged support BEFORE building the placebo matrix. This is
    # a defensive invariant — the candidate-pool exclusion above already
    # guarantees zero overlap by construction; the explicit check
    # converts the property into an observable failure mode for the
    # downstream Phase 0 / R2-B audits.
    overlap_count = int(np.intersect1d(chosen, original_support_indices).size)
    if overlap_count != 0:
        raise D002GNullInvalid(
            f"M6: placebo-vs-support overlap_count={overlap_count} ≠ 0 "
            f"despite candidate-pool exclusion. This is an internal "
            f"invariant violation; refusing the cell."
        )

    placebo_upper = np.zeros(n_total_upper, dtype=np.float64)
    placebo_upper[chosen] = magnitudes

    delta_placebo = np.zeros_like(K_0, dtype=np.float64)
    delta_placebo[iu_r, iu_c] = placebo_upper
    delta_placebo = delta_placebo + delta_placebo.T  # symmetrize

    K_placebo = K_0 + delta_placebo

    placebo_frobenius = float(np.linalg.norm(delta_placebo))

    if precursor_frobenius > 0.0:
        rel_err = abs(placebo_frobenius - precursor_frobenius) / precursor_frobenius
        if rel_err > 1e-9:
            raise D002GNullInvalid(
                f"M6: Frobenius preservation violated: "
                f"precursor={precursor_frobenius:.6e}, "
                f"placebo={placebo_frobenius:.6e}, rel_err={rel_err:.3e}"
            )

    return K_placebo, {
        "placebo_edges_count": int(n_support),
        "placebo_frobenius_norm": placebo_frobenius,
        "precursor_frobenius_norm": precursor_frobenius,
        # P0-1 Codex review fix: explicit metadata so downstream audits
        # can verify the support-exclusion contract was honoured.
        "null_strategy": "M6_PLACEBO_COUPLING",
        "original_support_count": int(n_support),
        "placebo_support_count": int(n_support),
        "placebo_overlap_count": int(overlap_count),
        "placebo_overlap_forbidden": True,
        "candidate_pool_size": int(candidate_pool_size),
    }


# ---------------------------------------------------------------------------
# Public API — realize_null
# ---------------------------------------------------------------------------


def _default_null_seed(strategy: NullStrategy, base_seed: int) -> int:
    """Locked null-seed formula. Tests can override; canonical sweep MUST NOT."""
    if strategy == "M1_INDEPENDENT_SEED":
        return int(base_seed) + NULL_SEED_OFFSET
    # M6: deterministic mix of base_seed with the locked salt.
    return deterministic_mix(int(base_seed), M6_PLACEBO_SALT)


def realize_null(
    substrate: Substrate,
    *,
    strategy: NullStrategy,
    base_seed: int,
    N: int,
    lambda_value: float,
    null_seed: int | None = None,
    metadata_extra: Mapping[str, Any] | None = None,
) -> NullRealization:
    """Realise one M1 or M6 null cohort K_baseline.

    Parameters
    ----------
    substrate
        Any object implementing :class:`d002c_substrates.Substrate`.
    strategy
        Mechanism choice. MUST be explicit — there is NO default.
    base_seed
        The precursor cohort seed this null is paired to. Integer in
        the signed-int64 range.
    N
        Cohort size (required).
    lambda_value
        Cell coordinate. Must be ≥ 0; M6 additionally requires > 0.
    null_seed
        Optional override of the locked null-seed formula. ``None`` →
        the formula:

          * M1: ``base_seed + NULL_SEED_OFFSET`` (offset = 10000).
          * M6: ``deterministic_mix(base_seed, M6_PLACEBO_SALT)``.

        Tests that probe edge cases can override; the canonical sweep
        MUST pass ``None``.
    metadata_extra
        Optional caller-supplied provenance merged into the strategy
        metadata before sha computation.

    Returns
    -------
    NullRealization
        Frozen dataclass with K_baseline + content-addressed sha.

    Raises
    ------
    D002GNullInvalid
        Invalid strategy, ``lambda_value < 0``, ``lambda_value == 0``
        under M6, non-square / non-finite K, dtype mismatch, N < 2.
    BitIdenticalNullError
        M1 produced ``K_null == K_precursor`` bit-identically.
    """
    if strategy not in _VALID_STRATEGIES:
        raise D002GNullInvalid(
            f"strategy must be one of {sorted(_VALID_STRATEGIES)}; got {strategy!r}"
        )
    if not math.isfinite(lambda_value) or lambda_value < 0.0:
        raise D002GNullInvalid(f"lambda_value must be finite and >= 0; got {lambda_value!r}")
    if int(N) < 2:
        raise D002GNullInvalid(f"N must be >= 2; got {N!r}")
    if strategy == "M6_PLACEBO_COUPLING" and lambda_value <= 0.0:
        raise D002GNullInvalid(
            "M6_PLACEBO_COUPLING requires lambda_value > 0 "
            "(no precursor delta to permute at lambda=0)"
        )

    effective_null_seed = (
        int(null_seed) if null_seed is not None else _default_null_seed(strategy, base_seed)
    )

    if strategy == "M1_INDEPENDENT_SEED":
        K_null, mech_meta = _realize_m1(
            substrate,
            base_seed=int(base_seed),
            null_seed=effective_null_seed,
            lambda_value=float(lambda_value),
            N=int(N),
        )
    else:
        K_null, mech_meta = _realize_m6(
            substrate,
            base_seed=int(base_seed),
            null_seed=effective_null_seed,
            lambda_value=float(lambda_value),
            N=int(N),
        )

    if K_null.dtype != np.float64:
        K_null = K_null.astype(np.float64, copy=False)

    metadata: dict[str, Any] = dict(mech_meta)
    if metadata_extra is not None:
        for k, v in metadata_extra.items():
            metadata[str(k)] = v

    sha = _realization_sha(
        strategy=strategy,
        base_seed=int(base_seed),
        null_seed=effective_null_seed,
        lambda_value=float(lambda_value),
        substrate_id=str(substrate.id),
        N=int(N),
        metadata=metadata,
        K_baseline=K_null,
    )

    return NullRealization(
        K_baseline=K_null,
        strategy=strategy,
        base_seed=int(base_seed),
        null_seed=int(effective_null_seed),
        lambda_value=float(lambda_value),
        substrate_id=str(substrate.id),
        N=int(N),
        metadata=metadata,
        generated_at=_now_iso(),
        payload_sha256=sha,
    )


__all__ = [
    "BitIdenticalNullError",
    "D002GNullInvalid",
    "M6InsufficientCandidatePool",
    "M6_PLACEBO_SALT",
    "NULL_SEED_OFFSET",
    "NullRealization",
    "NullStrategy",
    "R2_B_RANDOM_SITE_SEED",
    "deterministic_mix",
    "realize_null",
]
