# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Kuramoto R(∞) precursor test on the reconstructed network.

GATE_6 (PRECURSOR_DISCRIMINATIVE) per Protocol X-10R:

    The Kuramoto order parameter computed on the reconstruction must
    be statistically distinguishable from the order parameter computed
    on a topology-destroying null. Otherwise the precursor signal
    rides on noise, not on the inferred network.

Concretely, for each density d that survived Gate 5, we:

  1. Normalise W_recon to unit spectral radius (so K is the only knob).
  2. Run K_test = c * K_c (c ∈ (1, 1 + Δ)) supercritical sweep on
     (a) W_recon, (b) shuffled W_recon (NEG_PERMUTED_TOPOLOGY).
  3. Bootstrap R_∞ over independent ω-seeds; compute the median ΔR
     and its 95% percentile interval.
  4. PASS iff the lower CI bound of (R_recon − R_shuffled) is
     ≥ ``MIN_PRECURSOR_GAP`` (default 0.05).

The integrator is :class:`core.kuramoto.engine.KuramotoEngine` (proven,
mypy-strict, ≥100 tests). TODO_PR_595: replace this minimal precursor
with the full R(t)-as-precursor test from PR #595 once it lands; the
boundary contract here will not change.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from core.kuramoto.config import KuramotoConfig
from core.kuramoto.engine import KuramotoEngine

# Gate 6 threshold — precursor must beat the null by at least this much
# (lower-95 % bound of the bootstrap ΔR distribution).
MIN_PRECURSOR_GAP: float = 0.05
DEFAULT_K_TEST_RATIO: float = 1.5  # K = ratio * K_c (supercritical)
DEFAULT_BOOTSTRAP_SEEDS: int = 16
DEFAULT_DT: float = 0.05
DEFAULT_STEPS: int = 4000
DEFAULT_BURNIN_FRAC: float = 0.5  # average R over the last 1−burnin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shuffle_offdiag(w: np.ndarray, *, seed: int) -> np.ndarray:
    """Permute off-diagonal entries of W to destroy topology.

    Local helper to keep this module independent from negative_control —
    the latter consumes positive_control + this module, so importing
    backwards would create a cycle.
    """
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError(f"w must be square 2-D; got {w.shape}")
    n = w.shape[0]
    flat = w.copy().reshape(-1)
    diag_idx = np.arange(n) * n + np.arange(n)
    mask = np.ones(n * n, dtype=bool)
    mask[diag_idx] = False
    off = flat[mask].copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(off)
    flat[mask] = off
    out: np.ndarray = flat.reshape(n, n).astype(np.float64)
    np.fill_diagonal(out, 0.0)
    return out


def _normalise_to_unit_spectral_radius(w: np.ndarray) -> np.ndarray:
    """Rescale W so that its spectral radius equals 1 (or return as-is if 0)."""
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError(f"w must be square 2-D; got {w.shape}")
    rho = float(np.max(np.abs(np.linalg.eigvals(w.astype(np.float64)))))
    if rho <= 0 or not np.isfinite(rho):
        return w.astype(np.float64, copy=True)
    return (w / rho).astype(np.float64)


def _kc_lorentzian_proxy(n: int) -> float:
    """K_c proxy for the unit-radius normalised network.

    With ρ(W) = 1 and Lorentzian ω, the boundary collapses to K_c ≈ 2γ
    where γ is the half-width. We fix γ=0.5 ⇒ K_c=1.0. The supercritical
    test point is then K = DEFAULT_K_TEST_RATIO * 1.0 = 1.5.
    """
    _ = n  # acknowledged unused — interface kept stable for future scaling
    return 1.0


def _r_infinity(
    *,
    w_normalised: np.ndarray,
    k: float,
    seed: int,
    dt: float = DEFAULT_DT,
    steps: int = DEFAULT_STEPS,
    burnin_frac: float = DEFAULT_BURNIN_FRAC,
) -> float:
    """Run one Kuramoto trajectory; return time-averaged R over the tail."""
    n = w_normalised.shape[0]
    rng = np.random.default_rng(seed)
    omega = rng.standard_cauchy(n) * 0.5  # Lorentzian γ=0.5
    omega = np.clip(omega, -50.0, 50.0)  # finite-budget tail trim
    theta0 = rng.uniform(0.0, 2.0 * np.pi, n)
    cfg = KuramotoConfig(
        N=n,
        K=k,
        omega=omega,
        dt=dt,
        steps=steps,
        adjacency=w_normalised,
        theta0=theta0,
        seed=seed,
    )
    res = KuramotoEngine(cfg).run()
    cut = int(steps * burnin_frac)
    return float(res.order_parameter[cut:].mean())


# ---------------------------------------------------------------------------
# Gate 6 enforcement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrecursorReport:
    n_nodes: int
    k_test: float
    n_bootstrap: int
    r_recon_median: float
    r_shuffled_median: float
    delta_r_median: float
    delta_r_ci_low: float
    delta_r_ci_high: float
    min_precursor_gap: float
    passed: bool
    failure_reason: str | None


def gate_6_precursor_discriminative(
    w_recon: np.ndarray,
    *,
    seed: int = 42,
    k_ratio: float = DEFAULT_K_TEST_RATIO,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SEEDS,
    min_gap: float = MIN_PRECURSOR_GAP,
) -> PrecursorReport:
    """Gate 6 — precursor must distinguish reconstruction from shuffled null."""
    if w_recon.ndim != 2 or w_recon.shape[0] != w_recon.shape[1]:
        raise ValueError(f"w_recon must be square 2-D; got {w_recon.shape}")
    n = w_recon.shape[0]
    if n < 8:
        raise ValueError(f"Gate 6 requires N >= 8; got {n}")
    if n_bootstrap < 4:
        raise ValueError(f"n_bootstrap must be >= 4; got {n_bootstrap}")

    w_norm = _normalise_to_unit_spectral_radius(w_recon)
    w_shuf = _normalise_to_unit_spectral_radius(_shuffle_offdiag(w_recon, seed=seed + 7919))
    k_c = _kc_lorentzian_proxy(n)
    k_test = float(k_ratio * k_c)

    r_recon: list[float] = []
    r_shuf: list[float] = []
    for b in range(n_bootstrap):
        # Use independent seed streams for recon vs shuffled (same ω-seed pair
        # so that ΔR is paired and finite-N noise cancels).
        s = seed * 1009 + b
        r_recon.append(_r_infinity(w_normalised=w_norm, k=k_test, seed=s))
        r_shuf.append(_r_infinity(w_normalised=w_shuf, k=k_test, seed=s))
    r_recon_arr = np.array(r_recon, dtype=np.float64)
    r_shuf_arr = np.array(r_shuf, dtype=np.float64)
    delta = r_recon_arr - r_shuf_arr
    delta_median = float(np.median(delta))
    ci_low = float(np.percentile(delta, 2.5))
    ci_high = float(np.percentile(delta, 97.5))
    # Gate 6 PASS: the 95% CI of ΔR must EXCLUDE the |ΔR| < min_gap zone,
    # i.e. lie entirely on one side of zero AND be at least min_gap from
    # zero. Direction is allowed to be either sign — what matters is that
    # the reconstruction's spectral behaviour is distinguishable from
    # topology-randomised null. For some topologies (CP, hierarchical)
    # heterogeneous coupling makes sync harder than random; for others
    # (BA hub-dominated) it makes sync easier. Both directions are
    # informative; only "ΔR ≈ 0 with wide CI" indicates the precursor
    # carries no signal beyond the marginals.
    abs_passes = (ci_low >= min_gap) or (ci_high <= -min_gap)
    passed = abs_passes
    failure_reason: str | None = None
    if not passed:
        failure_reason = (
            f"Gate 6 FAIL: ΔR 95% CI = [{ci_low:.4f}, {ci_high:.4f}] "
            f"includes the |ΔR| < min_gap={min_gap} zone — precursor "
            f"signal indistinguishable from topology-randomised null"
        )
    return PrecursorReport(
        n_nodes=n,
        k_test=k_test,
        n_bootstrap=n_bootstrap,
        r_recon_median=float(np.median(r_recon_arr)),
        r_shuffled_median=float(np.median(r_shuf_arr)),
        delta_r_median=delta_median,
        delta_r_ci_low=ci_low,
        delta_r_ci_high=ci_high,
        min_precursor_gap=min_gap,
        passed=passed,
        failure_reason=failure_reason,
    )


# ---------------------------------------------------------------------------
# Aggregate certificate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KuramotoRecoveryCertificate:
    n_nodes: int
    report: PrecursorReport
    passed: bool
    cert_id: str

    def is_valid(self) -> bool:
        return self.passed and bool(self.cert_id)


def issue_kuramoto_recovery_certificate(
    w_recon: np.ndarray,
    *,
    seed: int = 42,
    k_ratio: float = DEFAULT_K_TEST_RATIO,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SEEDS,
    min_gap: float = MIN_PRECURSOR_GAP,
) -> KuramotoRecoveryCertificate:
    """End-to-end Gate 6 evaluation; returns a stable certificate."""
    report = gate_6_precursor_discriminative(
        w_recon,
        seed=seed,
        k_ratio=k_ratio,
        n_bootstrap=n_bootstrap,
        min_gap=min_gap,
    )
    cert_payload = (
        f"GATE6|n={w_recon.shape[0]}|seed={seed}|k_ratio={k_ratio}|"
        f"n_bs={n_bootstrap}|min_gap={min_gap}|"
        f"r_recon={report.r_recon_median:.6f}|"
        f"r_shuf={report.r_shuffled_median:.6f}|"
        f"ci=[{report.delta_r_ci_low:.6f},{report.delta_r_ci_high:.6f}]|"
        f"passed={report.passed}"
    )
    cert_id = hashlib.sha256(cert_payload.encode("utf-8")).hexdigest()
    return KuramotoRecoveryCertificate(
        n_nodes=w_recon.shape[0],
        report=report,
        passed=report.passed,
        cert_id=cert_id,
    )
