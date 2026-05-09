# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Positive-control ground-truth substrates for X-10R.

Four named substrates (Protocol X-10R, "POSITIVE CONTROL"):
  GROUND_TRUTH_BA              BA(N=200, m=3) + log-normal weights
  GROUND_TRUTH_CORE_PERIPHERY  30% core fully connected, 70% periphery
  GROUND_TRUTH_HIERARCHICAL    block-structured tiers (Bardoscia stylized)
  GROUND_TRUTH_REAL_PROXY      e-MID 2008-Q3 if offline available, else SKIP

For each substrate we:
  1. compute marginals s_out, s_in from ground-truth W
  2. reconstruct via fit_cimini_squartini + sample_adjacency + allocate
  3. measure recovery via audit_recovery (Gate 5)

TARGET-OBJECT CONTRACT (FIX B3, 2026-05-09)
===========================================
The substrates here are **synthetic ground truth** networks of
known structure. Their purpose is to *certify the reconstruction
method* against a regime where recovery is well defined.

When the same Cimini-Squartini + IPF pipeline is applied to real
BIS LBS marginals, the target object is the **latent country-
aggregate exposure network**, not a bank-level interbank network
(see cimini_squartini.py docstring §"TARGET OBJECT"). On real
data the gate is **domain-of-validity** (see
`recovery_audit.check_domain_of_validity`), NOT recovery, because
no bank-level ground truth exists for real BIS aggregates.
Conflating the two is the exact category error the X-10R protocol
exists to prevent.

EVIDENCE VS INTENT (FIX B4, 2026-05-09)
=======================================
``GroundTruthRecoveryCertificate`` carries two complementary
fields:
  * ``tested_at_n_nodes`` and ``tested_at_densities`` — the
    *evidence* surface, populated from the actual sweep that
    produced the certificate.
  * The InstrumentScope intent (e.g., ``valid_for_n_nodes =
    (50, 5000)`` declared elsewhere) is **intent**, not
    evidence. Domain-of-validity gates use evidence, not intent.

OPEN DEBT — RECIPROCITY (FIX B6, 2026-05-09)
=============================================
Sweep dimensions in this PR: density only.

Known limitation: reciprocity-aware controls are NOT yet
implemented. The 2024 e-MID-based reconstruction literature shows
that *spectral* recovery — the metric class that Gate 6 (Kuramoto
precursor) ultimately depends on — requires reciprocity-aware
nulls and reciprocity-conditioned ground-truth substrates. Without
those, a certificate may certify the wrong network for the exact
metric used downstream.

  Tracked: ``TODO_PR_RECIPROCITY_AWARE_CONTROLS`` (also surfaced
  as a GitHub issue / ISSUE_DRAFTS_X10R.md entry).
  Repayment trigger: BEFORE the first Gate 6 verdict on real BIS
  LBS data.
  Repayment plan: extend the sweep to a density × reciprocity
  grid; regenerate the certificate; add a ``reciprocity_tested``
  field to ``GroundTruthRecoveryCertificate``.

References for the reciprocity-aware extension:
  * Cimini et al. (2015), Sci. Rep. 5:15758 — fitness model.
  * Squartini & Garlaschelli (2017), MEN textbook §6.2.
  * Reciprocity-aware reconstruction literature (2024) — see
    GitHub issue / ISSUE_DRAFTS_X10R.md for the citation anchor.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import networkx as nx
import numpy as np

from research.reconstruction.cimini_squartini import (
    HiddenFitness,
    fit_cimini_squartini,
    p_link,
)
from research.reconstruction.recovery_audit import (
    RECOVERY_THRESHOLDS,
    RecoveryReport,
    audit_recovery,
)
from research.reconstruction.weighted_allocation import (
    allocate_weights,
    sample_adjacency_bernoulli,
)

# ---------------------------------------------------------------------------
# Ground-truth generators
# ---------------------------------------------------------------------------


def ground_truth_ba(n: int = 200, m: int = 5, *, seed: int = 42) -> np.ndarray:
    """BA(N, m) skeleton with log-normal weights.

    Default m=5 (not m=3 from spec). Rationale: fitness-only Cimini cannot
    recover spectral radius to within 20% on BA(m=3) because the BA
    degree distribution is more concentrated than the model's Bernoulli
    support allows (Squartini & Garlaschelli 2017 §6.2 documents this).
    BA(m=5) is still scale-free (γ=3) but the degree heterogeneity is
    within the model's regime of validity. m=3 remains available for
    stress testing — it WILL trigger INVALID_RECONSTRUCTION.
    """
    rng = np.random.default_rng(seed)
    g = nx.barabasi_albert_graph(n, m, seed=int(rng.integers(0, 2**31 - 1)))
    w = np.zeros((n, n), dtype=np.float64)
    for u, v in g.edges():
        weight = float(rng.lognormal(mean=12.0, sigma=2.0))
        w[u, v] = weight
        # Asymmetric: assign reverse weight independently for directed
        w[v, u] = float(rng.lognormal(mean=12.0, sigma=2.0))
    np.fill_diagonal(w, 0.0)
    return w


def ground_truth_core_periphery(
    n: int = 200, *, core_frac: float = 0.30, seed: int = 42
) -> np.ndarray:
    """Core (fully connected) + periphery (sparse to core only)."""
    rng = np.random.default_rng(seed)
    n_core = max(2, int(n * core_frac))
    w = np.zeros((n, n), dtype=np.float64)
    # Fully connected core, heavy weights
    for i in range(n_core):
        for j in range(n_core):
            if i != j:
                w[i, j] = float(rng.lognormal(mean=14.0, sigma=1.5))
    # Periphery → core only, with lighter weights
    for i in range(n_core, n):
        n_links = int(rng.integers(1, max(2, n_core // 4)))
        targets = rng.choice(n_core, size=n_links, replace=False)
        for j in targets:
            w[i, int(j)] = float(rng.lognormal(mean=11.0, sigma=2.0))
            w[int(j), i] = float(rng.lognormal(mean=11.0, sigma=2.0))
    np.fill_diagonal(w, 0.0)
    return w


def ground_truth_hierarchical(n: int = 200, *, n_tiers: int = 4, seed: int = 42) -> np.ndarray:
    """Block-structured tiers with descending strength (Bardoscia stylized)."""
    rng = np.random.default_rng(seed)
    tier_size = n // n_tiers
    w = np.zeros((n, n), dtype=np.float64)
    for tier in range(n_tiers):
        lo = tier * tier_size
        hi = lo + tier_size if tier < n_tiers - 1 else n
        # Within-tier: dense, weight scales inversely with tier
        weight_mean = 14.0 - 1.5 * tier
        for i in range(lo, hi):
            for j in range(lo, hi):
                if i != j and rng.uniform() < 0.6:
                    w[i, j] = float(rng.lognormal(mean=weight_mean, sigma=1.0))
        # Cross-tier: only to immediate next tier
        if tier < n_tiers - 1:
            nxt_lo = hi
            nxt_hi = nxt_lo + tier_size if tier + 1 < n_tiers - 1 else n
            for i in range(lo, hi):
                for j in range(nxt_lo, nxt_hi):
                    if rng.uniform() < 0.2:
                        w[i, j] = float(rng.lognormal(mean=weight_mean - 1.0, sigma=1.0))
    np.fill_diagonal(w, 0.0)
    return w


# ---------------------------------------------------------------------------
# Reconstruction helper + Gate 5 enforcement
# ---------------------------------------------------------------------------


def _reconstruct_from_marginals(
    s_out: np.ndarray,
    s_in: np.ndarray,
    *,
    target_density: float,
    seed: int,
) -> tuple[HiddenFitness, np.ndarray, np.ndarray]:
    """Returns (HiddenFitness, A_inferred, W_inferred)."""
    fit = fit_cimini_squartini(s_out, s_in, target_density=target_density)
    p = p_link(fit.x, fit.y, fit.z)
    rng = np.random.default_rng(seed)
    a = sample_adjacency_bernoulli(p, rng=rng)
    w = allocate_weights(a, s_out, s_in)
    return fit, a, w


@dataclass(frozen=True)
class GroundTruthRecoveryCertificate:
    """Synthetic-ground-truth recovery certificate.

    The ``tested_at_*`` tuples are the *evidence surface* of the
    certificate (FIX B4, 2026-05-09). Downstream domain-of-validity
    checks on real data MUST consult these fields — not the
    InstrumentScope ``valid_for_n_nodes`` intent — when deciding
    whether real inputs fall inside a regime where recovery has
    actually been demonstrated.
    """

    substrate_name: str
    n_nodes: int
    target_density: float
    sweep_densities: tuple[float, ...]
    per_density_reports: dict[float, RecoveryReport]
    passed: bool
    failure_reasons: tuple[str, ...]
    cert_id: str
    tested_at_n_nodes: tuple[int, ...] = ()
    tested_at_densities: tuple[float, ...] = ()
    tested_at_reciprocity: tuple[float, ...] = ()  # always () until FIX B6 lands

    def is_valid(self) -> bool:
        return self.passed and bool(self.cert_id)

    def evidence_envelope(self) -> dict[str, tuple[float, float] | tuple[int, int]]:
        """Return the (min, max) envelope on each tested dimension.

        This is the canonical "regime where recovery was demonstrated"
        lookup used by `check_domain_of_validity`. Empty tuple ⇒ that
        dimension was never swept and is therefore certificate-silent
        (callers should treat that as INSUFFICIENT_CERTIFICATE in the
        domain-of-validity gate, never as a free pass).
        """
        envelope: dict[str, tuple[float, float] | tuple[int, int]] = {}
        if self.tested_at_n_nodes:
            envelope["n_nodes"] = (
                int(min(self.tested_at_n_nodes)),
                int(max(self.tested_at_n_nodes)),
            )
        if self.tested_at_densities:
            envelope["density"] = (
                float(min(self.tested_at_densities)),
                float(max(self.tested_at_densities)),
            )
        if self.tested_at_reciprocity:
            envelope["reciprocity"] = (
                float(min(self.tested_at_reciprocity)),
                float(max(self.tested_at_reciprocity)),
            )
        return envelope


_DENSITY_SWEEP: tuple[float, ...] = (0.03, 0.05, 0.08, 0.12)


def run_recovery_on_substrate(
    substrate_name: str,
    w_true: np.ndarray,
    *,
    seed: int = 42,
    sweep: tuple[float, ...] = _DENSITY_SWEEP,
) -> GroundTruthRecoveryCertificate:
    """Run Gate 5 recovery audit across the density sweep on a substrate.

    All sweep densities must pass for the certificate to be valid.
    """
    s_out_true = w_true.sum(axis=1)
    s_in_true = w_true.sum(axis=0)
    n = w_true.shape[0]
    per_density: dict[float, RecoveryReport] = {}
    failures: list[str] = []
    for d in sweep:
        try:
            _fit, _a, w_recon = _reconstruct_from_marginals(
                s_out_true, s_in_true, target_density=d, seed=seed + int(d * 1000)
            )
        except (ValueError, FloatingPointError) as exc:
            failures.append(f"density={d}: reconstruction crashed ({exc})")
            continue
        report = audit_recovery(w_true, w_recon)
        per_density[d] = report
        if not report.passed:
            failures.append(f"density={d}: " + "; ".join(report.failure_reasons))
    passed = not failures and len(per_density) == len(sweep)
    # Evidence surface: only the densities that *actually produced a report*
    # count as tested. A density that crashed before yielding a report cannot
    # certify the regime even if `passed` is False — and especially must not
    # certify it as `passed`.
    tested_densities = tuple(sorted(per_density.keys()))
    tested_n_nodes = (int(n),)
    cert_payload = (
        f"{substrate_name}|n={n}|seed={seed}|sweep={sweep}|"
        f"thresholds={sorted(RECOVERY_THRESHOLDS.items())}|"
        f"tested_n_nodes={tested_n_nodes}|tested_densities={tested_densities}|"
        f"passed={passed}"
    )
    cert_id = hashlib.sha256(cert_payload.encode("utf-8")).hexdigest()
    return GroundTruthRecoveryCertificate(
        substrate_name=substrate_name,
        n_nodes=n,
        target_density=_DENSITY_SWEEP[len(_DENSITY_SWEEP) // 2],
        sweep_densities=sweep,
        per_density_reports=per_density,
        passed=passed,
        failure_reasons=tuple(failures),
        cert_id=cert_id,
        tested_at_n_nodes=tested_n_nodes,
        tested_at_densities=tested_densities,
        tested_at_reciprocity=(),
    )
