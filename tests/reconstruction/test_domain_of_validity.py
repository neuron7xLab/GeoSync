# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Tests for the real-data domain-of-validity gate (FIX B2).

INV-RECONSTRUCTION-2:
    "Recovery" is defined only on synthetic substrates with known
    ground truth. On real data with unobserved truth, the strongest
    available gate is DOMAIN-OF-VALIDITY: do real inputs fall inside
    the regime where synthetic recovery was demonstrated?

These tests pin the verdict surface for that gate (WITHIN_/OUT_OF_/
INSUFFICIENT_) and certify the contract that the real-data path is
FORBIDDEN from emitting GROUND_TRUTH_RECOVERED.
"""

from __future__ import annotations

import numpy as np
import pytest

from research.reconstruction.positive_control import (
    GroundTruthRecoveryCertificate,
    ground_truth_core_periphery,
    run_recovery_on_substrate,
)
from research.reconstruction.reconstruction_capsule import (
    ReconstructionStatus,
    assert_real_data_status_legal,
    assert_synthetic_status_legal,
)
from research.reconstruction.recovery_audit import (
    DomainCheck,
    DomainOfValidityStatus,
    check_domain_of_validity,
)

# ---------------------------------------------------------------------------
# Helpers — synthesise marginals + a "real-like" certificate envelope
# ---------------------------------------------------------------------------


def _synthetic_certificate_at(n: int, seed: int = 42) -> GroundTruthRecoveryCertificate:
    """Real positive-control certificate at the requested N (Gate 5 path)."""
    w = ground_truth_core_periphery(n=n, core_frac=0.30, seed=seed)
    return run_recovery_on_substrate(f"CP_{n}", w, seed=seed)


def _marginals_for_n(n: int, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate balanced (Σs_in = Σs_out) lognormal marginals."""
    rng = np.random.default_rng(seed)
    s_out = rng.lognormal(mean=10.0, sigma=1.0, size=n)
    s_in = rng.lognormal(mean=10.0, sigma=1.0, size=n)
    # Re-balance to satisfy GATE_2 conservation of mass (cheap rescale).
    s_in = s_in * (s_out.sum() / s_in.sum())
    return s_out, s_in


def _stub_certificate_only_n(n_set: tuple[int, ...]) -> GroundTruthRecoveryCertificate:
    """Certificate with tested_at_n_nodes set, density envelope absent."""
    return GroundTruthRecoveryCertificate(
        substrate_name="stub",
        n_nodes=int(n_set[0]),
        target_density=0.05,
        sweep_densities=(),
        per_density_reports={},
        passed=True,
        failure_reasons=(),
        cert_id="0" * 64,
        tested_at_n_nodes=n_set,
        tested_at_densities=(),
        tested_at_reciprocity=(),
    )


def _empty_certificate() -> GroundTruthRecoveryCertificate:
    """Certificate with no evidence on any dimension (FIX B4 'silent' case)."""
    return GroundTruthRecoveryCertificate(
        substrate_name="empty",
        n_nodes=0,
        target_density=0.05,
        sweep_densities=(),
        per_density_reports={},
        passed=False,
        failure_reasons=("no sweep was run",),
        cert_id="0" * 64,
        tested_at_n_nodes=(),
        tested_at_densities=(),
        tested_at_reciprocity=(),
    )


# ---------------------------------------------------------------------------
# Verdict surface — the four canonical paths (per FIX B2 spec)
# ---------------------------------------------------------------------------


def test_within_domain_when_all_dims_inside_certified_range() -> None:
    """N inside [min, max] of tested_at_n_nodes AND density inside envelope
    ⇒ WITHIN_VALIDATED_DOMAIN."""
    cert = _synthetic_certificate_at(n=120, seed=1)
    s_out, s_in = _marginals_for_n(n=120, seed=11)
    # Pick an inferred density inside the certified sweep envelope.
    densities = cert.tested_at_densities
    inside = float((min(densities) + max(densities)) / 2.0)
    check = check_domain_of_validity(s_out, s_in, cert, inferred_density=inside)
    assert isinstance(check, DomainCheck)
    assert check.status is DomainOfValidityStatus.WITHIN_VALIDATED_DOMAIN
    assert check.checks["n_nodes"] is True
    assert check.checks["density"] is True
    assert check.out_of_range_dims == ()


def test_out_of_domain_when_n_nodes_exceed_certified() -> None:
    """N outside the certificate's tested_at_n_nodes envelope ⇒ OUT_OF_."""
    cert = _synthetic_certificate_at(n=120, seed=2)
    s_out, s_in = _marginals_for_n(n=600, seed=12)
    inside = float((min(cert.tested_at_densities) + max(cert.tested_at_densities)) / 2.0)
    check = check_domain_of_validity(s_out, s_in, cert, inferred_density=inside)
    assert check.status is DomainOfValidityStatus.OUT_OF_VALIDATED_DOMAIN
    assert "n_nodes" in check.out_of_range_dims
    assert check.checks["n_nodes"] is False


def test_out_of_domain_when_density_below_floor() -> None:
    """Inferred density well below the smallest tested density ⇒ OUT_OF_."""
    cert = _synthetic_certificate_at(n=120, seed=3)
    s_out, s_in = _marginals_for_n(n=120, seed=13)
    floor = float(min(cert.tested_at_densities))
    check = check_domain_of_validity(s_out, s_in, cert, inferred_density=floor / 10.0)
    assert check.status is DomainOfValidityStatus.OUT_OF_VALIDATED_DOMAIN
    assert "density" in check.out_of_range_dims
    assert check.checks["density"] is False


def test_insufficient_when_certificate_lacks_tested_ranges() -> None:
    """Empty evidence surface AND a required-dim ask ⇒ INSUFFICIENT_."""
    cert = _empty_certificate()
    s_out, s_in = _marginals_for_n(n=80, seed=14)
    check = check_domain_of_validity(s_out, s_in, cert, inferred_density=0.05)
    assert check.status is DomainOfValidityStatus.INSUFFICIENT_CERTIFICATE
    assert "n_nodes" in check.missing_dims
    assert "density" in check.missing_dims


def test_partial_certificate_only_n_dim_yields_insufficient_for_density() -> None:
    """Certificate with only N evidence: density is missing ⇒ INSUFFICIENT_."""
    cert = _stub_certificate_only_n(n_set=(50, 200))
    s_out, s_in = _marginals_for_n(n=120, seed=15)
    check = check_domain_of_validity(s_out, s_in, cert, inferred_density=0.05)
    assert check.status is DomainOfValidityStatus.INSUFFICIENT_CERTIFICATE
    assert "density" in check.missing_dims
    assert check.checks.get("n_nodes") is True


# ---------------------------------------------------------------------------
# Real-data emission contract — the forbidden category error
# ---------------------------------------------------------------------------


def test_real_data_path_emits_within_or_out_never_recovered() -> None:
    """INV-RECONSTRUCTION-2: real-data status must be a domain-of-validity
    status; emitting GROUND_TRUTH_RECOVERED on real data is forbidden."""
    legal = {
        ReconstructionStatus.WITHIN_VALIDATED_DOMAIN,
        ReconstructionStatus.OUT_OF_VALIDATED_DOMAIN,
        ReconstructionStatus.INSUFFICIENT_CERTIFICATE,
    }
    for s in legal:
        assert_real_data_status_legal(s)

    forbidden = {
        ReconstructionStatus.GROUND_TRUTH_RECOVERED,
        ReconstructionStatus.GROUND_TRUTH_NOT_RECOVERED,
        ReconstructionStatus.INVALID_RECONSTRUCTION,
        ReconstructionStatus.OUT_OF_DENSITY_BOUND,
    }
    for s in forbidden:
        with pytest.raises(ValueError, match="INV-RECONSTRUCTION-2"):
            assert_real_data_status_legal(s)


def test_synthetic_path_emits_recovery_status_never_domain_status() -> None:
    """The mirror contract: synthetic path forbids domain-of-validity
    statuses (those are reserved for real data)."""
    legal = {
        ReconstructionStatus.GROUND_TRUTH_RECOVERED,
        ReconstructionStatus.GROUND_TRUTH_NOT_RECOVERED,
        ReconstructionStatus.INVALID_RECONSTRUCTION,
        ReconstructionStatus.OUT_OF_DENSITY_BOUND,
    }
    for s in legal:
        assert_synthetic_status_legal(s)

    forbidden = {
        ReconstructionStatus.WITHIN_VALIDATED_DOMAIN,
        ReconstructionStatus.OUT_OF_VALIDATED_DOMAIN,
        ReconstructionStatus.INSUFFICIENT_CERTIFICATE,
    }
    for s in forbidden:
        with pytest.raises(ValueError, match="INV-RECONSTRUCTION-1"):
            assert_synthetic_status_legal(s)


# ---------------------------------------------------------------------------
# DomainCheck behaviour & input contract
# ---------------------------------------------------------------------------


def test_domain_check_serialises_to_dict() -> None:
    cert = _synthetic_certificate_at(n=80, seed=4)
    s_out, s_in = _marginals_for_n(n=80, seed=16)
    inside = float((min(cert.tested_at_densities) + max(cert.tested_at_densities)) / 2.0)
    check = check_domain_of_validity(s_out, s_in, cert, inferred_density=inside)
    payload = check.as_dict()
    assert payload["status"] == check.status.value
    assert "checks" in payload
    assert "measured" in payload
    assert "certified_envelope" in payload
    assert isinstance(payload["out_of_range_dims"], list)


def test_domain_check_rejects_shape_mismatch() -> None:
    cert = _synthetic_certificate_at(n=80, seed=5)
    s_out = np.zeros(50)
    s_in = np.zeros(60)
    with pytest.raises(ValueError, match="shape mismatch"):
        check_domain_of_validity(s_out, s_in, cert)


def test_domain_check_rejects_too_small_n() -> None:
    cert = _synthetic_certificate_at(n=80, seed=6)
    s = np.array([1.0])
    with pytest.raises(ValueError, match="at least 2 nodes"):
        check_domain_of_validity(s, s, cert)


def test_domain_check_rejects_non_finite_marginals() -> None:
    cert = _synthetic_certificate_at(n=80, seed=7)
    s_out = np.array([1.0, np.nan, 3.0])
    s_in = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="finite"):
        check_domain_of_validity(s_out, s_in, cert)


def test_evidence_envelope_reports_min_max() -> None:
    """The envelope helper returns (min, max) per dimension that has evidence."""
    cert = _synthetic_certificate_at(n=80, seed=8)
    env = cert.evidence_envelope()
    assert "n_nodes" in env
    assert env["n_nodes"] == (80, 80)
    assert "density" in env
    lo, hi = env["density"]
    assert lo <= hi
    # Reciprocity not yet swept (FIX B6 placeholder).
    assert "reciprocity" not in env


def test_within_domain_when_real_n_at_envelope_boundary() -> None:
    """N exactly at the envelope max passes (closed interval contract)."""
    cert = _stub_certificate_only_n(n_set=(50, 200))
    cert_with_density = GroundTruthRecoveryCertificate(
        substrate_name=cert.substrate_name,
        n_nodes=cert.n_nodes,
        target_density=cert.target_density,
        sweep_densities=cert.sweep_densities,
        per_density_reports=cert.per_density_reports,
        passed=cert.passed,
        failure_reasons=cert.failure_reasons,
        cert_id=cert.cert_id,
        tested_at_n_nodes=cert.tested_at_n_nodes,
        tested_at_densities=(0.05,),
        tested_at_reciprocity=cert.tested_at_reciprocity,
    )
    s_out, s_in = _marginals_for_n(n=200, seed=17)
    check = check_domain_of_validity(s_out, s_in, cert_with_density, inferred_density=0.05)
    assert check.status is DomainOfValidityStatus.WITHIN_VALIDATED_DOMAIN
    assert check.checks["n_nodes"] is True
    assert check.checks["density"] is True


def test_certified_envelope_is_populated_on_within_verdict() -> None:
    cert = _synthetic_certificate_at(n=80, seed=9)
    s_out, s_in = _marginals_for_n(n=80, seed=18)
    inside = float((min(cert.tested_at_densities) + max(cert.tested_at_densities)) / 2.0)
    check = check_domain_of_validity(s_out, s_in, cert, inferred_density=inside)
    assert "n_nodes" in check.certified_envelope
    assert "density" in check.certified_envelope
    n_lo, n_hi = check.certified_envelope["n_nodes"]
    assert n_lo == n_hi == 80


def test_reciprocity_required_but_missing_returns_insufficient() -> None:
    """Caller can demand a reciprocity gate even though FIX B6 hasn't shipped;
    the gate must then report INSUFFICIENT_CERTIFICATE rather than a free pass."""
    cert = _synthetic_certificate_at(n=80, seed=10)
    s_out, s_in = _marginals_for_n(n=80, seed=19)
    inside = float((min(cert.tested_at_densities) + max(cert.tested_at_densities)) / 2.0)
    check = check_domain_of_validity(
        s_out,
        s_in,
        cert,
        inferred_density=inside,
        require_dims=("n_nodes", "density", "reciprocity"),
    )
    assert check.status is DomainOfValidityStatus.INSUFFICIENT_CERTIFICATE
    assert "reciprocity" in check.missing_dims
