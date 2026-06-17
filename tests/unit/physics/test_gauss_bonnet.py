# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Falsification battery for the discrete Gauss-Bonnet self-verification substrate.

The tests assert the theorem on closed manifolds of two distinct Euler
characteristics (sphere χ=2 via tetrahedron/octahedron, torus χ=0) and then
adversarially attack every fail-closed leg: topological corruption,
curvature-field tamper, inverse probe, and forged-certificate integrity.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from core.physics.gauss_bonnet import (
    ClaimStatus,
    GaussBonnetCertificate,
    GaussBonnetViolation,
    TriangulatedSurface,
    angular_defects,
    assert_certificate_integrity,
    canonical_field_digest,
    inverse_curvature_probe,
    verify_gauss_bonnet,
)

TWO_PI = 2.0 * np.pi


def _tetrahedron() -> TriangulatedSurface:
    vertices: NDArray[np.float64] = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    )
    faces: NDArray[np.int64] = np.array(
        [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], dtype=np.int64
    )
    return TriangulatedSurface(vertices=vertices, faces=faces)


def _octahedron() -> TriangulatedSurface:
    vertices: NDArray[np.float64] = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    faces: NDArray[np.int64] = np.array(
        [
            [4, 0, 2],
            [4, 2, 1],
            [4, 1, 3],
            [4, 3, 0],
            [5, 2, 0],
            [5, 1, 2],
            [5, 3, 1],
            [5, 0, 3],
        ],
        dtype=np.int64,
    )
    return TriangulatedSurface(vertices=vertices, faces=faces)


def _torus(n: int = 12, m: int = 24, big_r: float = 3.0, small_r: float = 1.0) -> TriangulatedSurface:
    """Embedded torus tube (χ = 0); every edge bounds exactly two triangles."""
    coords: list[list[float]] = []
    for i in range(n):
        theta = TWO_PI * i / n
        for j in range(m):
            phi = TWO_PI * j / m
            x = (big_r + small_r * np.cos(theta)) * np.cos(phi)
            y = (big_r + small_r * np.cos(theta)) * np.sin(phi)
            z = small_r * np.sin(theta)
            coords.append([float(x), float(y), float(z)])
    vertices: NDArray[np.float64] = np.array(coords, dtype=np.float64)

    def idx(i: int, j: int) -> int:
        return (i % n) * m + (j % m)

    tris: list[list[int]] = []
    for i in range(n):
        for j in range(m):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            tris.append([a, b, d])
            tris.append([b, c, d])
    faces: NDArray[np.int64] = np.array(tris, dtype=np.int64)
    return TriangulatedSurface(vertices=vertices, faces=faces)


class TestInvariantOnClosedManifolds:
    def test_tetrahedron_is_sphere(self) -> None:
        surface = _tetrahedron()
        assert surface.is_closed_manifold()
        assert surface.euler_characteristic() == 2
        cert = verify_gauss_bonnet(surface)
        assert cert.status is ClaimStatus.VERIFIED
        assert cert.euler_characteristic == 2
        assert abs(cert.total_defect - 2.0 * TWO_PI) < 1e-9
        assert abs(cert.residual) < 1e-9
        assert len(cert.field_sha256) == 64

    def test_tetrahedron_defects_are_pi(self) -> None:
        defects = angular_defects(_tetrahedron())
        # Each vertex sees three equilateral faces: angle sum = π, defect = π.
        assert np.allclose(defects, np.pi, atol=1e-9)

    def test_octahedron_is_sphere(self) -> None:
        surface = _octahedron()
        assert surface.is_closed_manifold()
        assert surface.euler_characteristic() == 2
        cert = verify_gauss_bonnet(surface)
        assert cert.status is ClaimStatus.VERIFIED
        assert abs(cert.total_defect - 2.0 * TWO_PI) < 1e-9

    def test_torus_has_zero_total_curvature(self) -> None:
        surface = _torus()
        assert surface.is_closed_manifold()
        assert surface.euler_characteristic() == 0
        cert = verify_gauss_bonnet(surface)
        assert cert.status is ClaimStatus.VERIFIED
        assert abs(cert.total_defect) < 1e-9
        assert abs(cert.residual) < 1e-9


class TestFailClosedTopology:
    def test_open_surface_is_rejected(self) -> None:
        full = _tetrahedron()
        open_surface = TriangulatedSurface(
            vertices=full.vertices, faces=full.faces[:-1]
        )
        assert not open_surface.is_closed_manifold()
        with pytest.raises(GaussBonnetViolation, match="closed manifold"):
            verify_gauss_bonnet(open_surface)

    def test_open_surface_accepted_only_when_closure_not_required(self) -> None:
        # With the manifold guard disabled the residual leg still fires,
        # because an open surface no longer satisfies Σ K = 2πχ.
        full = _tetrahedron()
        open_surface = TriangulatedSurface(
            vertices=full.vertices, faces=full.faces[:-1]
        )
        with pytest.raises(GaussBonnetViolation, match="residual"):
            verify_gauss_bonnet(open_surface, require_closed_manifold=False)


class TestFailClosedCurvatureField:
    def test_tampered_field_is_rejected(self) -> None:
        surface = _tetrahedron()
        tampered = angular_defects(surface).copy()
        tampered[0] += 0.5
        with pytest.raises(GaussBonnetViolation, match="tamper"):
            verify_gauss_bonnet(surface, claimed_defects=tampered)

    def test_clean_field_is_accepted(self) -> None:
        surface = _octahedron()
        clean = angular_defects(surface)
        cert = verify_gauss_bonnet(surface, claimed_defects=clean)
        assert cert.status is ClaimStatus.VERIFIED

    def test_wrong_shape_field_is_rejected(self) -> None:
        surface = _tetrahedron()
        with pytest.raises(GaussBonnetViolation, match="shape"):
            verify_gauss_bonnet(
                surface, claimed_defects=np.zeros(99, dtype=np.float64)
            )


class TestInverseProbe:
    def test_probe_catches_a_real_violation(self) -> None:
        surface = _octahedron()
        caught = inverse_curvature_probe(surface, vertex=0, delta=0.75)
        assert isinstance(caught, GaussBonnetViolation)
        assert "tamper" in str(caught)

    def test_probe_rejects_subtolerance_delta(self) -> None:
        surface = _tetrahedron()
        with pytest.raises(GaussBonnetViolation, match="not a falsifier"):
            inverse_curvature_probe(surface, vertex=0, delta=1e-15)

    def test_probe_rejects_out_of_range_vertex(self) -> None:
        surface = _tetrahedron()
        with pytest.raises(GaussBonnetViolation, match="out of range"):
            inverse_curvature_probe(surface, vertex=99, delta=1.0)


class TestCertificateIntegrity:
    def test_genuine_certificate_passes(self) -> None:
        surface = _tetrahedron()
        field = angular_defects(surface)
        cert = verify_gauss_bonnet(surface)
        assert_certificate_integrity(cert, field)  # no raise

    def test_verified_without_hash_is_breach(self) -> None:
        surface = _tetrahedron()
        field = angular_defects(surface)
        forged = GaussBonnetCertificate(
            total_defect=2.0 * TWO_PI,
            euler_characteristic=2,
            expected_total=2.0 * TWO_PI,
            residual=0.0,
            status=ClaimStatus.VERIFIED,
            field_sha256="",
            tolerance=1e-9,
        )
        with pytest.raises(GaussBonnetViolation, match="without a SHA-256"):
            assert_certificate_integrity(forged, field)

    def test_verified_with_wrong_hash_is_breach(self) -> None:
        surface = _tetrahedron()
        field = angular_defects(surface)
        forged = GaussBonnetCertificate(
            total_defect=2.0 * TWO_PI,
            euler_characteristic=2,
            expected_total=2.0 * TWO_PI,
            residual=0.0,
            status=ClaimStatus.VERIFIED,
            field_sha256="0" * 64,
            tolerance=1e-9,
        )
        with pytest.raises(GaussBonnetViolation, match="mismatch"):
            assert_certificate_integrity(forged, field)

    def test_non_verified_certificate_is_noop(self) -> None:
        surface = _tetrahedron()
        field = angular_defects(surface)
        unverified = GaussBonnetCertificate(
            total_defect=0.0,
            euler_characteristic=2,
            expected_total=2.0 * TWO_PI,
            residual=-2.0 * TWO_PI,
            status=ClaimStatus.UNVERIFIED,
            field_sha256="",
            tolerance=1e-9,
        )
        assert_certificate_integrity(unverified, field)  # no raise


class TestDigestAndValidation:
    def test_digest_is_deterministic_and_sensitive(self) -> None:
        field = angular_defects(_tetrahedron())
        assert canonical_field_digest(field) == canonical_field_digest(field.copy())
        bumped = field.copy()
        bumped[0] += 1e-6
        assert canonical_field_digest(field) != canonical_field_digest(bumped)

    def test_float32_vertices_rejected(self) -> None:
        verts = np.zeros((4, 3), dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        with pytest.raises(GaussBonnetViolation, match="float64"):
            TriangulatedSurface(vertices=verts, faces=faces)

    def test_bad_vertex_shape_rejected(self) -> None:
        with pytest.raises(GaussBonnetViolation, match="shape"):
            TriangulatedSurface(
                vertices=np.zeros((4, 2), dtype=np.float64),
                faces=np.array([[0, 1, 2]], dtype=np.int64),
            )

    def test_out_of_range_face_index_rejected(self) -> None:
        with pytest.raises(GaussBonnetViolation, match="out-of-range"):
            TriangulatedSurface(
                vertices=np.zeros((3, 3), dtype=np.float64),
                faces=np.array([[0, 1, 9]], dtype=np.int64),
            )
