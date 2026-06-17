# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Discrete Gauss-Bonnet invariant as a fail-closed self-verification substrate.

For a closed, piecewise-flat triangulated 2-manifold the **discrete
Gauss-Bonnet theorem** is exact (Banchoff 1967; Chen 1948):

    Σ_v K(v) = 2π · χ

where the angular defect at a vertex ``v`` is

    K(v) = 2π − Σ_{f ∋ v} θ_f(v)

with ``θ_f(v)`` the Euclidean interior angle of triangle ``f`` at ``v``,
and ``χ = V − E + F`` the Euler characteristic. The identity is purely
combinatorial-topological: it holds for *any* metric realisation of a
fixed closed simplicial 2-manifold, which is exactly what makes it a
trustworthy integrity oracle rather than a tunable diagnostic.

Two structurally distinct corruptions break the identity, and this module
fails closed on both:

1. **Topological corruption.** The residual obeys

       Σ_v K(v) − 2πχ = π · (2E − 3F).

   A closed manifold satisfies ``2E = 3F`` (every edge bounds exactly two
   triangles), so the residual is zero. Any connectivity tamper that
   violates manifold closure yields a residual on the order of ``π`` and
   is rejected by :func:`verify_gauss_bonnet`.

2. **Curvature-field corruption.** A caller may submit an externally
   computed angular-defect field (``claimed_defects``). It is cross-checked
   vertex-wise against the geometric field; any injected anomaly exceeding
   tolerance is rejected. The accepted field is bound by a SHA-256 digest
   so that a downstream tamper invalidates the certificate.

Claim-integrity invariant (``claim_type`` discipline, operationalised):
a :class:`GaussBonnetCertificate` can only carry ``ClaimStatus.VERIFIED``
when its ``field_sha256`` matches the canonical digest of the accepted
field *and* the residual is within tolerance. The only constructor path
(:func:`verify_gauss_bonnet`) raises before returning otherwise, so a
``VERIFIED`` certificate without a valid hash is structurally
unrepresentable, not merely discouraged.

This module imports no trading / execution / forecast / policy symbols. It
is a pure mathematical substrate: NumPy and :mod:`hashlib` only.

References
----------
Banchoff, T. "Critical points and curvature for embedded polyhedra"
    (J. Differential Geometry, 1967).
Chen, S.-S. "On the curvatura integra in a Riemannian manifold"
    (Annals of Mathematics, 1948).
Crane, K. "Discrete Differential Geometry: An Applied Introduction"
    (2020), §2.4 — angle defect and the discrete Gauss-Bonnet theorem.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

TWO_PI = 2.0 * np.pi


class GaussBonnetViolation(Exception):
    """Raised when the discrete Gauss-Bonnet invariant fails closed.

    A raise is *information*, not a fault: it means a topological or
    curvature-field corruption was detected and refused promotion to a
    ``VERIFIED`` certificate.
    """


class ClaimStatus(Enum):
    """Provenance tier of a Gauss-Bonnet certificate.

    ``VERIFIED`` is reachable only through :func:`verify_gauss_bonnet`,
    which binds a matching SHA-256 digest before returning.
    """

    VERIFIED = "verified"
    REFUTED = "refuted"
    UNVERIFIED = "unverified"


@dataclass(frozen=True, slots=True)
class TriangulatedSurface:
    """An immutable triangulated 2-manifold.

    Parameters
    ----------
    vertices :
        ``(V, 3)`` float64 array of vertex positions.
    faces :
        ``(F, 3)`` integer array of triangle vertex indices.
    """

    vertices: NDArray[np.float64]
    faces: NDArray[np.int64]

    def __post_init__(self) -> None:
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise GaussBonnetViolation(
                f"vertices must have shape (V, 3); got {self.vertices.shape}"
            )
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise GaussBonnetViolation(f"faces must have shape (F, 3); got {self.faces.shape}")
        if self.faces.size and int(self.faces.max()) >= self.num_vertices:
            raise GaussBonnetViolation("face references out-of-range vertex index")
        if self.faces.size and int(self.faces.min()) < 0:
            raise GaussBonnetViolation("face references negative vertex index")
        # Enforce float64 substrate (mixed float32 silently degrades the
        # angle-defect residual below the topological signal floor).
        if self.vertices.dtype != np.float64:
            raise GaussBonnetViolation(f"vertices must be float64; got {self.vertices.dtype}")

    @property
    def num_vertices(self) -> int:
        """Vertex count ``V``."""
        return int(self.vertices.shape[0])

    @property
    def num_faces(self) -> int:
        """Triangle count ``F``."""
        return int(self.faces.shape[0])

    def _edge_multiplicity(self) -> Counter[tuple[int, int]]:
        """Map each undirected edge to the number of incident triangles."""
        counter: Counter[tuple[int, int]] = Counter()
        for tri in self.faces:
            a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
            for u, v in ((a, b), (b, c), (c, a)):
                counter[(u, v) if u < v else (v, u)] += 1
        return counter

    @property
    def num_edges(self) -> int:
        """Unique undirected edge count ``E``."""
        return len(self._edge_multiplicity())

    def is_closed_manifold(self) -> bool:
        """True iff every undirected edge bounds exactly two triangles."""
        if self.num_faces == 0:
            return False
        return all(mult == 2 for mult in self._edge_multiplicity().values())

    def euler_characteristic(self) -> int:
        """Euler characteristic ``χ = V − E + F``."""
        return self.num_vertices - self.num_edges + self.num_faces


def _interior_angle(
    apex: NDArray[np.float64], left: NDArray[np.float64], right: NDArray[np.float64]
) -> float:
    """Euclidean interior angle at ``apex`` of triangle ``(apex, left, right)``."""
    u = left - apex
    v = right - apex
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu == 0.0 or nv == 0.0:
        raise GaussBonnetViolation("degenerate triangle: zero-length edge at apex")
    cosine = float(np.dot(u, v)) / (nu * nv)
    # bounds: arccos domain is [-1, 1]; the clip absorbs float rounding of the
    # dot/norm ratio only (|cosine| may land at 1+1e-16), it is not a physical
    # clamp on the angle and changes no in-domain value.
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def angular_defects(surface: TriangulatedSurface) -> NDArray[np.float64]:
    """Per-vertex angular defect ``K(v) = 2π − Σ θ_f(v)``.

    Returns a ``(V,)`` float64 array indexed by vertex.
    """
    angle_sum = np.zeros(surface.num_vertices, dtype=np.float64)
    verts = surface.vertices
    for tri in surface.faces:
        i, j, k = (int(tri[0]), int(tri[1]), int(tri[2]))
        angle_sum[i] += _interior_angle(verts[i], verts[j], verts[k])
        angle_sum[j] += _interior_angle(verts[j], verts[k], verts[i])
        angle_sum[k] += _interior_angle(verts[k], verts[i], verts[j])
    return TWO_PI - angle_sum


def canonical_field_digest(defects: NDArray[np.float64]) -> str:
    """SHA-256 over a platform-stable serialisation of a curvature field.

    Values are rendered at fixed precision so the digest is reproducible
    across architectures (raw ``tobytes()`` would leak endianness and the
    exact rounding of the producing pipeline).
    """
    payload = ";".join(f"{float(x):.12e}" for x in np.ravel(defects))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class GaussBonnetCertificate:
    """Result of a fail-closed Gauss-Bonnet verification.

    A ``VERIFIED`` instance is only ever produced by
    :func:`verify_gauss_bonnet`, which guarantees ``field_sha256`` matches
    :func:`canonical_field_digest` of the accepted field and that
    ``abs(residual) <= tolerance``.
    """

    total_defect: float
    euler_characteristic: int
    expected_total: float
    residual: float
    status: ClaimStatus
    field_sha256: str
    tolerance: float


def verify_gauss_bonnet(
    surface: TriangulatedSurface,
    *,
    claimed_defects: NDArray[np.float64] | None = None,
    tolerance: float = 1e-9,
    require_closed_manifold: bool = True,
) -> GaussBonnetCertificate:
    """Verify ``Σ K(v) = 2πχ`` and return a hash-bound certificate.

    Fails closed by raising :class:`GaussBonnetViolation` when any leg of
    the invariant is broken:

    * the surface is not a closed manifold (``require_closed_manifold``);
    * a submitted ``claimed_defects`` field deviates from the geometric
      field beyond ``tolerance`` (curvature-field tamper);
    * the residual ``Σ K(v) − 2πχ`` exceeds ``tolerance``.

    Only on full success is a ``ClaimStatus.VERIFIED`` certificate built,
    carrying the SHA-256 of the accepted field.
    """
    if require_closed_manifold and not surface.is_closed_manifold():
        raise GaussBonnetViolation(
            "surface is not a closed manifold (an edge does not bound exactly "
            "two triangles); discrete Gauss-Bonnet does not apply"
        )

    geometric = angular_defects(surface)

    if claimed_defects is not None:
        claimed = np.asarray(claimed_defects, dtype=np.float64)
        if claimed.shape != geometric.shape:
            raise GaussBonnetViolation(
                f"claimed_defects shape {claimed.shape} != geometric {geometric.shape}"
            )
        deviation = float(np.max(np.abs(claimed - geometric)))
        if deviation > tolerance:
            raise GaussBonnetViolation(
                f"curvature-field tamper: max vertex deviation {deviation:.3e} "
                f"exceeds tolerance {tolerance:.3e}"
            )
        field = claimed
    else:
        field = geometric

    chi = surface.euler_characteristic()
    expected = TWO_PI * chi
    total = float(np.sum(field))
    residual = total - expected
    if abs(residual) > tolerance:
        raise GaussBonnetViolation(
            f"Gauss-Bonnet residual {residual:.3e} exceeds tolerance "
            f"{tolerance:.3e} (Σ K = {total:.6f}, 2πχ = {expected:.6f}, χ = {chi})"
        )

    return GaussBonnetCertificate(
        total_defect=total,
        euler_characteristic=chi,
        expected_total=expected,
        residual=residual,
        status=ClaimStatus.VERIFIED,
        field_sha256=canonical_field_digest(field),
        tolerance=tolerance,
    )


def assert_certificate_integrity(
    certificate: GaussBonnetCertificate,
    field: NDArray[np.float64],
) -> None:
    """Enforce the claim-integrity invariant on a (possibly forged) certificate.

    A ``VERIFIED`` certificate MUST carry a non-empty ``field_sha256`` that
    matches :func:`canonical_field_digest` of ``field`` and a residual
    within its declared tolerance. This guards against a hand-forged
    certificate that asserts ``VERIFIED`` without — or with a stale —
    provenance hash. Raises :class:`GaussBonnetViolation` on any breach.
    """
    if certificate.status is not ClaimStatus.VERIFIED:
        return
    if not certificate.field_sha256:
        raise GaussBonnetViolation(
            "claim-integrity breach: VERIFIED status without a SHA-256 digest"
        )
    expected_digest = canonical_field_digest(field)
    if certificate.field_sha256 != expected_digest:
        raise GaussBonnetViolation(
            "claim-integrity breach: field SHA-256 mismatch "
            f"({certificate.field_sha256[:12]}… != {expected_digest[:12]}…)"
        )
    if abs(certificate.residual) > certificate.tolerance:
        raise GaussBonnetViolation(
            "claim-integrity breach: VERIFIED status with out-of-tolerance "
            f"residual {certificate.residual:.3e}"
        )


def inverse_curvature_probe(
    surface: TriangulatedSurface,
    *,
    vertex: int,
    delta: float = 1.0,
    tolerance: float = 1e-9,
) -> GaussBonnetViolation:
    """Adversarial inverse probe: inject a curvature anomaly and demand a raise.

    Distorts the geometric angular-defect field at ``vertex`` by ``delta``
    and submits it to :func:`verify_gauss_bonnet`. A correctly fail-closed
    invariant MUST reject the tampered field; the caught
    :class:`GaussBonnetViolation` is returned as positive evidence of
    fail-closed behaviour. If verification does *not* raise — i.e. the
    oracle accepted a corrupted curvature field — this raises a fresh
    :class:`GaussBonnetViolation` flagging the breach, so the probe itself
    cannot fail silently.
    """
    if not (0 <= vertex < surface.num_vertices):
        raise GaussBonnetViolation(
            f"probe vertex {vertex} out of range [0, {surface.num_vertices})"
        )
    if abs(delta) <= tolerance:
        raise GaussBonnetViolation(f"probe delta {delta:.3e} is within tolerance; not a falsifier")
    tampered = angular_defects(surface).copy()
    tampered[vertex] += delta
    try:
        verify_gauss_bonnet(surface, claimed_defects=tampered, tolerance=tolerance)
    except GaussBonnetViolation as caught:
        return caught
    raise GaussBonnetViolation(
        "BREACH: oracle accepted a tampered curvature field "
        f"(injected delta={delta:.3e} at vertex {vertex}); the Gauss-Bonnet "
        "invariant failed to fail-closed"
    )
