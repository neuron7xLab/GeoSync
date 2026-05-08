# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Interbank network topology — empirical adapter + null baselines.

Two construction paths:

1. **Empirical**: weighted, possibly directed exposure matrix
   (``A[i, j]`` = lending from bank *i* to bank *j*) is symmetrised with
   :math:`A_{sym} = (A + A^T) / 2` and binarised at a documented
   threshold to give the undirected coupling support used by the
   Kuramoto model. The binary skeleton is what enters the inverse
   problem; magnitudes are kept on a parallel ``weights`` matrix for
   diagnostics.

2. **Barabási–Albert null** (preferential attachment, *m* edges per
   step). Empirical interbank networks are well documented to be
   approximately scale-free with γ ≈ 2.0–2.3 (Boss et al. 2004,
   *Quant. Finance* 4: 677; Soramäki et al. 2007, *Physica A* 379:
   317). BA at *m* ≈ 2–4 reproduces this regime and is used **only** as
   the topology-null baseline of the falsification battery, never as a
   model of the real economy.

Pure-function API. No I/O. No randomness except where ``rng`` is passed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "InterbankTopology",
    "from_exposure_matrix",
    "barabasi_albert_null",
]


@dataclass(frozen=True, slots=True)
class InterbankTopology:
    """Frozen interbank topology.

    Attributes
    ----------
    adjacency
        Binary symmetric adjacency, ``int8`` shape ``(N, N)``, zero
        diagonal. ``adjacency[i, j] == 1`` iff there is a coupling
        between *i* and *j* in the support graph.
    weights
        Real, non-negative, symmetric exposure magnitudes shape
        ``(N, N)``, zero diagonal. ``None`` when the topology was
        generated synthetically (BA null).
    node_labels
        Tuple of node identifiers; length ``N``.
    source_label
        Free-form provenance tag, e.g. ``"e-MID_2011-Q3"`` or
        ``"BA_null_m=3_seed=42"``.

    Invariants
    ----------
    INV-TOP1: ``adjacency`` is symmetric, binary, zero-diagonal.
    INV-TOP2: ``weights`` (when present) is symmetric, non-negative,
              zero-diagonal, with the same shape as ``adjacency``.
    INV-TOP3: ``len(node_labels) == adjacency.shape[0]``.
    """

    adjacency: NDArray[np.int8]
    node_labels: tuple[str, ...]
    source_label: str
    weights: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        a = self.adjacency
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError(f"INV-TOP1: adjacency must be square 2-D, got shape={a.shape}")
        n = a.shape[0]
        if not np.array_equal(a, a.T):
            raise ValueError("INV-TOP1: adjacency must be symmetric")
        if not np.all((a == 0) | (a == 1)):
            raise ValueError("INV-TOP1: adjacency must be binary {0,1}")
        if np.any(np.diag(a) != 0):
            raise ValueError("INV-TOP1: adjacency must have zero diagonal")
        if len(self.node_labels) != n:
            raise ValueError(f"INV-TOP3: len(node_labels)={len(self.node_labels)} != N={n}")
        w = self.weights
        if w is not None:
            if w.shape != a.shape:
                raise ValueError(f"INV-TOP2: weights shape={w.shape} != adjacency {a.shape}")
            if not np.allclose(w, w.T, equal_nan=False):
                raise ValueError("INV-TOP2: weights must be symmetric")
            if np.any(w < 0):
                raise ValueError("INV-TOP2: weights must be non-negative")
            if np.any(np.diag(w) != 0):
                raise ValueError("INV-TOP2: weights must have zero diagonal")
        # Freeze arrays to prevent silent mutation downstream.
        a_frozen = np.array(a, dtype=np.int8, copy=True)
        a_frozen.flags.writeable = False
        object.__setattr__(self, "adjacency", a_frozen)
        if w is not None:
            w_frozen = np.array(w, dtype=np.float64, copy=True)
            w_frozen.flags.writeable = False
            object.__setattr__(self, "weights", w_frozen)

    @property
    def n_nodes(self) -> int:
        return int(self.adjacency.shape[0])

    @property
    def degree(self) -> NDArray[np.int64]:
        out: NDArray[np.int64] = self.adjacency.sum(axis=1, dtype=np.int64)
        return out


def from_exposure_matrix(
    exposures: NDArray[np.float64],
    node_labels: tuple[str, ...],
    *,
    threshold: float = 0.0,
    source_label: str = "empirical_exposure",
) -> InterbankTopology:
    """Build an :class:`InterbankTopology` from a possibly directed exposure matrix.

    Parameters
    ----------
    exposures
        Real, non-negative matrix shape ``(N, N)``. ``exposures[i, j]``
        is the magnitude of *i*'s exposure to *j* (e.g. lending volume).
        May be directed; symmetrisation by averaging is applied.
    node_labels
        Length-``N`` tuple of node identifiers.
    threshold
        Inclusive lower bound on a symmetrised entry for it to enter
        the binary support. Default ``0.0`` keeps every non-zero edge.
    source_label
        Provenance tag stored on the resulting topology.
    """
    e = np.asarray(exposures, dtype=np.float64)
    if e.ndim != 2 or e.shape[0] != e.shape[1]:
        raise ValueError(f"exposures must be square 2-D, got shape={e.shape}")
    if e.shape[0] != len(node_labels):
        raise ValueError(f"node_labels length {len(node_labels)} != exposures dim {e.shape[0]}")
    if np.any(e < 0):
        raise ValueError("exposures must be non-negative")
    if not np.isfinite(e).all():
        raise ValueError("exposures must be finite (no NaN/Inf)")
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold}")
    sym = 0.5 * (e + e.T)
    np.fill_diagonal(sym, 0.0)
    adj = (sym > threshold).astype(np.int8)
    np.fill_diagonal(adj, 0)
    return InterbankTopology(
        adjacency=adj,
        weights=sym,
        node_labels=tuple(node_labels),
        source_label=source_label,
    )


def barabasi_albert_null(
    n_nodes: int,
    m: int,
    *,
    seed: int,
    source_label: str | None = None,
) -> InterbankTopology:
    """Barabási–Albert preferential attachment, *m* edges per new node.

    Implementation is self-contained (no NetworkX dependency) so the
    module remains importable in minimal install environments and
    the seed semantics are unambiguous: ``rng = default_rng(seed)``.

    Empirical anchor: :math:`m \\in \\{2, 3, 4\\}` reproduces interbank
    degree distributions observed in Boss et al. 2004 and Soramäki et
    al. 2007. The mean degree of the resulting graph is ``2m`` for
    large ``n``.

    Parameters
    ----------
    n_nodes
        Final number of nodes (must be > ``m``).
    m
        Edges added per new node (must be >= 1).
    seed
        Seed for ``np.random.default_rng``. Required — there is no
        sensible default for a null model.
    source_label
        Optional override for the resulting ``source_label`` field.
    """
    if n_nodes <= m:
        raise ValueError(f"n_nodes={n_nodes} must be > m={m}")
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")

    rng = np.random.default_rng(seed)
    adj = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    # Seed graph: complete on m+1 nodes so every initial node has degree m.
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            adj[i, j] = 1
            adj[j, i] = 1
    # Repeated nodes list for the standard PA draw without replacement.
    repeated: list[int] = []
    for i in range(m + 1):
        repeated.extend([i] * m)
    for new_node in range(m + 1, n_nodes):
        targets: set[int] = set()
        while len(targets) < m:
            t = int(repeated[int(rng.integers(0, len(repeated)))])
            if t != new_node:
                targets.add(t)
        for t in targets:
            adj[new_node, t] = 1
            adj[t, new_node] = 1
            repeated.append(t)
        repeated.extend([new_node] * m)
    label = source_label if source_label is not None else f"BA_null_m={m}_seed={seed}"
    return InterbankTopology(
        adjacency=adj,
        weights=None,
        node_labels=tuple(f"n{i}" for i in range(n_nodes)),
        source_label=label,
    )
