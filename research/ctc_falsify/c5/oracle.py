# SPDX-License-Identifier: MIT
"""Near-oracle upper bound: best OOS linear discriminant over the full
gamma cross-spectral representation. Dependency-free (numpy/scipy only);
ROC-AUC via the Mann–Whitney U statistic; LDA via the pooled-covariance
pseudo-inverse. Train/test seeds are disjoint — the AUC must be a real
out-of-sample property, never a memorised one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import csd

from research.ctc_falsify import config as l1
from research.ctc_falsify.c5 import config_c5 as c5
from research.ctc_falsify.generative import draw_n_plus, draw_null

_FS: float = 1.0 / l1.DT
_LO: float = l1.F0 - l1.GAMMA_BAND_HALFWIDTH
_HI: float = l1.F0 + l1.GAMMA_BAND_HALFWIDTH


@dataclass(frozen=True)
class OracleResult:
    oos_auc: float
    n_train: int
    n_test: int
    n_features: int
    train_test_disjoint: bool


def _feature(sig_a: np.ndarray, sig_b: np.ndarray) -> np.ndarray:
    f, pxy = csd(sig_a, sig_b, fs=_FS, nperseg=c5.CSD_NPERSEG)
    band = (f >= _LO) & (f <= _HI)
    z = np.asarray(pxy, dtype=np.complex128)[band]
    return np.concatenate([np.abs(z), np.angle(z)]).astype(np.float64)


def _features_for(seeds: list[int]) -> tuple[np.ndarray, np.ndarray]:
    pos = [draw_n_plus(s) for s in seeds]
    neg = [draw_null(s, fam) for fam in l1.NULL_FAMILIES for s in seeds]
    x_pos = np.array([_feature(p.sig_a, p.sig_b) for p in pos], dtype=np.float64)
    x_neg = np.array([_feature(n.sig_a, n.sig_b) for n in neg], dtype=np.float64)
    x = np.vstack([x_pos, x_neg])
    y = np.concatenate([np.ones(len(x_pos)), np.zeros(len(x_neg))])
    return x, y


def _lda_direction(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xp, xn = x[y == 1], x[y == 0]
    mu = xp.mean(0) - xn.mean(0)
    cov = np.cov(np.vstack([xp - xp.mean(0), xn - xn.mean(0)]).T)
    w = np.linalg.pinv(cov) @ mu
    return np.asarray(w, dtype=np.float64)


def _auc(scores: np.ndarray, y: np.ndarray) -> float:
    pos = scores[y == 1]
    neg = scores[y == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    order = np.argsort(np.concatenate([pos, neg]), kind="stable")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1)
    r_pos = ranks[: pos.size].sum()
    u = r_pos - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def run_oracle() -> OracleResult:
    tr, te = c5.train_seeds(), c5.test_seeds()
    disjoint = set(tr).isdisjoint(set(te))

    x_tr, y_tr = _features_for(tr)
    x_te, y_te = _features_for(te)
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-12
    w = _lda_direction((x_tr - mu) / sd, y_tr)
    s_te = ((x_te - mu) / sd) @ w
    auc = _auc(s_te, y_te)
    # AUC is symmetric about 0.5 w.r.t. orientation; report discriminability.
    auc = max(auc, 1.0 - auc)
    return OracleResult(
        oos_auc=auc,
        n_train=len(x_tr),
        n_test=len(x_te),
        n_features=int(x_tr.shape[1]),
        train_test_disjoint=disjoint,
    )
