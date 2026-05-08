# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Tests for :mod:`research.systemic_risk.falsification`.

Coverage:
* :func:`auc_mann_whitney` algebraic identities (perfect / inverted /
  uniform).
* :func:`benjamini_hochberg` monotonicity + clipping under standard
  edge cases (Benjamini & Hochberg 1995).
* End-to-end ``run_falsification`` null-AUC sanity:
  on i.i.d. random scores the per-crisis AUC distribution should be
  centred near 0.5 and the BH-corrected verdict should not be
  ``HARD_PASS``. This is the test that *catches* implementation bugs
  that systematically inflate AUC.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from research.systemic_risk.event_ledger import (
    BankingCrisisEvent,
    BankingCrisisLedger,
)
from research.systemic_risk.falsification import (
    FalsificationConfig,
    auc_mann_whitney,
    benjamini_hochberg,
    run_falsification,
)


class TestAucMannWhitney:
    def test_perfect_separation_yields_one(self) -> None:
        pos = np.array([5.0, 6.0, 7.0])
        neg = np.array([1.0, 2.0, 3.0])
        assert auc_mann_whitney(pos, neg) == pytest.approx(1.0)

    def test_perfect_inversion_yields_zero(self) -> None:
        pos = np.array([1.0, 2.0, 3.0])
        neg = np.array([5.0, 6.0, 7.0])
        assert auc_mann_whitney(pos, neg) == pytest.approx(0.0)

    def test_identical_yields_half(self) -> None:
        pos = np.array([2.0, 2.0, 2.0])
        neg = np.array([2.0, 2.0, 2.0])
        assert auc_mann_whitney(pos, neg) == pytest.approx(0.5)

    def test_empty_returns_half_by_convention(self) -> None:
        assert auc_mann_whitney(np.array([]), np.array([1.0])) == 0.5
        assert auc_mann_whitney(np.array([1.0]), np.array([])) == 0.5

    def test_random_iid_centred_at_half(self) -> None:
        # Statistical: under H0 (same distribution) E[AUC] = 0.5.
        rng = np.random.default_rng(123)
        aucs = []
        for _ in range(100):
            x = rng.standard_normal(50)
            y = rng.standard_normal(50)
            aucs.append(auc_mann_whitney(x, y))
        assert abs(float(np.mean(aucs)) - 0.5) < 0.05, (
            f"INV-AUC-IID VIOLATED: mean(AUC | H0) = {np.mean(aucs):.4f}, "
            f"expected 0.5 ± 0.05 over 100 reps of size 50, seed=123"
        )


class TestBenjaminiHochberg:
    def test_all_zero_input(self) -> None:
        out = benjamini_hochberg(np.zeros(5))
        assert np.all(out == 0.0)

    def test_all_one_input(self) -> None:
        out = benjamini_hochberg(np.ones(5))
        assert np.all(out == 1.0)

    def test_classic_example(self) -> None:
        # B-H (1995) Table 1 mini-case: 4 p-values [0.005, 0.01, 0.04, 0.5].
        # Adjusted: 4/1*0.005=0.02, 4/2*0.01=0.02, 4/3*0.04≈0.0533, 4/4*0.5=0.5.
        # After enforcing monotonicity from the largest, expected:
        # [0.02, 0.02, 0.0533, 0.5].
        p = np.array([0.005, 0.01, 0.04, 0.5])
        out = benjamini_hochberg(p)
        np.testing.assert_allclose(out, [0.02, 0.02, 0.05333333, 0.5], atol=1e-7)

    def test_order_preserved(self) -> None:
        # Output order matches input order, even with shuffled input.
        p = np.array([0.5, 0.005, 0.04, 0.01])
        out = benjamini_hochberg(p)
        # The smallest input is at index 1; its adjusted value should be 0.02.
        assert out[1] == pytest.approx(0.02)

    def test_invalid_p_rejected(self) -> None:
        with pytest.raises(ValueError):
            benjamini_hochberg(np.array([-0.1, 0.5]))
        with pytest.raises(ValueError):
            benjamini_hochberg(np.array([0.5, 1.5]))


class TestRunFalsificationSanity:
    def _build_synthetic_ledger_and_score(
        self,
        seed: int,
    ) -> tuple[BankingCrisisLedger, tuple[date, ...], np.ndarray]:
        # 5 crises spread across an 8-year synthetic series; pure noise score.
        # No physics → expected verdict ≠ HARD_PASS.
        start = date(2010, 1, 1)
        n_days = 365 * 8
        dates = tuple(start + timedelta(days=i) for i in range(n_days))
        events = (
            BankingCrisisEvent(
                country="ABC",
                start=date(2011, 6, 1),
                end=date(2011, 12, 31),
                source="LV2018",
                label="A_2011",
            ),
            BankingCrisisEvent(
                country="ABC",
                start=date(2013, 3, 1),
                end=date(2013, 9, 30),
                source="LV2018",
                label="A_2013",
            ),
            BankingCrisisEvent(
                country="ABC",
                start=date(2015, 1, 1),
                end=date(2015, 6, 30),
                source="LV2018",
                label="A_2015",
            ),
        )
        ledger = BankingCrisisLedger(events=events)
        rng = np.random.default_rng(seed)
        score = rng.standard_normal(n_days).astype(np.float64)
        return ledger, dates, score

    def test_random_scores_do_not_pass(self) -> None:
        ledger, dates, score = self._build_synthetic_ledger_and_score(seed=42)
        cfg = FalsificationConfig(
            pre_event_window_days=60,
            null_window_count=10,
            min_distance_from_event_days=180,
            n_permutations=200,
            seed=7,
        )
        report = run_falsification(score, dates, ledger, config=cfg, country_filter="ABC")
        # Random noise must not produce HARD_PASS — that would mean a leak.
        assert report.verdict != "HARD_PASS"
        # Either an outcome is below the fail threshold (HARD_FAIL) or none
        # of the AUCs are convincingly above the pass threshold (UNDECIDED).
        assert report.verdict in {"HARD_FAIL", "UNDECIDED"}

    def test_score_length_mismatch_rejected(self) -> None:
        ledger, dates, _ = self._build_synthetic_ledger_and_score(seed=0)
        with pytest.raises(ValueError, match="score length"):
            run_falsification(np.zeros(10, dtype=np.float64), dates, ledger)

    def test_non_monotone_dates_rejected(self) -> None:
        ledger = BankingCrisisLedger(events=tuple())
        bad_dates = (date(2010, 1, 1), date(2010, 1, 1))
        with pytest.raises(ValueError, match="strictly increasing"):
            run_falsification(np.zeros(2, dtype=np.float64), bad_dates, ledger)
