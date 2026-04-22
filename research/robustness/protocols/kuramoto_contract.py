# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Kuramoto frozen-artifact contract.

Loads the cross-asset Kuramoto evidence bundle, verifies each file's
SHA-256 against ``results/cross_asset_kuramoto/offline_robustness/SOURCE_HASHES.json``,
and exposes typed views on the demo equity curve, fold metrics, and
parameter lock.

Fail-closed: any missing file or hash mismatch raises
:class:`FrozenArtifactMismatch` and the gate runner records a
``FAIL`` verdict on the hash-integrity axis.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST: Final[Path] = (
    REPO_ROOT / "results" / "cross_asset_kuramoto" / "offline_robustness" / "SOURCE_HASHES.json"
)
DEMO_EQUITY_REL: Final[str] = "results/cross_asset_kuramoto/demo/equity_curve.csv"
DEMO_FOLDS_REL: Final[str] = "results/cross_asset_kuramoto/demo/fold_metrics.csv"
DEMO_RISK_REL: Final[str] = "results/cross_asset_kuramoto/demo/risk_metrics.csv"
PARAM_LOCK_REL: Final[str] = "results/cross_asset_kuramoto/PARAMETER_LOCK.json"


class FrozenArtifactMismatch(RuntimeError):
    """Raised when a frozen artifact fails sha256 verification or is missing."""


@dataclass(frozen=True)
class FrozenArtifactManifest:
    """Parsed SOURCE_HASHES.json plus the repo root used to resolve rels."""

    generated_utc: str
    regenerated_utc: str | None
    hashes: dict[str, str]
    repo_root: Path

    @classmethod
    def load(
        cls,
        manifest_path: Path = DEFAULT_MANIFEST,
        repo_root: Path = REPO_ROOT,
    ) -> FrozenArtifactManifest:
        if not manifest_path.is_file():
            raise FrozenArtifactMismatch(f"frozen-artifact manifest missing: {manifest_path}")
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return cls(
            generated_utc=str(data["generated_utc"]),
            regenerated_utc=data.get("regenerated_utc"),
            hashes=dict(data["hashes"]),
            repo_root=repo_root,
        )

    def verify_all(self) -> None:
        """Raise :class:`FrozenArtifactMismatch` on any missing/mismatched file."""
        mismatches: list[str] = []
        missing: list[str] = []
        for rel, expected in self.hashes.items():
            path = self.repo_root / rel
            if not path.is_file():
                missing.append(rel)
                continue
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            if sha != expected:
                mismatches.append(f"{rel}: got {sha}, expected {expected}")
        if missing or mismatches:
            parts: list[str] = []
            if missing:
                parts.append("missing:\n  " + "\n  ".join(missing))
            if mismatches:
                parts.append("sha256 mismatch:\n  " + "\n  ".join(mismatches))
            raise FrozenArtifactMismatch("\n".join(parts))


@dataclass(frozen=True)
class KuramotoRobustnessContract:
    """Typed view on the frozen evidence bundle for robustness suites."""

    manifest: FrozenArtifactManifest
    equity_curve: pd.DataFrame
    fold_metrics: pd.DataFrame
    risk_metrics: pd.DataFrame
    parameter_lock: dict[str, object]

    @classmethod
    def from_frozen_artifacts(
        cls,
        manifest_path: Path = DEFAULT_MANIFEST,
        repo_root: Path = REPO_ROOT,
    ) -> KuramotoRobustnessContract:
        """Load contract and verify hashes in one fail-closed step."""
        manifest = FrozenArtifactManifest.load(manifest_path, repo_root)
        manifest.verify_all()

        equity = pd.read_csv(
            repo_root / DEMO_EQUITY_REL,
            parse_dates=["date"],
        )
        folds = pd.read_csv(
            repo_root / DEMO_FOLDS_REL,
            parse_dates=["is_start", "os_start", "os_end"],
        )
        risk = pd.read_csv(repo_root / DEMO_RISK_REL)
        param_lock = json.loads((repo_root / PARAM_LOCK_REL).read_text(encoding="utf-8"))
        contract = cls(
            manifest=manifest,
            equity_curve=equity,
            fold_metrics=folds,
            risk_metrics=risk,
            parameter_lock=param_lock,
        )
        contract.assert_frozen_consistency()
        return contract

    def assert_frozen_consistency(self) -> None:
        """Sanity checks that panic on drift inside the loaded frames."""
        required_equity = {"date", "strategy_cumret", "benchmark_cumret", "drawdown"}
        missing = required_equity - set(self.equity_curve.columns)
        if missing:
            raise FrozenArtifactMismatch(f"equity_curve.csv missing columns: {sorted(missing)}")
        required_folds = {"fold_id", "sharpe", "max_dd", "pass_fail"}
        missing = required_folds - set(self.fold_metrics.columns)
        if missing:
            raise FrozenArtifactMismatch(f"fold_metrics.csv missing columns: {sorted(missing)}")
        if len(self.fold_metrics) < 2:
            raise FrozenArtifactMismatch(
                f"fold_metrics.csv needs at least 2 folds, has {len(self.fold_metrics)}"
            )

    def daily_strategy_returns(self) -> pd.Series:
        """Strategy daily returns from ``strategy_cumret`` (pct_change)."""
        s = self.equity_curve["strategy_cumret"].astype(float).pct_change().dropna()
        s.index = self.equity_curve["date"].iloc[1:].to_numpy()
        s.name = "strategy_ret"
        return s

    def daily_benchmark_returns(self) -> pd.Series:
        """Benchmark daily returns from ``benchmark_cumret``."""
        s = self.equity_curve["benchmark_cumret"].astype(float).pct_change().dropna()
        s.index = self.equity_curve["date"].iloc[1:].to_numpy()
        s.name = "benchmark_ret"
        return s
