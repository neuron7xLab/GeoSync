# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Regression tests for the contract-manifest hash gate."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.ci import check_manifest_hashes as gate


def _write_manifest(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_manifest_hash_gate_skips_evidence_freshness_manifests(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        {
            "schema": "evidence_artifact_freshness.v1",
            "artifacts": [
                {
                    "path": "missing.json",
                    "sha256": "0" * 64,
                }
            ],
        },
    )
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "_discover_manifest_paths", lambda: [manifest])

    assert gate.main([]) == 0


def test_manifest_hash_gate_rejects_missing_contract_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        {
            "schema": "contract_manifest.v1",
            "artifacts": [
                {
                    "path": "missing.json",
                    "sha256": "0" * 64,
                }
            ],
        },
    )
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "_discover_manifest_paths", lambda: [manifest])

    assert gate.main([]) == 1
