# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""The deterministic evidence-artifact freshness gate must have teeth (P0-7).

Proves the committed deterministic evidence artifacts equal their regenerated
output, and that a hand-edit (the exact failure mode the gate exists to catch)
is detected. The negative test restores the original bytes in ``finally`` so it
never leaves the tree dirty.
"""

from __future__ import annotations

from scripts.ci.check_artifact_freshness import (
    DETERMINISTIC_ARTIFACTS,
    ROOT,
    build_manifest,
    check,
    regenerated_sha,
)


def test_committed_artifacts_are_fresh() -> None:
    """All declared deterministic artifacts match their regenerated output."""
    assert check() == []


def test_manifest_is_consistent_with_artifacts() -> None:
    """artifacts/.manifest.json equals the freshly-built manifest (no drift)."""
    import json

    committed = json.loads((ROOT / "artifacts" / ".manifest.json").read_text(encoding="utf-8"))
    assert committed == build_manifest()


def test_hand_edit_is_detected() -> None:
    """A tampered artifact is flagged STALE (the gate's whole reason to exist)."""
    entry = DETERMINISTIC_ARTIFACTS[0]
    path = ROOT / str(entry["artifact"])
    original = path.read_bytes()
    try:
        path.write_bytes(original + b'\n{"hand_edited": true}\n')  # corrupt
        committed_sha, regen_sha = regenerated_sha(entry)  # restores to corrupted bytes
        assert committed_sha != regen_sha, "tampered artifact must differ from regeneration"
    finally:
        path.write_bytes(original)  # leave the tree clean
    # And the original is genuinely fresh again.
    again_committed, again_regen = regenerated_sha(entry)
    assert again_committed == again_regen
