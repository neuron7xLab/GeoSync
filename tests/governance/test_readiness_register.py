from __future__ import annotations

import json
from pathlib import Path


def test_readiness_register_is_structured() -> None:
    data = json.loads(Path("governance/readiness_register.json").read_text(encoding="utf-8"))

    assert data["schema_version"] == "readiness.v1"
    assert data["system"] == "GeoSync"
    assert data["current_level"] == "L4-minus"
    assert data["target_direction"] == "L5-candidate"

    entries = data["entries"]
    assert len(entries) >= 5

    seen_ids: set[str] = set()
    for entry in entries:
        assert entry["id"] not in seen_ids
        seen_ids.add(entry["id"])
        assert entry["status"] in {"open", "closed"}
        assert isinstance(entry["summary"], str)
        assert len(entry["summary"]) >= 20
        assert isinstance(entry["required_artifacts"], list)
        assert entry["required_artifacts"]
        if entry["status"] == "closed":
            assert entry.get("evidence_artifacts")


def test_readiness_register_starts_without_promotion() -> None:
    data = json.loads(Path("governance/readiness_register.json").read_text(encoding="utf-8"))

    assert data["current_level"] != data["target_direction"]
    assert any(entry["status"] == "open" for entry in data["entries"])
