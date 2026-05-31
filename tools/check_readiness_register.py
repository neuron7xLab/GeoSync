#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

path = Path("governance/readiness_register.json")
data = json.loads(path.read_text(encoding="utf-8"))

assert data["schema_version"] == "readiness.v1"
assert data["system"] == "GeoSync"
assert isinstance(data["entries"], list)
assert data["entries"]

ids = set()
for entry in data["entries"]:
    assert isinstance(entry["id"], str)
    assert entry["id"] not in ids
    ids.add(entry["id"])
    assert entry["status"] in {"open", "closed"}
    assert isinstance(entry["summary"], str)
    assert len(entry["summary"]) >= 20
    assert isinstance(entry["required_artifacts"], list)
    assert entry["required_artifacts"]
    if entry["status"] == "closed":
        assert entry.get("evidence_artifacts")

print("readiness register ok")
