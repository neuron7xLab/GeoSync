#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

path = Path("governance/readiness_register.json")
data = json.loads(path.read_text(encoding="utf-8"))
sha256_re = re.compile(r"^[0-9a-f]{64}$")

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

    records = entry.get("evidence_artifacts")
    if entry["status"] == "closed":
        assert isinstance(records, list)
        assert records
    if records is not None:
        assert isinstance(records, list)
        for record in records:
            assert isinstance(record, dict)
            assert isinstance(record.get("kind"), str)
            assert record["kind"].strip()
            assert isinstance(record.get("path"), str)
            assert record["path"].strip()
            assert isinstance(record.get("verification_command"), str)
            assert record["verification_command"].strip()
            assert isinstance(record.get("sha256"), str)
            assert sha256_re.fullmatch(record["sha256"])

print("readiness register ok")
