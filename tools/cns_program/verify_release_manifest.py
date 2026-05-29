#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

OUT = Path("results/cns_manifest_verification.json")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def verify(path: Path) -> dict[str, Any]:
    errors: list[str] = []
    if not path.exists():
        return {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "manifest": str(path),
            "valid": False,
            "errors": ["manifest_not_found"],
            "checked_artifacts": 0,
            "total_artifacts": 0,
        }

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        data = {}
        errors.append("manifest_not_object")

    sha_map = data.get("sha256")
    if not isinstance(sha_map, dict) or not sha_map:
        sha_map = {}
        errors.append("sha_map_invalid")

    checked = 0
    for raw_path, expected in sha_map.items():
        if not isinstance(raw_path, str):
            errors.append("path_invalid")
            continue
        if not isinstance(expected, str) or not SHA_RE.match(expected):
            errors.append(f"digest_invalid:{raw_path}")
            continue
        target = Path(raw_path)
        if not target.exists():
            errors.append(f"artifact_not_found:{raw_path}")
            continue
        if sha256_file(target) != expected:
            errors.append(f"artifact_changed:{raw_path}")
            continue
        checked += 1

    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "manifest": str(path),
        "valid": not errors,
        "errors": errors,
        "checked_artifacts": checked,
        "total_artifacts": len(sha_map),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="results/cns_release_manifest.json")
    args = parser.parse_args()
    payload = verify(Path(args.manifest))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
