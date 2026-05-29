#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

DEPLOY_GATE_RESULT = Path("results/cns_deploy_gate.json")
RELEASE_MANIFEST_RESULT = Path("results/cns_release_manifest.json")
MANIFEST_FILES = (
    Path(".github/workflows/cns-release-gate.yml"),
    Path("conf/cns_program/default.yaml"),
    Path("conf/cns_program/invariants.yaml"),
    Path("reports/CNS_RELEASE_GATE_CONTRACT.md"),
    Path("tools/cns_program/cns_contract.py"),
    Path("tools/cns_program/deploy_release_gate.py"),
    Path("tools/cns_program/build_release_manifest.py"),
    Path("tools/cns_program/verify_release_manifest.py"),
    Path("tools/cns_program/verify_reports_contract.py"),
    Path("tools/cns_program/quality_gate.py"),
    Path("tools/cns_program/simple_yaml.py"),
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    missing = [str(path) for path in MANIFEST_FILES if not path.exists()]
    if missing:
        print(json.dumps({"valid": False, "missing": missing}))
        return 1
    if not DEPLOY_GATE_RESULT.exists():
        print(json.dumps({"valid": False, "missing": [str(DEPLOY_GATE_RESULT)]}))
        return 1
    manifest = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "deploy_gate": json.loads(DEPLOY_GATE_RESULT.read_text(encoding="utf-8")),
        "sha256": {str(path): sha256_file(path) for path in MANIFEST_FILES},
    }
    RELEASE_MANIFEST_RESULT.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {"manifest": str(RELEASE_MANIFEST_RESULT), "files_hashed": len(MANIFEST_FILES)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
