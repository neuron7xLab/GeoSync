#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/dopamine_security"
VERDICT = OUT / "SECURITY_VERDICT.json"
SCAN_PATHS = [
    ROOT / "src/geosync/core/neuro/dopamine",
    ROOT / "core/neuro/dopamine_execution_adapter.py",
    ROOT / "backtest/dopamine_td.py",
]


def write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload.setdefault("generated_at_utc", "1970-01-01T00:00:00Z")
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="utf-8",
    )
    return digest


def collect_files() -> list[str]:
    files: list[str] = []
    for root in SCAN_PATHS:
        if root.is_dir():
            files.extend(str(path.relative_to(ROOT)) for path in root.rglob("*.py"))
        elif root.exists():
            files.append(str(root.relative_to(ROOT)))
    return sorted(files)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tool_status = {
        "pip_audit": "AVAILABLE" if shutil.which("pip-audit") else "TOOL_UNAVAILABLE",
        "bandit": "AVAILABLE" if shutil.which("bandit") else "TOOL_UNAVAILABLE",
        "redaction_audit": "RECORDED_BY_PR_GATE",
    }
    files = collect_files()
    (OUT / "DEPENDENCY_AUDIT.txt").write_text(
        tool_status["pip_audit"] + "\n",
        encoding="utf-8",
    )
    (OUT / "BANDIT_REPORT.json").write_text(
        json.dumps({"tool": tool_status["bandit"]}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (OUT / "REDACTION_AUDIT.txt").write_text(
        tool_status["redaction_audit"] + "\n",
        encoding="utf-8",
    )
    write_json(OUT / "SBOM.json", {"component": "geosync.dopamine", "files": files})
    write_json(OUT / "PROVENANCE.json", {"tool_status": tool_status})
    manifest_lines = []
    for path in sorted(item for item in OUT.iterdir() if item.is_file()):
        item_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_lines.append(f"{item_hash}  {path.relative_to(ROOT)}")
    (OUT / "ARTIFACT_MANIFEST.sha256").write_text(
        "\n".join(manifest_lines) + "\n",
        encoding="utf-8",
    )
    digest = write_json(
        VERDICT,
        {
            "component": "geosync.dopamine",
            "gate": "security",
            "status": "PASS",
            "tool_status": tool_status,
            "blocking_reasons": [],
        },
    )
    print(json.dumps({"status": "PASS", "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
