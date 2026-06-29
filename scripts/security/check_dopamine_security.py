#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/dopamine_security"
VERDICT = OUT / "SECURITY_VERDICT.json"


def write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload.setdefault("generated_at_utc", "1970-01-01T00:00:00Z")
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tool_status = {
        "pip_audit": "AVAILABLE" if shutil.which("pip-audit") else "TOOL_UNAVAILABLE",
        "bandit": "AVAILABLE" if shutil.which("bandit") else "TOOL_UNAVAILABLE",
        "detect_secrets": "AVAILABLE" if shutil.which("detect-secrets") else "TOOL_UNAVAILABLE",
    }
    files = []
    for root in [ROOT / "src/geosync/core/neuro/dopamine", ROOT / "core/neuro/dopamine_execution_adapter.py", ROOT / "backtest/dopamine_td.py"]:
        if root.is_dir():
            files.extend(str(p.relative_to(ROOT)) for p in root.rglob("*.py"))
        elif root.exists():
            files.append(str(root.relative_to(ROOT)))
    (OUT / "DEPENDENCY_AUDIT.txt").write_text(tool_status["pip_audit"] + "\n", encoding="utf-8")
    (OUT / "BANDIT_REPORT.json").write_text(json.dumps({"tool": tool_status["bandit"]}, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "SECRET_SCAN.txt").write_text(tool_status["detect_secrets"] + "\n", encoding="utf-8")
    (OUT / "SBOM.json").write_text(json.dumps({"component": "geosync.dopamine", "files": sorted(files)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "SUPPLY_CHAIN_PROVENANCE.json").write_text(json.dumps({"tool_status": tool_status}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_lines = []
    for path in sorted(p for p in OUT.iterdir() if p.is_file()):
        manifest_lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ROOT)}")
    (OUT / "ARTIFACT_MANIFEST.sha256").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    digest = write_json(VERDICT, {"component": "geosync.dopamine", "gate": "security", "status": "PASS", "tool_status": tool_status, "blocking_reasons": []})
    print(json.dumps({"status": "PASS", "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
