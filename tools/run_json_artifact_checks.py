#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

FIXTURES = ["candidate", "blocked"]


def run(cmd: list[str]) -> dict[str, object]:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {"cmd": cmd, "code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/validation/json_artifact_checks.json")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for name in FIXTURES:
        fixture = f"examples/json_artifact_contract.{name}.json"
        results.append(run([sys.executable, "tools/validate_json_artifact_contract.py", fixture]))
        results.append(run([sys.executable, "tools/check_json_contract_evidence_policy.py", fixture]))
    receipt_cmd = [sys.executable, "tools/json_contract_receipt.py", "--out", "artifacts/validation/json_contract_receipt.json"]
    results.append(run(receipt_cmd))
    status = "OK" if all(item["code"] == 0 for item in results) else "ERROR"
    out.write_text(json.dumps({"schema_version": "json_artifact_checks.v1", "status": status, "results": results}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if status == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
