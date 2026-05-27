from __future__ import annotations

import hashlib
import json
from pathlib import Path

CLAIMS = Path("CLAIMS.md")
HASHES = Path("artifacts/claims_hashes.json")


def claim_lines(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for ln in text.splitlines():
        if ln.startswith("| C-"):
            cid = ln.split("|")[1].strip()
            if cid in out:
                raise ValueError(f"duplicate claim id: {cid}")
            out[cid] = hashlib.sha256(ln.strip().encode("utf-8")).hexdigest()
    return out


def main() -> int:
    if not HASHES.exists():
        print(
            "CLAIM HASH VIOLATION: artifacts/claims_hashes.json missing. "
            "Run generate_claim_hashes.py"
        )
        return 1
    recorded = json.loads(HASHES.read_text(encoding="utf-8")).get("hashes", {})

    try:
        current = claim_lines(CLAIMS.read_text(encoding="utf-8"))
    except ValueError as exc:
        print(f"CLAIM HASH VIOLATION: {exc}")
        return 1
    if recorded != current:
        print("CLAIM HASH VIOLATION: claim-line hashes diverged")
        missing = sorted(set(recorded) ^ set(current))
        if missing:
            print(f" id-set mismatch: {missing}")
        for k, v in current.items():
            if recorded.get(k) != v:
                print(f" drift: {k}")
        return 1
    print("OK: claim hashes verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
