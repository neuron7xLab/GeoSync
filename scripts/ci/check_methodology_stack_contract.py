from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs" / "governance" / "methodology_stack_contract.md"

REQUIRED_TOKENS = (
    "adaptive verified operation",
    "observability-first instrumentation",
    "alternative-model elimination",
    "phenomenon-to-engineering transfer",
    "target lock",
    "observation window",
    "alternative models",
    "discriminating test",
    "controlled parameters",
    "negative evidence path",
    "claim boundary",
    "Pending CI blocks merge interpretation.",
)


def validate_contract_text(text: str) -> list[str]:
    missing = [token for token in REQUIRED_TOKENS if token not in text]
    if "No claim may skip levels." not in text:
        missing.append("No claim may skip levels.")
    return missing


def main() -> int:
    text = CONTRACT.read_text(encoding="utf-8")
    missing = validate_contract_text(text)
    if missing:
        for token in missing:
            print(f"METHODOLOGY_STACK_CONTRACT_MISSING: {token}")
        return 1
    print("METHODOLOGY_STACK_CONTRACT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
