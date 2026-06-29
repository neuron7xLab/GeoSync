#!/usr/bin/env python3
from __future__ import annotations

from scripts.dopamine_contract_core import (
    ARTIFACT,
    CONTRACT,
    DENIED_EDGES_KEY,
    PROMOTION_FLAG_KEY,
    ROOT,
    VALID_STATUSES,
    load_contract,
    main,
    sha256,
    validate_contract,
    validate_required_artifacts,
    write_artifact,
)

__all__ = [
    "ARTIFACT",
    "CONTRACT",
    "DENIED_EDGES_KEY",
    "PROMOTION_FLAG_KEY",
    "ROOT",
    "VALID_STATUSES",
    "load_contract",
    "main",
    "sha256",
    "validate_contract",
    "validate_required_artifacts",
    "write_artifact",
]


if __name__ == "__main__":
    raise SystemExit(main())
