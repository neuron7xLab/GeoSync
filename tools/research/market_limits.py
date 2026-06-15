# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Loader and well-formedness validator for ``physics_contracts/market_limits.yaml``.

The market-limits contract states the boundary under which GeoSync's geometric
and dynamical descriptors (Kuramoto, Ricci, topology, thermo) may be interpreted
as statements about real markets at all. This module loads that contract and
exposes a fail-closed validator so a witness test can assert the contract stays
well-formed and aligned with the canonical claim-tier vocabulary.

It asserts nothing new about markets; it only makes the limits machine-checkable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - plumbed by env, not tests
    raise RuntimeError("market_limits requires PyYAML. Install via `pip install pyyaml`.") from exc

_CONTRACT_PATH = Path(__file__).resolve().parents[2] / "physics_contracts" / "market_limits.yaml"

REQUIRED_FIELDS: tuple[str, ...] = (
    "limit_id",
    "mathematical_object",
    "market_interpretation",
    "valid_when",
    "invalid_when",
    "witness_required",
    "falsifier_required",
    "forbidden_claims",
    "max_claim_tier_violated",
)

# The four pre-registered nulls a falsifier_required entry must name. Mirrors
# ``tools/research/ricci_preregistration_guard.REQUIRED_NULLS`` so the contract
# cannot drift to a falsifier that the prereg pipeline does not run.
KNOWN_FALSIFIERS: frozenset[str] = frozenset(
    {"permutation_null", "lag_sweep_no_future_data", "cost_model", "multi_session_replay"}
)


@dataclass(frozen=True)
class MarketLimit:
    """A single market approximation limit drawn from the contract."""

    limit_id: str
    mathematical_object: str
    market_interpretation: str
    valid_when: tuple[str, ...]
    invalid_when: tuple[str, ...]
    witness_required: str
    falsifier_required: str
    forbidden_claims: tuple[str, ...]
    max_claim_tier_violated: str


@dataclass(frozen=True)
class MarketLimitsContract:
    """The loaded contract: claim-tier vocabulary plus the limit entries."""

    version: int
    claim_tiers: tuple[str, ...]
    limits: tuple[MarketLimit, ...]

    def by_id(self, limit_id: str) -> MarketLimit | None:
        for limit in self.limits:
            if limit.limit_id == limit_id:
                return limit
        return None


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"expected a list[str], got {value!r}")
    return tuple(value)


def load_contract(path: Path | None = None) -> MarketLimitsContract:
    """Load and parse the market-limits contract, fail-closed on shape errors."""
    contract_path = path or _CONTRACT_PATH
    raw: Any = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("market_limits.yaml must parse to a mapping")

    version = raw.get("version")
    if not isinstance(version, int):
        raise ValueError("market_limits.yaml requires an integer `version`")

    claim_tiers = _as_str_tuple(raw.get("claim_tiers"))
    raw_limits = raw.get("limits")
    if not isinstance(raw_limits, list) or not raw_limits:
        raise ValueError("market_limits.yaml requires a non-empty `limits` list")

    limits: list[MarketLimit] = []
    for entry in raw_limits:
        if not isinstance(entry, dict):
            raise ValueError(f"each limit must be a mapping, got {entry!r}")
        missing = [field for field in REQUIRED_FIELDS if field not in entry]
        if missing:
            raise ValueError(
                f"limit {entry.get('limit_id', '<unknown>')!r} missing fields: {missing}"
            )
        limits.append(
            MarketLimit(
                limit_id=str(entry["limit_id"]),
                mathematical_object=str(entry["mathematical_object"]),
                market_interpretation=str(entry["market_interpretation"]),
                valid_when=_as_str_tuple(entry["valid_when"]),
                invalid_when=_as_str_tuple(entry["invalid_when"]),
                witness_required=str(entry["witness_required"]),
                falsifier_required=str(entry["falsifier_required"]),
                forbidden_claims=_as_str_tuple(entry["forbidden_claims"]),
                max_claim_tier_violated=str(entry["max_claim_tier_violated"]),
            )
        )

    return MarketLimitsContract(
        version=version,
        claim_tiers=claim_tiers,
        limits=tuple(limits),
    )
