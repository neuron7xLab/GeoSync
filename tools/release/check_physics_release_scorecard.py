# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Physics release scorecard validator and readiness gate.

The scorecard (``physics_release_scorecard.yml``) records one discrete readiness
state per release-critical dimension. This tool enforces three properties:

1. **Schema** — every dimension declares an ``id``, ``required`` flag, and a
   ``state`` drawn from the allowed ladder
   (FALSE < UNTESTED < PARTIAL < LOCAL_VERIFIED < CI_VERIFIED < EVIDENCE_BEARING).
2. **Anti-theater** — a dimension may only claim ``CI_VERIFIED`` or
   ``EVIDENCE_BEARING`` if its ``binding`` artifact actually exists on disk. A
   verified state with a missing/empty binding is a hard error: a green label
   cannot float free of the gate that earns it.
3. **Readiness gate** (``--assert-physics-ready``) — a release is physics-ready
   only when *every* ``required`` dimension is ``CI_VERIFIED`` or
   ``EVIDENCE_BEARING``. A ``PARTIAL`` required dimension cannot be upgraded by
   assertion; the gate fails closed and names the blockers.

Default invocation validates the scorecard (schema + anti-theater) and exits 0
even when the repository is not yet physics-ready — the honest PARTIAL snapshot
is a valid scorecard. ``--assert-physics-ready`` is the release-time gate.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCORECARD = ROOT / "physics_release_scorecard.yml"

#: Ordered readiness ladder; index encodes monotonic strength.
LADDER: tuple[str, ...] = (
    "FALSE",
    "UNTESTED",
    "PARTIAL",
    "LOCAL_VERIFIED",
    "CI_VERIFIED",
    "EVIDENCE_BEARING",
)
#: States that count as "earned" — they require an existing binding artifact and
#: satisfy the physics-ready gate.
VERIFIED_STATES = frozenset({"CI_VERIFIED", "EVIDENCE_BEARING"})


@dataclass(frozen=True)
class Dimension:
    """A single release-critical dimension and its discrete readiness state."""

    id: str
    required: bool
    state: str
    binding: str | None
    evidence_command: str | None
    note: str


class ScorecardError(ValueError):
    """Raised when the scorecard is malformed or makes an unbacked claim."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScorecardError(message)


def load_scorecard(path: Path) -> list[Dimension]:
    """Parse and schema-validate the scorecard at ``path``.

    Raises :class:`ScorecardError` on any malformed entry. Anti-theater binding
    existence is checked separately by :func:`check_bindings` so that callers can
    distinguish a schema fault from a missing artifact.
    """
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(raw, dict), "scorecard root must be a mapping")
    allowed = raw.get("allowed_states")
    _require(
        isinstance(allowed, list) and list(allowed) == list(LADDER),
        f"allowed_states must equal the canonical ladder {list(LADDER)}",
    )
    dims_raw = raw.get("dimensions")
    _require(
        isinstance(dims_raw, list) and len(dims_raw) > 0, "dimensions must be a non-empty list"
    )

    dimensions: list[Dimension] = []
    seen: set[str] = set()
    for entry in dims_raw:
        _require(isinstance(entry, dict), "each dimension must be a mapping")
        dim_id = entry.get("id")
        _require(
            isinstance(dim_id, str) and bool(dim_id), "dimension 'id' must be a non-empty string"
        )
        _require(dim_id not in seen, f"duplicate dimension id: {dim_id!r}")
        seen.add(dim_id)
        required = entry.get("required")
        _require(isinstance(required, bool), f"{dim_id}: 'required' must be a bool")
        state = entry.get("state")
        _require(
            isinstance(state, str) and state in LADDER,
            f"{dim_id}: 'state' must be one of {list(LADDER)}; got {state!r}",
        )
        binding = entry.get("binding")
        _require(
            binding is None or isinstance(binding, str),
            f"{dim_id}: 'binding' must be a string or null",
        )
        command = entry.get("evidence_command")
        _require(
            command is None or isinstance(command, str),
            f"{dim_id}: 'evidence_command' must be a string or null",
        )
        note = entry.get("note", "")
        _require(isinstance(note, str), f"{dim_id}: 'note' must be a string")
        dimensions.append(
            Dimension(
                id=dim_id,
                required=required,
                state=state,
                binding=binding,
                evidence_command=command,
                note=note,
            )
        )
    return dimensions


def check_bindings(dimensions: list[Dimension], root: Path) -> list[str]:
    """Return anti-theater violations: verified states whose binding is absent."""
    violations: list[str] = []
    for dim in dimensions:
        if dim.state not in VERIFIED_STATES:
            continue
        if not dim.binding:
            violations.append(
                f"{dim.id}: state {dim.state} requires a non-empty 'binding' artifact"
            )
            continue
        if not (root / dim.binding).exists():
            violations.append(f"{dim.id}: state {dim.state} binds missing artifact {dim.binding!r}")
    return violations


def readiness_blockers(dimensions: list[Dimension]) -> list[str]:
    """Required dimensions not yet CI_VERIFIED/EVIDENCE_BEARING (physics-ready blockers)."""
    return [
        f"{dim.id} is {dim.state} (required >= CI_VERIFIED)"
        for dim in dimensions
        if dim.required and dim.state not in VERIFIED_STATES
    ]


def summarize(dimensions: list[Dimension]) -> dict[str, Any]:
    blockers = readiness_blockers(dimensions)
    return {
        "physics_ready": not blockers,
        "blockers": blockers,
        "dimensions": {dim.id: dim.state for dim in dimensions},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scorecard", type=Path, default=SCORECARD, help="Path to the scorecard YAML."
    )
    parser.add_argument(
        "--assert-physics-ready",
        action="store_true",
        help="Exit 1 unless every required dimension is CI_VERIFIED or EVIDENCE_BEARING.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the summary as JSON.")
    args = parser.parse_args(argv)

    try:
        dimensions = load_scorecard(args.scorecard)
    except ScorecardError as exc:
        print(f"[ERROR] malformed scorecard: {exc}")
        return 1

    binding_violations = check_bindings(dimensions, args.scorecard.resolve().parent)
    summary = summarize(dimensions)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        for dim in dimensions:
            flag = "*" if dim.required else " "
            print(f"[{dim.state:<16}] {flag} {dim.id}")
        print(f"\nphysics_ready: {summary['physics_ready']}")

    if binding_violations:
        for v in binding_violations:
            print(f"[ERROR] anti-theater: {v}")
        return 1

    if args.assert_physics_ready and summary["blockers"]:
        print("\n[BLOCKED] release is NOT physics-ready:")
        for b in summary["blockers"]:
            print(f"  - {b}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
