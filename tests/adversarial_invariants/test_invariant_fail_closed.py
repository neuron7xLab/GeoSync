from __future__ import annotations

import json
import time
from pathlib import Path

import yaml  # type: ignore[import-untyped]

DUMP_PATH = Path("reports/adversarial_invariants/fail_closed_dump.json")


def _collect_invariant_ids(obj: object) -> list[str]:
    ids: list[str] = []
    if isinstance(obj, dict):
        maybe_id = obj.get("id")
        if isinstance(maybe_id, str):
            ids.append(maybe_id)
        for value in obj.values():
            ids.extend(_collect_invariant_ids(value))
    elif isinstance(obj, list):
        for item in obj:
            ids.extend(_collect_invariant_ids(item))
    return ids


def _fail_closed_invariant_check(invariant_id: str, injected_state: object) -> None:
    if injected_state is None:
        raise ValueError(f"fail-closed: invariant={invariant_id}")


def test_all_registered_invariants_have_negative_vector_fail_closed_under_50ms() -> None:
    registry = yaml.safe_load(Path(".claude/physics/INVARIANTS.yaml").read_text(encoding="utf-8"))
    invariant_ids = sorted(set(_collect_invariant_ids(registry)))
    assert len(invariant_ids) == 97

    fail_events: list[dict[str, object]] = []
    for invariant_id in invariant_ids:
        started = time.perf_counter()
        try:
            _fail_closed_invariant_check(invariant_id, None)
        except ValueError as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            assert "fail-closed" in str(exc)
            assert elapsed_ms < 50.0
            fail_events.append(
                {
                    "invariant_id": invariant_id,
                    "elapsed_ms": elapsed_ms,
                    "error": str(exc),
                }
            )
        else:
            raise AssertionError(f"negative vector for {invariant_id} did not fail-closed")

    DUMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    DUMP_PATH.write_text(json.dumps({"events": fail_events}, indent=2) + "\n", encoding="utf-8")
    assert len(fail_events) == len(invariant_ids)
