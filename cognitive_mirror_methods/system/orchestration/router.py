from __future__ import annotations

from typing import Any

from cognitive_mirror_methods.system.orchestration.registry import module_exists

STATUS_BLOCKED = "BLOCKED"
STATUS_PASS = "PASS"


def _response(
    request: dict[str, Any],
    status: str,
    output: dict[str, Any],
    errors: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "request_id": str(request.get("request_id", "")),
        "module": str(request.get("module", "")),
        "status": status,
        "output": output,
        "artifacts": [],
        "errors": errors or [],
        "warnings": [],
        "validation": {
            "schema_valid": bool(request.get("request_id") and request.get("module")),
            "contract_valid": status == STATUS_PASS,
            "tests_passed": [],
        },
        "next_action": "inspect output" if status == STATUS_PASS else "repair request",
    }


def route_request(request: dict[str, Any]) -> dict[str, Any]:
    module_id = str(request.get("module", ""))
    if not module_exists(module_id):
        return _response(request, STATUS_BLOCKED, {}, ["unknown module"])
    text = str(request.get("input", {}).get("text", ""))
    return _response(
        request,
        STATUS_PASS,
        {"text": " ".join(text.split()), "module": module_id},
    )
