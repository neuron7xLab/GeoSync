from __future__ import annotations

from typing import Any

from cognitive_mirror_methods.system.orchestration.registry import ordered_modules
from cognitive_mirror_methods.system.orchestration.router import route_request


HALT_STATES = {"BLOCKED", "FAILED"}


def run_pipeline(request: dict[str, Any]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    text = str(request.get("input", {}).get("text", ""))
    for module_id in ordered_modules():
        step_request = dict(request)
        step_request["module"] = module_id
        step_request["input"] = {"text": text}
        result = route_request(step_request)
        results.append(result)
        if result["status"] in HALT_STATES:
            return {
                "request_id": str(request.get("request_id", "")),
                "module": "pipeline",
                "status": result["status"],
                "output": {"results": results},
                "artifacts": [],
                "errors": result["errors"],
                "warnings": [],
                "validation": result["validation"],
                "next_action": "repair pipeline input",
            }
        text = str(result["output"].get("text", text))
    return {
        "request_id": str(request.get("request_id", "")),
        "module": "pipeline",
        "status": "PASS",
        "output": {"results": results},
        "artifacts": [],
        "errors": [],
        "warnings": [],
        "validation": {"schema_valid": True, "contract_valid": True, "tests_passed": []},
        "next_action": "create artifact",
    }
