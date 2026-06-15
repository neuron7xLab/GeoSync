from __future__ import annotations

from typing import Any

from cognitive_mirror_methods.system.orchestration.router import route_request


def execute(request: dict[str, Any]) -> dict[str, Any]:
    """Run one module through the public system boundary."""
    return route_request(request)
