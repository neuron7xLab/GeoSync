"""Stable runtime boundary for BBB-NVU deterministic inference.

This module is intentionally thin: it isolates integration-facing API shape from
CLI/file-system adapters while delegating all risk logic to deterministic_engine.
"""
# fmt: off
# ruff: noqa: I001

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, StrictStr

from BBB_NVU_Cognitive_Noise_Gate_2026.src.deterministic_engine import DeterministicInferenceEngine, load_yaml

OutputProfile = Literal["full", "risk", "actions"]


class RuntimeRequest(BaseModel):
    """Single request accepted by the integration runtime boundary."""

    model_config = ConfigDict(extra="forbid")

    input_doc: dict[str, Any]
    source_id: StrictStr = "memory.json"


class RuntimeBoundary:
    """Integration-facing runtime API with explicit timestamp and output profile."""

    def __init__(self, rules: Mapping[str, Any], engine_hash: str | None = None):
        self.engine = DeterministicInferenceEngine.from_rules(dict(rules), engine_hash=engine_hash)

    @classmethod
    def from_rule_file(cls, rules_path: str, engine_hash: str | None = None) -> "RuntimeBoundary":
        """Build a runtime boundary from a pinned rules file; file I/O stays here."""
        return cls(load_yaml(rules_path), engine_hash=engine_hash)

    def evaluate_run(
        self,
        input_doc: dict[str, Any],
        *,
        source_id: str = "memory.json",
        created_at: str,
        profile: OutputProfile = "full",
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Evaluate one input using an explicit UTC timestamp and output profile."""
        output = self.engine.build_output(input_doc, source_id=source_id, created_at=created_at)
        if profile == "full":
            return output
        if profile == "risk":
            return output["risk"]
        return output["actions"]

    def evaluate_batch(
        self,
        requests: Sequence[RuntimeRequest | Mapping[str, Any]],
        *,
        created_at: str,
        profile: OutputProfile = "full",
    ) -> list[dict[str, Any] | list[dict[str, Any]]]:
        """Evaluate a deterministic batch; each request is validated at the boundary."""
        outputs: list[dict[str, Any] | list[dict[str, Any]]] = []
        for request in requests:
            runtime_request = request if isinstance(request, RuntimeRequest) else RuntimeRequest.model_validate(request)
            outputs.append(
                self.evaluate_run(
                    runtime_request.input_doc,
                    source_id=runtime_request.source_id,
                    created_at=created_at,
                    profile=profile,
                )
            )
        return outputs
