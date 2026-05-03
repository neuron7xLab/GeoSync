# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Cognitive bridge between GeoSync runtime and an external LLM agent host.

This package wires GeoSync to a process-isolated cognitive sidecar (default
target: local OpenClaw gateway at loopback) under the gradient ontology
maintenance/processing split:

* The sidecar is an **advisory** Layer-2 primitive. It never overrides the
  deterministic agent state machine; outputs are SPECULATIVE-tier per the
  Inference Discipline Protocol v1.0 and must be reconciled by the caller.
* Transport is fail-closed. A timeout, schema violation, kill-switch, or
  stressed-state condition collapses the bridge into ADVISORY_UNAVAILABLE
  with no side effect on the host system.
* Protocol payloads are versioned and content-addressable (sha256 over the
  canonical JSON serialization) so that audit logs can prove provenance.

Public surface:

    - protocol.AdvisoryRequest, AdvisoryResponse, AdvisoryStatus
    - transport.Transport, InMemoryTransport, LoopbackHttpTransport
    - sidecar.CognitiveSidecar, SidecarConfig
    - invariants.CB_INVARIANTS, BridgeInvariantError
"""

from runtime.cognitive_bridge.errors import (
    BridgeInvariantError,
    BridgeTimeoutError,
    BridgeTransportError,
)
from runtime.cognitive_bridge.invariants import CB_INVARIANTS, InvariantId
from runtime.cognitive_bridge.protocol import (
    PROTOCOL_VERSION,
    AdvisoryRequest,
    AdvisoryResponse,
    AdvisoryStatus,
    EvidenceTier,
)
from runtime.cognitive_bridge.sidecar import CognitiveSidecar, SidecarConfig
from runtime.cognitive_bridge.transport import (
    InMemoryTransport,
    LoopbackHttpTransport,
    Transport,
)

__all__ = [
    "AdvisoryRequest",
    "AdvisoryResponse",
    "AdvisoryStatus",
    "BridgeInvariantError",
    "BridgeTimeoutError",
    "BridgeTransportError",
    "CB_INVARIANTS",
    "CognitiveSidecar",
    "EvidenceTier",
    "InMemoryTransport",
    "InvariantId",
    "LoopbackHttpTransport",
    "PROTOCOL_VERSION",
    "SidecarConfig",
    "Transport",
]
