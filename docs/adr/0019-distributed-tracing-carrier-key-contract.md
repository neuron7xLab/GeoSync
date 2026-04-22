# ADR-0019: Distributed-tracing carrier-key contract

## Status
Accepted

**Date:** 2026-04-22

**Decision makers:** Yaroslav Vasylenko (repo owner); Claude Code (executor)

## Context

`core/tracing/distributed.py` implements OpenTelemetry-compatible inject
and extract helpers for correlation IDs and baggage. The original code
called `existing_key.lower()` on carrier keys and `str(value)` on carrier
values without restricting key types. In practice upstream carriers
include:

* ASGI/WSGI scope headers — typically `bytes` keys
* gRPC metadata — often `bytes`/`bytearray` keys
* OpenTelemetry test harnesses — sometimes non-string keys
* legacy WSGI middleware — mixed `str` and `bytes`

Three concrete defects surfaced:

1. Non-`str` keys cause `AttributeError` on `.lower()` in the Python
   `str` sense, or match spuriously via implicit `str(key)` coercion.
2. `b"..."` values rendered through `str(value)` emit the literal
   Python repr (`"b'payload'"`), leaking the byte prefix on the wire.
3. CR/LF/NUL inside correlation IDs or baggage values is silently
   injected and can be smuggled through a non-strict HTTP/1.1 proxy
   to split a single header into two wire headers.

## Decision

Extraction is **read-tolerant**; injection is **write-canonical**.

Read path:
* supported key types: `str`, `bytes`, `bytearray`
* keys are ASCII-decoded when byte-shaped, lower-cased, then validated
  against the RFC 7230 § 3.2.6 *token* grammar via `_HEADER_TOKEN_RE`;
  keys that fail the grammar are silently ignored
* values are normalised via `_normalize_header_value`: `str` passes
  through, bytes decode via latin-1 (the canonical HTTP surrogate),
  other types fall back to `str()`

Write path:
* only `str` keys and `str` values are emitted
* every outgoing value passes through `_reject_crlf`, which raises
  `ValueError` on CR/LF/NUL

Type signatures on extraction helpers widen from `Mapping[str, str]`
to `Mapping[Any, Any]` to reflect the real carrier contract; setters
retain `MutableMapping[str, str]`.

## Consequences

### Positive
- Accepts bytes-shaped carriers from ASGI/gRPC without crashing.
- Never emits `b'...'` repr strings on the wire.
- Header-splitting via CR/LF/NUL is blocked at inject time.
- Silent `str(key)` coercion of non-header-like objects is gone —
  integer / tuple / arbitrary-object keys are ignored cleanly.

### Negative
- Marginal CPU cost: one regex `fullmatch` per key comparison. Amortises
  to sub-μs per header in typical propagation loads.

### Neutral
- Public API unchanged. Existing callers with `Mapping[str, str]`
  carriers continue to work identically.
- Read-tolerant signature means mypy cannot statically prevent a
  carrier with nonsense keys; that trade-off is deliberate.

## Alternatives Considered

### Alternative 1: Narrow the signatures to `Mapping[str, str]`
**Pros:** strongest static type guarantee.
**Cons:** bytes carriers from WSGI/ASGI cannot be passed without a
manual copy at every call site — a footgun that led to the original
bug.
**Reason for rejection:** the real-world carrier shape is not
`Mapping[str, str]`; pretending it is leaves the runtime defenceless.

### Alternative 2: Percent-encode CR/LF instead of rejecting
**Pros:** preserves the (malformed) value round-trip.
**Cons:** a correlation ID or baggage value carrying CR/LF is almost
certainly an attack or a bug; silently mutating it masks the signal.
**Reason for rejection:** fail-closed is safer for a security
primitive.

## Implementation

### Required changes
- Add `_HEADER_TOKEN_RE`, `_FORBIDDEN_HEADER_VALUE_CHARS`,
  `_normalize_header_key`, `_header_key_matches`,
  `_normalize_header_value`, `_current_local_baggage`, `_reject_crlf`
  helpers in `core/tracing/distributed.py`.
- Update `_DictGetter.get`, `_first_correlation_value`,
  `_extract_local_baggage`, `extract_distributed_context` to accept
  `Mapping[Any, Any]` and dispatch through the helpers.
- Update `_DictSetter.set`, `inject_distributed_context`,
  `_inject_local_baggage` to run values through `_reject_crlf`.
- `_LOCAL_BAGGAGE` default changes from `{}` to `None` (via
  `_current_local_baggage`) so no global mutable default dict leaks.

### Validation Criteria
- `tests/unit/tracing/test_distributed.py`: 129 tests pass (62 new
  around carrier contract, CRLF hardening, Hypothesis properties,
  and inject→extract roundtrip).
- Synthetic bytes-keyed carrier roundtrips cleanly.
- CRLF in correlation/baggage values raises `ValueError`.
- No existing test regressed.

## Related Decisions
- ADR-0010: Observability unified telemetry fabric (parent context
  for this module's public surface).

## References
- RFC 7230 § 3.2.6 (token grammar), § 3.2 (header field value syntax).
- OWASP header-injection guidance on CR/LF smuggling.
- OpenTelemetry `TextMapGetter` / `TextMapSetter` protocols.

## Notes
The fix is additive — every existing string-only caller continues to
pass. The new contract is a strict superset of the old one on the read
side, and a strict subset (reject CR/LF) on the write side. Downstream
consumers that were already well-formed observe no behaviour change.
