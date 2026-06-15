# Validation

Required checks:

1. Public API returns the response envelope.
2. Router uses the registry before dispatch.
3. Pipeline returns one envelope for the full run.
4. Unknown modules return a structured error.
5. Shape adapter rejects missing required fields.

Runtime commands remain blocked until this branch can run CI.
