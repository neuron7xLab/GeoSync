# Formal Interpretation

Let artifact A be promotable iff all constraints hold:

Promote(A) :=
  SchemaValid(A)
  AND ShaValid(A)
  AND ContractValid(A)
  AND PurposeAligned(A)
  AND ExternalFalsificationPass(A)
  AND HyperdirectConflict(A) <= threshold

If any conjunct is false => reject with fail-closed behavior (exit code 2).

Parse failures => exit code 3.
