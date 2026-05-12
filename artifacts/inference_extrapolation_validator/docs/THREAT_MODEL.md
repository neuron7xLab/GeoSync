# Threat Model

## Adversaries
- Overconfident model outputs presented as evidence without falsification.
- Insider skipping witness review for high-risk claim.
- Pipeline tampering with artifact body after signing.
- Integration team misreading "formal consistency" as "real-world truth".

## Attack surfaces
- CLI arguments (`generate`) and malformed artifacts (`verify`).
- Missing/null witness metadata.
- Null-model omission for high-risk evidence.
- SHA drift between generation and downstream storage.

## Controls
- Fail-closed exit code 2 on contract violation.
- Exit code 3 on artifact parse failure.
- Mandatory schema gate before write and during verify.
- External falsification drift gate (`drift_score <= 0.5`) for evidence promotion.
