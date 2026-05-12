# Before / After Hallucination Claim Example

## Before (unsafe)
"Model predicts regime flip tomorrow; confidence high; execute immediately."

## After (IEV-gated)
- status: `UNVERIFIED` until tests + falsifiers + null models + witness + external probe pass.
- if any gate fails -> `KILLED` artifact preserved.
- only if all pass -> `EVIDENCE` with bounded claim.
