# Integration Demo

## Minimal sequence
1. upstream model generates extrapolated hypothesis
2. orchestration runs falsifiers and null models
3. IEV `generate` emits attested artifact
4. admission controller runs IEV `verify`
5. only `VERIFY PASS` + claim_status EVIDENCE can pass to execution

## One-command gate
`make iev-gate`
