# State Diagram (one-page)

```text
[CONTEXT_BOUND]
   | inference
   v
[HYPOTHESIS_UNVERIFIED] --(claim as evidence)-> [REJECTED_POLICY_VIOLATION]
   | falsification battery
   +-- fail -------------------------------> [KILLED_WITH_COUNTEREXAMPLE]
   |                                         | artifact writer
   |                                         v
   |                                      [ATTESTED_NEGATIVE_ARTIFACT]
   |
   +-- pass --> [SURVIVED_PENDING_WITNESS if high risk]
                     | witness fail ---> [KILLED_WITH_COUNTEREXAMPLE]
                     | witness pass
                     v
                 [SURVIVED]
                     | artifact writer
                     v
                 [ATTESTED_EVIDENCE_ARTIFACT]
```
