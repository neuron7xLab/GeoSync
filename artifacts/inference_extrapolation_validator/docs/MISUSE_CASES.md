# Misuse Cases

1. **"Looks plausible" -> mark EVIDENCE**
   - Blocked by required tests/null models/witness/external falsification.
2. **High risk with `not_required` witness**
   - Blocked by high-risk witness policy.
3. **Corrupt SHA and resubmit**
   - Blocked by verify SHA round-trip.
4. **Claim success with high drift**
   - Blocked by external falsification threshold.
5. **Treat module as truth oracle**
   - Blocked in claim-boundary docs and epistemic model.
