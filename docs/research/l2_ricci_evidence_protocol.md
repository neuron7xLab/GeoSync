# L2 Ricci Evidence-Bearing Session Protocol

> **Status (locked until a real depth-5 session lands):**
> `claim_tier: HYPOTHESIS` · `semantic_validation_status: PLACEHOLDER` · `decision: OBSERVE`
>
> The README states that L2 order-book microstructure Ricci is **not**
> evidence-bearing and "must stay at `HYPOTHESIS` until a real depth-5 session
> with a real data hash is recorded." This document is the **promotion path**,
> not another label: it specifies, field by field, the artifact a real session
> must produce and the gates that artifact must clear before any L2 Ricci claim
> may move beyond `HYPOTHESIS`. It fabricates no evidence.
>
> Machine-checkable mirror: `schemas/research/l2_ricci_session.schema.json`,
> the recorder/gate `tools/research/run_l2_ricci_session.py`, and the contract
> tests in `tests/research/test_l2_ricci_session_schema.py`. The schema is
> line-specific (`depth_level` pinned to `5`) and complements the line-agnostic
> envelope `schemas/research/research_inference_artifact.schema.json` and the
> `ricci_microstructure_v1` preregistration; it does not replace either.

## 1. Hypothesis (under test, not asserted)

Ollivier–Ricci curvature features computed over a venue-native L2 **depth-5**
order book carry a falsifiable, out-of-sample signal for short-horizon mid-price
direction, beyond what a registered null model and a no-lookahead baseline
already explain. The graph-topology Ricci adapter that ships today is provenance
over correlation structure — it is **not** this L2 microstructure result, which
remains unimplemented and at `HYPOTHESIS`.

## 2. Forbidden claims

Until the full evidence chain exists (real data hash ∧ replayable session ∧
declared baseline ∧ registered null model ∧ executable falsifier ∧ recorded
negative evidence ∧ `semantic_validation_status: PASS`), none of the following
may be asserted from this line:

- "validated", "profitable", "alpha", "edge", "market-predictive", "deployable";
- any IC / Sharpe / PnL / hit-rate number presented as realized rather than as a
  pre-registered target;
- promotion of `claim_tier` above `HYPOTHESIS` on placeholder or zero-hash data.

## 3. Required session artifact

A session artifact MUST carry every field below; the schema rejects any artifact
missing one (`additionalProperties: false`), and the validator rejects shapes
that are valid but evidentially dishonest.

| Field | Meaning / promotion role |
| --- | --- |
| `git_sha` | 40-hex commit the session was produced at (provenance). |
| `data_sha256` | sha256 of the depth-5 capture. Zero = placeholder; real = precondition for evidence. |
| `config_sha256` | sha256 of the pinned run configuration. |
| `venue` | Lowercase venue the depth-5 book came from. |
| `symbol` | Captured instrument symbol. |
| `depth_level` | Pinned to `5` — this contract is exclusively the depth-5 session. |
| `session_start_utc` / `session_end_utc` | UTC capture window; end ≥ start (validator-enforced). |
| `baseline_id` | Reference the score is judged against; no baseline ⇒ asserted, not falsified. |
| `null_model_id` | Registered null the session is tested against; a signal that beats no null is not evidence. |
| `falsifier_id` | Executable falsifier that could have killed the hypothesis. |
| `replay_command` | Command that recomputes the session from committed inputs. |
| `negative_evidence` | Recorded falsification attempts (survival paths and kills), ≥ 1 entry. |
| `semantic_validation_status` | Honest evidentiary self-label; only `PASS` authorises promotion. |

## 4. Promotion gates (all must hold)

A session may promote past `HYPOTHESIS` only when **all** hold:

1. **Real data** — `data_sha256`, `config_sha256` non-zero; `git_sha` non-zero.
2. **Replay** — `replay_command` recomputes the session from committed inputs.
3. **Baseline** — `baseline_id` names a reference the score is judged against.
4. **Null model** — `null_model_id` names a registered null, and the
   `null_model` falsification axis appears in `negative_evidence`.
5. **Falsifier** — `falsifier_id` names an executable falsifier, and the
   `lookahead` (no-future-data) axis appears in `negative_evidence`.
6. **Negative evidence** — at least one non-`BLOCKED` attempt is recorded; an
   all-`BLOCKED` session has not been adversarially tested.
7. **Honest status** — `semantic_validation_status: PASS`; a `PLACEHOLDER` must
   not carry a real data hash.

Any unmet gate keeps the artifact at `HYPOTHESIS`. The validator is fail-closed.

## 5. Rejection (the line can kill itself)

The session is rejected — not silently downgraded — when a declared evidence
session carries zero hashes, omits a required falsification axis, records only
`BLOCKED` attempts, or has `session_end_utc` before `session_start_utc`. Survival
of these attacks is information and is recorded in `negative_evidence`; it is not
a victory lap.

## 6. Replay

```sh
# Emit a schema-valid, honestly-placeholder skeleton (never evidence):
python tools/research/run_l2_ricci_session.py --dry-run

# Validate a recorded session (shape + promotion-readiness):
python tools/research/run_l2_ricci_session.py --artifact path/to/session.json

# Contract tests:
pytest -q tests/research/test_l2_ricci_session_schema.py
```
