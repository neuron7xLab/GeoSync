# Canonical Seven — Foundation Layer

The **Canonical Seven** are the seven institutional preconditions an
empirical claim in this module must survive before it is allowed to
graduate from `HYPOTHESIS` to `THEORY`, and from `THEORY` to `FACT`.
They operationalise the falsificationist contract that the module
*as a whole* exists to enforce: every claim must carry, in code,
the specific conditions for its own death.

| # | Pillar                            | Module                          | Canonical role                                                        |
|---|-----------------------------------|---------------------------------|----------------------------------------------------------------------|
| 1 | Hypothesis Death Engine           | `death_conditions.py`           | Five typed kill / demote / quarantine triggers + precedence rule.     |
| 2 | Bayesian Evidence Ledger          | `evidence_ledger.py`            | Append-only, log-odds, monotonic Bayesian update with kill triggers.  |
| 3 | Data Reality Firewall             | `temporal_panel.py` + ext.      | Eight-gate ingress contract on every panel before any score is built. |
| 4 | Adversarial Baseline Ladder       | `adversarial_ladder.py`         | Paired-bootstrap delta-AUC against a canonical roster of prosecutors. |
| 5 | Leakage Sentinel                  | `leakage_sentinel.py`           | Six runnable checks for time-flow / contamination / centred windows.  |
| 6 | Replication Capsule               | `replication.py`                | Bit-identical rerun of a sealed manifest; KILL on divergence.         |
| 7 | Claim Governance Constitution     | `governance.py`                 | Pre-merge science gate; promotes/blocks tier transitions.             |

The first three pillars in this PR (1, 2, 5) form the **foundation
layer** that every later pillar depends on. They are pure-function,
fail-closed, and free of I/O.

## Precedence

When more than one trigger fires, the registry resolves to a single
tier action under the strict ordering:

```
KILL > INVALIDATE > QUARANTINE > DEMOTE > STOP > NONE
```

This ordering encodes the operational rule that *destroying a claim
always wins over preserving it*. The aggregator never silently nets
demotions against demotions — five distinct actions, five distinct
operational meanings.

## Foundation invariants

* **F1** — every Bayes factor is finite and ≥ 0; `inf` is rejected at
  the constructor; `0` is reserved for the replication-mismatch /
  KILL trigger.
* **F2** — the evidence ledger is append-only and chronologically
  ordered (strict `<`); reordering is impossible by construction.
* **F3** — the death-conditions registry is frozen; new triggers are
  added only via `extend(...)`, which returns a new registry and
  never mutates the existing one.
* **F4** — the leakage sentinel returns `detected=True` on every
  *positive* detection, and `LeakageReport.detected` is the
  disjunction of its component outcomes — there is no quiet-fail
  path.
* **F5** — every public dataclass in this layer is `frozen=True`
  with `slots=True`; instances cannot be mutated after construction.

## Module wiring

```
DataFirewall ─┐
              ▼
LeakageSentinel ─► DeathConditionsRegistry ─► TierTransition
              ▲
EvidenceLedger ┘
```

The registry consumes outcomes from the leakage sentinel and the
evidence ledger (plus the adversarial ladder, fragility audit, and
replication capsule once they are wired). It returns a single
`TierTransition`, which the governance layer then either promotes to
the claim ledger or rejects.

## Reading order

1. Start with [`evidence_ledger.py`](evidence_ledger.py) — the
   formal Bayesian skeleton; everything downstream is an evidence
   producer.
2. Then [`death_conditions.py`](death_conditions.py) — the tier-
   action arbiter; this is where the claim either survives a round
   or doesn't.
3. Finally [`leakage_sentinel.py`](leakage_sentinel.py) — the most
   common producer of `INVALIDATE`-tier evidence; six independent
   sentinel checks guarding against the six most common
   time-leakage failure modes.

Each module ships its own test file under
`tests/research/systemic_risk/`. Property-style invariants (F1–F5)
are checked there.
