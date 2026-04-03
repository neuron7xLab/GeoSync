# Quality Definition of Ready (DoR) & Definition of Done (DoD)

Maintaining a disciplined engineering cadence requires unambiguous entry and exit
criteria for every quality-focused task. This checklist clarifies what must be
true before work is pulled into an iteration and the evidence required before a
quality issue is considered complete. Use it when planning, grooming, or
reviewing work so the team can ship with confidence.

## Definition of Ready (DoR)

A quality task is ready to start only when the following statements hold true:

1. **Problem statement captured** – The defect, risk, or improvement objective is
   described in user-impact terms with logs, metrics, or screenshots attached.
2. **Scope and boundaries agreed** – Acceptance criteria list what is in scope
   and explicitly call out non-goals or deferred follow-ups.
3. **Data sources identified** – Required datasets, fixtures, or synthetic
   samples are documented and accessible, including retention or privacy
   constraints.
4. **Reproduction path verified** – Steps or scripts to reproduce the current
   failure (or demonstrate the gap) are validated on a non-production
   environment.
5. **Quality signals defined** – Owners have chosen measurable signals (tests,
   KPIs, SLO thresholds) that will prove the issue is resolved.
6. **Dependencies unblocked** – Cross-team inputs, configuration flags, secrets,
   or infrastructure changes are accounted for with owners and timelines.
7. **Risk & rollout plan drafted** – Mitigation steps and rollback triggers are
   documented, including who will monitor the change once deployed.

If any criterion is missing, the task returns to backlog refinement instead of
being started.

## Definition of Done (DoD)

A quality task may be closed only when the following outcomes are demonstrated:

1. **Fix implemented and reviewed** – Code, configuration, or runbook updates are
   merged with peer review and lint/test gates passing.
2. **Automated coverage in place** – Regression tests (unit, integration, or
   contract) fail when the original issue resurfaces and run in CI.
3. **Manual validation captured** – Screenshots, logs, or dashboards confirm the
   expected behaviour in the target environment, attached to the task record.
4. **Observability updated** – Alerts, metrics, or traces have thresholds tuned
   and dashboards updated to watch the corrected path.
5. **Documentation synchronized** – README, runbooks, or user guides reflect the
   new behaviour and any operational caveats.
6. **Rollout executed** – Deployments or configuration flips follow the approved
   plan with monitoring during and after release; rollback plan is not needed.
7. **Post-change review completed** – Owners recorded learnings, potential
   follow-up work, and confirmed with stakeholders that service levels are back
   within target.

Only when every exit criterion is satisfied can the task move to "Done". Missing
items must be split into follow-up work with the same rigor applied.
