# Cognitive Dynamics Stack Completion Check

Status: OPERATIONALLY_INTEGRATED

Active governance file: .claude/commit_acceptors/cognitive-dynamics-lab.yaml

Active executable files:
- scripts/cognitive_dynamics_lab/simulation_runner.py
- scripts/cognitive_dynamics_lab/parameter_review_runner.py

Replay commands:
- python scripts/cognitive_dynamics_lab/simulation_runner.py --out artifacts/cognitive_dynamics_lab
- python scripts/cognitive_dynamics_lab/parameter_review_runner.py --out artifacts/cognitive_dynamics_lab

Expected outputs:
- summary.json
- metrics_table.csv
- parameter_review_summary.json
- parameter_review.json
- parameter_review.csv

Boundaries:
- claim tier stays simulation-only
- production trading and execution paths are outside this stack

Verdict: the cognitive dynamics stack is integrated into main as a bounded deterministic research and replay layer with governance binding, replay commands, parameter review outputs, and rollback scope.
