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

Normalized scale:
- simulation summary includes metric_scale 0..1
- simulation summary includes normalized_metrics
- parameter review includes normalized_value
- parameter review includes normalized_signal_delta
- parameter review includes normalized_limit_delta

Quantized states:
- simulation summary includes quantization_bins
- simulation summary includes quantized_states low, medium, high
- simulation summary includes quantized_metrics
- parameter review includes quantized_value
- parameter review includes quantized_signal_delta
- parameter review includes quantized_limit_delta

Boundaries:
- claim tier stays simulation-only
- production trading and execution paths are outside this stack

Verdict: the cognitive dynamics stack is integrated into main as a bounded deterministic research and replay layer with governance binding, replay commands, normalized metric outputs, quantized discrete state outputs, parameter review outputs, and rollback scope.
