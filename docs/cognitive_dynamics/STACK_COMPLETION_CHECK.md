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

Calibration profile:
- version: calibration-v0.1
- sensitivity: balanced
- low state upper threshold: 0.25
- high state lower threshold: 0.75
- parameter signal pass minimum: 1.5
- parameter limit pass maximum: 0.5

Quantized states:
- simulation summary includes calibrated quantization_bins
- simulation summary includes quantized_states low, medium, high
- simulation summary includes quantized_metrics
- parameter review includes quantized_value
- parameter review includes quantized_signal_delta
- parameter review includes quantized_limit_delta

Optimization profile:
- version: objective-v0.1
- simulation summary includes optimization_profile
- simulation summary includes optimized_metrics
- simulation summary includes objective_score
- simulation summary includes objective_state
- minimized metrics: mean_mae, mean_realism_gap, mean_outside_ei_fraction
- maximized metrics: mean_ei_balance, mean_energy_efficiency, mean_stability_margin
- parameter review includes objective_contribution
- parameter review includes recommended_action
- parameter review summary includes recommended_actions

Boundaries:
- claim tier stays simulation-only
- production trading and execution paths are outside this stack

Verdict: the cognitive dynamics stack is integrated into main as a bounded deterministic research and replay layer with governance binding, replay commands, normalized metric outputs, calibrated quantized state outputs, objective optimization outputs, parameter review actions, and rollback scope.
