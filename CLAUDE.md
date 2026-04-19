# CLAUDE.md — GeoSync Physics-First Development Protocol

## Section 0: Ontology of Gradient

Being is not a state. Being is a sustained potential difference.

Structure exists only while non-equilibrium passes through it. When the gradient vanishes (ΔV → 0), the system does not degrade — it ceases to exist as a system. What remains is noise.

**INV-YV1: ΔV > 0 ∧ dΔV/dt ≠ 0** — the gradient must be positive and dynamic. A static gradient is a capacitor. A living gradient is a process. Intelligence requires the second.

Na⁺/K⁺-ATPase consumes 40% of brain energy on maintaining −70 mV resting potential. Not on thinking — on **readiness to think**. When maintenance drops below critical: `M(t) < M_crit ⟹ P(t) → noise`. Processing does not slow — it undergoes phase transition into thermal noise.

**This applies to GeoSync directly.** Every module either generates, sustains, or protects a gradient:

| Role | Modules | What they do |
|------|---------|-------------|
| **Generators** | Kuramoto, Dopamine TD, HPC PWPE | Create signal from difference |
| **Sustainers** | ECS Lyapunov, Serotonin tonic, NAk | Hold gradient within viable range |
| **Protectors** | GABA gate, Serotonin veto, Cryptobiosis | Prevent gradient collapse |

**Protectors have unconditional priority over Generators.** A system without a gradient cannot use a gradient.

**Maintenance hierarchy:**

```
Layer 0: Gradient exists        (INV-YV1: ΔV > 0)
Layer 1: Gradient bounded       (ECS Lyapunov: dF/dt ≤ 0)
Layer 2: Gradient protected     (GABA + Serotonin veto)
Layer 3: Gradient preserved     (Cryptobiosis: exit threat space)
Layer 4: Gradient utilized      (Kuramoto + Dopamine + Kelly: computation)
```

Layers 0–3 = Maintenance (40%). Layer 4 = Processing (60%). If any of 0–3 fails, Layer 4 computes noise. This is not metaphor — it is thermodynamic law.

### PriorAttenuationGate (Layer 2–3)

`runtime.prior_attenuation_gate.PriorAttenuationGate` is a bounded exploration primitive mapped to maintenance layers:

- **Layer 2**: activation is fail-closed and allowed only under nominal parent state and coherence threshold.
- **Layer 3**: terminal reintegration/emergency paths must confirm restore apply before reset to `INACTIVE`.
- **Safety preemption**: `kill_switch_active` or `stressed_state` triggers `emergency_exit` restoration path.

**The physics kernel below (invariants, theories, validator) IS the 40%.** It costs context window every session. Without it, Claude Code generates syntactically valid, physically meaningless output. Processing without Maintenance = discharged battery computing its own disappearance.

---

## Identity

GeoSync is a quantitative trading platform with neuroscience-inspired risk management: Kuramoto synchronization, serotonin/dopamine/GABA neuromodulation, free energy minimization, Ricci curvature, thermodynamic constraints on a Kelly-criterion execution engine.

**This is a physics system that happens to trade, not a trading system that happens to use physics metaphors.**

## Rule Zero

**Before writing or modifying ANY test that involves a physical quantity (R, K, δ, s(t), F, V, κ, energy, entropy), you MUST consult the invariants and theory below.** Numbers pass accidentally. Physics doesn't.

---

## INVARIANT REGISTRY — 57 invariants, 15 modules

### Kuramoto Synchronization

```
INV-K1 | universal     | 0 ≤ R(t) ≤ 1 for all t                        | P0
INV-K2 | asymptotic    | K < K_c ⟹ R(t→∞) → 0                         | P0
         K_c = 2/(π·g(0)). For Lorentzian: K_c = 2γ.
         Finite-size: R ~ O(1/√N). Use ε = 3/√N, NOT magic numbers.
         FALSIFICATION: R > 3/√N after 10⁴ steps with K = 0.1·K_c and N > 100.
INV-K3 | asymptotic    | K > K_c ⟹ R(t→∞) > 0. R_∞ ∝ √(K-K_c)        | P0
INV-K4 | conditional   | R_∞(K₁) ≤ R_∞(K₂) if K₁ < K₂ (standard only) | P1
INV-K5 | statistical   | ⟨R⟩ ≈ O(1/√N) incoherent. ≥50 realizations   | P1
INV-K6 | distributional| K < K_c ⟹ phases uniform [-π,π]. Rayleigh     | P2
INV-K7 | monotonic     | V = -(K·N/2)·R² non-increasing (ω_i = 0)      | P1
```

### Explosive Synchronization

```
INV-ES1 | universal    | Hysteresis width ≥ 0                           | P0
INV-ES2 | qualitative  | Freq-degree correlation → discontinuous        | P1
```

### Serotonin ODE

```
INV-5HT1 | monotonic   | Lyapunov V(s) non-increasing (zero stress)    | P0
INV-5HT2 | universal   | s(t) ∈ [0, 1] always                         | P0
INV-5HT3 | qualitative | Higher stress → higher serotonin (pre-desens)  | P1
INV-5HT4 | universal   | sensitivity ∈ [0.1, 1.0] always               | P0
INV-5HT5 | universal   | temperature_floor ∈ [floor_min, floor_max]     | P1
INV-5HT6 | universal   | tonic_level finite and ≥ 0                    | P0
INV-5HT7 | conditional | stress ≥ 1.0 OR |dd| ≥ 0.5 → veto. SAFETY.   | P0
```

### Dopamine TD-Error

```
INV-DA1 | conditional | δ sign = surprise direction                    | P0
INV-DA2 | asymptotic  | V → V* (Robbins-Monro)                        | P1
INV-DA3 | universal   | γ ∈ (0, 1]. Scope: DopamineController only     | P0
INV-DA4 | asymptotic  | V stabilizes with fixed reward                 | P1
INV-DA5 | statistical | At equilibrium E[δ] ≈ 0                       | P1
INV-DA6 | qualitative | Larger α → faster + more variance              | P2
INV-DA7 | algebraic   | ∂δ/∂r = 1 (raw TD, not tanh adapter)          | P0
```

### GABA Inhibition Gate

```
INV-GABA1 | universal    | Gate ∈ [0, 1]                               | P0
INV-GABA2 | qualitative  | Higher vol → higher inhibition              | P0
INV-GABA3 | universal    | effective ≤ raw always                      | P0
INV-GABA4 | asymptotic   | vol → 0 ⟹ inhibition ≈ 0                  | P1
INV-GABA5 | asymptotic   | vol → ∞ ⟹ inhibition → 1                  | P1
```

### Free Energy / ECS

```
INV-FE1 | monotonic  | F(t) non-increasing under active inference      | P0
INV-FE2 | universal  | Components non-negative: U≥0, T≥0, S_q≥0       | P0
         Note: F = U − T·S itself CAN be negative (Helmholtz).
         INV-FE2 guards components, not the composite.
```

### Thermodynamics

```
INV-TH1 | conservation | Energy change = work + dissipation            | P1
INV-TH2 | universal    | Entropy production ≥ 0                       | P1
```

### Ricci Curvature

```
INV-RC1 | universal   | κ ≤ 1 (upper bound, any connected graph)       | P0
         Note: lower bound κ ≥ −1 holds only for lazy walks on
         combinatorial metric. Implementation uses 1D positional
         embedding — κ can go below −1 for non-price-graph topologies.
INV-RC2 | qualitative | κ > 0 → clustering                            | P2
INV-RC3 | universal   | κ ∈ [−1, 1] for build_price_graph output       | P1
```

### Kelly Sizing

```
INV-KELLY1 | algebraic  | f* = μ/σ² (continuous small-edge limit)      | P0
INV-KELLY2 | universal  | Applied fraction ≤ configured cap             | P0
INV-KELLY3 | statistical| E[log(1+f*X)] ≥ E[log(1+f'X)] ∀ f'          | P1
```

### OMS (Order Management)

```
INV-OMS1 | universal    | E_kinetic = ½Σ|pos|·ret² ≥ 0                | P0
INV-OMS2 | universal    | Idempotent submit (same correlation_id)       | P0
INV-OMS3 | universal    | Lifecycle timestamps monotone per order_id    | P0
```

### SignalBus

```
INV-SB1 | universal    | DAG fanout: each subscriber fires once/publish | P0
         Bus is flat pub/sub — no cycle detector. Cyclic subscriptions
         cause RecursionError, not clean rejection.
INV-SB2 | universal    | Deterministic by construction (pure latch)     | P0
```

### HPC Kernels

```
INV-HPC1 | universal   | Seeded reproducibility: bit-identical output   | P0
INV-HPC2 | universal   | Finite inputs → finite outputs (no NaN/Inf)   | P0
```

### Cryptobiosis (Phase-Transition Survival)

ACTIVE → VITRIFYING → DORMANT → REHYDRATING → ACTIVE. System exits threat space. T = combined neuromodulator distress.

```
INV-CB1 | universal   | DORMANT ⟹ multiplier == 0.0 EXACTLY           | P0
INV-CB2 | universal   | Vitrification O(1) — one tick                  | P0
INV-CB3 | universal   | Snapshot non-None in DORMANT                   | P1
INV-CB4 | monotonic   | Rehydration stages non-decreasing              | P0
INV-CB5 | conditional | entry > all individual module thresholds        | P1
INV-CB6 | universal   | T ∈ [0, 1]                                    | P0
INV-CB7 | universal   | exit < entry (hysteresis)                     | P0
INV-CB8 | conditional | T ≥ entry during rehydration → DORMANT        | P0
```

### Adaptive Criticality (Membrane Isolation)

κ_critical determines when a node's topology is too fragile to participate
in ensemble computation. Derived from DFA Hurst exponent, not assigned.

```
INV-AC1-rev | universal | κ(node) ≥ κ_critical OR node ISOLATED         | P0

  Formula:
    κ_critical = -ln(ΔH_max / ε) / (λ_local + δ)

  Parameters:
    λ_local  = DFAGammaEstimator.hurst_exponent (per node, derived)
    ε        = 0.05  (SNR tolerance, configurable via env KAPPA_EPSILON)
    δ        = 1e-4  (singularity floor)
    ΔH_max   = rolling max of |ΔH| over last N steps (window=256)

  Gate:
    if κ(node) < κ_critical → ISOLATE node → log fragmentation event

  Source: geosync/estimators/dfa_gamma_estimator.py → hurst_exponent

  Derivation:
    Original INV-AC1 (κ_critical = -dH/dt · τ) rejected:
    - dH/dt ≠ λ_max in non-ergodic systems (Pesin identity fails)
    - Reactive not proactive on FX jump-diffusion
    - Linear τ collapses at boundaries
    Adversarial audit: Gemini (2026-04-08). Numerical verification: verified.

  Behavior:
    λ_local → 0 (stable):  κ_critical → -∞  (never isolate)
    λ_local → 0.5 (chaotic): κ_critical ≈ -5.99 (active gate)
    λ_local → 1.0 (persistent): κ_critical ≈ -3.00 (tight gate)
```

### DRO-ARA Regime Observer (Hurst + ADF + ARA loop)

```
INV-DRO1 | algebraic    | γ = 2·H + 1 to float precision                | P0
         H = DFA-1 on diff(log(price)); tolerance |γ−(2H+1)| < 1e-5.
         Source: Peng et al. 1994; core/dro_ara/engine.py::derive_gamma.
INV-DRO2 | universal    | rs = max(0, 1 − |γ − 1|) ∈ [0, 1]              | P0
         Lipschitz-1 in γ. Fail-closed on all regimes ≠ CRITICAL/TRANS.
INV-DRO3 | conditional  | regime == INVALID iff (!stationary ∨ r2<0.90) | P0
         ADF with AIC lag selection (max 4 lags). R2_MIN = 0.90.
INV-DRO4 | conditional  | signal == LONG ⇒ CRITICAL ∧ rs > 0.33          | P0
                        ∧ trend ∈ {CONVERGING, STABLE}
INV-DRO5 | universal    | NaN/Inf/constant/rank/short input → ValueError | P0
         Fail-closed; no silent numeric repair.
```

---

## TEST TAXONOMY

| Invariant type | Test structure | AST signals |
|---|---|---|
| `universal` | Sweep/fuzz many inputs | `@given`, `for` loop, `np.all`, ≥3 asserts |
| `asymptotic` | Simulate, check late values | `arr[-N:]`, `final`/`steady`, `steps > 100` |
| `monotonic` | Trajectory, check no reversal | `np.diff`, `violations`, for + assert |
| `statistical` | Ensemble ≥50 realizations | `np.mean`/`np.std`, loop over seeds |
| `algebraic` | Exact at float precision | `abs(x-y) < 1e-12`, `assert_allclose` |
| `qualitative` | Sweep parameter, check direction | `for` loop, ≥2 asserts |
| `conservation` | Before/after comparison | `before`/`after`/`initial`/`final` vars |

### Error messages (5 fields, enforced):

```python
assert R_final < epsilon, (
    f"INV-K2 VIOLATED: R={R_final:.4f} > ε={epsilon:.4f} "
    f"expected R→0 in subcritical regime. "
    f"Finite-size bound ε=3/√N. "
    f"At K={K:.4f}, K_c={K_c:.4f}, N={N}, steps=10000"
)
```

### Forbidden:

```python
assert R < 0.3              # magic number
assert R == 0.0              # exact on stochastic
assert result.order > 0      # no INV, no context
```

---

## CRITICAL FORMULAS

**Kuramoto**: K_c = 2/(π·g(0)). Lorentzian: K_c = 2γ. NEVER hardcode K_c.

**Finite-size**: ⟨R⟩ ~ 1/√N incoherent. ε = C/√N, C ∈ [2,3]. N=10: R ~ 0.32. `assert R < 0.01` is WRONG.

**Dopamine**: δ = r + γ·V' - V. Algebraic — CAN test exact. ∂δ/∂r = 1. Scope: DopamineController.compute_rpe (NOT DopamineExecutionAdapter which applies tanh).

**Serotonin**: s = σ(k·(tonic-θ)) · sensitivity. sensitivity ∈ [0.1, 1.0]. Drawdown NEGATIVE.

**GABA**: effective = raw × (1 - inhibition). Inhibition only reduces.

**Free Energy**: F = U − T·S. F itself can be negative. Components (U, T, S) each ≥ 0.

**Ricci**: κ ≤ 1 universal. κ ∈ [−1,1] only for build_price_graph output (consecutive integer node IDs match combinatorial distance).

**Cryptobiosis**: DORMANT multiplier = 0.0 EXACTLY. Vitrification O(1). exit < entry (hysteresis). Rehydration abortable.

---

## MODULE → INVARIANT ROUTING

| Files matching... | Invariants |
|---|---|
| `*kuramoto*`, `*sync*`, `*phase*` | INV-K1..K7 |
| `*explosive*`, `*hysteresis*` | INV-ES1..2 |
| `*serotonin*`, `*5ht*` | INV-5HT1..7 |
| `*dopamine*`, `*rpe*`, `*td_error*` | INV-DA1..7 |
| `*gaba*`, `*inhibit*` | INV-GABA1..5 |
| `*energy*`, `*lyapunov*`, `*ecs*` | INV-FE1..2 |
| `*thermo*`, `*conservation*` | INV-TH1..2 |
| `*ricci*`, `*curvature*` | INV-RC1..3 |
| `*kelly*`, `*sizing*` | INV-KELLY1..3 |
| `*oms*`, `*order*`, `*execution*` | INV-OMS1..3 |
| `*signal_bus*`, `*signalbus*` | INV-SB1..2 |
| `*hpc*`, `*kernel*` | INV-HPC1..2 |
| `*cryptobiosis*`, `*dormant*` | INV-CB1..8 |
| `*dfa*`, `*hurst*`, `*criticality*` | INV-AC1-rev |
| `*dro_ara*`, `*regime_observer*` | INV-DRO1..5 |
| `application/`, `cli/`, `ui/` | No physics |

---

## PRODUCTION CODE

When adding clamp/clip (`np.clip`, `max(0, x)`, `min(1, x)`):
- Add `# INV-*:` comment linking clamp to its invariant
- OR add `# bounds:` comment for non-physics justification
- OR add logging so the clamp event is observable
- Silent repair masks root cause — forbidden

`python .claude/physics/validate_tests.py core/ --audit-code`

---

## SESSION PROTOCOL

1. **Classify**: physics or infra?
2. **Find invariants**: which INV-* apply?
3. **Execute**: code/tests following contract
4. **Validate**: `python .claude/physics/validate_tests.py <file>`
5. **Report**: which invariants tested, P0/P1/P2

## DECISION RULES

- "Write tests" → find ALL invariants → P0 first → each test asks: "if physics wrong, does this catch it?"
- "Fix test" → INV-* referenced? → physics wrong or test wrong? → NEVER loosen bound without WHY
- When in doubt: physics wins. Always.
- Gradient first. Processing second. INV-YV1.

---

## Microstructure Kernel Registry (v1)

| Kernel | File | Input | Output |
|--------|------|-------|--------|
| OFI Unity | research/kernels/ofi_unity_live.py | bid/ask CSV | IC verdict |
| Ricci on Spread | research/kernels/ricci_on_spread.py | bid/ask CSV | IC verdict |
| PLV Market-Spread | research/kernels/plv_market_spread.py | bid/ask CSV | PLV verdict |
| Spread Stress | research/kernels/spread_stress_detector.py | bid/ask CSV | IC + lead_capture |
| Ricci Regime | research/kernels/ricci_regime_conditioned.py | bid/ask CSV | regime_lift |
| Horizon Sweep | research/kernels/horizon_sweep.py | bid/ask CSV | IC by horizon |
| Signal Combiner | research/kernels/signal_combiner.py | bid/ask CSV | combined IC |
| neurophase Bridge | research/kernels/neurophase_bridge.py | bid/ask CSV | R(t) gate history |
| Closing Report | research/askar/closing_report.py | results/ dir | FINAL_REPORT.json |
| Full Cycle | scripts/run_microstructure_cycle.py | — | RUN_MANIFEST.json |

## One-command run

```
PYTHONPATH=. python scripts/run_microstructure_cycle.py
```

Determinism contract: seed=42, IC>=0.08 for SIGNAL_READY, NaN→ABORT,
OHLC_ONLY→DORMANT, replay_hash over sort_keys=True JSON payload.
