# CLAUDE.md — GeoSync Physics-First Development Protocol

## Identity

GeoSync is a quantitative trading platform with neuroscience-inspired risk
management. It implements Kuramoto synchronization, serotonin/dopamine/GABA
neuromodulation, free energy minimization, Ricci curvature, and thermodynamic
constraints on top of a Kelly-criterion execution engine.

**This is a physics system that happens to trade, not a trading system
that happens to use physics metaphors.**

## Rule Zero

**Before writing or modifying ANY test that involves a physical quantity
(R, K, δ, s(t), F, V, κ, energy, entropy), you MUST:**

1. Read `.claude/physics/INVARIANTS.yaml` — find ALL invariants for that quantity
2. Read the relevant theory file in `.claude/physics/` (e.g., `KURAMOTO_THEORY.md`)
3. Read `.claude/physics/TEST_TAXONOMY.md` — determine the correct test TYPE
4. Only then write the test

If you skip this, you will write tests that check numbers instead of physics.
Numbers pass accidentally. Physics doesn't.

## The Physics Kernel (`.claude/physics/`)

```
.claude/physics/
├── INVARIANTS.yaml        # Machine-readable invariant registry (ALL modules)
├── KURAMOTO_THEORY.md     # Kuramoto synchronization: K_c, R, finite-size, Lyapunov
├── SEROTONIN_THEORY.md    # Serotonin ODE: aversive state, sigmoid, desensitization
├── DOPAMINE_THEORY.md     # Dopamine TD-error: δ = r + γV' - V, convergence
├── GABA_THEORY.md         # GABA inhibition: sigmoid gate, monotonicity, position reduction
├── TEST_TAXONOMY.md       # How to write each test type (falsification → regression)
├── EXAMPLES.md            # Before/after transformations of real tests from this codebase
└── validate_tests.py      # 7-level validator: L1-L5 tests, C1-C2 code audit, --self-check
```

### Key Invariants (P0 — block merge)

| ID | Statement | Type | Module |
|----|-----------|------|--------|
| INV-K1 | R(t) ∈ [0,1] always | universal | kuramoto |
| INV-K2 | K < K_c → R(t→∞) → 0 | asymptotic | kuramoto |
| INV-K3 | K > K_c → R(t→∞) > 0 | asymptotic | kuramoto |
| INV-ES1 | Hysteresis width ≥ 0 | universal | explosive_sync |
| INV-5HT2 | s(t) ∈ [0, 1] always | universal | serotonin |
| INV-5HT4 | sensitivity ∈ [0.1, 1.0] | universal | serotonin |
| INV-5HT6 | tonic_level finite and ≥ 0 | universal | serotonin |
| INV-5HT7 | stress ≥ 1 OR |dd| ≥ 0.5 → veto | conditional | serotonin |
| INV-DA1 | δ = r + γV' − V (sign correct) | conditional | dopamine |
| INV-DA3 | γ ∈ (0, 1] enforced | universal | dopamine |
| INV-DA7 | ∂δ/∂r = 1 (linearity) | algebraic | dopamine |
| INV-GABA1 | Gate output ∈ [0, 1] | universal | gaba |
| INV-GABA2 | Higher vol → higher inhibition | qualitative | gaba |
| INV-GABA3 | effective ≤ raw position | universal | gaba |
| INV-FE1 | Free energy non-increasing | monotonic | free_energy |
| INV-FE2 | Free energy ≥ 0 | universal | free_energy |
| INV-RC1 | κ ∈ [-1, 1] (unweighted) | universal | ricci |

Full registry: 34 invariants across 8 modules. See INVARIANTS.yaml.

### Critical Formula

**Kuramoto critical coupling**: K_c = 2 / (π · g(0))

where g(0) is the value of the frequency distribution at zero.
For Lorentzian g(ω) = (γ/π)/(ω² + γ²): g(0) = 1/(πγ), so K_c = 2γ.

**Never hardcode K_c. Always compute from the distribution.**

## Testing Discipline

### Priorities
- **P0** (CI gate, blocks merge): Universal bounds, Lyapunov monotonicity, critical transitions
- **P1** (warning): Finite-size scaling, monotonicity sweeps, convergence rates
- **P2** (informational): Distribution tests, exact scaling exponents

### Test Error Messages Must Include (L4 enforced)
Every physics test assertion error MUST contain:
1. Invariant ID (e.g., "INV-K2 VIOLATED") — L4 checks regex
2. Expected value/behavior (from theory) — L4 checks "expect|should|must|violat"
3. Observed value — L4 checks "R=|delta=|level=|got|actual" + digits
4. Physical reasoning for why it's wrong
5. Parameters (K, N, steps, seed) — L4 checks "N=|seed=|K_c=|at|with"

### Finite-Size Awareness
- R = 0 exactly is impossible for finite N
- Incoherent regime: R fluctuates as O(1/√N)
- Use ε = C/√N (C ∈ [2,3]) not arbitrary thresholds
- For ensemble statistics: ≥ 50 realizations at minimum

### Production Code Audit (--audit-code)
When modifying physics production code (not tests), run:
```
python .claude/physics/validate_tests.py core/neuro/ --audit-code
```
This catches:
- **C1**: Silent clamp/clip without logging — may hide physics violation
- **C2**: Numeric bounds without INV-* comment — undocumented constraint

If you add `np.clip(R, 0, 1)` or `max(0.1, sensitivity)` — add a comment
explaining which invariant justifies the bound, or add logging so the
clamp event is observable. Silent projection of invalid state into valid
range masks the root cause.

### Forbidden Patterns
```python
# BAD: arbitrary threshold, no physics
assert R < 0.3

# BAD: exact equality on stochastic quantity  
assert R == 0.0

# BAD: no invariant reference, no error context
assert result.order_parameter > 0

# BAD: testing implementation detail, not physics
assert len(phases) == N  # this is a shape test, not physics
```

```python
# GOOD: physics-grounded, identified, contextualized
epsilon = 3.0 / np.sqrt(N)
assert R_final < epsilon, (
    f"INV-K2 VIOLATED: R={R_final:.4f} > ε={epsilon:.4f} "
    f"at K={K:.4f} < K_c={K_c:.4f} with N={N}. "
    f"Subcritical regime requires R→0."
)
```

## Module → Theory Routing Table

**Use this to decide WHICH theory file to read. Don't read all — read the relevant one.**

| If working on files matching... | Read this theory file | Key invariants |
|---|---|---|
| `*kuramoto*`, `*sync*`, `*phase*`, `*order_param*` | KURAMOTO_THEORY.md | INV-K1..K7 |
| `*explosive*`, `*hysteresis*` | KURAMOTO_THEORY.md §4 | INV-ES1..ES2 |
| `*serotonin*`, `*5ht*`, `*aversive*` | SEROTONIN_THEORY.md | INV-5HT1..7 |
| `*dopamine*`, `*rpe*`, `*td_error*`, `*reward_pred*` | DOPAMINE_THEORY.md | INV-DA1..7 |
| `*gaba*`, `*inhibit*`, `*position_gate*` | GABA_THEORY.md | INV-GABA1..5 |
| `*energy*`, `*free_energy*`, `*lyapunov*`, `*ecs*` | INVARIANTS.yaml §free_energy | INV-FE1..2 |
| `*thermo*`, `*conservation*`, `*entropy*` | INVARIANTS.yaml §thermodynamics | INV-TH1..2 |
| `*ricci*`, `*curvature*` | INVARIANTS.yaml §ricci | INV-RC1..2 |
| `*regime*` | KURAMOTO_THEORY.md + SEROTONIN_THEORY.md | Multiple |
| `application/`, `connectors/`, `cli/`, `ui/` | None (infra, no physics) | Standard tests |

### Module Map

**Physics Core** (theory-heavy, invariant-dense):
- `core/physics/` — conservation laws, higher-order Kuramoto, explosive sync
- `core/indicators/kuramoto*.py` — order parameter, Hilbert phases
- `core/indicators/kuramoto_ricci_composite.py` — Kuramoto + Ricci composite

**Neuromodulation** (ODE stability, Lyapunov proofs):
- `core/neuro/`, `src/geosync/core/neuro/` — serotonin, dopamine, GABA, ECS
- `rl/core/` — actor-critic, reward prediction error, modulation signal

**Market Structure** (thermodynamic constraints):
- `markets/` — orderbook, regime detection
- `analytics/regime/` — regime classification, phase transitions
- `backtest/physics_validation.py` — physics checks in backtesting

**Infrastructure** (standard software tests, no physics):
- `application/`, `connectors/`, `cli/`, `ui/`

## When Asked to "Write Tests for Module X"

1. Classify: is X physics-core, neuromodulation, market-structure, or infra?
2. If physics-related → read the physics kernel files first
3. Identify which invariants apply
4. Write P0 tests first (falsification), then P1 (convergence), then P2 (stats)
5. For each test, ask: "What physics does this test? If the physics is wrong, does this test catch it?"
6. If the answer to (5) is "no" → the test has zero physics value, rewrite it

## When Asked to "Fix a Failing Test"

1. Is the test checking physics? → Read the invariant it references
2. Is the PHYSICS wrong, or is the TEST wrong?
   - Physics violation = BUG IN IMPLEMENTATION → fix the code
   - Test using wrong threshold/method = BUG IN TEST → fix the test
3. Never "fix" a test by loosening a physics bound without understanding WHY

## Architecture Notes

- See `CANONICAL_OBJECT.md` for full system architecture
- See `PHYSICS_IMPLEMENTATION_SUMMARY.md` for current implementation status
- See `tests/TEST_PLAN.md` for existing test organization

## Collaboration Protocol

This project uses Adversarial Orchestration:
- **Creator** (Claude Code) → generates code and tests
- **Critic** (human or this protocol) → checks physics grounding
- **Auditor** → verifies invariants hold across modules
- **Verifier** → runs CI, checks no regressions

Claude Code is the Creator. The physics kernel is the embedded Critic.
When in doubt, the physics wins. Always.

## Session Protocol (FOLLOW THIS SEQUENCE)

When starting ANY task involving physics modules:

### Step 1: Classify
```
Is this task about: kuramoto/sync/phase/regime → KURAMOTO_THEORY.md
                     serotonin/5ht/aversive    → SEROTONIN_THEORY.md
                     dopamine/rpe/td_error      → DOPAMINE_THEORY.md
                     gaba/inhibition/gate        → GABA_THEORY.md
                     energy/lyapunov/thermo      → INVARIANTS.yaml
                     infra/api/cli/ui            → No physics, standard tests
```

### Step 2: Load Theory
Read the relevant theory file. Confirm you understand:
- What physical quantity is computed
- What invariants MUST hold (check INVARIANTS.yaml for IDs)
- What the falsification criteria are
- What common bugs the theory prevents

### Step 3: Execute Task
Write code/tests following the physics contract.

### Step 4: Self-Validate
```bash
# Test validation (L1-L5):
python .claude/physics/validate_tests.py <your_test_file>

# Production code audit (C1-C2):
python .claude/physics/validate_tests.py <module_dir> --audit-code

# Kernel self-check (34 invariants, cross-refs, dispatch):
python .claude/physics/validate_tests.py --self-check
```
Fix any issues BEFORE presenting results.

### Step 5: Report
In your response, state which invariants you tested and which theory
file you consulted. Example:
"Tested INV-K1 (bounds), INV-K2 (subcritical decay), INV-K4 (monotonicity).
 Consulted KURAMOTO_THEORY.md. All P0 invariants covered."

## Prompt Templates for Task Delegation

### When Yaroslav says "write tests for X":
```
1. cat .claude/physics/INVARIANTS.yaml | grep -A5 "<module>"
2. cat .claude/physics/<RELEVANT>_THEORY.md
3. cat .claude/physics/EXAMPLES.md
4. Write tests following TEST_TAXONOMY.md patterns
5. python .claude/physics/validate_tests.py <test_file>
```

### When Yaroslav says "fix failing test in X":
```
1. Read the test — does it reference INV-*?
2. If yes → read that invariant's theory, determine if PHYSICS or TEST is wrong
3. If no → the test is physics-blind, needs rewrite not just fix
4. cat .claude/physics/<RELEVANT>_THEORY.md for context
```

### When Yaroslav says "add physics validation to module X":
```
1. Read the module source code
2. cat .claude/physics/INVARIANTS.yaml — identify ALL applicable invariants
3. For each invariant: write the appropriate test type (see TEST_TAXONOMY.md)
4. Priority order: P0 first, then P1, then P2
5. Run validate_tests.py to confirm all tests reference invariants
```
