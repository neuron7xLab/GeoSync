# Reset-Wave як objective-criterion функція

> Per IERD §4.2 lexicon: "truth function" → "objective criterion". This document uses the IERD-compliant phrasing throughout.

## 1) Однофразова постановка

Перетворити шумний фазовий стан мережі на керований рух до базового еталона так, щоб **bounded phase potential** не зростала, а небезпечні режими автоматично блокувалися.

## 2) Абстрактна objective-criterion функція

Нехай:
- `x` — поточний вектор фаз вузлів,
- `x*` — базовий (еталонний) вектор фаз,
- `E(x, x*) = mean(1 − cos(x* − x))` — bounded phase potential (energy не у термодинамічному сенсі — ми не претендуємо на закон термодинаміки, лише на чисельну функцію Ляпунова на компактному [-π, π)),
- `L(x)` — predicate безпеки (`True`, якщо є критичне відхилення).

**Tier per IERD §2:** EXTRAPOLATED. ANCHORED-частина — стабільна область `coupling_gain · dt ≤ 0.2`. Поза нею не гарантовано монотонне зменшення.

Тоді objective-criterion механізму:

`V(x) = 1`, якщо одночасно:
1. **Safety criterion (ANCHORED):** `L(x) ⇒ lock` (fail-closed),
2. **Energy criterion (EXTRAPOLATED, scoped):** `not L(x) ∧ within_stable_region ⇒ E_{t+1} ≤ E_t`,
3. **Convergence criterion (EXTRAPOLATED, scoped):** за скінченне число кроків `||x_t − x*|| → tol`.

Інакше `V(x) = 0`.

Інтуїція: система задовольняє objective criterion тоді, коли вона або безпечно блокує небезпеку, або всередині stability bound гарантовано зменшує відхилення до еталону.

## 3) Механізм як цілісна дія

1. **Сканування стану:** `d = x* − x`, `|d|`.
2. **Decision на режим:**
   - якщо `max(|d|) > max_phase_error` → **safety-lock**, write_ops_inhibited=True, no active updates;
   - інакше → **damped phase synchronization**.
3. **Step:** Euler `x ← x + dt · coupling_gain · sin(d)` або RK4-fixed.
4. **Audit:** після кожного кроку:
   - `V(θ_{t+1}) ≤ V(θ_t)` (всередині stable region),
   - детермінізм траєкторії (same input ⇒ same output),
   - досягнута збіжність або коректне блокування.

## 4) Контракт I/O

- **Вхід:** `node_phases`, `baseline_phases`, `coupling_gain`, `dt`, `steps`, `convergence_tol`, `max_phase_error`, `integrator ∈ {euler, rk4_fixed}`.
- **Вихід:** `ResetWaveResult(converged, locked, initial_potential, final_potential, trajectory)`.
- **Boundary conditions:** додатні `coupling_gain / dt / steps / max_phase_error`; однакова довжина векторів; `integrator` з whitelist.

## 5) Falsification (коли модель вважається хибною)

Механізм вважається falsified якщо хоч раз:
1. `not safety_lock ∧ within_stable_region ∧ final_potential > initial_potential`,
2. `safety_lock=True ∧ final_potential ≠ initial_potential` (lock дозволяє active update),
3. однаковий вхід дає різний вихід (порушення детермінізму),
4. `max(|d|) > max_phase_error ∧ not safety_lock` (lock не сцрацював на extreme drift).

Falsifier тестується у `tests/test_reset_wave_physics_laws.py` + `tests/test_reset_wave_stress_validation.py` (Monte Carlo n=400 + grid 4×3×3×3=108 cells × 20 seeds).

## 6) Практичний сенс

Це не "магія фізики" — це інженерна дисципліна:
- **safety first** (lock),
- **bounded градієнт до еталона** всередині stability region (reset),
- **перевірювані інваріанти** (energy non-increase, determinism, convergence, lock fail-closed).

У такій формі механізм читається як єдина система рішень:
**ризик > поріг → lock; інакше → зменшуй помилку до нуля з audit objective criterion.**
