# Adaptive Market Mind (AMM)

**Інтуїція:** мозок зважує сенсорні помилки за прецизією (довірою), яка залежить від невизначеності середовища. AMM переносить це до ринку.

## Формули
- Прогноз: $\hat x_t = EMA(x_t)$
- Помилка: $PE_t = x_t - \hat x_t$
- Волатильність: $\sigma_t^2 = \lambda_\sigma \sigma_{t-1}^2 + (1-\lambda_\sigma) PE_t^2$
- Прецизія: $\pi_t = \mathrm{clip}\Big(\frac{\alpha}{\sigma_t^2+\varepsilon} e^{-\beta H_t} (1+\lambda(R_t-\bar R)) e^{\eta \kappa_t}\Big)$
- Валентність: $a_t=\tanh(k_t \pi_t PE_t)$
- Пульс уваги: $S_t = \lambda_S S_{t-1} + (1-\lambda_S)\max(0, a_t-\theta_t)$
- Гомеостатика: $k_{t+1}=k_t e^{\eta_k(S_t-\rho)},\ \theta_{t+1}=\theta_t+\eta_\theta(S_t-\rho)$

## Виходи
`amm_pulse`, `amm_precision`, `amm_valence`, `pred`, `pe`, `entropy`

## Онлайн-пороги без буферів
P²-квантилі (`core/neuro/quantile.py`) для `q_lo/q_hi` — O(1), без deque.

## Сайзинг
$L = w_\text{pulse}(S_t)\cdot w_\text{prec}(\pi_t) \cdot \frac{\text{target\_vol}}{\hat\sigma}$,
де $\hat\sigma$ — EW оцінка |PE|.

## Метрики та SLO
- `amm_pulse`, `amm_precision`, `amm_gain`, `amm_threshold`
- `amm_update_seconds` (Histogram) — латентність оновлення; SLO: p95 < 200µs/крок на CPU.
