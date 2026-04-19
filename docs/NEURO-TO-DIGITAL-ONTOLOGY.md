# Neuro-to-Digital Ontology

## 1. Джерело натхнення
- Молекула: N,N-Dimethyltryptamine
- Основний механізм: 5-HT2A agonism → зменшення top-down priors → збільшення ентропії мозку

## 2. Таблиця відповідності

| Механізм ЦНС | Цифровий аналог | Інваріант | Математична форма |
|---|---|---|---|
| Зменшення впливу priors | PriorAttenuationGate activation guard | INV-PA-1 | coherence ≥ activation_threshold |
| Ентропійний сплеск | Entropy ceiling forced reintegration | INV-PA-3 | entropy > max_entropy → phase=REINTEGRATION |
| Тривалість вікна дослідження обмежена | Duration cap reintegration | INV-PA-4 | bars_elapsed ≥ max_duration_bars |
| Закриті terminal paths | reintegrate()/emergency_exit only | INV-PA-5 | phase=INACTIVE iff terminal_apply_confirmed |
| Точне відновлення priors | restore from backup | INV-PA-6 | restored_weights == prior_backup |
| Safety preemption | kill-switch / stressed_state | INV-PA-7 | safety_signal=True → emergency_exit |
