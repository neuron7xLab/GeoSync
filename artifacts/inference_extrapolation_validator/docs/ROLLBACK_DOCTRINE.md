# Rollback Doctrine

Immediate rollback trigger conditions:
- sudden drift increase in external probe,
- witness pipeline outage,
- schema/verify failures in production.

Rollback action:
1. force `result=killed_with_counterexample`
2. block evidence promotions
3. run falsifier + verify on last known-good release
