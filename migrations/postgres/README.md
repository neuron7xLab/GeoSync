# Kill-switch state migrations

This directory contains SQL migrations required by the PostgreSQL-backed kill-switch
state store. Apply the migrations using your organisation's preferred schema
management tooling before enabling the PostgreSQL persistence backend in
production environments.

1. `0001_create_kill_switch_state_table.sql` – creates the `kill_switch_state`
   table and an index on `updated_at` used for staleness monitoring. The script
   is idempotent and can be safely re-run during deployments.
