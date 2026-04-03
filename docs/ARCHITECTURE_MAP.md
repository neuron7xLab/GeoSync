# GeoSync Architecture Map (Canonical Code Root)

## Canonical package
- **Root:** `src/geosync`
- **Import prefix:** `import geosync...`
- **Purpose:** Single source of truth for runtime, controllers, SDKs, and services.

## Legacy packages
- **`core/` (deprecated):** Thin shims that forward to `geosync.core.*`. Kept for backward compatibility only.
- **`geosync/` (repository root):** Shim that forwards to `src/geosync` to keep existing tooling working during transition.

## Subsystem map
- **Serotonin (TACL/5-HT):** `src/geosync/core/neuro/serotonin/` (legacy shim: `core/neuro/serotonin/`)
- **Thermo / TACL:** `runtime/` (API + controller), bridged through canonical runtime entrypoint
- **TACL Behavior Contracts:** `tacl/` (unchanged)
- **NAK controller:** `nak_controller/`
- **Neurotrade / Cortex:** `cortex_service/` and `geosync/neural_controller/` (shimmed via canonical root)
- **Risk/Execution:** `application/`, `execution/`, `runtime/`
- **Observability:** `observability/`
- **Experimental/Sandbox:** `sandbox/`, `examples/`

## Guidance
- New code MUST import from `geosync...`.
- Legacy imports under `core...` are deprecated and emit warnings; they resolve to the canonical modules where possible.
- See README for canonical run command and import examples.
