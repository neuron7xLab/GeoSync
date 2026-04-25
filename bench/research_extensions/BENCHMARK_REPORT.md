# GeoSync research extensions — microbenchmark report

- seed: `20260425`
- warmup iters: `2`
- measure iters (cap): `30`
- per-config wall time cap: `60` s
- python: `3.12.3` (CPython)
- numpy: `2.4.3`
- platform: `Linux-6.8.0-106-lowlatency-x86_64-with-glibc2.39` / `x86_64`
- started UTC: `2026-04-25T09:49:36+00:00`
- total wall time: `1455.29` s

## HEADLINE

**Headline measurements** (all values are medians over 30 samples after 2 warm-up iterations, deterministic seed `20260425`):

  - sparse triadic log-log slope vs N (p=0.05): **2.233** (target < 2.0; sub-quadratic=False)
  - sparse triadic log-log slope vs N (p=0.1): **2.219** (target < 2.0; sub-quadratic=False)
  - sparse triadic log-log slope vs triangle count T2: **0.672** (target ~ 1.0; linear-in-work=False)
  - beta-coupling build cost @ N=1024, L=5: **86.626 ms**
  - DR-FREE per-eval (ambiguity_dim=7): **16.75 us**

**Skipped benches (wall-time budget exceeded):** pipeline, ricci_flow

## Per-bench tables

### bench_capital_weighted

| N | L | median (ms) | p99 (ms) | baseline median (ms) | overhead ratio |
|---:|---:|---:|---:|---:|---:|
| 16 | 1 | 0.1190 | 0.1478 | 0.0027 | 44.80x |
| 16 | 5 | 0.1162 | 0.1395 | 0.0027 | 43.50x |
| 16 | 20 | 0.1149 | 0.1217 | 0.0025 | 46.47x |
| 64 | 1 | 0.1875 | 0.2034 | 0.0069 | 27.17x |
| 64 | 5 | 0.2006 | 0.3326 | 0.0072 | 27.84x |
| 64 | 20 | 0.1968 | 0.2528 | 0.0071 | 27.55x |
| 256 | 1 | 1.9887 | 4.6747 | 0.1163 | 17.10x |
| 256 | 5 | 1.7687 | 2.0418 | 0.0975 | 18.14x |
| 256 | 20 | 1.9249 | 2.4733 | 0.0936 | 20.56x |
| 1024 | 1 | 87.2786 | 133.0006 | 15.6252 | 5.59x |
| 1024 | 5 | 86.6264 | 144.9370 | 12.2036 | 7.10x |
| 1024 | 20 | 74.8178 | 92.9393 | 11.2711 | 6.64x |
| 4096 | 1 | 2656.4275 | 2976.3753 | 332.2527 | 8.00x |
| 4096 | 5 | 2866.0934 | 3204.7967 | 328.4076 | 8.73x |
| 4096 | 20 | 2790.8511 | 2976.2005 | 362.0512 | 7.71x |

### bench_ricci_flow

| N | p | total median (ms) | p99 (ms) | flow only (ms) | surgery only (ms) | active edges |
|---:|---:|---:|---:|---:|---:|---:|

### bench_dr_free

| ambiguity dim | median (us) | p99 (us) | robust margin |
|---:|---:|---:|---:|
| 0 | 25.584 | 41.605 | 0.000000e+00 |
| 1 | 25.359 | 34.937 | 5.314879e-03 |
| 2 | 27.867 | 29.929 | 1.313841e-02 |
| 3 | 28.442 | 35.591 | 1.525606e-02 |
| 4 | 15.050 | 108.589 | 1.906782e-02 |
| 5 | 25.186 | 34.548 | 2.210854e-02 |
| 6 | 16.459 | 30.897 | 2.427031e-02 |
| 7 | 16.747 | 20.202 | 2.624678e-02 |

log-log slope vs dim: `-0.2501331906039045` (O(1) per-metric: True)

### bench_sparse_simplicial

| N | p | T2 | sparse median (us) | sparse p99 (us) | dense median (us) | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.05 | 0 | 0.660 | 1.059 | 1.865 | 2.82x |
| 32 | 0.10 | 4 | 26.683 | 37.192 | 7.052 | 0.26x |
| 64 | 0.05 | 4 | 39.683 | 163.978 | 8.591 | 0.22x |
| 64 | 0.10 | 29 | 34.733 | 42.110 | 38.460 | 1.11x |
| 128 | 0.05 | 48 | 35.943 | 46.068 | 64.478 | 1.79x |
| 128 | 0.10 | 298 | 85.365 | 130.019 | 338.593 | 3.97x |
| 256 | 0.05 | 323 | 84.222 | 95.698 | n/a | n/a |
| 256 | 0.10 | 2596 | 577.321 | 612.352 | n/a | n/a |
| 512 | 0.05 | 2843 | 618.083 | 652.860 | n/a | n/a |
| 512 | 0.10 | 22026 | 5308.453 | 6630.090 | n/a | n/a |
| 1024 | 0.05 | 22135 | 5438.602 | 5634.902 | n/a | n/a |
| 1024 | 0.10 | 175682 | 42295.577 | 50396.239 | n/a | n/a |

**log-log slopes (sparse runtime vs N at fixed p):**

  - `p=0.05`: slope = **2.2328** (sub-quadratic = False)
  - `p=0.1`: slope = **2.2193** (sub-quadratic = False)

**log-log slope (sparse runtime vs triangle count T2, work-faithful axis):**

  - slope vs T2 = **0.6716** (target ~ 1.0; linear-in-work = False)

*Note: Erdos-Renyi at fixed p has T2 ~ p^3 N^3 / 6, so the runtime-vs-N slope reflects triangle growth, not per-triangle algorithmic cost. The runtime-vs-T2 slope is the work-faithful measurement of the kernel itself.*

### bench_pipeline

| N | total median (ms) | p99 (ms) | capital (ms) | ricci (ms) | sparse-50 (ms) | active edges |
|---:|---:|---:|---:|---:|---:|---:|
