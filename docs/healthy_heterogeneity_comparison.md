# Healthy optimization: heterogeneity comparison

Two consecutive headline runs of the healthy CMA-ES optimization with the
same `configs/healthy.yaml` (1500 trials, 400 ms × 450 neurons,
`beta_band = [13, 30]` Hz, `beta_threshold = 0.05`). The only change
between them is `NetworkConfig.heterogeneity_pct` — Gaussian per-neuron
variation in g_L / g_Na / g_K, sampled once at network build.

The question this comparison was supposed to answer: does breaking the
identical-neuron degeneracy with biophysical heterogeneity let the
optimizer find STN-active configurations that satisfy
`STN beta < 0.05`? Pre-registered decision rule: ≥15 Hz STN under
feasibility → proceed to PD optimization; <5 Hz → relax the healthy
beta threshold (move to "option 2").

## Side-by-side numbers

| Metric | run 1: no het | run 2: het = 10% |
|---|---:|---:|
| Run dir | `results/healthy/20260509_071150/` | `results/healthy/20260509_133717/` |
| Trials complete | 1500 | 1500 |
| Trials feasible | 1471 (98.1%) | 1431 (95.4%) |
| Best feasible loss | **1.0320** | **1.0320** |
| Best STN rate (Hz) | **0.00** | **0.00** |
| Best GPe rate (Hz) | 64.93 | 65.08 |
| Best GPi rate (Hz) | 67.18 | 66.84 |
| Best STN beta | 0.000 | 0.000 |
| Wall-clock (s) | 1191 | 1208 |

Best feasible loss is **identical** to four decimal places. STN at the
best feasible config is silent in both runs. GPe and GPi nail their
targets in both runs.

## STN-rate distribution across all 1500 trials (each run)

| STN rate threshold | run 1: no het | run 2: het = 10% |
|---|---:|---:|
| max STN rate (Hz) | 11.63 | 11.77 |
| trials with STN > 0 Hz | 29 | 79 |
| trials with STN > 5 Hz | 11 | 12 |
| trials with STN > 10 Hz | 2 | 2 |
| trials with STN > 15 Hz | **0** | **0** |
| feasible **and** STN ≥ 5 Hz | 0 | 0 |
| feasible **and** STN ≥ 10 Hz | 0 | 0 |

Heterogeneity makes more trials *touch* nonzero STN rates (29 → 79),
but doesn't extend the active-STN ceiling (11.6 → 11.8 Hz) and doesn't
change the binary "active-STN ⇒ infeasible" outcome. Every single
STN-active trial in either run produces `beta > 0.05`.

## Sample top trials, run 2 (het = 10%)

Top 10 trials with STN ≥ 5 Hz, sorted by loss:

| trial | feas | STN | GPe | GPi | beta |
|---:|:-:|---:|---:|---:|---:|
| 18 | F | 5.93 | 81.47 | 61.11 | 0.530 |
| 10 | F | 9.07 | 98.17 | 93.84 | 0.512 |
| 109 | F | 7.07 | 112.12 | 114.84 | 0.400 |
| 116 | F | 8.13 | 108.78 | 114.60 | 0.410 |
| 93 | F | 7.07 | 123.67 | 97.24 | 0.461 |
| 51 | F | 7.30 | 119.45 | 124.82 | 0.364 |
| 83 | F | 10.93 | 118.27 | 119.76 | 0.493 |
| 86 | F | 7.10 | 104.43 | 154.04 | 0.468 |
| 17 | F | 6.00 | 108.30 | 175.53 | 0.426 |
| 62 | F | 8.10 | 95.32 | 201.87 | 0.414 |

All STN-active trials are infeasible with beta in [0.36, 0.53] — the
network is still locking into a coherent ~20 Hz rhythm whenever STN
fires. Compare to run 1's matching list which showed the same pattern
(STN=11.6, beta=0.47 etc.).

## Read

**Heterogeneity_pct = 0.10 is not enough to break the STN-GPe loop's
synchronization.** A 10% Gaussian spread on intrinsic conductances
shifts individual neurons' f-I curves by a small amount but the
collective network response is still strongly coherent at the loop's
natural ~20 Hz frequency once STN drive is non-trivial.

This is consistent with the pre-Phase-3 sanity check
(`docs/network_beta_sanity_check.md`): the STN-GPe loop's natural
frequency lies inside [13, 30] Hz, so any active STN regime that
maintains the loop will have substantial in-band power.

Per the user's pre-registered fallback ("If still < 5 Hz, surface and
we go to option 2"): the data unambiguously falls under "still < 5 Hz
feasible." Surfacing for next decision.
