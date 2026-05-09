# Healthy optimization — status, diagnosis, and future options

**Date:** 2026-05-09. Status capture at the Phase 3 step 2-3 decision point
in `AGENTS.md`. The healthy optimization is paused pending an architectural
or scope decision; this document is the hand-off so future-us can pick up
cleanly.

## 1. What has been tried

Two consecutive headline runs of `scripts/01_run_healthy_optimization.py`
with `configs/healthy.yaml` (1500 trials each, 400 ms simulation, 100 ms
burn-in, 450 neurons). The configuration is unchanged between runs except
for `NetworkConfig.heterogeneity_pct`.

| | Run 1 | Run 2 |
|---|---|---|
| Run dir | `results/healthy/20260509_071150/` | `results/healthy/20260509_133717/` |
| `heterogeneity_pct` | 0.0 | 0.10 |
| Trials feasible | 1471 / 1500 | 1431 / 1500 |
| **Best feasible loss** | **1.0320** | **1.0320** (4-dp identical) |
| **Best feasible STN rate** | **0.00 Hz** | **0.00 Hz** |
| Best GPe rate | 64.9 Hz | 65.1 Hz |
| Best GPi rate | 67.2 Hz | 66.8 Hz |
| Max STN rate observed (any trial) | 11.6 Hz | 11.8 Hz |
| Feasible trials with STN ≥ 5 Hz | 0 | 0 |
| Feasible trials with STN ≥ 15 Hz | 0 | 0 |

Comparison detail in `docs/healthy_heterogeneity_comparison.md`. Both
optuna_study.db files and full per-trial pickle records are in the run
dirs above.

**Heterogeneity at 0.10 did not break the network synchronization.**

## 2. Diagnosis

The optimizer is doing its job. CMA-ES has converged (best feasible loss
plateaus from trial ~1000 onward; both runs settle on the same value to 4
decimal places). The result is a global optimum given the current
architecture, constraint, and search bounds — not a local-minimum trap.

The structural conflict:

- The STN-GPe loop's natural frequency under canonical synaptic time
  constants and conduction delays of the Terman-Rubin lineage is in
  the 13–30 Hz band — confirmed by the pre-Phase-3 sanity check at
  `docs/network_beta_sanity_check.md` (peak 19.5 Hz under
  literature-strong PD coupling).
- Once STN drive is non-trivial, the loop entrains GPe and GPi into
  coherent oscillation in that band.
- Healthy STN (target 20 Hz) requires non-trivial drive to a cell whose
  spontaneous tonic rate is ~10 Hz.
- Therefore: any STN-active configuration produces beta > 0.05 in
  [13, 30] Hz and is infeasible. The only feasible regime in the
  current architecture is `STN-silent + GPe/GPi at target`, which the
  optimizer correctly finds.

Heterogeneity at 10% Gaussian spread on intrinsic g_L / g_Na / g_K
shifts individual cells' f-I curves by small amounts but does not
desynchronize the collective network response: across both runs, every
STN-active trial (n = 11+12 = 23 between the two runs) has
beta ∈ [0.36, 0.68]. The synchronization happens at the
network-architecture level, not the cell level.

This is a network-architecture issue, not an optimization issue.

## 3. Options on the table

Listed with the trade-offs each carries. Future work picks one (or some
combination); this document does not prescribe.

### Option a — Relax the healthy beta threshold (0.05 → 0.10)

The smallest change. Edit one line in `configs/healthy.yaml` and re-run.

Trade-offs:
- Easiest. Honest about model limits ("the rebuild's network produces
  some baseline beta even in the healthy regime"). Real basal ganglia
  do have nonzero beta in the healthy state; the [0.05, 0.15] gap
  between healthy and PD thresholds is the contrast we care about,
  not the absolute zero floor.
- Constraint relation `healthy < PD` preserved (0.10 < 0.15).
- Risk: even moderate STN drive may produce beta > 0.10 if the loop
  is strongly entrained at its natural frequency. The current data
  show STN ≈ 6 Hz already gives beta = 0.36 under run 2, so the
  threshold needs to be substantially higher (≥0.4) to admit STN
  activity at all — in which case the healthy/PD contrast effectively
  vanishes.
- Verdict: by itself probably not enough; useful as a fallback in
  combination with an architectural fix.

### Option b — Add GPe→GPe lateral inhibition

Adds a new connectivity matrix (`conn_gpe_gpe` with some indegree, e.g.
K_GPe→GPe = 10) and one new search parameter `g_gpe_gpe`. The
lateral inhibition lets GPe cells phase-shift relative to each other
under common excitatory input from STN, breaking the coherent rhythm.

Precedent in the model lineage:
- Terman, Rubin, Yew, Wilson 2002 (`wave.ode` distribution): explicit
  GPe-GPe coupling in the spatial-network variant.
- Hahn & McIntyre 2010, J Comput Neurosci 28:425-441.
- Kumaravelu et al. 2016, J Comput Neurosci 40:207-229.

Biological evidence:
- Sadek et al. 2007, J Neurosci 27:6352-6362 (in vivo).
- Bugaysen et al. 2013, Front Syst Neurosci 7:79 (intra-GPe inhibition).
- Mallet et al. 2012, Neuron 74:1075-1086.

Estimated effort: ~½ day (modify `bgnet/connectivity.py` to support
self-projections, extend `bgnet/network.py` builder, add the new
synapse pathway to `bgnet/integrator.py`, update `StudyConfig` /
configs YAMLs, write a sanity diagnostic showing reduced beta at
matched STN rates, run the headline optimization).

### Option c — Add prototypic / arkypallidal GPe split

GPe is biologically two subpopulations: prototypic GPe cells project to
STN/GPi, arkypallidal cells project back to striatum. They have
different firing characteristics. Modeling this would mean splitting
`gpe_*` in the config and integrator into `gpe_proto_*` and
`gpe_arky_*`, with separate connectivity, synapse parameters, and rate
targets.

References:
- Mallet et al. 2012, Neuron 74:1075-1086.
- Abdi et al. 2015, J Neurosci 35:6667-6688.
- Hernández et al. 2015, J Neurosci 35:11830-11847.

Most biologically motivated, especially for the PD-direction targets.
Effort: ~1-2 days plus test updates (new neuron parameter set, integrator
loop body grows by one pathway, all configs and validation tests need
revising). Higher risk of subtle bugs.

### Option d — Spatial / ring-topology connectivity

Replace random fixed-indegree with ring-topology + local (e.g.
distance-decaying) connectivity, in the style of `wave.ode` from
RT 2002. Breaks all-to-all synchronization through wave dynamics —
nearby cells entrain, distant cells don't, so population-mean beta
power is reduced even when local groups oscillate.

Effort: ~1 day. Modifies `bgnet/connectivity.py`. No new neuron model
or synapse pathway. Less biologically committed than option c.

Trade-off with c: c models a known anatomical split that's relevant to
the PD-direction question; d is a generic synchrony-breaking fix that
doesn't carry biological commitment but doesn't add modelling
overhead either.

### Option e — Expand optimization scope to synaptic time constants and delays

Currently we optimize 13 parameters (4 conductances, 3 drives, 3 OU mu,
3 OU sigma) and treat synaptic time constants and conduction delays
as fixed canonical values. Loosening them would add up to ~12 more
parameters (7 time constants × pathways + delays × pathways);
CMA-ES handles 25-dim fine.

Methodologically interesting (it's what the framework is *for*, after
all — searching biophysical hyperparameters), but doesn't address the
root architectural issue: even with optimal time constants, the
homogeneous-GPe + uniform-connectivity network will entrain.

Verdict: a separate methodological contribution / sensitivity study,
not a fix for this synchronization issue.

## 4. Recommended order

Not prescriptive — the next session decides:

1. **Option b first** (GPe→GPe lateral inhibition). Lowest cost, well-
   precedented across the model lineage, addresses the synchrony at
   the connectivity level without adding new cell types.
2. **Option c if b is insufficient** (prototypic/arkypallidal split).
   The most biologically committed answer; expensive but pays off
   especially for PD modeling.
3. **Option a (threshold relaxation) as fallback** at any point if b
   and c don't fully resolve the conflict. Acceptable as a documented
   model-vs-data gap in the manuscript.
4. **Option d as an alternative to c** if heterogeneity-of-cell-type
   feels like more commitment than warranted; comparable effort,
   different framing.
5. **Option e** as a separate downstream contribution, not a fix here.

## 5. Heterogeneity disposition

Keep `NetworkConfig.heterogeneity_pct = 0.10` as the default. Reasons:

- Well-precedented (Hahn & McIntyre 2010, Kumaravelu et al. 2016).
- Biologically realistic. No real population is identical.
- Costs nothing — the build-time multiplier sample is one numpy call
  per population.
- Although it didn't resolve the synchrony issue at this percent, it's
  a sensible model feature regardless. Removing it now would be
  reactive churn.

Future runs build on this default. If a sensitivity study later wants
to explore higher heterogeneity (e.g. 0.20-0.30) or zero, this is just
a config change at run time.

## 6. Run directories preserved

Both run dirs are intact and not overwritten:
- `results/healthy/20260509_071150/` — no-heterogeneity (Run 1)
- `results/healthy/20260509_133717/` — heterogeneity = 0.10 (Run 2)

Each contains `config.yaml`, `optuna_study.db`, `results.pkl`,
`metadata.json`, `log.txt`, and an empty `figures/` subdirectory. The
SQLite study files can be reopened with
`optuna.load_study(storage="sqlite:///<path>")`.

## 7. What is and isn't committed

Committed on `rebuild` branch:
- `bgnet/heterogeneity.py` and the wiring through `stn.py`,
  `pallidum.py`, `integrator.py`, `network.py` (commit `c099475`).
- Tests for the above, including the bumped two-seed-consistency
  tolerance (8 Hz, was 5 Hz).
- `scripts/validate_stn.py` updated to pass `homogeneous_het(1)`.
- The beta-band update from [8, 15] Hz to [13, 30] Hz across all
  configs, AGENTS.md §4.3, and `docs/rebuild_scope.md` §2.2.3
  (commit `6dc94e4`).
- Comparison doc `docs/healthy_heterogeneity_comparison.md`
  (commit `b479373`).

NOT committed / not yet done:
- No threshold change. `configs/healthy.yaml` still has
  `beta_threshold: 0.05`.
- No lateral inhibition (option b).
- No GPe split (option c).
- No spatial connectivity (option d).
- No expanded search space (option e).
- No further optimization re-runs since `c099475`.
- The `scripts/diagnose_stn_linear.py` and the sanity-check
  diagnostic scripts remain uncommitted (per earlier user
  instructions).

## 8. Pick-up checklist

Whoever resumes this work should:
1. Read this file end-to-end.
2. Read `docs/healthy_heterogeneity_comparison.md` and
   `docs/network_beta_sanity_check.md` for the empirical context.
3. Pick an option from §3 and write a brief plan before changing code.
4. Both run dirs above are preserved as the baseline; new runs go to
   fresh timestamps under `results/healthy/`.
