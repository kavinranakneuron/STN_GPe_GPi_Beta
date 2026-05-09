# AGENTS.md

Operational guide for a coding agent (Claude Code) executing the JNE-110355 rebuild.

This file is the source of truth for how to execute. The companion document `rebuild_scope.md` explains the *why* — read it first if you have not already. This file tells you *what to do, in what order, to what specification.*

The user is Kavineshvar Ranak Nakkeeran, first author of the manuscript. When this file says "ask the user," it means surface a decision point and wait for input. Do not invent answers.

---

## 0. Operating principles

Read these before every session. They take precedence over local convenience.

1. **No silent fudges.** No magic constants, no scaling factors that compensate for unit mismatches, no hardcoded values that make the math work without explanation. If you find yourself wanting to add `* 0.005` somewhere, stop and ask. The submitted paper has these and they are part of why we are rebuilding.

2. **One source of truth per number.** Every numeric result in the eventual paper must be traceable to (a) a specific run, (b) a specific config file, (c) a specific seed. The current codebase fails this; saved pickles disagree with figures. The rebuild must not.

3. **Honest reporting.** If the rebuild produces different numbers from the original paper, report what the rebuild produces. Do not tune to match. The original numbers are not reproducible from the submitted artifacts and reproducing them is not a goal.

4. **Test before optimizing.** Every neuron model, synapse, and integrator gets unit tests before it is used in an optimization. Catching a bug after a 30-minute optimization run is much more expensive than catching it in a 0.5-second test.

5. **Commit at clean checkpoints.** After each phase below produces a working artifact, commit with a clear message. The user should be able to walk back to any phase boundary cleanly.

6. **Ask, don't assume.** When something is genuinely ambiguous, surface it. The list of "decision points" in each phase below tells you what to surface; if something else feels ambiguous, surface that too.

7. **Respect the unit convention.** Current density throughout: conductances in mS/cm², currents in µA/cm², voltages in mV, time in ms, calcium in µM. No mixing with absolute units (pA, nS) anywhere. If a literature source uses different units, convert at the citation point and document the conversion.

8. **Skill files first.** Before creating Word/Excel/PowerPoint/PDF outputs or writing significant code, check `/mnt/skills/public/` for relevant skills. Read them before proceeding.

---

## 1. Project context (compressed)

The JNE-110355 manuscript is a methods paper presenting a JAX/GPU framework for spiking-network simulation of the basal ganglia, with CMA-ES optimization to fit electrophysiological targets. Reviewers requested major revisions. A codebase audit found the saved artifacts inconsistent with the manuscript on several headline numbers, undocumented scaling factors in the integrator, and a mismatch between the optimization target (GPe beta) and the reported result (STN beta). The rebuild scope decision was to rebuild the pipeline cleanly, run it honestly, and report what the framework actually produces.

The paper's contribution is **methodological**: a GPU-accelerated, optimization-driven framework for basal ganglia network research. The STN-GPe-GPi healthy/PD demonstration is a *worked example* of the framework operating under one biological hypothesis (STN-GPe loop generates beta), not a novel biological claim about beta generation.

Targets are **all primate, all MPTP**, anchored primarily to Tachibana et al. (2014). Beta band is **8–15 Hz**. Beta is treated as a **constraint** in optimization, not an objective term. Six parameters (4 synaptic conductances, 3 tonic drives, 3 OU means, 3 OU sigmas — total 13) are searched in absolute biophysical units; no multipliers on baselines.

Read `rebuild_scope.md` for full justification of each decision.

---

## 2. Repository layout

The rebuild lives in a new package called `bgnet/` that replaces the existing `jax_models/`, `optimization/`, and `numpy_baseline/` packages. The old packages move to `legacy/` for reference but are not imported.

```
project_root/
├── AGENTS.md                  # This file
├── rebuild_scope.md           # Why-document
├── README.md                  # User-facing, written last
├── pyproject.toml             # Modern Python packaging, not requirements.txt
├── bgnet/
│   ├── __init__.py
│   ├── neurons/
│   │   ├── __init__.py
│   │   ├── stn.py             # Single-compartment STN (Gillies-Willshaw inspired)
│   │   └── pallidum.py        # Rubin-Terman GPe and GPi
│   ├── synapses.py            # Double-exponential conductance-based, no scaling factors
│   ├── noise.py               # Ornstein-Uhlenbeck, mu and sigma both free
│   ├── connectivity.py        # Fixed-indegree
│   ├── integrator.py          # Forward Euler, single jax.lax.scan loop
│   ├── network.py             # Build, simulate, return spikes/Vm/etc.
│   ├── observables.py         # Firing rates, CV, LFP proxies, beta fraction
│   ├── lfp.py                 # Three LFP proxies: pop rate (primary), Vm, syn currents
│   ├── objective.py           # Loss + constraints for optimizer
│   ├── optimize.py            # CMA-ES via Optuna, constraint handling
│   └── config.py              # YAML loader, validation, dataclasses
├── configs/
│   ├── base_network.yaml      # Network sizes, dt, duration — shared
│   ├── healthy.yaml           # Healthy optimization config
│   ├── pd_asymmetric.yaml     # PD with literature-motivated asymmetric bounds
│   ├── pd_symmetric.yaml      # PD with symmetric bounds
│   ├── parameter_recovery.yaml
│   ├── convergence_study.yaml
│   ├── loss_sensitivity.yaml
│   ├── lfp_proxy_comparison.yaml
│   ├── timestep_sensitivity.yaml
│   ├── sampler_comparison.yaml
│   ├── scaling_validation.yaml
│   ├── statistical_validation.yaml
│   └── dbs_simulation.yaml
├── scripts/
│   ├── 01_run_healthy_optimization.py
│   ├── 02_run_pd_optimization.py        # Takes config arg for asymmetric/symmetric
│   ├── 03_validate_top_configs.py       # 10-seed validation, robustness selection
│   ├── 04_run_scaling_validation.py
│   ├── 05_run_statistical_validation.py
│   ├── 06_run_dbs_simulation.py
│   ├── 07_run_benchmarks.py
│   ├── 08_run_sampler_comparison.py
│   ├── 09_run_timestep_sensitivity.py
│   ├── 10_run_loss_weight_sensitivity.py
│   ├── 11_run_lfp_proxy_comparison.py
│   ├── 12_run_parameter_recovery.py
│   ├── 13_run_convergence_characterization.py
│   └── 14_generate_figures.py
├── tests/
│   ├── unit/
│   │   ├── test_stn_neuron.py
│   │   ├── test_pallidum_neuron.py
│   │   ├── test_synapses.py
│   │   ├── test_ou_noise.py
│   │   ├── test_connectivity.py
│   │   └── test_observables.py
│   ├── integration/
│   │   ├── test_short_simulation.py
│   │   └── test_optimization_smoke.py
│   └── conftest.py
├── results/                   # Generated; gitignored except for headline runs
│   └── <study>/<timestamp>/{config.yaml, log.txt, results.pkl, figures/}
├── legacy/                    # Old code, read-only reference
│   ├── jax_models/
│   ├── optimization/
│   └── numpy_baseline/
└── docs/
    ├── design_decisions.md    # Living document of choices and rationale
    └── reproduction_guide.md  # How to reproduce paper numbers from scratch
```

Keep this layout. Do not add top-level scripts outside `scripts/`. Do not put configs anywhere else.

---

## 3. Phase plan

Six phases. Each has explicit entry conditions, deliverables, and exit conditions. Do not start phase N+1 until phase N's exit conditions are met.

### Phase 1: Foundation rebuild (Week 1)

**Goal:** Working `bgnet/` package with neurons, synapses, noise, connectivity, integrator. Tests passing. No optimization yet.

**Entry conditions:** None. Start here.

**Steps:**

1. **Set up the repository skeleton.** Create the layout in §2. Move existing `jax_models/`, `optimization/`, `numpy_baseline/` to `legacy/`. Set up `pyproject.toml` with `bgnet` as the package name. Dependencies: `jax`, `jaxlib`, `optuna`, `cma`, `numpy`, `scipy`, `pyyaml`, `matplotlib`, `pytest`. Add `pytest`, `ruff` as dev dependencies.

2. **Implement STN neuron** (`bgnet/neurons/stn.py`). Single-compartment Hodgkin-Huxley from Terman, Rubin, Yew & Wilson (2002), parameters from the canonical `episodic.ode` source (ModelDB 182758). Currents: I_L, I_Na, I_K, I_AHP, I_Ca, I_T (the canonical RT 2002 set — no I_H, no separate I_CaH; m, s, and a are instantaneous). Membrane equation: `C_m dV/dt = -(I_L + I_Na + I_K + I_AHP + I_Ca + I_T) - I_syn + I_drive + I_noise`. C_m = 1.0 µF/cm². Spike detection: upward crossing of 0 mV with 2 ms minimum ISI. All conductances and currents in current-density units. Validation in `docs/stn_validation.md`. Note: the original Phase 1 plan called for a Gillies–Willshaw-inspired STN; Phase 1.5 replaced it with the RT 2002 model after the GW f-I curve proved silent at the optimizer's drive bounds (see `docs/phase1_review.md`).

3. **Implement Pallidum neurons** (`bgnet/neurons/pallidum.py`). Rubin-Terman formalism (Rubin & Terman, 2004). Single function with parameters that differ between GPe and GPi (g_T, g_AHP, baseline I_app). Currents: I_Na, I_K, I_L, I_T, I_Ca, I_AHP. Spike detection: upward crossing of -20 mV. Reference values from Ebert et al. (2014) Tables 1–2.

4. **Implement synapses** (`bgnet/synapses.py`). Double-exponential conductance-based. `I_syn = g_max * s * (E_syn - V_post)`. AMPA (E_syn = 0 mV, τ_rise = 1 ms, τ_decay = 3 ms) and GABA_A (E_syn = -70 mV, τ_rise = 5 ms, τ_decay = 8 ms). NO SCALING FACTORS. The synaptic conductances are absolute values searched by the optimizer.

5. **Implement OU noise** (`bgnet/noise.py`). Euler-Maruyama integration. τ = 5 ms. Both µ and σ are parameters. Each neuron gets independent noise. Document why OU vs. white/Poisson: temporally correlated noise approximates the integration of many independent afferent spikes; τ = 5 ms is short relative to neuronal time constants but long enough to avoid white-noise artifacts (cite Destexhe et al. 2001 or similar).

6. **Implement connectivity** (`bgnet/connectivity.py`). Fixed-indegree only. K_STN→GPe = 15, K_GPe→STN = 14, K_STN→GPi = 30, K_GPe→GPi = 10. For each postsynaptic neuron, sample exactly K presynaptic neurons uniformly without replacement. Store as sparse COO. Verify exact indegree at all network sizes.

7. **Implement integrator** (`bgnet/integrator.py`). Forward Euler, dt = 0.025 ms. Single `jax.lax.scan` loop over timesteps. JIT compiled. Returns full state trajectories (Vm, spike times, optionally synaptic currents) for the requested duration. NO scaling factors anywhere in this file. If you find yourself wanting to add one, stop and audit the unit conventions.

8. **Implement network builder** (`bgnet/network.py`). Takes a config object, returns a callable that runs the simulation. Network sizes: STN:GPe:GPi in 2:4:3 ratio. Default optimization size: 100/200/150 = 450 neurons. Validation size: 10000/20000/15000 = 45000 neurons.

9. **Write unit tests for each module.** For each neuron type: test that an isolated neuron with sufficient drive produces spikes at expected rate (within 10% tolerance). For synapses: test that a single presynaptic spike produces a postsynaptic conductance waveform matching the analytical double-exponential. For OU: test that long-time mean and variance match µ and σ². For connectivity: test that every postsynaptic neuron has exactly K inputs at network sizes 450, 4500, 45000. For integrator: test that a dt=0.025 simulation matches dt=0.0125 within 2 Hz on firing rates.

10. **Write a smoke test** for the integrated simulation: 100 ms, 450 neurons, default parameters, expect non-zero firing in all populations and no NaN values.

**Exit conditions:**
- All unit tests pass.
- Smoke test produces sensible output (non-zero firing in all populations, no NaN, consistent across two seeds with different OU realizations).
- A 600 ms simulation at 450 neurons completes in under 5 seconds on the L4 GPU.
- The `bgnet` package can be imported cleanly with no warnings.

**Decision point at end of phase:** None automatically; surface anything that surprises you.

---

### Phase 2: Optimization scaffolding (Week 2)

**Goal:** Working `bgnet/optimize.py` with CMA-ES via Optuna, constraint handling, YAML config loading. Optimization scripts (01-02) runnable end-to-end on tiny configs.

**Entry conditions:** Phase 1 exit conditions met.

**Steps:**

1. **Implement observables** (`bgnet/observables.py`). Firing rate per population, CV of ISI per neuron averaged over population, LFP proxies (population firing rate primary, mean Vm and synaptic-current sum secondary). Beta fraction = integrated power in [8, 15] Hz / integrated power in [1, 100] Hz, after DC removal. Use Welch's method (scipy.signal.welch) with Hann window, 50% overlap, segment length min(signal length, 8192).

2. **Implement objective function** (`bgnet/objective.py`). Weighted-sum loss for rate and CV terms. Beta as a constraint, not a loss term. Loss formulation:
   ```
   loss = w_rate * sum(((rate_i - target_rate_i) / target_rate_i)**2 for i in [STN, GPe, GPi])
        + w_cv * sum((cv_i - target_cv_i)**2 for i in [STN, GPe, GPi])
   ```
   with default `w_rate = 1.0, w_cv = 0.2` (CV soft per scope decision). Constraint:
   ```
   c_beta = target_beta_threshold - actual_beta  # negative means satisfied
   ```
   Healthy: target_beta_threshold = 0.05 (so c_beta ≤ 0 means STN beta < 0.05).
   PD: target_beta_threshold = -(0.15 - actual_beta) so positive means satisfied. Wait — this needs care; see Optuna constraint conventions: Optuna treats constraint values ≤ 0 as feasible. So:
   - Healthy: `c_beta = actual_beta - 0.05` (≤ 0 means actual < 0.05, feasible).
   - PD: `c_beta = 0.15 - actual_beta` (≤ 0 means actual > 0.15, feasible).
   Document this in a docstring with explicit sign conventions.

3. **Implement Optuna driver** (`bgnet/optimize.py`). Use `optuna.samplers.CmaEsSampler` with `consider_pruned_trials=False`. Constraints handled via `set_user_attr("constraint", (c_beta,))` and `CmaEsSampler` configured with `constraints_func`. If the constraint formulation proves intractable in early testing (excessive infeasible trials, no convergence), fall back to one-sided threshold-with-bonus (penalty proportional to `max(0, threshold - beta)` added to loss, no penalty for exceeding threshold). Document which formulation was used in the run config.

4. **Implement config loader** (`bgnet/config.py`). YAML loader that produces typed dataclasses. Validate at load time: ranges checked, required fields present, units commented in YAML. Each config records the package version (`bgnet.__version__`) and gets copied verbatim into the run's results directory.

5. **Implement results manager.** Each script call produces a directory at `results/<study_name>/<YYYYMMDD_HHMMSS>/`. Contents:
   - `config.yaml` — verbatim copy of the input config
   - `log.txt` — structured logging output
   - `optuna_study.db` — full Optuna study (SQLite)
   - `results.pkl` — best params, best loss, all trial metadata
   - `metadata.json` — git SHA, package version, GPU info, host info, wall-clock time
   - `figures/` — any figures generated during the run

6. **Write the healthy optimization script** (`scripts/01_run_healthy_optimization.py`). Loads `configs/healthy.yaml`. Runs the optimization. Saves results. Logs progress. Should be a thin wrapper around `bgnet.optimize.run_study()`.

7. **Write the PD optimization script** (`scripts/02_run_pd_optimization.py`). Takes a config path as a CLI argument so the same script handles asymmetric and symmetric variants.

8. **Smoke test the scripts.** Create a tiny config: 100 trials, 450 neurons, 200 ms simulation. Run each script. Verify the results directory contains everything specified in step 5 and that the loaded study matches the saved one.

**Exit conditions:**
- All Phase 1 tests still pass.
- Smoke runs of scripts 01 and 02 produce valid results directories.
- Constraint handling works: trials with violated constraint are correctly marked infeasible by Optuna.
- A run can be fully reproduced from its saved config + same git SHA + same hardware.

**Decision points:**
- After the first real-sized smoke test (~50 trials), report whether the constraint formulation appears to be working or whether fallback to threshold-with-bonus is needed. Surface to user.

---

### Phase 3: Healthy and PD optimization (Week 2–3)

**Goal:** Three completed optimization runs (healthy, PD asymmetric, PD symmetric) with full validation across 10 seeds at 45,000 neurons.

**Entry conditions:** Phase 2 exit conditions met. Constraint handling confirmed working (or fallback locked in).

**Steps:**

1. **Finalize healthy config.** `configs/healthy.yaml`:
   - Network: 100/200/150 = 450 neurons
   - Simulation: 400 ms with 100 ms burn-in
   - Trials: 1500 (headline run)
   - Targets: STN 20 Hz, GPe 65 Hz, GPi 67 Hz; CV 0.4/0.35/0.20; beta < 0.05 in 8-15 Hz
   - Source citations: Tachibana et al. 2014 (rates), Brown 2003 / Kühn et al. 2006 / Mallet et al. 2008 (band)
   - Search bounds (all symmetric for healthy):
     - Synaptic conductances: 0.005–0.5 mS/cm² each (4 params)
     - Tonic drives: -5 to +5 µA/cm² each (3 params)
     - OU means: -5 to +5 µA/cm² each (3 params)
     - OU sigmas: 0 to 5 µA/cm² each (3 params)
   - CMA-ES seed: 42 (headline)
   - Network seed during optimization: 42 (fixed)
   - OU seed: fresh per trial

2. **Run the healthy optimization.** Script 01. Wall-clock should be ~30–45 minutes. Monitor convergence: log best loss every 50 trials. If best loss stalls before trial 500, surface to user — may indicate model or config bug.

3. **Validate top healthy configurations.** Script 03. Take the top 10 trials (by loss, among feasible). Re-simulate each at 45,000 neurons across 10 seeds. Compute mean and SD of all metrics. Select the configuration with the lowest mean loss whose metrics have the smallest seed-to-seed SD (a tiebreaker formula like `mean_loss + 0.5 * std(rate_metrics)` is reasonable; document whatever you choose). This selected configuration is the **headline healthy result**.

4. **Finalize PD asymmetric config.** `configs/pd_asymmetric.yaml`. Same structure as healthy except:
   - Targets: STN 27 Hz, GPe 41 Hz, GPi 63 Hz; CV 0.6/0.40/0.30; beta > 0.15 in 8-15 Hz
   - Asymmetric synaptic bounds reflecting literature priors:
     - g_STN→GPe ∈ [0.05, 0.5] (biased toward higher excitation)
     - g_GPe→STN ∈ [0.005, 0.1] (biased toward lower inhibition)
     - g_STN→GPi ∈ [0.05, 0.5]
     - g_GPe→GPi ∈ [0.01, 0.3]
   - Other bounds same as healthy
   - Document asymmetric-bounds rationale in YAML comments with citations (Albin et al. 1989; Tachibana et al. 2014; Bergman et al. 1994).

5. **Finalize PD symmetric config.** `configs/pd_symmetric.yaml`. Identical to PD asymmetric except synaptic conductance bounds are 0.005–0.5 mS/cm² for all four pathways (same as healthy). This is the "no prior" run that addresses Reviewer 1's "known directions" critique.

6. **Run PD asymmetric optimization.** Script 02 with `configs/pd_asymmetric.yaml`.

7. **Run PD symmetric optimization.** Script 02 with `configs/pd_symmetric.yaml`.

8. **Validate top PD configurations** for both bound conditions. Same procedure as step 3. Two headline PD results.

9. **Generate comparison summary.** Save a markdown table in `results/comparison_summary.md` with:
   - Healthy headline configuration (parameters, achieved metrics, seed-to-seed variability)
   - PD asymmetric headline configuration (same)
   - PD symmetric headline configuration (same)
   - PD/Healthy ratios for synaptic conductances under both bound conditions

10. **Compare PD asymmetric vs. PD symmetric.** Are the discovered configurations qualitatively similar? Specifically: do both runs find g_STN→GPe higher and g_GPe→STN lower than healthy? Document the comparison. This is the answer to Reviewer 1 #4.

**Exit conditions:**
- Three complete optimization studies, each with feasible best trials and validated headline configurations.
- All three headline configurations satisfy their constraints across at least 8 of 10 validation seeds.
- The comparison summary is written.

**Decision points:**
- **After step 3:** present the healthy headline configuration to user. Confirm it looks biologically reasonable (rates within ~15% of targets, beta < 0.05, no extreme parameter values pinned to bound edges) before proceeding to PD.
- **After step 8:** present both PD headline configurations to user. Confirm before proceeding to validations. If asymmetric and symmetric disagree qualitatively, this is a finding — surface and discuss before proceeding.

---

### Phase 4: Validation suite (Week 3–4)

**Goal:** All B1 (carry-over), B2 (reviewer-driven), and B3 (above-and-beyond) validations completed and figures generated.

**Entry conditions:** Phase 3 exit conditions met. Three headline configurations locked.

**Steps:**

1. **Scaling validation** (script 04). Apply each headline configuration to networks of 450, 1800, 4500, 15000, 45000 neurons. 600 ms simulation, 100 ms burn-in. Verify firing rate variation < 3 Hz across sizes; verify beta condition (< 0.05 healthy, > 0.15 PD) holds across sizes. Save table.

2. **Statistical validation** (script 05). For each headline configuration at 45000 neurons, run 10 seeds. Independent t-tests between healthy and each PD condition for all metrics. Save table with mean ± SD and p-values.

3. **DBS qualitative sanity check** (script 06). Take the PD asymmetric configuration. Apply DBS as in original paper: increase I_STN substantially, further reduce GPe→STN coupling. 600 ms at 45000 neurons. Verify beta suppression and GPi rate normalization. Frame in script docstring as "qualitative sanity check, not a DBS contribution."

4. **Performance benchmarks** (script 07). Wall-clock vs. network size: 450, 1800, 4500, 15000, 45000 neurons, 600 ms simulation, 5 repeats each. Compare against a simple NumPy single-threaded baseline at 450 neurons (this is what the speedup claim should reference; do NOT compare against the legacy `numpy_baseline/` package which uses different models). The "X-fold speedup" claim must be honest: same models, same parameters, same network, JAX-on-GPU vs NumPy-on-CPU. Document this clearly.

5. **Sampler comparison** (script 08). 5 seeds × 500 trials × 3 samplers (CMA-ES, TPE, Random) on the healthy objective. Mean ± SD of best loss for each sampler. This addresses one of the original paper's claims and Reviewer 2's concern about CMA-ES specifics.

6. **Timestep sensitivity** (script 09). Headline configurations simulated at dt = 0.025, 0.0125, 0.00625 ms at 45000 neurons. Report firing rates and beta fractions at each dt. The original paper hid divergence here; the rebuild reports honestly. If beta fraction varies by more than 5 percentage points across timesteps, this is a real finding to report.

7. **Loss-weight sensitivity** (script 10). Re-run PD asymmetric optimization with weight grid: w_rate ∈ {0.5, 1.0, 2.0}, w_cv ∈ {0.05, 0.2, 0.5}. Nine configurations. Reduced to 500 trials each (sweep mode, not headline). Report how the discovered configuration shifts. This directly addresses Reviewer 2 #3.

8. **LFP proxy comparison** (script 11). Re-analyze the headline simulations using all three LFP proxies (population firing rate, mean Vm, synaptic current sum). Report beta fraction under each. Show that qualitative result (PD beta enhancement) is robust to proxy choice; report any quantitative differences.

9. **Parameter recovery** (script 12, B3a). Take the PD asymmetric headline configuration. Generate "synthetic ground truth" data from it (10 seeds, 45000 neurons). Compute target metrics (rates, CV, beta) from synthetic data. Re-run the PD asymmetric optimization with these synthetic-data-derived targets. Does the optimizer recover the original parameters? Report parameter-by-parameter recovery error. This is a strong methodological contribution — the framework should be able to recover known ground truth.

10. **Convergence characterization** (script 13, B3b). 10 independent CMA-ES runs with different sampler seeds, all with the PD asymmetric config (1000 trials each). Report distribution of discovered configurations: tight clustering = framework reliable, scattered = downstream concern. Compute pairwise distances in parameter space, visualize as a heatmap or PCA.

**Exit conditions:**
- All ten scripts run successfully and produce results directories.
- All planned tables and headline numbers exist.
- Performance benchmark's speedup claim is documented and defensible (same-model comparison).

**Decision points:**
- **After step 6 (timestep sensitivity):** if beta varies dramatically with dt, surface to user. May indicate need for finer dt as default.
- **After step 9 (parameter recovery):** if recovery error is large (> 30% on key parameters), the framework has an identifiability problem — surface to user before continuing.
- **After step 10 (convergence):** if discovered configurations scatter widely across runs, surface to user before continuing.

---

### Phase 5: Figures and tables (Week 4–5)

**Goal:** All figures regenerated from rebuild data. Consistent style. All abbreviations defined. Tables with full author lists in references.

**Entry conditions:** Phase 4 exit conditions met. All headline numbers known.

**Steps:**

1. **Read the matplotlib skill** at `/mnt/skills/public/` if any exists, and check for relevant figure-making conventions.

2. **Define the visual style.** Single style file at `bgnet/plotting.py`. Colors: blue for healthy, red for PD, green for DBS. Font: sans-serif, size 9-10 pt. Line widths consistent. All units on axes. All abbreviations in caption.

3. **Generate figures from saved results.** Script 14. Each figure regenerates from results directories — no hand-edited intermediate data. The script must be re-runnable without intermediate manual steps.

   Figure list (numbering may shift as paper structure firms up):
   - F1: Computational workflow diagram (architectural; can be hand-drawn but reproducible from script)
   - F2: Cloud infrastructure / accessibility metrics (similar)
   - F3: Raster plots, healthy vs. PD asymmetric, all three populations
   - F4: STN LFP proxy traces, healthy vs. PD
   - F5: Power spectral density, STN, healthy vs. PD; (B) PD with DBS
   - F6: Bar charts of firing rates and STN beta fraction
   - F7: Synaptic configuration schematic with discovered ratios for both PD conditions
   - F8: Statistical validation across 10 seeds
   - F9: Scaling validation
   - F10 (new): Parameter recovery plot — discovered vs. true parameters
   - F11 (new): Convergence characterization — distribution of discovered configurations across CMA-ES seeds
   - F12 (new): LFP proxy comparison
   - F13 (new): Loss-weight sensitivity heatmap
   - F14 (new): Bound sensitivity comparison (asymmetric vs. symmetric PD)

4. **Generate tables.**
   - T1a, T1b: Neuron parameters (STN; GPe/GPi)
   - T2: Synaptic parameters (now in absolute units, no multipliers)
   - T3: Discovered parameters (healthy, PD asymmetric, PD symmetric) and PD/healthy ratios
   - T4: Scaling validation
   - T5: Statistical validation
   - T6: DBS results

5. **Update references** to comply with editor instructions: full author lists for ≤10 authors, "et al." only for >10 authors. Audit every reference in the bibliography.

6. **Audit captions.** Every abbreviation defined on first use in each caption. Every panel labeled. Every axis labeled with units.

**Exit conditions:**
- Script 14 regenerates all figures from saved results in one command.
- All figures, tables, and captions reviewed.
- Reference list audited for author-list compliance.

---

### Phase 6: Manuscript revision and response (Week 5–6)

**Goal:** Revised manuscript, response-to-reviewers document, all submission files.

**Entry conditions:** Phase 5 exit conditions met.

**Steps:**

1. **Read the relevant skill** at `/mnt/skills/public/docx/SKILL.md` before producing Word output.

2. **Write the response-to-reviewers document.** Address every reviewer point individually. For each point: (a) quote the reviewer, (b) describe the change made, (c) point to the manuscript section/figure/table where the change appears. Be direct. Where we disagree with a reviewer, say so with reasoning, but mostly we are agreeing and showing what we did.

3. **Reframe Section 1 (Introduction).** Methodological framing throughout. The framework is the contribution. STN-GPe-GPi is a worked example. Cortex/striatum exclusion is a deliberate scope choice for the demonstration, not a model limitation.

4. **Rewrite Section 2 (Methods).** Explicit definitions of all parameters including I_drive (Reviewer 1 #2a) and OU mean (Reviewer 1 #2b). Optimization procedure described in detail: CMA-ES initialization (mean = midpoint of bounds, initial sigma per parameter, population size from default), constraint handling (Optuna `constraints_func` with sign convention), stopping criterion (1500 trials), seed handling (single seed during search, 10-seed validation at end), parameter list (all 13 with bounds in a table). Address Reviewer 2 #2 directly.

5. **Update Section 3 (Results).** All numbers from the rebuild. Acknowledge if any qualitative conclusion has changed (e.g., if symmetric PD finds the same asymmetric reorganization, or if it doesn't). Report the bound-sensitivity comparison as a Results subsection. Report parameter recovery and convergence as Results subsections. Beta peak frequency reported correctly throughout (Reviewer 1 #1).

6. **Rewrite Section 4 (Discussion).** Methodological contribution emphasized. Biological interpretation phrased as "the framework operating under hypothesis X discovered configuration Y," not "we discovered the mechanism of beta." GPi controversy acknowledged with citations to Tachibana 2014, Wichmann et al. 2011, Raz et al. 2000, Rivlin-Etzion et al. 2008. Limitations section expanded per Reviewer 2 #4 — fixed-indegree connectivity discussed explicitly, with citation to Gerstner et al. 2014 for justification and acknowledgment of anatomical-specificity simplifications.

7. **Update Use of Generative AI statement** to reflect the rebuild process.

8. **Compile the final manuscript** as docx using the docx skill. Single source file.

9. **Compile the highlighted PDF** (changes from original highlighted) per the revision checklist.

10. **Compile the clean PDF** (no highlights) per the revision checklist.

11. **Verify submission requirements** against the revision checklist:
    - Author response (anonymized if double-anonymous was selected)
    - Highlighted PDF (anonymized, designation "Complete Document for Review (PDF Only)")
    - Source file (clean Word file with full author list, corresponding author marked, funding info, ethics statement, editable tables/figure captions/equations, no colored text in tables)
    - Additional source files (high-res images, designation "Source Files")
    - Clean PDF (designation "Source Files")
    - Supplementary material (with title and description, designation "Supplementary Data Files")

**Exit conditions:**
- Manuscript is complete, internally consistent, and addresses every reviewer point.
- Response document covers every point with reference to the change.
- All submission files prepared per the revision checklist.

**Decision points:**
- After step 6 (Discussion rewrite): present full Discussion to user for review before finalizing.
- Before step 11 (final compilation): present full revision package to user.

---

## 4. Reference values

These values are referenced throughout the rebuild. Single source of truth.

### 4.1 Firing rate targets (primate MPTP, Tachibana et al. 2014 unless noted)

| Population | Healthy (Hz) | PD (Hz) |
|---|---|---|
| STN | 20 | 27 |
| GPe | 65 | 41 |
| GPi | 67 | 63 |

GPi target of 63 Hz is the more recent careful primate finding. Older work (Filion & Tremblay 1991; Boraud et al. 2002) reports ~80 Hz; this controversy is acknowledged in the manuscript Discussion but no separate sensitivity run is performed.

### 4.2 CV targets (approximate, soft constraints)

| Population | Healthy | PD |
|---|---|---|
| STN | 0.4 | 0.6 |
| GPe | 0.35 | 0.40 |
| GPi | 0.20 | 0.30 |

Order-of-magnitude only. CV gets w_cv = 0.2 in the objective.

### 4.3 Beta band

**13–30 Hz** ("PD beta"). Citation: Brown 2003; Kühn et al. 2006; Mallet et al. 2008.

The choice is based on the Phase 2 / Phase 3 entry sanity check
(`docs/network_beta_sanity_check.md`): under literature-canonical
synaptic time constants and delays from the Terman-Rubin lineage, the
STN-GPe loop spontaneously oscillates at 15–25 Hz (peak 19.5 Hz in the
hand-tuned strong-coupling test). 13–30 Hz aligns with the broader PD
beta literature (the human-clinical convention) and with the natural
oscillation frequency of the model lineage; forcing the model into the
narrower 8–15 Hz primate "low-beta" band would require non-canonical
synaptic time constants or delays — exactly the kind of model tuning
this rebuild avoids. The all-primate firing-rate targets (Tachibana
et al. 2014, §4.1) are unaffected; only the band-frequency choice
reflects the model's intrinsic dynamics under canonical parameters.

Constraint thresholds (unchanged):
- Healthy: STN beta fraction < 0.05
- PD: STN beta fraction > 0.15

### 4.4 Search bounds

| Parameter group | Bound |
|---|---|
| Synaptic conductances (mS/cm²) | 0.005 to 0.5 (symmetric); asymmetric variants for PD |
| Tonic drives I_pop (µA/cm²) | -5 to +5 |
| OU means µ_pop (µA/cm²) | -5 to +5 |
| OU sigmas σ_pop (µA/cm²) | 0 to 5 |

PD asymmetric variant: g_STN→GPe ∈ [0.05, 0.5], g_GPe→STN ∈ [0.005, 0.1], g_STN→GPi ∈ [0.05, 0.5], g_GPe→GPi ∈ [0.01, 0.3].

### 4.5 Connectivity

Fixed-indegree:
- K_STN→GPe = 15
- K_GPe→STN = 14
- K_STN→GPi = 30
- K_GPe→GPi = 10

Network sizes:
- Optimization: 100 STN / 200 GPe / 150 GPi = 450
- Validation: 10000 STN / 20000 GPe / 15000 GPi = 45000
- Ratio always 2:4:3

### 4.6 Synaptic time constants and reversal potentials

| Type | E_syn (mV) | τ_rise (ms) | τ_decay (ms) | Delay (ms) |
|---|---|---|---|---|
| AMPA (STN→GPe, STN→GPi) | 0 | 1 | 3 | 5 |
| GABA_A (GPe→STN) | -70 | 5 | 8 | 8 |
| GABA_A (GPe→GPi) | -70 | 5 | 8 | 5 |

### 4.7 Numerical

- dt = 0.025 ms (standard)
- Integration: forward Euler
- Spike detection: STN threshold 0 mV with 2 ms min ISI; GPe/GPi threshold -20 mV
- Burn-in: 100 ms
- Optimization simulation duration: 400 ms (post burn-in)
- Validation simulation duration: 600 ms (post burn-in)

### 4.8 Optimization

- Sampler: CMA-ES via Optuna's `CmaEsSampler`
- Trials: 1000 standard, 1500 headline
- Population size: Optuna default (4 + floor(3 * ln(n_dim))) ≈ 11 for n=13
- Initial mean: midpoint of bounds for each parameter
- Initial sigma: 1/4 of bound range for each parameter
- Constraint handling: Optuna `constraints_func`, sign convention "≤ 0 means feasible"
- Fallback if constraint formulation intractable: one-sided threshold-with-bonus added to loss

---

## 5. Citations and references (for code comments and config files)

When you reference biological values in code comments or YAML configs, use these citation strings consistently:

- Tachibana et al. 2014: `# Tachibana et al. 2014, Front Syst Neurosci 8:74`
- Bergman et al. 1994: `# Bergman et al. 1994, J Neurophysiol 72(2):507-520`
- Stein & Bar-Gad 2013: `# Stein & Bar-Gad 2013, Exp Neurol 245:52-59`
- Devergnas et al. 2014: `# Devergnas et al. 2014, Neurobiol Dis 68:156-166`
- Brown 2003: `# Brown 2003, Mov Disord 18(4):357-363`
- Kühn et al. 2006: `# Kühn et al. 2006, Eur J Neurosci 23:1956-1960`
- Mallet et al. 2008: `# Mallet et al. 2008, J Neurosci 28(18):4795-4806`
- Rubin & Terman 2004: `# Rubin & Terman 2004, J Comput Neurosci 16(3):211-235`
- Terman et al. 2002: `# Terman et al. 2002, J Neurosci 22:2963-2976`
- Gillies & Willshaw 2006: `# Gillies & Willshaw 2006, J Neurophysiol 95(4):2352-2365` (considered but not used; see Phase 1.5)
- Ebert et al. 2014: `# Ebert et al. 2014, Front Comput Neurosci 8:154`
- Gerstner et al. 2014: `# Gerstner et al. 2014, Neuronal Dynamics, Cambridge UP, Ch 12.3`
- Hansen 2016: `# Hansen 2016, arXiv:1604.00772 (CMA-ES tutorial)`
- Albin et al. 1989: `# Albin et al. 1989, Trends Neurosci 12(10):366-375`

Full citations are in `rebuild_scope.md` references section.

---

## 6. Things to never do

- Never add a scaling factor to compensate for unit mismatch. Audit units instead.
- Never report numbers that don't trace back to a specific run. Every value in figures/tables comes from a results directory.
- Never tune parameters by hand to match the original paper. The original paper's numbers are not the goal.
- Never use the legacy `numpy_baseline/` for the speedup claim. Build a fresh CPU baseline using the same models.
- Never optimize GPe beta and report STN beta. Optimize STN beta, report STN beta.
- Never call the constraint formulation "working" without verifying that infeasible trials are correctly rejected by the optimizer.
- Never run an optimization without saving the config alongside it.
- Never claim a finding from a single seed when 10-seed validation is available.
- Never use "et al." in references for ≤10 authors (per editor instruction).
- Never use bounded value targets for beta (the original formulation that caused over-prioritization).

---

## 7. Things to always do

- Read `rebuild_scope.md` if you have not seen this project before.
- Check skills (`/mnt/skills/public/`) before producing files.
- Run unit tests before optimizations.
- Save full configs alongside results.
- Log structured output, not prints.
- Document every numeric constant with a citation if it has one.
- Surface decision points to the user when something is genuinely ambiguous.
- Commit at clean checkpoints with clear messages.
- When something surprises you, stop and verify rather than push through.

---

## 8. Troubleshooting

If you hit problems:

- **Optimization not converging:** check whether the objective function is correctly normalized (rate errors as relative deviation, not absolute). Check whether constraints are being correctly evaluated by sampling 10 random parameter vectors and printing constraint values.
- **Beta fraction implausibly high or low:** check the LFP proxy. The default is population firing rate; if you see Vm-style values, the proxy code is wrong.
- **NaN values during simulation:** voltage clamping in neuron models should prevent this. If you see NaN, integrate dt = 0.0125 ms and see if it persists. Voltage clamping range: [-100, 60] mV. Gating variables clamped to [0, 1].
- **Different runs of same config produce different best parameters:** this is expected with stochastic objective. The 10-seed validation step exists to handle this. Don't try to make optimization deterministic.
- **Results directory missing files:** the results manager should be defensive about saving. If `metadata.json` or `config.yaml` are missing, the saving logic has a bug — fix at source, don't paper over.
- **Speedup claim seems off:** the comparison must be JAX-on-GPU vs same-model NumPy-on-CPU, not vs the legacy `numpy_baseline/` package. If wall-clock numbers seem off, audit the comparison setup.

For genuinely novel problems, surface to user.

---

End of AGENTS.md.
