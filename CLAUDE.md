# CLAUDE.md — CBGTC Project Context

## What This Project Is

A JAX-accelerated biophysical model of the STN-GPe-GPi basal ganglia circuit for studying pathological beta oscillations in Parkinson's disease. The code accompanies a manuscript by Kavin Nakkeeran (Johns Hopkins, Functional Neurosurgery Lab) currently in final revision with PI Dr. Anderson (Stan).

### Key Scientific Finding
An 82% reduction in GPe→STN inhibitory conductance (multiplier: 0.182) is necessary and sufficient to transition the network from healthy (~0% beta power) to Parkinsonian (~30% beta power in GPe) oscillatory states. This is consistent with dopamine-dependent plasticity at the GPe→STN synapse.

---

## Critical Context: Codebase Is Ground Truth

The manuscript was drafted based on rough methodology. **All numbers, figures, results, and analyses need to be re-generated from scratch with a full paper trail.** The codebase defines what the model actually does; the manuscript will be updated to match.

### Two-Stage Workflow
1. **Optimization** on 450-neuron model (100 STN, 200 GPe, 150 GPi) — fast parameter search
2. **Validation/Results** on 1800-neuron model (400 STN, 800 GPe, 600 GPi) — publication-quality results

The synaptic weight scaling formula (`g_max * N_ref / N_actual`, where reference is 10/20/15) ensures consistent total synaptic drive across model sizes. Parameters found at 450 neurons should produce the same qualitative dynamics at 1800.

---

## Current Task: Full Cleanup & Re-run

See SCOPE.md for the complete plan. Summary of phases:
1. Codebase cleanup (remove cruft, reorganize, clean AdEx refs)
2. Write new scripts (timestep sensitivity, CMA-ES vs TPE, benchmarks)
3. Re-run everything on GCP with metadata
4. Generate all figures from fresh results
5. Update manuscript to match

---

## Key Decisions (Finalized)

### DBS Implementation
The paper will describe what the code actually does: an "informational lesion" approach where DBS is modeled by increasing ISTN (80→150) and further reducing g_gpe_stn_mult (0.182→0.05). This represents DBS disrupting the STN-GPe feedback loop. The `create_dbs_simulator` function (130 Hz pulse injection) exists but is NOT used for the paper results. It can remain in the codebase as an alternative but should be clearly labeled.

### Beta Power Computation
**Switch from plain FFT to Welch's method.** Current code uses a simple FFT on ~300ms of post-burn-in data, which gives coarse frequency resolution. Update `metrics_jax.py` to use `scipy.signal.welch` for PSD estimation. This is standard in the electrophysiology literature and gives more reliable spectral estimates. All simulations should run for at least 600ms total (500ms post-burn-in) for adequate spectral resolution.

### Network Sizes
- Optimization: 450 neurons (100/200/150)
- Final results & figures: 1800 neurons (400/800/600)

### NumPy Models
Keep `numpy_models/` as a reference implementation. It's needed for the performance benchmarking script (measuring JAX speedup vs CPU baseline). Archive in a `reference/` or `numpy_baseline/` directory rather than deleting.

### Supplementary Data
The sensitivity analysis table (S2) and CMA-ES vs TPE comparison in the current manuscript are rough estimates from other projects. They must be re-run from this codebase with proper methodology for the final paper.

---

## Files That ARE Relevant to the Paper

### Core Model (`jax_models/`)
- `stn_jax.py` — STN neuron (modified Hodgkin-Huxley with T-type Ca, h-current). Gillies & Willshaw (2006).
- `gpe_gpi_hh.py` — GPe/GPi neuron (Rubin-Terman HH). Rubin & Terman (2004).
- `integrator.py` — Network step function. **Needs cleanup: remove AdEx branches, make HH the only path.**
- `network_builder.py` — Builds network state + config. Synaptic scaling. **Needs cleanup: remove `use_hh` param, AdEx imports, `build_network_state_adex()`.**
- `synapses_jax.py` — Sparse conductance-based synapses using `SynapseConfig` NamedTuple.
- `noise_jax.py` — Ornstein-Uhlenbeck noise process.
- `observables.py` — Firing rates, beta power (to be updated).

### Optimization (`optimization/`)
- `sim_jax.py` — Simulation wrapper. **Needs cleanup: remove `run_simulation_python_loop()`, fix duplicate `I_app`/`I_baseline` assignments.**
- `metrics_jax.py` — Metrics computation. **Needs update: switch beta computation from FFT to Welch's method.**
- `optuna_driver.py` — Generic Optuna driver.

### Top-Level Scripts (to be moved to `scripts/`)
- `optuna_hh_healthy.py` → `scripts/run_healthy_optimization.py`
- `optuna_hh_parkinsonian.py` → `scripts/run_parkinsonian_optimization.py`
- `dbs_simulation.py` → `scripts/run_dbs_simulation.py`
- `statistical_validation.py` → `scripts/run_statistical_validation.py`
- `generate_figures.py` → `scripts/generate_figures.py`

### NumPy Reference (`numpy_baseline/` — renamed from `numpy_models/`)
Keep for performance benchmarking only.

### Results (`results/`)
Clear all old results. Re-generate everything with metadata.

---

## Files to Remove

- `cortex_bg_pac.py` — PAC model, not in paper
- `ping_model.py` — PING model, not in paper
- `jax_models/adex_jax.py` — AdEx neurons, paper uses HH only
- `optimization/optuna_full_biological.py` — earlier attempt
- `optimization/optuna_parkinsonian.py` — v1
- `optimization/optuna_parkinsonian_v2.py` — v2
- `optimization/optuna_parkinsonian_v3.py` — v3
- `optimization/optuna_wide_search.py` — exploratory
- `optimization/sim_jax.py.bak` — backup
- `results/*` — all old results (will be re-generated)
- `docs/ASSEMBLY_GUIDE.md`, `docs/CBGTC_Publication_Draft.md`, `docs/FINAL_SUMMARY.md`, `docs/TEST_RESULTS.md`, `docs/VALIDATION.md` — outdated

---

## Target Clean Structure

```
cbgtc_project/
├── README.md                              # Updated from README_JAX.md
├── CLAUDE.md                              # This file
├── SCOPE.md                               # Full plan
├── requirements.txt
│
├── jax_models/                            # Core model (HH only)
│   ├── __init__.py
│   ├── stn_jax.py                         # STN Hodgkin-Huxley
│   ├── gpe_gpi_hh.py                      # GPe/GPi Rubin-Terman
│   ├── integrator.py                      # Network step (HH only, no AdEx)
│   ├── network_builder.py                 # Network construction (HH only)
│   ├── synapses_jax.py                    # Sparse synapses
│   ├── noise_jax.py                       # OU noise
│   └── observables.py                     # Firing rates, beta power
│
├── optimization/                          # Optimization infrastructure
│   ├── __init__.py
│   ├── sim_jax.py                         # JIT-compiled simulation wrapper
│   ├── metrics_jax.py                     # Metrics (Welch's beta, CV, rates)
│   └── optuna_driver.py                   # Generic Optuna driver
│
├── scripts/                               # All runnable scripts
│   ├── run_healthy_optimization.py        # Study 1: 6 params, 500 trials
│   ├── run_parkinsonian_optimization.py   # Study 2: 10 params, 1000 trials
│   ├── run_dbs_simulation.py              # DBS effect
│   ├── run_statistical_validation.py      # 10-seed validation
│   ├── run_timestep_sensitivity.py        # dt sensitivity (NEW)
│   ├── run_cmaes_vs_tpe.py               # Sampler comparison (NEW)
│   ├── run_benchmarks.py                  # Performance table (NEW)
│   └── generate_figures.py                # All paper figures
│
├── numpy_baseline/                        # CPU reference (for benchmarking)
│   └── (moved from numpy_models/)
│
├── tests/
│   └── test_sim_complete.py
│
├── results/                               # All re-generated with metadata
│   ├── optimization/
│   ├── simulations/
│   ├── validation/
│   ├── benchmarks/
│   └── figures/
│
└── docs/
    └── optimization_report.md
```

---

## Model Details

### Network Architecture
- **Optimization model**: 450 neurons (100 STN, 200 GPe, 150 GPi)
- **Publication model**: 1800 neurons (400 STN, 800 GPe, 600 GPi)
- dt = 0.025 ms, total simulation ≥ 600ms, burn-in = 100ms

### Neuron Models
- **STN**: Modified HH — Na, K, T-type Ca, high-threshold Ca, AHP, leak, h-current. Gillies & Willshaw (2006).
- **GPe/GPi**: Rubin-Terman HH — Na, K, T-type Ca, Ca, AHP, leak. No h-current. Rubin & Terman (2004).

### Connectivity (from code — these are the correct values)
| Pathway | Type | P(conn) | g_max (ref) | τ_rise | τ_decay | E_syn |
|---------|------|---------|-------------|--------|---------|-------|
| STN→GPe | Excitatory | 0.15 | 2.0 | 5.0 ms | 3.0 ms | 0 mV |
| GPe→STN | Inhibitory | 0.07 | 9.0 | 8.0 ms | 8.0 ms | -70 mV |
| STN→GPi | Excitatory | 0.30 | 2.0 | 5.0 ms | 3.0 ms | 0 mV |
| GPe→GPi | Inhibitory | 0.05 | 3.0 | 5.0 ms | 8.0 ms | -70 mV |

g_max values tuned for reference population (10/20/15). Scaled as `g_ref * (N_ref / N_actual)`.
Synaptic current scale factor: 0.005 for HH units. Noise scale: 0.01 for HH.

### Optimization Parameters
**Healthy (6 params):**
| Parameter | Range | Description |
|-----------|-------|-------------|
| ISTN | 80-200 | STN tonic drive current (μA/cm²) |
| I_gpe | 1-8 | GPe applied current (μA/cm²) |
| I_gpi | 1-8 | GPi applied current (μA/cm²) |
| noise_stn_sigma | 0.5-5 | STN OU noise amplitude |
| noise_gpe_sigma | 10-100 | GPe OU noise amplitude |
| noise_gpi_sigma | 10-100 | GPi OU noise amplitude |

**Parkinsonian (10 params = same 6 + 4 synaptic multipliers):**
| Parameter | Range | Description |
|-----------|-------|-------------|
| g_stn_gpe_mult | 1.5-5.0 | STN→GPe weight multiplier |
| g_gpe_stn_mult | 0.1-0.8 | GPe→STN weight multiplier (CRITICAL) |
| g_stn_gpi_mult | 1.0-4.0 | STN→GPi weight multiplier |
| g_gpe_gpi_mult | 0.2-1.2 | GPe→GPi weight multiplier |

**Loss function:**
- Firing rates: squared relative error, weight = 1.0
- CV: squared relative error, weight = 0.2 (PD) / 0.5 (healthy)
- Beta: weight = 15.0 in PD (penalize insufficient, reward excess)
- Constraints: silent population penalty (+100), PD rate bounds

### Best Parameters Found (from optimization)
**Healthy:** ISTN=140.0, I_gpe=3.379, I_gpi=2.188, noise_stn=0.996, noise_gpe=97.760, noise_gpi=69.678

**Parkinsonian:** ISTN=80.0, I_gpe=0.672, I_gpi=2.430, noise_stn=4.333, noise_gpe=139.364, noise_gpi=109.012, g_stn_gpe_mult=1.592, **g_gpe_stn_mult=0.182**, g_stn_gpi_mult=1.975, g_gpe_gpi_mult=0.419

**DBS (informational lesion):** Same as PD but ISTN→150.0, g_gpe_stn_mult→0.05

### Targets
**Healthy:** STN 20 Hz, GPe 70 Hz, GPi 80 Hz, CV ~0.2-0.4, <5% beta
**Parkinsonian:** STN 27.5 Hz, GPe 42.5 Hz, GPi 82.5 Hz, CV ~0.25-0.6, >20% beta

---

## Pending Manuscript Tasks (from PI Dr. Anderson)

1. ✅ Decided → Schematic diagram of computational workflow (GCP, GPU, vCPUs)
2. ✅ Decided → Computational performance table (JAX vs NumPy baseline)
3. ✅ Decided → CMA-ES vs TPE comparison data (new script needed)
4. 📝 Text edit → Healthy beta discussion (0% beta vs physiological baseline)
5. 📝 Refs needed → GPe interneuron diversity (prototypic vs arkypallidal: Mallet et al., Abdi et al.)
6. ✅ Decided → Timestep sensitivity analysis (new script needed)
7. ✅ In progress → Codebase cleanup

---

## Commands

```bash
# Install
pip install -r requirements.txt
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Run optimization (small model, GPU)
python scripts/run_healthy_optimization.py
python scripts/run_parkinsonian_optimization.py

# Run full-scale simulations (large model, GPU)
python scripts/run_dbs_simulation.py
python scripts/run_statistical_validation.py

# Validation analyses
python scripts/run_timestep_sensitivity.py
python scripts/run_cmaes_vs_tpe.py
python scripts/run_benchmarks.py

# Generate all figures
python scripts/generate_figures.py

# Tests
python -m pytest tests/
```
