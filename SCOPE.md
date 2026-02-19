# SCOPE: CBGTC Project — Full Cleanup & Re-run Plan

## Ground Truth

The **codebase is ground truth**. The manuscript will be updated to match what the code actually does. All results will be re-generated fresh.

---

## Phase 1: Codebase Cleanup

### 1A. Remove non-paper files
- `cortex_bg_pac.py` — PAC model, not in paper
- `ping_model.py` — PING model, not in paper
- `jax_models/adex_jax.py` — AdEx neurons, paper uses Rubin-Terman HH only
- `numpy_models/` → rename to `numpy_baseline/` and keep for performance benchmarking (JAX vs CPU speedup comparison)
- `optimization/optuna_full_biological.py` — earlier optimization attempt
- `optimization/optuna_parkinsonian.py` — v1 (superseded)
- `optimization/optuna_parkinsonian_v2.py` — v2 (superseded)
- `optimization/optuna_parkinsonian_v3.py` — v3 (superseded)
- `optimization/optuna_wide_search.py` — exploratory
- `optimization/sim_jax.py.bak` — backup file
- `results/pac_v3.png`, `results/ping_v2.png` — non-paper figures
- `results/full_biological_*`, `results/healthy_10param_*`, `results/healthy_1800_*` — old runs
- `results/parkinsonian_10param_*`, `results/parkinsonian_constrained_*` — old runs
- `results/parkinsonian_full_*`, `results/parkinsonian_v2_*`, `results/parkinsonian_v3_*` — old runs
- `docs/ASSEMBLY_GUIDE.md`, `docs/CBGTC_Publication_Draft.md`, `docs/FINAL_SUMMARY.md`, `docs/TEST_RESULTS.md`, `docs/VALIDATION.md` — outdated docs

### 1B. Clean up AdEx references in remaining code
- `integrator.py`: Remove all AdEx branches (the `if 'Ca' in state` / `else` blocks). Make HH the only path. Remove `use_hh` flag.
- `network_builder.py`: Remove `use_hh` parameter, AdEx imports, `build_network_state_adex()`. HH-only.
- `sim_jax.py`: Remove `run_simulation_python_loop()` (unused baseline). Clean up duplicate `I_app`/`I_baseline` assignments (lines 35-41 have redundant writes).

### 1C. Reorganize into clean structure
```
cbgtc_project/
├── README.md
├── CLAUDE.md
├── requirements.txt
├── jax_models/
│   ├── __init__.py
│   ├── stn_jax.py
│   ├── gpe_gpi_hh.py
│   ├── integrator.py          (HH-only, no AdEx)
│   ├── network_builder.py     (HH-only, no AdEx)
│   ├── synapses_jax.py
│   ├── noise_jax.py
│   └── observables.py
├── optimization/
│   ├── __init__.py
│   ├── sim_jax.py             (cleaned)
│   ├── metrics_jax.py
│   └── optuna_driver.py
├── scripts/
│   ├── run_healthy_optimization.py
│   ├── run_parkinsonian_optimization.py
│   ├── run_dbs_simulation.py
│   ├── run_statistical_validation.py
│   ├── run_timestep_sensitivity.py      (NEW)
│   ├── run_cmaes_vs_tpe.py             (NEW)
│   └── generate_figures.py
├── tests/
│   └── test_sim_complete.py
├── results/                             (all re-generated, with metadata)
│   └── (empty until re-run)
└── docs/
    └── optimization_report.md
```

---

## Phase 2: New Scripts Needed

### 2A. Timestep sensitivity analysis (`run_timestep_sensitivity.py`)
Stan flagged this (task 6). Test dt = 0.025 ms against dt = 0.0125 ms and dt = 0.00625 ms. 
- Run the same simulation (healthy params + PD params) at each timestep
- Compare: firing rates, CV, beta fraction
- Show convergence (results should be stable across timesteps)
- Save results table + figure

### 2B. CMA-ES vs TPE comparison (`run_cmaes_vs_tpe.py`)
Stan wants this data (task 3). Run the same optimization problem with:
- CMA-ES sampler (current approach)
- TPE sampler (Optuna default)
- Random sampler (baseline)
Compare convergence speed: trials to reach threshold loss, final loss after N trials.
- Use the healthy optimization (6-param, faster) as the benchmark
- Run each sampler 3-5 times with different seeds for error bars
- Save comparison table + convergence curves figure

### 2C. Performance benchmarking script (`run_benchmarks.py`)
For computational table (task 2). Measure wall-clock time for:
- JAX JIT simulation at different network sizes (450, 1800, 5000, 18000)
- NumPy baseline (Python loop) at 450 neurons (for comparison ratio)
- Report: neurons, sim time (ms), wall clock (s), speedup factor
- Include memory usage
- Save table as CSV/JSON + formatted for paper

---

## Phase 3: Re-run Everything

All runs must be executed on GCP with proper GPU. Each result must include metadata:
- Date/time of run
- Git commit hash
- Hardware (GPU type, vCPUs, RAM)
- Software versions (JAX, Optuna, Python, CUDA)
- Random seed
- Network size
- All parameters

### 3A. Optimization (small model: 450 neurons = 100/200/150)

Optimization is efficient on the smaller model. This is where parameter search happens.

| Run | Script | Params | Trials | Expected time |
|-----|--------|--------|--------|---------------|
| Study 1: Healthy | `run_healthy_optimization.py` | 6 | 500 | ~10 min |
| Study 2: Parkinsonian | `run_parkinsonian_optimization.py` | 10 | 1000 | ~20 min |

Output: `results/optimization/healthy_study.pkl`, `results/optimization/parkinsonian_study.pkl`
Each .pkl includes: best_params, best_value, all trial history, targets, metadata dict.

### 3B. Full-scale simulations (large model: 1800 neurons = 400/800/600)

Take the best params from 3A, run on the larger model for publication results.

| Run | Description | Duration |
|-----|-------------|----------|
| Healthy simulation | 400/800/600, healthy params, 600ms | Single run |
| Parkinsonian simulation | 400/800/600, PD params, 600ms | Single run |
| DBS simulation | 400/800/600, PD params, PD+DBS params, 600ms | Two runs |

Output: `results/simulations/healthy_1800.pkl`, `results/simulations/parkinsonian_1800.pkl`, `results/simulations/dbs_1800.pkl`

**NOTE**: Need to verify that params optimized at 450 neurons produce the same qualitative results (beta emergence, correct firing rates) at 1800. The synaptic weight scaling (`g_max * REF_N / actual_N`) should handle this, but it must be validated. If firing rates or beta power drift significantly at the larger scale, may need a brief re-optimization or manual adjustment.

### 3C. Statistical validation (large model, 10 seeds)
Run both healthy and PD at 1800 neurons across 10 random seeds.
Output: `results/validation/statistical_validation_1800.pkl`

### 3D. Timestep sensitivity (large model)
Run at dt = 0.025, 0.0125, 0.00625 ms.
Output: `results/validation/timestep_sensitivity.pkl`

### 3E. CMA-ES vs TPE comparison (small model, fast)
Run at 450 neurons with multiple samplers.
Output: `results/validation/sampler_comparison.pkl`

### 3F. Performance benchmarks
Time simulations at multiple scales.
Output: `results/benchmarks/performance_table.json`

---

## Phase 4: Generate All Figures

From the large-model results. Each figure saved as PDF + PNG.

| Figure | Content | Data source |
|--------|---------|-------------|
| Fig 1 | Network schematic + connectivity diagram | Static (code-generated) |
| Fig 2 | Raster plots: Healthy vs Parkinsonian | 3B healthy + PD sims |
| Fig 3 | Power spectra: Healthy vs Parkinsonian | 3B healthy + PD sims |
| Fig 4 | Firing rate & beta bar charts | 3B sims |
| Fig 5 | LFP traces | 3B sims |
| Fig 6 | Statistical validation (error bars, p-values) | 3C |
| Fig 7 | DBS effect (spectra + bars) | 3B DBS sim |

Supplementary:
| Figure | Content | Data source |
|--------|---------|-------------|
| S1 | CMA-ES vs TPE convergence curves | 3E |
| S2 | Timestep sensitivity | 3D |
| S3 | Performance scaling (wall time vs neurons) | 3F |
| S4 | Optimization convergence (loss vs trial) | 3A |

---

## Phase 5: Manuscript Alignment

After all results are generated, update the manuscript to match:

### Numbers to update from actual results:
- Network sizes: "optimization performed on 450-neuron model (100 STN, 200 GPe, 150 GPi); results validated on 1800-neuron model (400 STN, 800 GPe, 600 GPi)"
- Connectivity probabilities: 0.15 (STN→GPe), 0.07 (GPe→STN), 0.30 (STN→GPi), 0.05 (GPe→GPi) — from code
- Firing rates, beta fractions, CV values — from 3B/3C results
- Performance numbers — from 3F benchmarks
- CMA-ES vs TPE speedup — from 3E results
- Timestep validation — from 3D

### New content to add:
- **Methods**: Two-stage approach (optimization on small model → validation on large model)
- **Methods**: Describe synaptic weight scaling formula (`g_max * N_ref / N_actual`)
- **Methods**: Timestep validation results
- **Methods**: DBS implementation details (current injection via noise pathway, "informational lesion" model — ISTN increase + g_gpe_stn further reduced)
- **Results**: CMA-ES vs TPE comparison (supplementary or main text)
- **Results**: Performance benchmarks table
- **Discussion**: Healthy beta — address that model shows 0% beta but physiological baseline beta exists
- **Discussion**: GPe interneuron diversity refs (prototypic vs arkypallidal)
- **Supplementary**: Computational workflow schematic figure

### RESOLVED: DBS implementation
The paper will describe the actual implementation: an "informational lesion" approach (ISTN 80→150, g_gpe_stn_mult 0.182→0.05). The `create_dbs_simulator` (130 Hz pulses) remains in codebase as an alternative but is not used for paper results.

### RESOLVED: Beta power computation
Switch code to Welch's method (`scipy.signal.welch`). Update `metrics_jax.py`. More scientifically rigorous, standard in electrophysiology literature. Requires simulations ≥600ms for adequate spectral resolution.

### RESOLVED: Connectivity probabilities
Code values are correct: 15%, 7%, 30%, 5%. Manuscript will be updated.

### RESOLVED: NumPy models
Rename `numpy_models/` to `numpy_baseline/`, keep for performance benchmarking.

---

## Summary: Order of Operations

1. **Codebase cleanup** (Phase 1) — Claude Code
2. **Write new scripts** (Phase 2) — Claude Code
3. **Re-run everything on GCP** (Phase 3) — You, on GPU
4. **Generate figures** (Phase 4) — You or Claude Code on GCP
5. **Update manuscript** (Phase 5) — This chat or Claude Code

---

## Open Questions for Kavin

~~1. DBS approach~~ → **Informational lesion** (what the code actually does). Manuscript will be updated.
~~2. Beta computation~~ → **Switch to Welch's method** in code. More scientifically accurate, standard in electrophysiology.
~~3. Large model size~~ → **1800 neurons (400/800/600)** confirmed.
~~4. generate_figures.py~~ → Will be regenerated from fresh results; hardcoded values will be removed.
~~5. Sensitivity analysis~~ → **Placeholder numbers from other projects. Must be re-run from this codebase.** Same for CMA-ES vs TPE comparison.
