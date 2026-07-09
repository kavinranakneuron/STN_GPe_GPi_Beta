# GPU-Accelerated Basal Ganglia Network Model for Parkinson's Disease

**Author:** Kavineshvar Ranak  
**Affiliation:** Functional Neurosurgery Lab, Johns Hopkins University  
**Date:** February 2026

---

## Overview

A JAX-accelerated computational model of the subthalamic nucleus–globus pallidus (STN-GPe-GPi) circuit for studying pathological beta oscillations in Parkinson's disease. The model uses **Hodgkin-Huxley neurons** with **fixed-indegree connectivity** and achieves a **490× speedup** over CPU implementations through GPU acceleration.

### Main Finding

Starting from identical baseline connectivity (all synaptic multipliers at 1.0), CMA-ES optimization independently discovered that the Parkinsonian state requires **coordinated synaptic reorganization** — not uniform degradation:

| Pathway | PD/Healthy Ratio | Interpretation |
|---------|-----------------|----------------|
| STN→GPe | **2.21×** | Excitation doubles |
| GPe→STN | **0.11×** | Inhibition collapses (89% loss) |
| STN→GPi | **1.06×** | Essentially unchanged |
| GPe→GPi | **1.45×** | Moderate increase |

This pattern — excitatory drive ramping up while the critical inhibitory brake on STN fails — is consistent with known dopamine-dependent plasticity and was discovered without prior constraints on which synapses should change.

### Key Results

- **Scale-invariant dynamics**: Firing rates vary <1 Hz and STN beta power remains ~50% across a 100× network size range (450 to 45,000 neurons)
- **DBS suppression**: Simulated deep brain stimulation eliminates 100% of pathological STN beta oscillations
- **Statistical robustness**: All metrics significant at p < 0.001 across 10 independent seeds (45,000 neurons)
- **Performance**: 490× speedup enables full parameter optimization in ~20 minutes on a single NVIDIA L4 GPU

---

## Quick Start

### Requirements

- Python 3.9–3.11
- NVIDIA GPU with CUDA 12.x (recommended) or CPU-only (slow)

### Installation

```bash
git clone https://github.com/neuronlab-cell/STN_GPe_GPi_Beta.git
cd cbgtc_project
pip install -r requirements.txt

# For GPU support (CUDA 12)
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Run a simulation

```python
python3 -c "
import sys; sys.path.insert(0, '.')
from jax_models.network_builder import build_network_state
from optimization.sim_jax import create_simulation_fn
from jax_models.observables import compute_all_metrics

state, config = build_network_state(n_stn=100, n_gpe=200, n_gpi=150, dt_ms=0.025)
params = {
    'ISTN': 101.789, 'I_gpe': 2.784, 'I_gpi': 2.261,
    'noise_stn_sigma': 3.116, 'noise_gpe_sigma': 98.463, 'noise_gpi_sigma': 68.379,
    'g_stn_gpe_mult': 1.87, 'g_gpe_stn_mult': 1.00,
    'g_stn_gpi_mult': 1.83, 'g_gpe_gpi_mult': 0.69,
}
simulator = create_simulation_fn(config, n_steps=24000)
obs = simulator(params, state)
m = compute_all_metrics(obs, dt_ms=0.025, burn_steps=4000)
for pop in ['stn', 'gpe', 'gpi']:
    print(f'{pop}: {float(m[\"firing_rates\"][pop]):.1f} Hz, beta={float(m[\"beta_fraction\"][pop])*100:.1f}%') 
"
```

---

## Reproducing Paper Results

All scripts are in `scripts/`. Run on a GPU instance (GCP with NVIDIA L4 recommended).

```bash
# 1. Optimization (450 neurons, ~30 min total)
python scripts/run_healthy_optimization.py
python scripts/run_parkinsonian_optimization.py

# 2. Scaling study (450–45,000 neurons)
python scripts/run_scaling_study.py

# 3. Statistical validation (45,000 neurons, 10 seeds)
python scripts/run_statistical_validation.py

# 4. DBS simulation (45,000 neurons)
python scripts/run_dbs_simulation.py

# 5. Benchmarks and sampler comparison
python scripts/run_benchmarks.py
python scripts/run_cmaes_vs_tpe.py

# 6. Timestep sensitivity
python scripts/run_timestep_sensitivity.py

# 7. Generate all figures
python scripts/generate_figures.py
```

Pre-computed results are included in `results/` so figures can be regenerated without re-running simulations.

---

## Project Structure

```
cbgtc_project/
├── jax_models/                          # Core neural models
│   ├── stn_jax.py                       # STN Hodgkin-Huxley (Gillies & Willshaw 2006)
│   ├── gpe_gpi_hh.py                    # GPe/GPi Rubin-Terman
│   ├── network_builder.py               # Fixed-indegree connectivity
│   ├── synapses_jax.py                  # Sparse conductance-based synapses
│   ├── integrator.py                    # Network step function
│   ├── noise_jax.py                     # Ornstein-Uhlenbeck noise
│   └── observables.py                   # Firing rates, spectral analysis
│
├── optimization/                        # Optimization infrastructure
│   ├── sim_jax.py                       # JIT-compiled simulation wrapper
│   ├── metrics_jax.py                   # Metrics (Welch PSD, CV, rates)
│   └── optuna_driver.py                 # Optuna CMA-ES driver
│
├── scripts/                             # Reproducibility scripts
│   ├── run_healthy_optimization.py      # Study 1: 10 params, 500 trials
│   ├── run_parkinsonian_optimization.py # Study 2: 10 params, 1000 trials
│   ├── run_scaling_study.py             # Scale invariance (450–45,000n)
│   ├── run_statistical_validation.py    # 10-seed validation at 45,000n
│   ├── run_dbs_simulation.py            # DBS beta suppression
│   ├── run_benchmarks.py                # JAX vs NumPy performance
│   ├── run_cmaes_vs_tpe.py             # Sampler comparison
│   ├── run_timestep_sensitivity.py      # Timestep convergence
│   ├── generate_figures.py              # All paper figures
│   └── test_scaling_invariance.py       # Indegree verification
│
├── numpy_baseline/                      # CPU reference (for benchmarking)
│
├── results/
│   ├── optimization/                    # Optimized parameters (.pkl)
│   ├── validation/                      # Scaling, stats, timestep (.pkl, .json)
│   ├── benchmarks/                      # Performance data (.json)
│   ├── simulations/                     # DBS results (.pkl)
│   └── figures/                         # All paper figures (.png, .pdf)
│
└── tests/
    └── test_sim_complete.py
```

---

## Model Details

### Neuron Models

- **STN**: Modified Hodgkin-Huxley with Na, K, T-type Ca, high-threshold Ca, AHP, leak, and h-current channels. Based on Gillies & Willshaw (2006).
- **GPe/GPi**: Rubin-Terman formulation with Na, K, T-type Ca, Ca, AHP, and leak channels. Based on Rubin & Terman (2004).

### Fixed-Indegree Connectivity

Each postsynaptic neuron receives input from a fixed number of presynaptic neurons (indegree K), preserving input statistics across network sizes (Gerstner et al., *Neuronal Dynamics*, Ch. 12.3).

| Pathway | Type | Indegree (K) | g_syn (mS/cm²) | E_syn (mV) |
|---------|------|-------------|-----------------|------------|
| STN→GPe | Excitatory | 15 | 0.2 | 0 |
| GPe→STN | Inhibitory | 14 | 0.9 | −70 |
| STN→GPi | Excitatory | 30 | 0.2 | 0 |
| GPe→GPi | Inhibitory | 10 | 0.3 | −70 |

### Optimized Parameters

**Healthy state** (baseline):

| Parameter | Value | Unit |
|-----------|-------|------|
| I_STN | 101.789 | µA/cm² |
| I_GPe | 2.784 | µA/cm² |
| I_GPi | 2.261 | µA/cm² |
| g_stn_gpe_mult | 1.87 | × |
| g_gpe_stn_mult | 1.00 | × |
| g_stn_gpi_mult | 1.83 | × |
| g_gpe_gpi_mult | 0.69 | × |

**Parkinsonian state**:

| Parameter | Value | Unit | PD/Healthy Ratio |
|-----------|-------|------|-----------------|
| I_STN | 61.927 | µA/cm² | — |
| I_GPe | 0.937 | µA/cm² | — |
| I_GPi | 1.838 | µA/cm² | — |
| g_stn_gpe_mult | 4.13 | × | 2.21× |
| g_gpe_stn_mult | 0.11 | × | **0.11×** |
| g_stn_gpi_mult | 1.94 | × | 1.06× |
| g_gpe_gpi_mult | 1.00 | × | 1.45× |

---

## Performance

| Neurons | Backend | Wall Time (s) | Speedup |
|---------|---------|---------------|---------|
| 450 | NumPy/CPU | 837.75 | 1.0× (ref) |
| 450 | JAX/GPU | 1.71 | 490× |
| 1,800 | JAX/GPU | 1.84 | — |
| 4,500 | JAX/GPU | 1.89 | — |
| 15,000 | JAX/GPU | 2.33 | — |
| 45,000 | JAX/GPU | 3.39 | — |

Hardware: NVIDIA L4 GPU, Google Cloud Platform (g2-standard-8)

---

## Cloud Setup (GCP)

```bash
# Create GPU VM
gcloud compute instances create cbgtc-hh \
    --zone=us-central1-c \
    --machine-type=g2-standard-8 \
    --accelerator=type=nvidia-l4,count=1 \
    --image-family=common-cuda121-debian-11 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --maintenance-policy=TERMINATE

# Connect
gcloud compute ssh cbgtc-hh --zone=us-central1-c

# Setup
git clone https://github.com/neuronlab-cell/cbgtc_project.git
cd cbgtc_project
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --break-system-packages
pip install optuna scipy matplotlib --break-system-packages

# Verify GPU
python3 -c "import jax; print(jax.devices())"  # Should show [cuda(id=0)]

# Stop VM when done (~$0.70/hr)
gcloud compute instances stop cbgtc-hh --zone=us-central1-c
```

---

## References

- Gillies A, Willshaw D (2006). Membrane channel interactions underlying rat subthalamic projection neuron rhythmic and bursting activity. *J Neurophysiol* 95:2352–2365.
- Rubin JE, Terman D (2004). High frequency stimulation of the subthalamic nucleus eliminates pathological thalamic rhythmicity in a computational model. *J Comput Neurosci* 16:211–235.
- Gerstner W, Kistler WM, Naud R, Paninski L (2014). *Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition*. Cambridge University Press.
- Bergman H, Wichmann T, Karmon B, DeLong MR (1994). The primate subthalamic nucleus. II. Neuronal activity in the MPTP model of parkinsonism. *J Neurophysiol* 72:507–520.
- Brown P (2003). Oscillatory nature of human basal ganglia activity. *Mov Disord* 18:357–363.
- Bradbury J et al. (2018). JAX: composable transformations of Python+NumPy programs. https://github.com/google/jax
- Akiba T et al. (2019). Optuna: A next-generation hyperparameter optimization framework. *KDD 2019*.

---

## AI Assistance Disclosure

Generative AI tools (Claude, Anthropic) were used to assist with code documentation, manuscript formatting, and readability improvements. All computational model design, simulation execution, data analysis, and scientific interpretation were performed by the authors. No AI tools were used to generate simulations, run experiments, or produce scientific conclusions.
