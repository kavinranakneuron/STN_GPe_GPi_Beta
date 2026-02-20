"""
CMA-ES vs TPE sampler comparison.

Runs the healthy optimization (6 params, 450 neurons) with three Optuna
samplers — CMA-ES, TPE, and Random — across multiple seeds. Compares
convergence speed and final loss.

Author: Kavin Nakkeeran, Johns Hopkins University
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import optuna
from optuna.samplers import CmaEsSampler, TPESampler, RandomSampler
import jax
import jax.numpy as jnp
import time
import pickle
import json
import platform
from datetime import datetime

from jax_models.network_builder import build_network_state
from optimization.sim_jax import create_simulation_fn
from optimization.metrics_jax import compute_all_metrics

print(f"JAX devices: {jax.devices()}")

# Suppress Optuna logs during bulk runs
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# CONFIGURATION
# =============================================================================

N_STN, N_GPE, N_GPI = 100, 200, 150  # Optimization model
DT_MS = 0.025
N_STEPS = 16000   # 400ms
BURN_STEPS = 4000  # 100ms burn-in

N_TRIALS = 500     # Trials per run
N_SEEDS = 5        # Repetitions per sampler

TARGETS = {
    'rate_stn': 20.0,
    'rate_gpe': 70.0,
    'rate_gpi': 80.0,
    'cv_stn': 0.40,
    'cv_gpe': 0.35,
    'cv_gpi': 0.20,
}

# =============================================================================
# NETWORK + SIMULATOR
# =============================================================================

print(f"Building {N_STN + N_GPE + N_GPI}-neuron network...")
state, config = build_network_state(N_STN, N_GPE, N_GPI, DT_MS)
simulator = create_simulation_fn(config, n_steps=N_STEPS)

# Warm up JIT
print("Warming up JIT...")
dummy_params = {
    'ISTN': 100.0, 'I_gpe': 3.0, 'I_gpi': 3.0,
    'noise_stn_sigma': 1.0, 'noise_gpe_sigma': 30.0, 'noise_gpi_sigma': 30.0,
}
obs = simulator(dummy_params, state)
obs['V_stn'].block_until_ready()
print("JIT ready!\n")

# =============================================================================
# OBJECTIVE (same as healthy optimization)
# =============================================================================

def create_objective():
    def objective(trial):
        params = {
            'ISTN': trial.suggest_float('ISTN', 80.0, 200.0),
            'I_gpe': trial.suggest_float('I_gpe', 1.0, 8.0),
            'I_gpi': trial.suggest_float('I_gpi', 1.0, 8.0),
            'noise_stn_sigma': trial.suggest_float('noise_stn_sigma', 0.5, 5.0),
            'noise_gpe_sigma': trial.suggest_float('noise_gpe_sigma', 10.0, 100.0),
            'noise_gpi_sigma': trial.suggest_float('noise_gpi_sigma', 10.0, 100.0),
        }

        try:
            obs = simulator(params, state)
            obs['V_stn'].block_until_ready()

            if jnp.any(jnp.isnan(obs['V_stn'])) or jnp.any(jnp.isnan(obs['V_gpe'])):
                return 1e6

            metrics = compute_all_metrics(obs, DT_MS, burn_steps=BURN_STEPS)

            r_stn = metrics['firing_rates']['stn']
            r_gpe = metrics['firing_rates']['gpe']
            r_gpi = metrics['firing_rates']['gpi']
            cv_stn = metrics['cv']['stn']
            cv_gpe = metrics['cv']['gpe']
            cv_gpi = metrics['cv']['gpi']

            loss = 0.0
            loss += ((r_stn - TARGETS['rate_stn']) / TARGETS['rate_stn']) ** 2
            loss += ((r_gpe - TARGETS['rate_gpe']) / TARGETS['rate_gpe']) ** 2
            loss += ((r_gpi - TARGETS['rate_gpi']) / TARGETS['rate_gpi']) ** 2
            loss += 0.5 * ((cv_stn - TARGETS['cv_stn']) / TARGETS['cv_stn']) ** 2
            loss += 0.5 * ((cv_gpe - TARGETS['cv_gpe']) / TARGETS['cv_gpe']) ** 2
            loss += 0.5 * ((cv_gpi - TARGETS['cv_gpi']) / TARGETS['cv_gpi']) ** 2

            if r_stn < 1.0:
                loss += 100.0
            if r_gpe < 10.0:
                loss += 100.0
            if r_gpi < 10.0:
                loss += 100.0

            return loss

        except Exception:
            return 1e6
    return objective

# =============================================================================
# SAMPLER DEFINITIONS
# =============================================================================

SAMPLERS = {
    'CMA-ES': lambda seed: CmaEsSampler(seed=seed),
    'TPE': lambda seed: TPESampler(seed=seed),
    'Random': lambda seed: RandomSampler(seed=seed),
}

# =============================================================================
# RUN COMPARISON
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CMA-ES vs TPE vs RANDOM — Sampler Comparison")
    print("=" * 70)
    print(f"Network: {N_STN}/{N_GPE}/{N_GPI} neurons")
    print(f"Trials per run: {N_TRIALS}")
    print(f"Seeds per sampler: {N_SEEDS}")
    print(f"Total runs: {len(SAMPLERS) * N_SEEDS}")
    print("=" * 70)

    all_results = {}

    for sampler_name, sampler_factory in SAMPLERS.items():
        print(f"\n{'='*50}")
        print(f"SAMPLER: {sampler_name}")
        print(f"{'='*50}")

        sampler_results = {
            'final_losses': [],
            'convergence_curves': [],  # best-so-far per trial
            'elapsed_seconds': [],
            'best_params_list': [],
        }

        for seed_idx in range(N_SEEDS):
            seed = seed_idx * 100 + 42
            print(f"\n  Seed {seed_idx + 1}/{N_SEEDS} (seed={seed})...")

            sampler = sampler_factory(seed)
            study = optuna.create_study(
                direction='minimize',
                sampler=sampler,
            )

            t0 = time.time()
            study.optimize(create_objective(), n_trials=N_TRIALS, show_progress_bar=False)
            elapsed = time.time() - t0

            # Extract convergence curve (best-so-far)
            best_so_far = []
            running_best = float('inf')
            for trial in study.trials:
                if trial.value is not None and trial.value < running_best:
                    running_best = trial.value
                best_so_far.append(running_best)

            sampler_results['final_losses'].append(study.best_value)
            sampler_results['convergence_curves'].append(best_so_far)
            sampler_results['elapsed_seconds'].append(elapsed)
            sampler_results['best_params_list'].append(study.best_params)

            print(f"    Loss: {study.best_value:.4f}  Time: {elapsed:.1f}s "
                  f"({elapsed/N_TRIALS*1000:.0f}ms/trial)")

        all_results[sampler_name] = sampler_results

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================

    print("\n" + "=" * 70)
    print("SAMPLER COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Sampler':<12} {'Final Loss':>16} {'Time (s)':>16} {'ms/trial':>10}")
    print("-" * 56)

    for name, res in all_results.items():
        losses = res['final_losses']
        times = res['elapsed_seconds']
        print(f"{name:<12} {np.mean(losses):>7.4f} +/- {np.std(losses):<6.4f} "
              f"{np.mean(times):>7.1f} +/- {np.std(times):<5.1f} "
              f"{np.mean(times)/N_TRIALS*1000:>7.0f}")

    # Trials to reach threshold
    threshold = 1.0  # Loss threshold for "converged"
    print(f"\nTrials to reach loss < {threshold}:")
    for name, res in all_results.items():
        trials_to_thresh = []
        for curve in res['convergence_curves']:
            reached = [i for i, v in enumerate(curve) if v < threshold]
            if reached:
                trials_to_thresh.append(reached[0] + 1)
            else:
                trials_to_thresh.append(N_TRIALS)  # didn't converge
        mean_t = np.mean(trials_to_thresh)
        std_t = np.std(trials_to_thresh)
        print(f"  {name:<12} {mean_t:.0f} +/- {std_t:.0f} trials")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================

    metadata = {
        'date': datetime.now().isoformat(),
        'network_size': (N_STN, N_GPE, N_GPI),
        'n_trials': N_TRIALS,
        'n_seeds': N_SEEDS,
        'targets': TARGETS,
        'jax_version': jax.__version__,
        'optuna_version': optuna.__version__,
        'platform': platform.platform(),
        'jax_devices': [str(d) for d in jax.devices()],
    }

    output = {
        'results': all_results,
        'metadata': metadata,
    }

    with open('results/validation/sampler_comparison.pkl', 'wb') as f:
        pickle.dump(output, f)
    print(f"\nSaved: results/validation/sampler_comparison.pkl")
