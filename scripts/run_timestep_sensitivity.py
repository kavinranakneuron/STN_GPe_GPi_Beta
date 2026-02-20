"""
Timestep sensitivity analysis.

Tests dt = 0.025, 0.0125, 0.00625 ms to verify numerical convergence.
Runs both healthy and Parkinsonian parameter sets at each timestep,
compares firing rates, CV, and beta fraction.

Author: Kavin Nakkeeran, Johns Hopkins University
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import jax
import jax.numpy as jnp
import time
import pickle
import json
import platform
from datetime import datetime

from jax_models.network_builder import build_network_state
from optimization.sim_jax import create_simulation_fn, apply_params_to_config
from optimization.metrics_jax import compute_all_metrics, compute_beta_fraction_all
from jax_models.integrator import network_step
from jax import lax

print(f"JAX devices: {jax.devices()}")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Timesteps to test (ms)
DT_VALUES = [0.025, 0.0125, 0.00625]

# Network size (large model for publication)
N_STN, N_GPE, N_GPI = 400, 800, 600

# Simulation duration: 600ms total, 100ms burn-in
SIM_DURATION_MS = 600.0
BURN_IN_MS = 100.0

# Best parameters from optimization
HEALTHY_PARAMS = {
    'ISTN': 140.0,
    'I_gpe': 3.379,
    'I_gpi': 2.188,
    'noise_stn_sigma': 0.996,
    'noise_gpe_sigma': 97.760,
    'noise_gpi_sigma': 69.678,
}

PD_PARAMS = {
    'ISTN': 80.0,
    'I_gpe': 0.672,
    'I_gpi': 2.430,
    'noise_stn_sigma': 4.333,
    'noise_gpe_sigma': 139.364,
    'noise_gpi_sigma': 109.012,
    'g_stn_gpe_mult': 1.592,
    'g_gpe_stn_mult': 0.182,
    'g_stn_gpi_mult': 1.975,
    'g_gpe_gpi_mult': 0.419,
}

# =============================================================================
# PD SIMULATOR (with synaptic multipliers)
# =============================================================================

def create_pd_simulator(base_config, n_steps):
    @jax.jit
    def simulate(trial_params, init_state):
        config = apply_params_to_config(trial_params, base_config)
        syn_configs = dict(config['synapses'])
        for syn_name, mult_name in [
            ('stn_to_gpe', 'g_stn_gpe_mult'),
            ('gpe_to_stn', 'g_gpe_stn_mult'),
            ('stn_to_gpi', 'g_stn_gpi_mult'),
            ('gpe_to_gpi', 'g_gpe_gpi_mult'),
        ]:
            old_cfg = syn_configs[syn_name]
            new_weights = old_cfg.weights * trial_params.get(mult_name, 1.0)
            syn_configs[syn_name] = old_cfg._replace(weights=new_weights)
        config['synapses'] = syn_configs

        def step_fn(carry_state, t_idx):
            t_ms = t_idx * config['dt_ms']
            new_state, obs = network_step(carry_state, config, t_ms)
            return new_state, obs
        _, obs_history = lax.scan(step_fn, init=init_state, xs=jnp.arange(n_steps))
        return obs_history
    return simulate

# =============================================================================
# RUN SENSITIVITY ANALYSIS
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TIMESTEP SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"Network: {N_STN}/{N_GPE}/{N_GPI} neurons ({N_STN + N_GPE + N_GPI} total)")
    print(f"Duration: {SIM_DURATION_MS}ms (burn-in: {BURN_IN_MS}ms)")
    print(f"Timesteps: {DT_VALUES} ms")
    print("=" * 70)

    results = {}

    for dt in DT_VALUES:
        n_steps = int(SIM_DURATION_MS / dt)
        burn_steps = int(BURN_IN_MS / dt)

        print(f"\n--- dt = {dt} ms ({n_steps} steps, burn-in {burn_steps}) ---")

        # Build network at this timestep
        state, config = build_network_state(N_STN, N_GPE, N_GPI, dt, seed=42)

        # Create simulators
        healthy_sim = create_simulation_fn(config, n_steps=n_steps)
        pd_sim = create_pd_simulator(config, n_steps)

        # Warm up JIT
        print("  JIT compiling...")
        t0 = time.time()
        obs = healthy_sim(HEALTHY_PARAMS, state)
        obs['V_stn'].block_until_ready()
        jit_time = time.time() - t0
        print(f"  JIT compile: {jit_time:.1f}s")

        # Run healthy
        print("  Running healthy...")
        t0 = time.time()
        obs_h = healthy_sim(HEALTHY_PARAMS, state)
        obs_h['V_stn'].block_until_ready()
        h_wall = time.time() - t0

        metrics_h = compute_all_metrics(obs_h, dt, burn_steps=burn_steps)
        beta_h = compute_beta_fraction_all(obs_h, dt, burn_steps=burn_steps)

        # Run PD (needs new JIT for this dt)
        print("  Running Parkinsonian...")
        t0 = time.time()
        obs_pd = pd_sim(PD_PARAMS, state)
        obs_pd['V_stn'].block_until_ready()
        pd_wall = time.time() - t0

        metrics_pd = compute_all_metrics(obs_pd, dt, burn_steps=burn_steps)
        beta_pd = compute_beta_fraction_all(obs_pd, dt, burn_steps=burn_steps)

        results[dt] = {
            'healthy': {
                'firing_rates': metrics_h['firing_rates'],
                'cv': metrics_h['cv'],
                'beta_fraction': {k: float(v) for k, v in beta_h.items()},
                'wall_time_s': h_wall,
            },
            'pd': {
                'firing_rates': metrics_pd['firing_rates'],
                'cv': metrics_pd['cv'],
                'beta_fraction': {k: float(v) for k, v in beta_pd.items()},
                'wall_time_s': pd_wall,
            },
            'n_steps': n_steps,
            'burn_steps': burn_steps,
        }

        print(f"  Healthy: STN={metrics_h['firing_rates']['stn']:.1f}Hz, "
              f"GPe={metrics_h['firing_rates']['gpe']:.1f}Hz, "
              f"GPe beta={beta_h['gpe']*100:.1f}% ({h_wall:.2f}s)")
        print(f"  PD:      STN={metrics_pd['firing_rates']['stn']:.1f}Hz, "
              f"GPe={metrics_pd['firing_rates']['gpe']:.1f}Hz, "
              f"GPe beta={beta_pd['gpe']*100:.1f}% ({pd_wall:.2f}s)")

    # =========================================================================
    # CONVERGENCE TABLE
    # =========================================================================

    print("\n" + "=" * 70)
    print("CONVERGENCE TABLE — HEALTHY")
    print("=" * 70)
    header = f"{'dt (ms)':<10} {'STN Hz':<10} {'GPe Hz':<10} {'GPi Hz':<10} {'STN CV':<10} {'GPe CV':<10} {'GPe beta%':<10}"
    print(header)
    print("-" * 70)
    for dt in DT_VALUES:
        r = results[dt]['healthy']
        print(f"{dt:<10.5f} {r['firing_rates']['stn']:<10.1f} {r['firing_rates']['gpe']:<10.1f} "
              f"{r['firing_rates']['gpi']:<10.1f} {r['cv']['stn']:<10.3f} {r['cv']['gpe']:<10.3f} "
              f"{r['beta_fraction']['gpe']*100:<10.1f}")

    print("\n" + "=" * 70)
    print("CONVERGENCE TABLE — PARKINSONIAN")
    print("=" * 70)
    print(header)
    print("-" * 70)
    for dt in DT_VALUES:
        r = results[dt]['pd']
        print(f"{dt:<10.5f} {r['firing_rates']['stn']:<10.1f} {r['firing_rates']['gpe']:<10.1f} "
              f"{r['firing_rates']['gpi']:<10.1f} {r['cv']['stn']:<10.3f} {r['cv']['gpe']:<10.3f} "
              f"{r['beta_fraction']['gpe']*100:<10.1f}")

    # Compute max relative change between finest two timesteps
    ref_dt = DT_VALUES[-1]  # finest
    prev_dt = DT_VALUES[-2]  # second finest
    print(f"\nMax relative change ({prev_dt} -> {ref_dt} ms):")
    for condition in ['healthy', 'pd']:
        ref = results[ref_dt][condition]
        prev = results[prev_dt][condition]
        max_change = 0.0
        for pop in ['stn', 'gpe', 'gpi']:
            rate_ref = ref['firing_rates'][pop]
            rate_prev = prev['firing_rates'][pop]
            if rate_ref > 0:
                change = abs(rate_ref - rate_prev) / rate_ref * 100
                max_change = max(max_change, change)
        print(f"  {condition}: {max_change:.1f}% (firing rates)")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================

    metadata = {
        'date': datetime.now().isoformat(),
        'network_size': (N_STN, N_GPE, N_GPI),
        'sim_duration_ms': SIM_DURATION_MS,
        'burn_in_ms': BURN_IN_MS,
        'dt_values': DT_VALUES,
        'jax_version': jax.__version__,
        'platform': platform.platform(),
        'jax_devices': [str(d) for d in jax.devices()],
    }

    output = {
        'results': results,
        'healthy_params': HEALTHY_PARAMS,
        'pd_params': PD_PARAMS,
        'metadata': metadata,
    }

    with open('results/validation/timestep_sensitivity.pkl', 'wb') as f:
        pickle.dump(output, f)
    print(f"\nSaved: results/validation/timestep_sensitivity.pkl")
