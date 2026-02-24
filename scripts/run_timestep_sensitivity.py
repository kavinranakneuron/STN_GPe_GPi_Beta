"""
Timestep sensitivity analysis.

Tests dt = 0.025, 0.0125, 0.00625 ms to verify numerical convergence.
Runs both healthy and Parkinsonian parameter sets at each timestep,
compares firing rates, CV, and beta fraction.

Uses STN beta as the primary beta metric.

Author: Kavin Nakkeeran, Johns Hopkins University
"""

import sys
sys.path.insert(0, '.')

import gc
import numpy as np
import jax
import jax.numpy as jnp
import time
import pickle
import platform
from datetime import datetime

from jax_models.network_builder import build_network_state
from optimization.sim_jax import apply_params_to_config
from optimization.metrics_jax import compute_all_metrics, compute_beta_fraction_all
from jax_models.integrator import network_step
from jax import lax

print(f"JAX devices: {jax.devices()}")

# =============================================================================
# CONFIGURATION
# =============================================================================

DT_VALUES = [0.025, 0.0125]

N_STN, N_GPE, N_GPI = 10000, 20000, 15000

SIM_DURATION_MS = 600.0
BURN_IN_MS = 100.0

HEALTHY_PARAMS = {
    'ISTN': 132.235, 'I_gpe': 3.039, 'I_gpi': 2.209,
    'noise_stn_sigma': 3.362, 'noise_gpe_sigma': 33.417, 'noise_gpi_sigma': 67.284,
    'g_stn_gpe_mult': 1.866, 'g_gpe_stn_mult': 0.999,
    'g_stn_gpi_mult': 1.834, 'g_gpe_gpi_mult': 0.687,
}

PD_PARAMS = {
    'ISTN': 70.048, 'I_gpe': 1.663, 'I_gpi': 1.670,
    'noise_stn_sigma': 1.971, 'noise_gpe_sigma': 66.383, 'noise_gpi_sigma': 96.429,
    'g_stn_gpe_mult': 4.132, 'g_gpe_stn_mult': 0.114,
    'g_stn_gpi_mult': 1.943, 'g_gpe_gpi_mult': 0.996,
}

# =============================================================================
# SIMULATOR WITH SYNAPTIC MULTIPLIERS
# =============================================================================

def create_simulator(base_config, n_steps):
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

        # Create simulator (applies multipliers for both conditions)
        simulator = create_simulator(config, n_steps)

        # Warm up JIT
        print("  JIT compiling...", end=" ", flush=True)
        t0 = time.time()
        obs = simulator(HEALTHY_PARAMS, state)
        obs['V_stn'].block_until_ready()
        jit_time = time.time() - t0
        print(f"{jit_time:.1f}s")

        del obs
        gc.collect()

        # Run healthy
        print("  Running healthy...", end=" ", flush=True)
        t0 = time.time()
        obs_h = simulator(HEALTHY_PARAMS, state)
        obs_h['V_stn'].block_until_ready()
        h_wall = time.time() - t0
        print(f"{h_wall:.2f}s")

        metrics_h = compute_all_metrics(obs_h, dt, burn_steps=burn_steps)
        beta_h = compute_beta_fraction_all(obs_h, dt, burn_steps=burn_steps)

        del obs_h
        gc.collect()
        jax.clear_caches()

        # Run PD
        print("  Running Parkinsonian...", end=" ", flush=True)
        t0 = time.time()
        obs_pd = simulator(PD_PARAMS, state)
        obs_pd['V_stn'].block_until_ready()
        pd_wall = time.time() - t0
        print(f"{pd_wall:.2f}s")

        metrics_pd = compute_all_metrics(obs_pd, dt, burn_steps=burn_steps)
        beta_pd = compute_beta_fraction_all(obs_pd, dt, burn_steps=burn_steps)

        del obs_pd
        gc.collect()
        jax.clear_caches()

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
            'jit_time_s': jit_time,
        }

        print(f"  Healthy: STN={metrics_h['firing_rates']['stn']:.1f}Hz, "
              f"GPe={metrics_h['firing_rates']['gpe']:.1f}Hz, "
              f"STN beta={beta_h['stn']*100:.1f}%")
        print(f"  PD:      STN={metrics_pd['firing_rates']['stn']:.1f}Hz, "
              f"GPe={metrics_pd['firing_rates']['gpe']:.1f}Hz, "
              f"STN beta={beta_pd['stn']*100:.1f}%")

    # =========================================================================
    # CONVERGENCE TABLE
    # =========================================================================

    print("\n" + "=" * 80)
    print("CONVERGENCE TABLE -- HEALTHY")
    print("=" * 80)
    header = f"{'dt (ms)':<10} {'STN Hz':<10} {'GPe Hz':<10} {'GPi Hz':<10} {'STN CV':<10} {'GPe CV':<10} {'STN beta%':<10}"
    print(header)
    print("-" * 80)
    for dt in DT_VALUES:
        r = results[dt]['healthy']
        print(f"{dt:<10.5f} {r['firing_rates']['stn']:<10.1f} {r['firing_rates']['gpe']:<10.1f} "
              f"{r['firing_rates']['gpi']:<10.1f} {r['cv']['stn']:<10.3f} {r['cv']['gpe']:<10.3f} "
              f"{r['beta_fraction']['stn']*100:<10.1f}")

    print("\n" + "=" * 80)
    print("CONVERGENCE TABLE -- PARKINSONIAN")
    print("=" * 80)
    print(header)
    print("-" * 80)
    for dt in DT_VALUES:
        r = results[dt]['pd']
        print(f"{dt:<10.5f} {r['firing_rates']['stn']:<10.1f} {r['firing_rates']['gpe']:<10.1f} "
              f"{r['firing_rates']['gpi']:<10.1f} {r['cv']['stn']:<10.3f} {r['cv']['gpe']:<10.3f} "
              f"{r['beta_fraction']['stn']*100:<10.1f}")

    # Max relative change between finest two timesteps
    ref_dt = DT_VALUES[-1]
    prev_dt = DT_VALUES[-2]
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
        # Also check STN beta convergence
        beta_ref = ref['beta_fraction']['stn']
        beta_prev = prev['beta_fraction']['stn']
        if beta_ref > 0:
            beta_change = abs(beta_ref - beta_prev) / beta_ref * 100
        else:
            beta_change = 0.0
        print(f"  {condition}: {max_change:.1f}% (rates), {beta_change:.1f}% (STN beta)")

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
