"""
Scale invariance study: verify that fixed-indegree connectivity preserves
network dynamics across network sizes from 450 to 45000 neurons.

Tests both healthy and Parkinsonian conditions at each size, recording
firing rates, CVs, beta fractions, and wall-clock times.
"""

import sys
sys.path.insert(0, '.')

import jax
import jax.numpy as jnp
from jax import lax
import time
import pickle
import traceback

from jax_models.network_builder import build_network_state
from optimization.sim_jax import apply_params_to_config
from jax_models.integrator import network_step
from optimization.metrics_jax import compute_all_metrics, compute_beta_fraction_all

print(f"JAX devices: {jax.devices()}")

# =============================================================================
# CONFIGURATION
# =============================================================================

DT_MS = 0.025
N_STEPS = 24000   # 600ms
BURN_STEPS = 4000  # 100ms burn-in

# Network sizes (STN:GPe:GPi = 2:4:3)
SIZES = [
    (100,   200,   150,   "450n"),
    (400,   800,   600,   "1800n"),
    (1000,  2000,  1500,  "4500n"),
    (3334,  6666,  5000,  "15000n"),
    (10000, 20000, 15000, "45000n"),
]

# Parameters from 450n optimization
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
# SIMULATOR FACTORY
# =============================================================================

def create_simulator(base_config, n_steps):
    """JIT-compiled simulator with synaptic multiplier support."""

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

        final_state, obs_history = lax.scan(step_fn, init=init_state, xs=jnp.arange(n_steps))
        return obs_history

    return simulate

# =============================================================================
# RUN STUDY
# =============================================================================

def run_condition(simulator, params, state, label):
    """Run a single simulation, return metrics and timing."""
    print(f"    Running {label}...", end=" ", flush=True)

    # Warm run (JIT compile)
    t_compile_start = time.time()
    obs = simulator(params, state)
    obs['V_stn'].block_until_ready()
    compile_time = time.time() - t_compile_start
    print(f"JIT {compile_time:.1f}s,", end=" ", flush=True)

    # Timed run (cached)
    t_run_start = time.time()
    obs = simulator(params, state)
    obs['V_stn'].block_until_ready()
    run_time = time.time() - t_run_start
    print(f"run {run_time:.3f}s,", end=" ", flush=True)

    # Metrics
    metrics = compute_all_metrics(obs, DT_MS, burn_steps=BURN_STEPS)
    beta_fractions = compute_beta_fraction_all(obs, DT_MS, burn_steps=BURN_STEPS)

    result = {
        'firing_rates': dict(metrics['firing_rates']),
        'cv': dict(metrics['cv']),
        'beta': dict(beta_fractions),
        'compile_time_s': compile_time,
        'run_time_s': run_time,
    }

    r = result['firing_rates']
    b = result['beta']
    print(f"STN={r['stn']:.1f}Hz GPe={r['gpe']:.1f}Hz GPi={r['gpi']:.1f}Hz beta={b['gpe']*100:.1f}%")

    return result


if __name__ == "__main__":
    print("=" * 80)
    print("SCALING INVARIANCE STUDY")
    print("Fixed-indegree connectivity across network sizes")
    print("=" * 80)
    print(f"Sizes: {', '.join(label for _, _, _, label in SIZES)}")
    print(f"Simulation: {N_STEPS} steps ({N_STEPS * DT_MS:.0f}ms), burn-in {BURN_STEPS} steps")
    print("=" * 80)

    all_results = {}

    for n_stn, n_gpe, n_gpi, label in SIZES:
        total_n = n_stn + n_gpe + n_gpi
        print(f"\n{'='*80}")
        print(f"  {label}: {n_stn} STN / {n_gpe} GPe / {n_gpi} GPi ({total_n} total)")
        print(f"{'='*80}")

        try:
            # Build network
            print(f"  Building network...", end=" ", flush=True)
            t0 = time.time()
            state, config = build_network_state(n_stn, n_gpe, n_gpi, DT_MS)
            build_time = time.time() - t0
            print(f"{build_time:.1f}s")

            # Fresh simulator for this size
            simulator = create_simulator(config, N_STEPS)

            # Healthy
            healthy_result = run_condition(simulator, HEALTHY_PARAMS, state, "Healthy")

            # PD
            pd_result = run_condition(simulator, PD_PARAMS, state, "PD")

            all_results[label] = {
                'n_stn': n_stn, 'n_gpe': n_gpe, 'n_gpi': n_gpi,
                'build_time_s': build_time,
                'healthy': healthy_result,
                'pd': pd_result,
            }

        except Exception as e:
            print(f"\n  FAILED: {e}")
            traceback.print_exc()
            all_results[label] = {'error': str(e)}
            # Clear JAX caches to free memory before next size
            jax.clear_caches()
            continue

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================

    print("\n\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)

    # Healthy table
    print("\n--- HEALTHY ---")
    header = f"{'Size':>8s} | {'STN Hz':>7s} {'GPe Hz':>7s} {'GPi Hz':>7s} | {'CV_STN':>6s} {'CV_GPe':>6s} {'CV_GPi':>6s} | {'B_STN%':>6s} {'B_GPe%':>6s} {'B_GPi%':>6s} | {'JIT(s)':>7s} {'Run(s)':>7s}"
    print(header)
    print("-" * len(header))
    for _, _, _, label in SIZES:
        res = all_results.get(label)
        if res is None or 'error' in res:
            print(f"{label:>8s} | {'FAILED':^73s} |")
            continue
        h = res['healthy']
        r, c, b = h['firing_rates'], h['cv'], h['beta']
        print(f"{label:>8s} | {r['stn']:7.1f} {r['gpe']:7.1f} {r['gpi']:7.1f} | "
              f"{c['stn']:6.2f} {c['gpe']:6.2f} {c['gpi']:6.2f} | "
              f"{b['stn']*100:6.1f} {b['gpe']*100:6.1f} {b['gpi']*100:6.1f} | "
              f"{h['compile_time_s']:7.1f} {h['run_time_s']:7.3f}")

    # PD table
    print("\n--- PARKINSONIAN ---")
    print(header)
    print("-" * len(header))
    for _, _, _, label in SIZES:
        res = all_results.get(label)
        if res is None or 'error' in res:
            print(f"{label:>8s} | {'FAILED':^73s} |")
            continue
        p = res['pd']
        r, c, b = p['firing_rates'], p['cv'], p['beta']
        print(f"{label:>8s} | {r['stn']:7.1f} {r['gpe']:7.1f} {r['gpi']:7.1f} | "
              f"{c['stn']:6.2f} {c['gpe']:6.2f} {c['gpi']:6.2f} | "
              f"{b['stn']*100:6.1f} {b['gpe']*100:6.1f} {b['gpi']*100:6.1f} | "
              f"{p['compile_time_s']:7.1f} {p['run_time_s']:7.3f}")

    # Paper-ready table (LaTeX-friendly)
    print("\n\n--- PAPER TABLE (copy-paste ready) ---")
    print(f"{'Size':>8s}  {'Cond':>7s}  {'STN':>5s}  {'GPe':>5s}  {'GPi':>5s}  "
          f"{'CV_S':>5s}  {'CV_G':>5s}  {'CV_I':>5s}  {'Beta%':>5s}")
    print("-" * 68)
    for _, _, _, label in SIZES:
        res = all_results.get(label)
        if res is None or 'error' in res:
            continue
        for cond_name, cond_key in [("Healthy", "healthy"), ("PD", "pd")]:
            d = res[cond_key]
            r, c, b = d['firing_rates'], d['cv'], d['beta']
            print(f"{label:>8s}  {cond_name:>7s}  {r['stn']:5.1f}  {r['gpe']:5.1f}  {r['gpi']:5.1f}  "
                  f"{c['stn']:5.2f}  {c['gpe']:5.2f}  {c['gpi']:5.2f}  {b['gpe']*100:5.1f}")

    # =========================================================================
    # SAVE
    # =========================================================================

    save_data = {
        'results': all_results,
        'healthy_params': HEALTHY_PARAMS,
        'pd_params': PD_PARAMS,
        'config': {
            'dt_ms': DT_MS,
            'n_steps': N_STEPS,
            'burn_steps': BURN_STEPS,
            'sizes': SIZES,
        },
    }

    with open('results/validation/scaling_study.pkl', 'wb') as f:
        pickle.dump(save_data, f)

    print(f"\nResults saved to results/validation/scaling_study.pkl")
