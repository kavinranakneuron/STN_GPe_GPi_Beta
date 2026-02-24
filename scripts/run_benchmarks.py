"""
Performance benchmarking -- JAX vs NumPy baseline.

Measures wall-clock time for JAX JIT simulation at multiple network sizes
and compares against the NumPy CPU baseline at 450 neurons.

Both conditions use the multiplier-applying simulator pattern.

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
from optimization.sim_jax import apply_params_to_config
from jax_models.integrator import network_step
from jax import lax

print(f"JAX devices: {jax.devices()}")

# =============================================================================
# CONFIGURATION
# =============================================================================

DT_MS = 0.025
SIM_DURATION_MS = 600.0
N_STEPS = int(SIM_DURATION_MS / DT_MS)  # 24000

NETWORK_SIZES = [
    (100,   200,   150),    # 450 neurons
    (400,   800,   600),    # 1800 neurons
    (1000,  2000,  1500),   # 4500 neurons
    (3334,  6666,  5000),   # 15000 neurons
    (10000, 20000, 15000),  # 45000 neurons
]

BENCH_PARAMS = {
    'ISTN': 132.235, 'I_gpe': 3.039, 'I_gpi': 2.209,
    'noise_stn_sigma': 3.362, 'noise_gpe_sigma': 33.417, 'noise_gpi_sigma': 67.284,
    'g_stn_gpe_mult': 1.866, 'g_gpe_stn_mult': 0.999,
    'g_stn_gpi_mult': 1.834, 'g_gpe_gpi_mult': 0.687,
}

N_WARMUP = 1
N_REPEATS = 3

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
# JAX BENCHMARKS
# =============================================================================

def benchmark_jax(n_stn, n_gpe, n_gpi):
    total = n_stn + n_gpe + n_gpi
    print(f"\n  JAX {total} neurons ({n_stn}/{n_gpe}/{n_gpi})...")

    state, config = build_network_state(n_stn, n_gpe, n_gpi, DT_MS, seed=42)
    simulator = create_simulator(config, N_STEPS)

    # JIT warmup
    print(f"    Compiling...", end=" ", flush=True)
    t0 = time.time()
    obs = simulator(BENCH_PARAMS, state)
    obs['V_stn'].block_until_ready()
    jit_compile_time = time.time() - t0
    print(f"{jit_compile_time:.1f}s")

    for _ in range(N_WARMUP - 1):
        obs = simulator(BENCH_PARAMS, state)
        obs['V_stn'].block_until_ready()

    # Timed runs
    times = []
    for i in range(N_REPEATS):
        t0 = time.time()
        obs = simulator(BENCH_PARAMS, state)
        obs['V_stn'].block_until_ready()
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"    Run {i+1}: {elapsed:.3f}s")

    median_time = float(np.median(times))
    print(f"    Median: {median_time:.3f}s")

    return {
        'n_stn': n_stn,
        'n_gpe': n_gpe,
        'n_gpi': n_gpi,
        'n_total': total,
        'jit_compile_s': jit_compile_time,
        'run_times_s': times,
        'median_s': median_time,
        'sim_duration_ms': SIM_DURATION_MS,
        'n_steps': N_STEPS,
    }


# =============================================================================
# NUMPY BASELINE
# =============================================================================

def benchmark_numpy_baseline():
    """Benchmark NumPy CPU baseline at 450 neurons using a Python loop."""
    n_stn, n_gpe, n_gpi = 100, 200, 150
    total = n_stn + n_gpe + n_gpi
    print(f"\n  NumPy baseline {total} neurons (Python loop)...")

    state, config = build_network_state(n_stn, n_gpe, n_gpi, DT_MS, seed=42)

    # Apply params including multipliers
    config = apply_params_to_config(BENCH_PARAMS, config)
    syn_configs = dict(config['synapses'])
    for syn_name, mult_name in [
        ('stn_to_gpe', 'g_stn_gpe_mult'),
        ('gpe_to_stn', 'g_gpe_stn_mult'),
        ('stn_to_gpi', 'g_stn_gpi_mult'),
        ('gpe_to_gpi', 'g_gpe_gpi_mult'),
    ]:
        old_cfg = syn_configs[syn_name]
        new_weights = old_cfg.weights * BENCH_PARAMS.get(mult_name, 1.0)
        syn_configs[syn_name] = old_cfg._replace(weights=new_weights)
    config['synapses'] = syn_configs

    # Shorter simulation for NumPy (it's slow)
    numpy_steps = min(N_STEPS, 4000)  # 100ms max
    numpy_sim_ms = numpy_steps * DT_MS

    print(f"    Running {numpy_steps} steps ({numpy_sim_ms}ms)...")
    t0 = time.time()
    current_state = state
    for i in range(numpy_steps):
        t_ms = i * DT_MS
        current_state, obs = network_step(current_state, config, t_ms)
    obs['V_stn'].block_until_ready()
    elapsed = time.time() - t0

    extrapolated = elapsed * (N_STEPS / numpy_steps)

    print(f"    Time ({numpy_steps} steps): {elapsed:.2f}s")
    print(f"    Extrapolated ({N_STEPS} steps): {extrapolated:.1f}s")

    return {
        'n_total': total,
        'actual_steps': numpy_steps,
        'actual_time_s': elapsed,
        'extrapolated_time_s': extrapolated,
        'sim_duration_ms': numpy_sim_ms,
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 70)
    print(f"Simulation: {SIM_DURATION_MS}ms ({N_STEPS} steps @ dt={DT_MS}ms)")
    print(f"Repeats: {N_REPEATS} (median)")
    print(f"Sizes: {[sum(s) for s in NETWORK_SIZES]} neurons")
    print("=" * 70)

    # JAX benchmarks
    jax_results = []
    for sizes in NETWORK_SIZES:
        try:
            result = benchmark_jax(*sizes)
            jax_results.append(result)
        except Exception as e:
            print(f"    FAILED: {e}")
            jax_results.append({
                'n_stn': sizes[0], 'n_gpe': sizes[1], 'n_gpi': sizes[2],
                'n_total': sum(sizes), 'error': str(e),
            })
            jax.clear_caches()

    # NumPy baseline
    print("\n" + "-" * 50)
    numpy_result = benchmark_numpy_baseline()

    # =========================================================================
    # PERFORMANCE TABLE
    # =========================================================================

    baseline_time = numpy_result['extrapolated_time_s']

    print("\n" + "=" * 70)
    print("PERFORMANCE TABLE")
    print("=" * 70)
    print(f"{'Neurons':<10} {'Backend':<10} {'Sim (ms)':<10} {'Wall (s)':<12} {'Speedup':<10}")
    print("-" * 52)

    print(f"{numpy_result['n_total']:<10} {'NumPy':<10} {SIM_DURATION_MS:<10.0f} "
          f"{baseline_time:<12.2f} {'1.0x (ref)':<10}")

    for r in jax_results:
        if 'error' in r:
            print(f"{r['n_total']:<10} {'JAX/JIT':<10} {'FAILED':<10}")
            continue
        speedup = baseline_time / r['median_s'] if r['n_total'] == 450 else None
        speedup_str = f"{speedup:.0f}x" if speedup else "--"
        print(f"{r['n_total']:<10} {'JAX/JIT':<10} {SIM_DURATION_MS:<10.0f} "
              f"{r['median_s']:<12.3f} {speedup_str:<10}")

    # JIT compile times
    print(f"\nJIT compile times:")
    for r in jax_results:
        if 'error' not in r:
            print(f"  {r['n_total']} neurons: {r['jit_compile_s']:.1f}s")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================

    metadata = {
        'date': datetime.now().isoformat(),
        'dt_ms': DT_MS,
        'sim_duration_ms': SIM_DURATION_MS,
        'n_steps': N_STEPS,
        'n_repeats': N_REPEATS,
        'jax_version': jax.__version__,
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'jax_devices': [str(d) for d in jax.devices()],
    }

    output = {
        'jax_results': jax_results,
        'numpy_baseline': numpy_result,
        'bench_params': BENCH_PARAMS,
        'metadata': metadata,
    }

    with open('results/benchmarks/performance_table.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    with open('results/benchmarks/performance_table.pkl', 'wb') as f:
        pickle.dump(output, f)

    print(f"\nSaved: results/benchmarks/performance_table.json")
    print(f"Saved: results/benchmarks/performance_table.pkl")
