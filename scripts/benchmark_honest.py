"""
Honest performance benchmark for JNE-110355 revision.

Three same-model JAX scenarios (GPU-jit, CPU-jit, GPU-nojit) measured at full
600 ms, plus a separately-labeled NumPy/AdEx reference (different model).

Run TWICE with different device env:
  # GPU scenarios (1 and 3):
  python scripts/benchmark_honest.py --mode gpu
  # CPU scenario (2):
  JAX_PLATFORMS=cpu python scripts/benchmark_honest.py --mode cpu

Results append to results/benchmarks/honest_benchmark.json
"""
import sys
sys.path.insert(0, '.')
import os, gc, json, time, platform, argparse
from datetime import datetime
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from jax_models.network_builder import build_network_state
from optimization.sim_jax import apply_params_to_config
from jax_models.integrator import network_step

DT_MS = 0.025
SIM_MS = 600.0
N_STEPS = int(SIM_MS / DT_MS)  # 24000
SIZE = (100, 200, 150)         # 450 neurons

BENCH_PARAMS = {
    'ISTN': 132.235, 'I_gpe': 3.039, 'I_gpi': 2.209,
    'noise_stn_sigma': 3.362, 'noise_gpe_sigma': 33.417, 'noise_gpi_sigma': 67.284,
    'g_stn_gpe_mult': 1.866, 'g_gpe_stn_mult': 0.999,
    'g_stn_gpi_mult': 1.834, 'g_gpe_gpi_mult': 0.687,
}

def _apply_multipliers(config, params):
    syn = dict(config['synapses'])
    for name, mult in [('stn_to_gpe','g_stn_gpe_mult'),('gpe_to_stn','g_gpe_stn_mult'),
                       ('stn_to_gpi','g_stn_gpi_mult'),('gpe_to_gpi','g_gpe_gpi_mult')]:
        old = syn[name]
        syn[name] = old._replace(weights=old.weights * params.get(mult, 1.0))
    config['synapses'] = syn
    return config

def make_jit_sim(base_config, n_steps):
    @jax.jit
    def sim(params, init_state):
        config = _apply_multipliers(apply_params_to_config(params, base_config), params)
        def step(carry, t_idx):
            return network_step(carry, config, t_idx * config['dt_ms'])
        _, obs = lax.scan(step, init=init_state, xs=jnp.arange(n_steps))
        return obs
    return sim

def time_jit(device_label, n_steps=N_STEPS, repeats=3):
    state, config = build_network_state(*SIZE, DT_MS, seed=42)
    sim = make_jit_sim(config, n_steps)
    t0 = time.time()
    obs = sim(BENCH_PARAMS, state); obs['V_stn'].block_until_ready()
    compile_s = time.time() - t0
    times = []
    for _ in range(repeats):
        t0 = time.time()
        obs = sim(BENCH_PARAMS, state); obs['V_stn'].block_until_ready()
        times.append(time.time() - t0)
    gc.collect(); jax.clear_caches()
    return {'scenario': f'JAX {device_label} (JIT+scan)', 'measured_steps': n_steps,
            'measured_ms': SIM_MS, 'compile_s': compile_s,
            'run_times_s': times, 'median_s': float(np.median(times)),
            'extrapolated': False}

def time_nojit_gpu(n_steps=N_STEPS):
    """Un-jitted Python loop, same JAX model, on whatever device is active."""
    state, config = build_network_state(*SIZE, DT_MS, seed=42)
    config = _apply_multipliers(apply_params_to_config(BENCH_PARAMS, config), BENCH_PARAMS)
    t0 = time.time()
    cur = state
    for i in range(n_steps):
        cur, obs = network_step(cur, config, i * DT_MS)
    obs['V_stn'].block_until_ready()
    elapsed = time.time() - t0
    gc.collect(); jax.clear_caches()
    return {'scenario': 'JAX GPU (no-jit Python loop)', 'measured_steps': n_steps,
            'measured_ms': SIM_MS, 'run_times_s': [elapsed], 'median_s': elapsed,
            'extrapolated': False}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['gpu','cpu'])
    args = ap.parse_args()

    dev = [str(d) for d in jax.devices()]
    print(f"Mode: {args.mode}  Devices: {dev}")
    # Safety check: in cpu mode, confirm we are actually on CPU
    if args.mode == 'cpu' and any('cuda' in d.lower() for d in dev):
        print("ERROR: --mode cpu but CUDA device active. Set JAX_PLATFORMS=cpu.")
        sys.exit(1)
    if args.mode == 'gpu' and not any('cuda' in d.lower() for d in dev):
        print("ERROR: --mode gpu but no CUDA device. Check GPU availability.")
        sys.exit(1)

    results = []
    if args.mode == 'gpu':
        print("\n[1/2] JAX GPU JIT+scan, full 600ms...")
        results.append(time_jit('GPU'))
        print(f"   median {results[-1]['median_s']:.3f}s, compile {results[-1]['compile_s']:.1f}s")
        print("\n[2/2] JAX GPU no-jit Python loop, full 600ms (SLOW ~14min)...")
        results.append(time_nojit_gpu())
        print(f"   {results[-1]['median_s']:.1f}s")
    else:
        print("\n[1/1] JAX CPU JIT+scan, full 600ms...")
        results.append(time_jit('CPU'))
        print(f"   median {results[-1]['median_s']:.3f}s, compile {results[-1]['compile_s']:.1f}s")

    meta = {'date': datetime.now().isoformat(), 'mode': args.mode,
            'devices': dev, 'platform': platform.platform(),
            'jax_version': jax.__version__, 'cpu_count': os.cpu_count(),
            'dt_ms': DT_MS, 'sim_ms': SIM_MS, 'n_steps': N_STEPS, 'size': SIZE}

    outpath = 'results/benchmarks/honest_benchmark.json'
    existing = []
    if os.path.exists(outpath):
        existing = json.load(open(outpath)).get('runs', [])
    existing.append({'meta': meta, 'results': results})
    json.dump({'runs': existing}, open(outpath, 'w'), indent=2, default=str)
    print(f"\nAppended to {outpath}")

if __name__ == '__main__':
    main()
