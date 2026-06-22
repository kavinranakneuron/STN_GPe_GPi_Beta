#!/usr/bin/env python3
"""
Diagnose the source of baseline disagreement across the three paper runs.

Run on the VM: cd ~/jne-revision/submitted && python3 scripts/diagnose_determinism.py

Two questions:
  D1. Is the simulator deterministic for a FIXED param vector?
      (tests GPU float nondeterminism — suspect #2)
  D2. Does CMA-ES with a fixed seed produce the SAME trial sequence twice?
      (tests sampler startup nondeterminism — suspect #1)
"""
import sys
sys.path.insert(0, '.')
import jax
import jax.numpy as jnp

from scripts.sweep_driver import (
    build_simulator, make_objective, DEFAULT_BOUNDS, DEFAULT_WEIGHTS
)

print("=" * 70)
print("D1: SIMULATOR DETERMINISM (fixed param vector, run 3x)")
print("=" * 70)
sim, base_state, _ = build_simulator()
params = {k: 0.5 * (lo + hi) for k, (lo, hi) in DEFAULT_BOUNDS.items()}

sums = []
for i in range(3):
    o = sim(params, base_state)
    o['V_stn'].block_until_ready()
    s = float(jnp.sum(o['V_stn']))
    sums.append(s)
    print(f"  run {i}: sum(V_stn) = {s:.6f}")

print(f"  All identical: {len(set(sums)) == 1}")
if len(set(sums)) != 1:
    spread = max(sums) - min(sums)
    print(f"  Spread: {spread:.6e}  <-- GPU float nondeterminism (suspect #2)")
print()

print("=" * 70)
print("D2: CMA-ES SEED REPRODUCIBILITY (same seed, 30 trials, run 2x)")
print("=" * 70)
import optuna
from optuna.samplers import CmaEsSampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

obj = make_objective(sim, base_state, {k: list(v) for k, v in DEFAULT_BOUNDS.items()},
                     dict(DEFAULT_WEIGHTS))

def run_study(seed, n=30):
    study = optuna.create_study(direction='minimize',
                                sampler=CmaEsSampler(seed=seed))
    study.optimize(obj, n_trials=n, show_progress_bar=False)
    # Return first-5 trial param vectors (the startup region) and best
    first5 = [t.params['g_stn_gpe_mult'] for t in study.trials[:5]]
    return first5, study.best_value

f5_a, best_a = run_study(42)
f5_b, best_b = run_study(42)

print("  First-5 g_stn_gpe_mult, study A:", [f"{x:.3f}" for x in f5_a])
print("  First-5 g_stn_gpe_mult, study B:", [f"{x:.3f}" for x in f5_b])
print(f"  Startup identical: {f5_a == f5_b}")
print(f"  Best A: {best_a:.4f}   Best B: {best_b:.4f}   Match: {abs(best_a-best_b)<1e-9}")
print()

print("=" * 70)
print("INTERPRETATION")
print("=" * 70)
print("""
If D1 NOT identical  -> GPU scatter-add float nondeterminism. The simulator
                        gives slightly different output each call. Over 1000
                        trials CMA-ES walks different paths. This is the cause.
                        Fix options: (a) accept it + characterize as stochastic
                        objective, (b) set XLA determinism flags and re-run.

If D1 identical but D2 startup differs -> CmaEsSampler isn't fully seeded
                        (independent startup sampler). Fix: pass
                        n_startup_trials=0 or seed the independent sampler.

If BOTH identical    -> the cross-run disagreement came from something
                        environmental between the separate process launches
                        (unlikely given frozen seed; would need deeper look).
""")
