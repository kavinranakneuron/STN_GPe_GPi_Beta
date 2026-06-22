"""
Sweep driver for JNE-110355 sensitivity analysis.

Factored version of run_parkinsonian_optimization.py that accepts
bounds, weight, and seed overrides. Loss function is logically identical
to the submitted script (hidden constraint penalties retained).

Usage:
    python scripts/sweep_driver.py --condition <name> \
        --output results/sensitivity/<sweep>/<condition>.pkl \
        [--bounds-override bounds.json] \
        [--weights-override weights.json] \
        [--seed 42] \
        [--n-trials 1000]
"""

import sys
sys.path.insert(0, '.')

import argparse
import json
import time
import pickle
from pathlib import Path

import optuna
from optuna.samplers import CmaEsSampler
import jax
import jax.numpy as jnp

from jax_models.network_builder import build_network_state
from optimization.metrics_jax import compute_all_metrics, compute_beta_fraction_all


DEFAULT_BOUNDS = {
    'ISTN':            [60.0, 150.0],
    'I_gpe':           [0.5, 4.0],
    'I_gpi':           [1.0, 5.0],
    'noise_stn_sigma': [1.0, 8.0],
    'noise_gpe_sigma': [20.0, 150.0],
    'noise_gpi_sigma': [20.0, 150.0],
    'g_stn_gpe_mult':  [1.5, 5.0],
    'g_gpe_stn_mult':  [0.1, 0.8],
    'g_stn_gpi_mult':  [1.0, 4.0],
    'g_gpe_gpi_mult':  [0.2, 1.2],
}

DEFAULT_WEIGHTS = {'rate': 5.0, 'cv': 0.2, 'beta': 15.0}

TARGETS = {
    'rate_stn': 27.5,
    'rate_gpe': 42.5,
    'rate_gpi': 82.5,
    'cv_stn': 0.60,
    'cv_gpe': 0.35,
    'cv_gpi': 0.25,
    'beta_gpe': 0.20,
}


def build_simulator(n_stn=100, n_gpe=200, n_gpi=150, dt_ms=0.025, n_steps=16000):
    from optimization.sim_jax import apply_params_to_config
    from jax_models.integrator import network_step
    from jax import lax

    base_state, base_config = build_network_state(n_stn, n_gpe, n_gpi, dt_ms)

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

    return simulate, base_state, base_config


def make_objective(simulator, base_state, bounds, weights, dt_ms=0.025, burn_steps=4000):
    """Build objective with given bounds and weights. Logic mirrors the
    submitted run_parkinsonian_optimization.py exactly."""
    def objective(trial):
        params = {
            name: trial.suggest_float(name, lo, hi)
            for name, (lo, hi) in bounds.items()
        }
        try:
            obs = simulator(params, base_state)
            obs['V_stn'].block_until_ready()

            if jnp.any(jnp.isnan(obs['V_stn'])) or jnp.any(jnp.isnan(obs['V_gpe'])):
                return 1e6

            metrics = compute_all_metrics(obs, dt_ms, burn_steps=burn_steps)
            beta_fractions = compute_beta_fraction_all(obs, dt_ms, burn_steps=burn_steps)

            r_stn = metrics['firing_rates']['stn']
            r_gpe = metrics['firing_rates']['gpe']
            r_gpi = metrics['firing_rates']['gpi']
            cv_stn = metrics['cv']['stn']
            cv_gpe = metrics['cv']['gpe']
            cv_gpi = metrics['cv']['gpi']
            beta_gpe = beta_fractions['gpe']
            beta_stn = beta_fractions['stn']

            loss = 0.0
            loss += weights['rate'] * ((r_stn - TARGETS['rate_stn']) / TARGETS['rate_stn']) ** 2
            loss += weights['rate'] * ((r_gpe - TARGETS['rate_gpe']) / TARGETS['rate_gpe']) ** 2
            loss += weights['rate'] * ((r_gpi - TARGETS['rate_gpi']) / TARGETS['rate_gpi']) ** 2
            loss += weights['cv'] * ((cv_stn - TARGETS['cv_stn']) / TARGETS['cv_stn']) ** 2
            loss += weights['cv'] * ((cv_gpe - TARGETS['cv_gpe']) / TARGETS['cv_gpe']) ** 2
            loss += weights['cv'] * ((cv_gpi - TARGETS['cv_gpi']) / TARGETS['cv_gpi']) ** 2

            if beta_gpe < TARGETS['beta_gpe']:
                loss += weights['beta'] * ((TARGETS['beta_gpe'] - beta_gpe) / TARGETS['beta_gpe']) ** 2
            if beta_gpe > 0.40:
                loss += 10.0 * ((beta_gpe - 0.40) / 0.40) ** 2

            if r_gpe > 55.0:
                loss += 5.0 * ((r_gpe - 55.0) / 20.0) ** 2
            if r_stn < 18.0:
                loss += 5.0 * ((18.0 - r_stn) / 18.0) ** 2
            if r_stn < 3.0 or r_gpe < 5.0 or r_gpi < 10.0:
                loss += 100.0

            trial.set_user_attr('r_stn', float(r_stn))
            trial.set_user_attr('r_gpe', float(r_gpe))
            trial.set_user_attr('r_gpi', float(r_gpi))
            trial.set_user_attr('cv_stn', float(cv_stn))
            trial.set_user_attr('cv_gpe', float(cv_gpe))
            trial.set_user_attr('cv_gpi', float(cv_gpi))
            trial.set_user_attr('beta_gpe', float(beta_gpe))
            trial.set_user_attr('beta_stn', float(beta_stn))

            return float(loss)
        except Exception as e:
            print(f"Trial failed: {e}")
            return 1e6

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--bounds-override', default=None,
                        help='Path to JSON with per-param [lo, hi] entries to override')
    parser.add_argument('--weights-override', default=None,
                        help='Path to JSON with {rate, cv, beta} entries to override')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-trials', type=int, default=1000)
    parser.add_argument('--k-stn-gpe', type=int, default=15)
    parser.add_argument('--k-gpe-stn', type=int, default=14)
    parser.add_argument('--k-stn-gpi', type=int, default=30)
    parser.add_argument('--k-gpe-gpi', type=int, default=10)
    args = parser.parse_args()

    bounds = {k: list(v) for k, v in DEFAULT_BOUNDS.items()}
    if args.bounds_override:
        with open(args.bounds_override) as f:
            overrides = json.load(f)
        for k, v in overrides.items():
            if k not in bounds:
                raise KeyError(f"Unknown bound key: {k}")
            bounds[k] = v

    weights = dict(DEFAULT_WEIGHTS)
    if args.weights_override:
        with open(args.weights_override) as f:
            weights.update(json.load(f))

    print(f"Condition: {args.condition}")
    print(f"Seed: {args.seed}, Trials: {args.n_trials}")
    print(f"Bounds:")
    for k, v in bounds.items():
        print(f"  {k}: {v}")
    print(f"Weights: {weights}")
    print(f"JAX devices: {jax.devices()}")

    import jax_models.network_builder as nb
    nb.K_STN_GPE = args.k_stn_gpe
    nb.K_GPE_STN = args.k_gpe_stn
    nb.K_STN_GPI = args.k_stn_gpi
    nb.K_GPE_GPI = args.k_gpe_gpi
    print(f"Connectivity (indegrees): STN->GPe={nb.K_STN_GPE} GPe->STN={nb.K_GPE_STN} "
          f"STN->GPi={nb.K_STN_GPI} GPe->GPi={nb.K_GPE_GPI}")

    print("\nBuilding simulator...")
    simulator, base_state, _ = build_simulator()

    print("Warming up JIT...")
    dummy = {k: 0.5 * (lo + hi) for k, (lo, hi) in bounds.items()}
    obs = simulator(dummy, base_state)
    obs['V_stn'].block_until_ready()
    print("JIT ready.\n")

    objective = make_objective(simulator, base_state, bounds, weights)

    study = optuna.create_study(
        direction='minimize',
        sampler=CmaEsSampler(seed=args.seed),
        study_name=f'sensitivity_{args.condition}',
    )

    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
    elapsed = time.time() - t0

    best = study.best_trial
    results = {
        'condition': args.condition,
        'seed': args.seed,
        'k_values': {'stn_gpe': args.k_stn_gpe, 'gpe_stn': args.k_gpe_stn, 'stn_gpi': args.k_stn_gpi, 'gpe_gpi': args.k_gpe_gpi},
        'n_trials': args.n_trials,
        'bounds': bounds,
        'weights': weights,
        'targets': TARGETS,
        'best_params': study.best_params,
        'best_value': study.best_value,
        'best_metrics': dict(best.user_attrs),
        'elapsed_seconds': elapsed,
        'loss_trajectory': [t.value if t.value is not None else float('inf')
                            for t in study.trials],
        'all_trials': [
            {
                'params': t.params,
                'value': t.value,
                'user_attrs': dict(t.user_attrs),
            }
            for t in study.trials
        ],
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)

    print(f"\n{'='*60}")
    print(f"DONE: {args.condition} (seed={args.seed})")
    print(f"  Time: {elapsed/60:.1f} min")
    print(f"  Best loss: {study.best_value:.4f}")
    print(f"  STN: {best.user_attrs['r_stn']:.1f} Hz (target 27.5)")
    print(f"  GPe: {best.user_attrs['r_gpe']:.1f} Hz (target 42.5)")
    print(f"  GPi: {best.user_attrs['r_gpi']:.1f} Hz (target 82.5)")
    print(f"  GPe beta: {best.user_attrs['beta_gpe']*100:.1f}%")
    print(f"  STN beta: {best.user_attrs['beta_stn']*100:.1f}%")
    print(f"  Synaptic multipliers:")
    for k in ['g_stn_gpe_mult', 'g_gpe_stn_mult', 'g_stn_gpi_mult', 'g_gpe_gpi_mult']:
        print(f"    {k}: {study.best_params[k]:.3f}")
    print(f"  Saved: {args.output}")


if __name__ == "__main__":
    main()
