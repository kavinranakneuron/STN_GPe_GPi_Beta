"""
Optuna optimization for HH GPe/GPi network - Healthy state.
10 parameters: 6 intrinsic + 4 synaptic multipliers
Target: STN ~20 Hz, GPe ~70 Hz, GPi ~80 Hz, <5% beta
"""

import sys
sys.path.insert(0, '.')

import optuna
from optuna.samplers import CmaEsSampler
import jax
import jax.numpy as jnp
import time
import pickle
import argparse
from pathlib import Path

from jax_models.network_builder import build_network_state
from optimization.sim_jax import create_simulation_fn
from optimization.metrics_jax import compute_all_metrics, compute_beta_fraction_all

print(f"JAX devices: {jax.devices()}")

# =============================================================================
# NETWORK SETUP (450 neurons)
# =============================================================================

N_STN, N_GPE, N_GPI = 100, 200, 150
DT_MS = 0.025
N_STEPS = 16000  # 400ms
BURN_STEPS = 4000  # 100ms burn-in

print(f"\nBuilding {N_STN + N_GPE + N_GPI}-neuron HH network...")
base_state, base_config = build_network_state(N_STN, N_GPE, N_GPI, DT_MS)

# =============================================================================
# CUSTOM SIMULATOR WITH SYNAPTIC SCALING
# =============================================================================

def create_simulator_with_multipliers(base_config, n_steps):
    from optimization.sim_jax import apply_params_to_config
    from jax_models.integrator import network_step
    from jax import lax

    @jax.jit
    def simulate(trial_params, init_state):
        config = apply_params_to_config(trial_params, base_config)
        syn_configs = dict(config['synapses'])

        # Scale synaptic weights
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

simulator = create_simulator_with_multipliers(base_config, N_STEPS)

# Warm-up JIT
print("Warming up JIT...")
dummy_params = {
    'ISTN': 100.0, 'I_gpe': 3.0, 'I_gpi': 3.0,
    'noise_stn_sigma': 1.0, 'noise_gpe_sigma': 30.0, 'noise_gpi_sigma': 30.0,
    'g_stn_gpe_mult': 1.0, 'g_gpe_stn_mult': 1.0,
    'g_stn_gpi_mult': 1.0, 'g_gpe_gpi_mult': 1.0,
}
obs = simulator(dummy_params, base_state)
obs['V_stn'].block_until_ready()
print("JIT ready!\n")

# =============================================================================
# TARGETS (Healthy state)
# =============================================================================

TARGETS = {
    'rate_stn': 20.0,
    'rate_gpe': 70.0,
    'rate_gpi': 80.0,
    'cv_stn': 0.40,
    'cv_gpe': 0.35,
    'cv_gpi': 0.20,
}

WEIGHTS = {
    'rate': 5.0,
    'cv': 0.2,
    'beta': 15.0,
}

# =============================================================================
# OBJECTIVE FUNCTION
# =============================================================================

def objective(trial):
    params = {
        'ISTN': trial.suggest_float('ISTN', 80.0, 200.0),
        'I_gpe': trial.suggest_float('I_gpe', 1.0, 8.0),
        'I_gpi': trial.suggest_float('I_gpi', 1.0, 8.0),
        'noise_stn_sigma': trial.suggest_float('noise_stn_sigma', 0.5, 5.0),
        'noise_gpe_sigma': trial.suggest_float('noise_gpe_sigma', 10.0, 100.0),
        'noise_gpi_sigma': trial.suggest_float('noise_gpi_sigma', 10.0, 100.0),
        # Synaptic multipliers - healthy should land near 1.0
        'g_stn_gpe_mult': trial.suggest_float('g_stn_gpe_mult', 0.5, 2.0),
        'g_gpe_stn_mult': trial.suggest_float('g_gpe_stn_mult', 0.5, 2.0),
        'g_stn_gpi_mult': trial.suggest_float('g_stn_gpi_mult', 0.5, 2.0),
        'g_gpe_gpi_mult': trial.suggest_float('g_gpe_gpi_mult', 0.5, 2.0),
    }

    try:
        obs = simulator(params, base_state)
        obs['V_stn'].block_until_ready()

        if jnp.any(jnp.isnan(obs['V_stn'])) or jnp.any(jnp.isnan(obs['V_gpe'])):
            return 1e6

        # Compute metrics
        metrics = compute_all_metrics(obs, DT_MS, burn_steps=BURN_STEPS)
        beta_fractions = compute_beta_fraction_all(obs, DT_MS, burn_steps=BURN_STEPS)

        r_stn = metrics['firing_rates']['stn']
        r_gpe = metrics['firing_rates']['gpe']
        r_gpi = metrics['firing_rates']['gpi']
        cv_stn = metrics['cv']['stn']
        cv_gpe = metrics['cv']['gpe']
        cv_gpi = metrics['cv']['gpi']
        beta_gpe = beta_fractions['gpe']
        beta_stn = beta_fractions['stn']

        # Loss
        loss = 0.0

        # Firing rate errors
        loss += WEIGHTS['rate'] * ((r_stn - TARGETS['rate_stn']) / TARGETS['rate_stn']) ** 2
        loss += WEIGHTS['rate'] * ((r_gpe - TARGETS['rate_gpe']) / TARGETS['rate_gpe']) ** 2
        loss += WEIGHTS['rate'] * ((r_gpi - TARGETS['rate_gpi']) / TARGETS['rate_gpi']) ** 2

        # CV errors
        loss += WEIGHTS['cv'] * ((cv_stn - TARGETS['cv_stn']) / TARGETS['cv_stn']) ** 2
        loss += WEIGHTS['cv'] * ((cv_gpe - TARGETS['cv_gpe']) / TARGETS['cv_gpe']) ** 2
        loss += WEIGHTS['cv'] * ((cv_gpi - TARGETS['cv_gpi']) / TARGETS['cv_gpi']) ** 2

        # Beta penalty: penalize beta above 5%
        if beta_gpe > 0.05:
            loss += WEIGHTS['beta'] * ((beta_gpe - 0.05) / 0.05) ** 2

        # Silent population penalty
        if r_stn < 1.0:
            loss += 100.0
        if r_gpe < 10.0:
            loss += 100.0
        if r_gpi < 10.0:
            loss += 100.0

        trial.set_user_attr('r_stn', float(r_stn))
        trial.set_user_attr('r_gpe', float(r_gpe))
        trial.set_user_attr('r_gpi', float(r_gpi))
        trial.set_user_attr('cv_stn', float(cv_stn))
        trial.set_user_attr('cv_gpe', float(cv_gpe))
        trial.set_user_attr('cv_gpi', float(cv_gpi))
        trial.set_user_attr('beta_gpe', float(beta_gpe))
        trial.set_user_attr('beta_stn', float(beta_stn))

        return loss

    except Exception as e:
        print(f"Trial failed: {e}")
        return 1e6

# =============================================================================
# RUN OPTIMIZATION
# =============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output', default='results/optimization/healthy_study.pkl')
    ap.add_argument('--n-trials', type=int, default=500)
    args = ap.parse_args()

    print("=" * 60)
    print("HH Network Optimization - Healthy State (10 params)")
    print("=" * 60)
    print(f"Network: {N_STN}/{N_GPE}/{N_GPI} neurons")
    print(f"Targets: STN={TARGETS['rate_stn']}Hz, GPe={TARGETS['rate_gpe']}Hz, GPi={TARGETS['rate_gpi']}Hz")
    print(f"         Beta_GPe < 5% (penalty weight={WEIGHTS['beta']}x)")
    print("=" * 60)

    study = optuna.create_study(
        direction='minimize',
        sampler=CmaEsSampler(seed=args.seed),
        study_name='hh_healthy_450'
    )

    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Time: {elapsed/60:.1f} minutes ({elapsed/max(len(study.trials),1)*1000:.0f}ms/trial)")
    print(f"Best score: {study.best_value:.4f}")

    print(f"\nBest intrinsic parameters:")
    for k in ['ISTN', 'I_gpe', 'I_gpi', 'noise_stn_sigma', 'noise_gpe_sigma', 'noise_gpi_sigma']:
        print(f"  {k}: {study.best_params[k]:.3f}")

    print(f"\nBest synaptic multipliers:")
    for k in ['g_stn_gpe_mult', 'g_gpe_stn_mult', 'g_stn_gpi_mult', 'g_gpe_gpi_mult']:
        print(f"  {k}: {study.best_params[k]:.3f}")

    best = study.best_trial
    print(f"\nBest metrics:")
    print(f"  STN: {best.user_attrs['r_stn']:.1f} Hz (target: {TARGETS['rate_stn']})")
    print(f"  GPe: {best.user_attrs['r_gpe']:.1f} Hz (target: {TARGETS['rate_gpe']})")
    print(f"  GPi: {best.user_attrs['r_gpi']:.1f} Hz (target: {TARGETS['rate_gpi']})")
    print(f"  STN CV: {best.user_attrs['cv_stn']:.2f} (target: {TARGETS['cv_stn']})")
    print(f"  GPe CV: {best.user_attrs['cv_gpe']:.2f} (target: {TARGETS['cv_gpe']})")
    print(f"  GPi CV: {best.user_attrs['cv_gpi']:.2f} (target: {TARGETS['cv_gpi']})")
    print(f"\n  GPe Beta: {best.user_attrs['beta_gpe']*100:.1f}% (target: <5%)")
    print(f"  STN Beta: {best.user_attrs['beta_stn']*100:.1f}%")

    # Save
    results = {
        'seed': args.seed,
        'best_params': study.best_params,
        'best_value': study.best_value,
        'best_metrics': dict(best.user_attrs),
        'targets': TARGETS,
        'weights': WEIGHTS,
        'network_size': (N_STN, N_GPE, N_GPI),
        'n_trials': len(study.trials),
        'elapsed_seconds': elapsed,
        'all_trials': [
            {'params': t.params, 'value': t.value, 'user_attrs': dict(t.user_attrs)}
            for t in study.trials
        ],
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)

    print(f"\nResults saved to {args.output}")
