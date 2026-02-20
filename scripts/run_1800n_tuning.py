"""
1800-neuron intrinsic parameter tuning.

Re-optimizes the 6 intrinsic parameters at full scale (400/800/600)
while holding the 4 synaptic multipliers fixed from the 450n optimization.

Two studies:
  1. Healthy: multipliers = 1.0, target low beta
  2. Parkinsonian: multipliers from 450n, target 20-40% beta

Author: Kavin Nakkeeran, Johns Hopkins University
"""

import sys
sys.path.insert(0, '.')

import optuna
from optuna.samplers import CmaEsSampler
import jax
import jax.numpy as jnp
import time
import pickle
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

N_STN, N_GPE, N_GPI = 400, 800, 600
DT_MS = 0.025
N_STEPS = 24000   # 600ms
BURN_STEPS = 4000  # 100ms burn-in
N_TRIALS = 500

# Fixed synaptic multipliers from 450n optimization
PD_MULTIPLIERS = {
    'g_stn_gpe_mult': 4.148,
    'g_gpe_stn_mult': 0.761,
    'g_stn_gpi_mult': 3.504,
    'g_gpe_gpi_mult': 1.077,
}

HEALTHY_MULTIPLIERS = {
    'g_stn_gpe_mult': 1.0,
    'g_gpe_stn_mult': 1.0,
    'g_stn_gpi_mult': 1.0,
    'g_gpe_gpi_mult': 1.0,
}

HEALTHY_TARGETS = {
    'rate_stn': 20.0,
    'rate_gpe': 70.0,
    'rate_gpi': 80.0,
    'cv_stn': 0.40,
    'cv_gpe': 0.35,
    'cv_gpi': 0.20,
}

PD_TARGETS = {
    'rate_stn': 27.5,
    'rate_gpe': 42.5,
    'rate_gpi': 82.5,
    'cv_stn': 0.60,
    'cv_gpe': 0.35,
    'cv_gpi': 0.25,
    'beta_gpe_low': 0.20,   # penalize below 20%
    'beta_gpe_high': 0.40,  # penalize above 40%
}

WEIGHTS = {
    'rate': 5.0,
    'cv': 0.2,
    'beta': 15.0,
}

# =============================================================================
# NETWORK + SIMULATORS
# =============================================================================

print(f"\nBuilding {N_STN + N_GPE + N_GPI}-neuron network...")
state, config = build_network_state(N_STN, N_GPE, N_GPI, DT_MS, seed=42)

# Healthy simulator (no synaptic scaling needed, multipliers are 1.0)
healthy_simulator = create_simulation_fn(config, n_steps=N_STEPS)

# PD simulator (with fixed synaptic multipliers)
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

pd_simulator = create_pd_simulator(config, N_STEPS)

# Warm up both
print("Warming up JIT (healthy)...")
dummy = {
    'ISTN': 100.0, 'I_gpe': 3.0, 'I_gpi': 3.0,
    'noise_stn_sigma': 1.0, 'noise_gpe_sigma': 30.0, 'noise_gpi_sigma': 30.0,
}
obs = healthy_simulator(dummy, state)
obs['V_stn'].block_until_ready()
print("Warming up JIT (PD)...")
dummy_pd = {**dummy, **PD_MULTIPLIERS}
obs = pd_simulator(dummy_pd, state)
obs['V_stn'].block_until_ready()
print("JIT ready!\n")

# =============================================================================
# PARAM SAMPLING (shared by both objectives)
# =============================================================================

def sample_intrinsic_params(trial):
    return {
        'ISTN': trial.suggest_float('ISTN', 60.0, 200.0),
        'I_gpe': trial.suggest_float('I_gpe', 0.5, 5.0),
        'I_gpi': trial.suggest_float('I_gpi', 1.0, 5.0),
        'noise_stn_sigma': trial.suggest_float('noise_stn_sigma', 0.5, 10.0),
        'noise_gpe_sigma': trial.suggest_float('noise_gpe_sigma', 20.0, 200.0),
        'noise_gpi_sigma': trial.suggest_float('noise_gpi_sigma', 20.0, 200.0),
    }

# =============================================================================
# HEALTHY OBJECTIVE
# =============================================================================

def healthy_objective(trial):
    params = sample_intrinsic_params(trial)

    try:
        obs = healthy_simulator(params, state)
        obs['V_stn'].block_until_ready()

        if jnp.any(jnp.isnan(obs['V_stn'])) or jnp.any(jnp.isnan(obs['V_gpe'])):
            return 1e6

        metrics = compute_all_metrics(obs, DT_MS, burn_steps=BURN_STEPS)
        beta_frac = compute_beta_fraction_all(obs, DT_MS, burn_steps=BURN_STEPS)

        r_stn = metrics['firing_rates']['stn']
        r_gpe = metrics['firing_rates']['gpe']
        r_gpi = metrics['firing_rates']['gpi']
        cv_stn = metrics['cv']['stn']
        cv_gpe = metrics['cv']['gpe']
        cv_gpi = metrics['cv']['gpi']
        beta_gpe = beta_frac['gpe']

        loss = 0.0

        # Firing rates
        loss += WEIGHTS['rate'] * ((r_stn - HEALTHY_TARGETS['rate_stn']) / HEALTHY_TARGETS['rate_stn']) ** 2
        loss += WEIGHTS['rate'] * ((r_gpe - HEALTHY_TARGETS['rate_gpe']) / HEALTHY_TARGETS['rate_gpe']) ** 2
        loss += WEIGHTS['rate'] * ((r_gpi - HEALTHY_TARGETS['rate_gpi']) / HEALTHY_TARGETS['rate_gpi']) ** 2

        # CV
        loss += WEIGHTS['cv'] * ((cv_stn - HEALTHY_TARGETS['cv_stn']) / HEALTHY_TARGETS['cv_stn']) ** 2
        loss += WEIGHTS['cv'] * ((cv_gpe - HEALTHY_TARGETS['cv_gpe']) / HEALTHY_TARGETS['cv_gpe']) ** 2
        loss += WEIGHTS['cv'] * ((cv_gpi - HEALTHY_TARGETS['cv_gpi']) / HEALTHY_TARGETS['cv_gpi']) ** 2

        # Beta: penalize if >5%
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

        return loss

    except Exception as e:
        print(f"Trial failed: {e}")
        return 1e6

# =============================================================================
# PARKINSONIAN OBJECTIVE
# =============================================================================

def pd_objective(trial):
    intrinsic = sample_intrinsic_params(trial)
    params = {**intrinsic, **PD_MULTIPLIERS}

    try:
        obs = pd_simulator(params, state)
        obs['V_stn'].block_until_ready()

        if jnp.any(jnp.isnan(obs['V_stn'])) or jnp.any(jnp.isnan(obs['V_gpe'])):
            return 1e6

        metrics = compute_all_metrics(obs, DT_MS, burn_steps=BURN_STEPS)
        beta_frac = compute_beta_fraction_all(obs, DT_MS, burn_steps=BURN_STEPS)

        r_stn = metrics['firing_rates']['stn']
        r_gpe = metrics['firing_rates']['gpe']
        r_gpi = metrics['firing_rates']['gpi']
        cv_stn = metrics['cv']['stn']
        cv_gpe = metrics['cv']['gpe']
        cv_gpi = metrics['cv']['gpi']
        beta_gpe = beta_frac['gpe']
        beta_stn = beta_frac['stn']

        loss = 0.0

        # Firing rates
        loss += WEIGHTS['rate'] * ((r_stn - PD_TARGETS['rate_stn']) / PD_TARGETS['rate_stn']) ** 2
        loss += WEIGHTS['rate'] * ((r_gpe - PD_TARGETS['rate_gpe']) / PD_TARGETS['rate_gpe']) ** 2
        loss += WEIGHTS['rate'] * ((r_gpi - PD_TARGETS['rate_gpi']) / PD_TARGETS['rate_gpi']) ** 2

        # CV
        loss += WEIGHTS['cv'] * ((cv_stn - PD_TARGETS['cv_stn']) / PD_TARGETS['cv_stn']) ** 2
        loss += WEIGHTS['cv'] * ((cv_gpe - PD_TARGETS['cv_gpe']) / PD_TARGETS['cv_gpe']) ** 2
        loss += WEIGHTS['cv'] * ((cv_gpi - PD_TARGETS['cv_gpi']) / PD_TARGETS['cv_gpi']) ** 2

        # Beta: penalize below 20% AND above 40%
        if beta_gpe < PD_TARGETS['beta_gpe_low']:
            loss += WEIGHTS['beta'] * ((PD_TARGETS['beta_gpe_low'] - beta_gpe) / PD_TARGETS['beta_gpe_low']) ** 2
        elif beta_gpe > PD_TARGETS['beta_gpe_high']:
            loss += WEIGHTS['beta'] * ((beta_gpe - PD_TARGETS['beta_gpe_high']) / PD_TARGETS['beta_gpe_high']) ** 2

        # PD constraints
        if r_gpe > 55.0:
            loss += 5.0 * ((r_gpe - 55.0) / 20.0) ** 2
        if r_stn < 18.0:
            loss += 5.0 * ((18.0 - r_stn) / 18.0) ** 2

        # Silent population penalty
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

        return loss

    except Exception as e:
        print(f"Trial failed: {e}")
        return 1e6

# =============================================================================
# PRINT FULL METRICS FROM A SIMULATION
# =============================================================================

def print_full_metrics(label, obs, dt_ms, burn_steps):
    metrics = compute_all_metrics(obs, dt_ms, burn_steps=burn_steps)
    beta_frac = compute_beta_fraction_all(obs, dt_ms, burn_steps=burn_steps)

    print(f"\n  {label} metrics:")
    print(f"    STN:  {metrics['firing_rates']['stn']:6.1f} Hz   CV={metrics['cv']['stn']:.3f}   beta={beta_frac['stn']*100:.1f}%")
    print(f"    GPe:  {metrics['firing_rates']['gpe']:6.1f} Hz   CV={metrics['cv']['gpe']:.3f}   beta={beta_frac['gpe']*100:.1f}%")
    print(f"    GPi:  {metrics['firing_rates']['gpi']:6.1f} Hz   CV={metrics['cv']['gpi']:.3f}   beta={beta_frac['gpi']*100:.1f}%")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    metadata = {
        'date': datetime.now().isoformat(),
        'network_size': (N_STN, N_GPE, N_GPI),
        'dt_ms': DT_MS,
        'n_steps': N_STEPS,
        'burn_steps': BURN_STEPS,
        'n_trials': N_TRIALS,
        'jax_version': jax.__version__,
        'platform': platform.platform(),
        'jax_devices': [str(d) for d in jax.devices()],
        'pd_multipliers': PD_MULTIPLIERS,
        'weights': WEIGHTS,
    }

    # =====================================================================
    # STUDY 1: HEALTHY
    # =====================================================================
    print("=" * 70)
    print("STUDY 1: HEALTHY — 1800 neurons, 6 intrinsic params")
    print("=" * 70)
    print(f"Targets: STN={HEALTHY_TARGETS['rate_stn']}Hz, "
          f"GPe={HEALTHY_TARGETS['rate_gpe']}Hz, "
          f"GPi={HEALTHY_TARGETS['rate_gpi']}Hz, beta<5%")

    healthy_study = optuna.create_study(
        direction='minimize',
        sampler=CmaEsSampler(seed=42),
        study_name='healthy_1800n',
    )

    t0 = time.time()
    healthy_study.optimize(healthy_objective, n_trials=N_TRIALS, show_progress_bar=True)
    healthy_elapsed = time.time() - t0

    hb = healthy_study.best_trial
    print(f"\nHealthy optimization: {healthy_elapsed/60:.1f} min "
          f"({healthy_elapsed/N_TRIALS*1000:.0f}ms/trial)")
    print(f"Best loss: {healthy_study.best_value:.4f}")
    print(f"Best params:")
    for k, v in healthy_study.best_params.items():
        print(f"  {k}: {v:.3f}")
    print(f"Metrics: STN={hb.user_attrs['r_stn']:.1f}Hz "
          f"GPe={hb.user_attrs['r_gpe']:.1f}Hz "
          f"GPi={hb.user_attrs['r_gpi']:.1f}Hz "
          f"beta_GPe={hb.user_attrs['beta_gpe']*100:.1f}%")

    healthy_results = {
        'best_params': healthy_study.best_params,
        'best_value': healthy_study.best_value,
        'best_metrics': hb.user_attrs,
        'targets': HEALTHY_TARGETS,
        'multipliers': HEALTHY_MULTIPLIERS,
        'weights': WEIGHTS,
        'n_trials': len(healthy_study.trials),
        'elapsed_seconds': healthy_elapsed,
        'metadata': metadata,
    }

    with open('results/optimization/healthy_1800n.pkl', 'wb') as f:
        pickle.dump(healthy_results, f)
    print("Saved: results/optimization/healthy_1800n.pkl")

    # =====================================================================
    # STUDY 2: PARKINSONIAN
    # =====================================================================
    print("\n" + "=" * 70)
    print("STUDY 2: PARKINSONIAN — 1800 neurons, 6 intrinsic + fixed multipliers")
    print("=" * 70)
    print(f"Fixed multipliers: {PD_MULTIPLIERS}")
    print(f"Targets: STN={PD_TARGETS['rate_stn']}Hz, "
          f"GPe={PD_TARGETS['rate_gpe']}Hz, "
          f"GPi={PD_TARGETS['rate_gpi']}Hz, "
          f"beta={PD_TARGETS['beta_gpe_low']*100:.0f}-{PD_TARGETS['beta_gpe_high']*100:.0f}%")

    pd_study = optuna.create_study(
        direction='minimize',
        sampler=CmaEsSampler(seed=42),
        study_name='pd_1800n',
    )

    t0 = time.time()
    pd_study.optimize(pd_objective, n_trials=N_TRIALS, show_progress_bar=True)
    pd_elapsed = time.time() - t0

    pb = pd_study.best_trial
    print(f"\nPD optimization: {pd_elapsed/60:.1f} min "
          f"({pd_elapsed/N_TRIALS*1000:.0f}ms/trial)")
    print(f"Best loss: {pd_study.best_value:.4f}")
    print(f"Best params:")
    for k, v in pd_study.best_params.items():
        print(f"  {k}: {v:.3f}")
    print(f"Metrics: STN={pb.user_attrs['r_stn']:.1f}Hz "
          f"GPe={pb.user_attrs['r_gpe']:.1f}Hz "
          f"GPi={pb.user_attrs['r_gpi']:.1f}Hz "
          f"beta_GPe={pb.user_attrs['beta_gpe']*100:.1f}%")

    pd_results = {
        'best_params': pd_study.best_params,
        'best_value': pd_study.best_value,
        'best_metrics': pb.user_attrs,
        'targets': PD_TARGETS,
        'multipliers': PD_MULTIPLIERS,
        'weights': WEIGHTS,
        'n_trials': len(pd_study.trials),
        'elapsed_seconds': pd_elapsed,
        'metadata': metadata,
    }

    with open('results/optimization/parkinsonian_1800n.pkl', 'wb') as f:
        pickle.dump(pd_results, f)
    print("Saved: results/optimization/parkinsonian_1800n.pkl")

    # =====================================================================
    # VALIDATION: run best params and print full metrics
    # =====================================================================
    print("\n" + "=" * 70)
    print("VALIDATION — simulating best params")
    print("=" * 70)

    # Healthy validation
    print("\nRunning healthy simulation with best params...")
    obs_h = healthy_simulator(healthy_study.best_params, state)
    obs_h['V_stn'].block_until_ready()
    print_full_metrics("HEALTHY", obs_h, DT_MS, BURN_STEPS)

    # PD validation
    print("\nRunning PD simulation with best params...")
    best_pd_full = {**pd_study.best_params, **PD_MULTIPLIERS}
    obs_pd = pd_simulator(best_pd_full, state)
    obs_pd['V_stn'].block_until_ready()
    print_full_metrics("PARKINSONIAN", obs_pd, DT_MS, BURN_STEPS)

    print("\n" + "=" * 70)
    print(f"Total time: {(healthy_elapsed + pd_elapsed)/60:.1f} minutes")
    print("=" * 70)
