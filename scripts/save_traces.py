"""Save raw simulation traces for figure generation."""
import sys
sys.path.insert(0, '.')
import jax
import jax.numpy as jnp
import numpy as np
import pickle
from jax_models.network_builder import build_network_state
from optimization.sim_jax import create_simulation_fn, apply_params_to_config
from optimization.metrics_jax import compute_all_metrics, compute_beta_fraction_all
from jax_models.integrator import network_step
from jax import lax

print(f"JAX devices: {jax.devices()}")

state, config = build_network_state(400, 800, 600, 0.025)
simulator = create_simulation_fn(config, n_steps=24000)

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

pd_simulator = create_pd_simulator(config, 24000)

healthy_params = {
    'ISTN': 140.0, 'I_gpe': 3.379, 'I_gpi': 2.188,
    'noise_stn_sigma': 0.996, 'noise_gpe_sigma': 97.760, 'noise_gpi_sigma': 69.678,
}
pd_params = {
    'ISTN': 80.0, 'I_gpe': 0.672, 'I_gpi': 2.430,
    'noise_stn_sigma': 4.333, 'noise_gpe_sigma': 139.364, 'noise_gpi_sigma': 109.012,
    'g_stn_gpe_mult': 1.592, 'g_gpe_stn_mult': 0.182,
    'g_stn_gpi_mult': 1.975, 'g_gpe_gpi_mult': 0.419,
}

print("Running healthy (1800n, 600ms)...")
obs_h = simulator(healthy_params, state)
obs_h['V_stn'].block_until_ready()

print("Running PD (1800n, 600ms)...")
obs_pd = pd_simulator(pd_params, state)
obs_pd['V_stn'].block_until_ready()

metrics_h = compute_all_metrics(obs_h, 0.025, burn_steps=4000)
metrics_pd = compute_all_metrics(obs_pd, 0.025, burn_steps=4000)
beta_h = compute_beta_fraction_all(obs_h, 0.025, burn_steps=4000)
beta_pd = compute_beta_fraction_all(obs_pd, 0.025, burn_steps=4000)

print(f"Healthy: STN={metrics_h['firing_rates']['stn']:.1f} GPe={metrics_h['firing_rates']['gpe']:.1f} GPi={metrics_h['firing_rates']['gpi']:.1f} beta={beta_h['gpe']*100:.1f}%")
print(f"PD:      STN={metrics_pd['firing_rates']['stn']:.1f} GPe={metrics_pd['firing_rates']['gpe']:.1f} GPi={metrics_pd['firing_rates']['gpi']:.1f} beta={beta_pd['gpe']*100:.1f}%")

# Save traces as numpy
traces = {
    'healthy': {k: np.array(v) for k, v in obs_h.items()},
    'pd': {k: np.array(v) for k, v in obs_pd.items()},
    'metrics_h': metrics_h, 'metrics_pd': metrics_pd,
    'beta_h': beta_h, 'beta_pd': beta_pd,
    'healthy_params': healthy_params, 'pd_params': pd_params,
    'network_size': (400, 800, 600), 'dt_ms': 0.025, 'burn_steps': 4000,
}
with open('results/simulations/traces_1800n.pkl', 'wb') as f:
    pickle.dump(traces, f)
print("Saved: results/simulations/traces_1800n.pkl")
