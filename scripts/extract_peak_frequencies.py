"""Extract peak PSD frequencies under both pipelines (figure vs metric)."""
import sys, pickle, json
sys.path.insert(0, '.')
import numpy as np
import jax, jax.numpy as jnp
from jax import lax
import scipy.signal as sps

from jax_models.network_builder import build_network_state
from optimization.sim_jax import apply_params_to_config
from jax_models.integrator import network_step

# Same params as fig generation:
healthy_params = {'ISTN': 132.235, 'I_gpe': 3.039, 'I_gpi': 2.209,
    'noise_stn_sigma': 3.362, 'noise_gpe_sigma': 33.417, 'noise_gpi_sigma': 67.284,
    'g_stn_gpe_mult': 1.866, 'g_gpe_stn_mult': 0.999,
    'g_stn_gpi_mult': 1.834, 'g_gpe_gpi_mult': 0.687}
pd_params = {'ISTN': 70.048, 'I_gpe': 1.663, 'I_gpi': 1.670,
    'noise_stn_sigma': 1.971, 'noise_gpe_sigma': 66.383, 'noise_gpi_sigma': 96.429,
    'g_stn_gpe_mult': 4.132, 'g_gpe_stn_mult': 0.114,
    'g_stn_gpi_mult': 1.943, 'g_gpe_gpi_mult': 0.996}

DT_MS = 0.025
N_STEPS = 24000  # 600 ms
BURN_STEPS = 4000  # 100 ms

# Sim at 45k for fidelity:
state, base_config = build_network_state(10000, 20000, 15000, DT_MS, seed=42)

def _apply_multipliers(config, params):
    syn = dict(config['synapses'])
    for name, mult in [('stn_to_gpe','g_stn_gpe_mult'),('gpe_to_stn','g_gpe_stn_mult'),
                       ('stn_to_gpi','g_stn_gpi_mult'),('gpe_to_gpi','g_gpe_gpi_mult')]:
        old = syn[name]; syn[name] = old._replace(weights=old.weights * params.get(mult, 1.0))
    config['synapses'] = syn
    return config

@jax.jit
def sim(params, init_state):
    config = _apply_multipliers(apply_params_to_config(params, base_config), params)
    def step(c, t): return network_step(c, config, t * config['dt_ms'])
    _, obs = lax.scan(step, init=init_state, xs=jnp.arange(N_STEPS))
    return obs

results = {}
fs = 1000.0 / DT_MS  # 40000 Hz

for name, params in [('healthy', healthy_params), ('pd', pd_params)]:
    print(f"Simulating {name}...")
    obs = sim(params, state)
    obs['V_stn'].block_until_ready()
    V_stn = np.asarray(obs['V_stn'])[BURN_STEPS:, :]  # (time, neurons)

    # Pipeline A: figure pipeline (500-subsample, large nperseg)
    rng = np.random.RandomState(42)
    n_sample = min(500, V_stn.shape[1])
    sample_idx = rng.choice(V_stn.shape[1], n_sample, replace=False)
    lfp_fig = V_stn[:, sample_idx].mean(axis=1)
    lfp_fig = lfp_fig - lfp_fig.mean()
    npseg_fig = min(len(lfp_fig), int(fs*0.5))  # ~20000
    f1, p1 = sps.welch(lfp_fig, fs=fs, nperseg=npseg_fig, noverlap=npseg_fig//2)
    mask_beta = (f1 >= 13) & (f1 <= 30)
    peak_fig = float(f1[mask_beta][np.argmax(p1[mask_beta])])
    resolution_fig = float(f1[1] - f1[0])

    # Pipeline B: metric pipeline (full pop, nperseg=8192)
    lfp_met = V_stn.mean(axis=1)
    lfp_met = lfp_met - lfp_met.mean()
    npseg_met = min(len(lfp_met), 8192)
    f2, p2 = sps.welch(lfp_met, fs=fs, nperseg=npseg_met, noverlap=npseg_met//2)
    mask_beta2 = (f2 >= 13) & (f2 <= 30)
    peak_met = float(f2[mask_beta2][np.argmax(p2[mask_beta2])])
    resolution_met = float(f2[1] - f2[0])

    results[name] = {
        'pipeline_figure': {'peak_hz': peak_fig, 'resolution_hz': resolution_fig,
                            'nperseg': npseg_fig, 'lfp': '500-neuron subsample'},
        'pipeline_metric': {'peak_hz': peak_met, 'resolution_hz': resolution_met,
                            'nperseg': npseg_met, 'lfp': 'full population'},
    }
    print(f"  {name} fig pipeline: peak={peak_fig:.2f} Hz, resolution={resolution_fig:.2f} Hz")
    print(f"  {name} met pipeline: peak={peak_met:.2f} Hz, resolution={resolution_met:.2f} Hz")

with open('results/sensitivity2/peak_frequencies.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved: results/sensitivity2/peak_frequencies.json")
