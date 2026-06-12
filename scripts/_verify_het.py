"""Verify het_sigma path: homogeneous vs het=0.10 firing rates with fixed PD params."""
import sys; sys.path.insert(0, '.')
import jax, jax.numpy as jnp
from jax import lax
import numpy as np

from jax_models.network_builder import build_network_state
from jax_models.integrator import network_step
from optimization.sim_jax import apply_params_to_config
from optimization.metrics_jax import compute_all_metrics, compute_beta_fraction_all

DT_MS = 0.025
N_STEPS = 16000
BURN = 4000

pd_params = {'ISTN': 70.048, 'I_gpe': 1.663, 'I_gpi': 1.670,
    'noise_stn_sigma': 1.971, 'noise_gpe_sigma': 66.383, 'noise_gpi_sigma': 96.429,
    'g_stn_gpe_mult': 4.132, 'g_gpe_stn_mult': 0.114,
    'g_stn_gpi_mult': 1.943, 'g_gpe_gpi_mult': 0.996}

def run(het_sigma):
    base_state, base_config = build_network_state(100, 200, 150, DT_MS, het_sigma=het_sigma)
    @jax.jit
    def simulate(tp, init_state):
        config = apply_params_to_config(tp, base_config)
        syn = dict(config['synapses'])
        for sn, mn in [('stn_to_gpe','g_stn_gpe_mult'),('gpe_to_stn','g_gpe_stn_mult'),
                       ('stn_to_gpi','g_stn_gpi_mult'),('gpe_to_gpi','g_gpe_gpi_mult')]:
            old = syn[sn]; syn[sn] = old._replace(weights=old.weights * tp.get(mn,1.0))
        config['synapses'] = syn
        def step(c, t): return network_step(c, config, t*config['dt_ms'])
        _, obs = lax.scan(step, init=init_state, xs=jnp.arange(N_STEPS))
        return obs
    obs = simulate(pd_params, base_state)
    obs['V_stn'].block_until_ready()
    m = compute_all_metrics(obs, DT_MS, burn_steps=BURN)
    b = compute_beta_fraction_all(obs, DT_MS, burn_steps=BURN)
    return m['firing_rates'], b

print("Running homogeneous (het=0.0)...")
r0, b0 = run(0.0)
print(f"  STN={r0['stn']:.2f} GPe={r0['gpe']:.2f} GPi={r0['gpi']:.2f}  betaGPe={b0['gpe']*100:.1f}% betaSTN={b0['stn']*100:.1f}%")
print("Running heterogeneous (het=0.10)...")
r1, b1 = run(0.10)
print(f"  STN={r1['stn']:.2f} GPe={r1['gpe']:.2f} GPi={r1['gpi']:.2f}  betaGPe={b1['gpe']*100:.1f}% betaSTN={b1['stn']*100:.1f}%")
print("Deltas (het - homo):")
dstn, dgpe, dgpi = r1['stn']-r0['stn'], r1['gpe']-r0['gpe'], r1['gpi']-r0['gpi']
print(f"  dSTN={dstn:+.2f} dGPe={dgpe:+.2f} dGPi={dgpi:+.2f}")
ok = all(abs(d) <= 2.0 for d in (dstn, dgpe, dgpi))
print(f"WITHIN_2HZ: {ok}")
