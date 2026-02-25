"""
Network builder with fixed-indegree connectivity.

Uses fixed indegree (constant number of presynaptic inputs per postsynaptic
neuron) instead of fixed connection probability. This preserves both mean
AND variance of synaptic input across network sizes, following the approach
described in Gerstner et al., Neuronal Dynamics, Ch 12.3.

The indegrees are calibrated to match the connectivity at the 450-neuron
optimization size (100 STN / 200 GPe / 150 GPi):
  STN->GPe: K=15  (was p=0.15 * 100 STN = 15)
  GPe->STN: K=14  (was p=0.07 * 200 GPe = 14)
  STN->GPi: K=30  (was p=0.30 * 100 STN = 30)
  GPe->GPi: K=10  (was p=0.05 * 200 GPe = 10)

The g_max values were originally tuned at a 10/20/15 reference size with
g_ref = {2.0, 9.0, 2.0, 3.0}. At the 100/200/150 optimization size the
old scaling formula gave g = g_ref * (N_ref_tuning / N_optimization), so
the per-synapse conductances are {0.2, 0.9, 0.2, 0.3}. With fixed indegree
no further scaling is needed — these values are used directly at all sizes.
"""

import jax.numpy as jnp
import numpy as np
from .stn_jax import create_population_state as create_stn_population, create_vectorized_stn, default_stn_params
from .gpe_gpi_hh import create_population_state as create_hh_population, create_vectorized_gpe_gpi, default_gpe_params, default_gpi_params
from .noise_jax import create_ou_for_population
from .synapses_jax import create_synapse_config, init_synapse_state


# Fixed indegrees (number of presynaptic inputs per postsynaptic neuron).
# These are constant regardless of network size.
K_STN_GPE = 15   # Each GPe neuron receives 15 STN inputs
K_GPE_STN = 14   # Each STN neuron receives 14 GPe inputs
K_STN_GPI = 30   # Each GPi neuron receives 30 STN inputs
K_GPE_GPI = 10   # Each GPi neuron receives 10 GPe inputs

# Per-synapse conductances (g_ref * N_ref_tuning / N_ref_optimization):
#   g_stn_gpe = 2.0 * (10/100) = 0.2
#   g_gpe_stn = 9.0 * (20/200) = 0.9
#   g_stn_gpi = 2.0 * (10/100) = 0.2
#   g_gpe_gpi = 3.0 * (20/200) = 0.3
G_STN_GPE = 0.2
G_GPE_STN = 0.9
G_STN_GPI = 0.2
G_GPE_GPI = 0.3


def build_network_state(n_stn, n_gpe, n_gpi, dt_ms, seed=42):
    """
    Build network with fixed-indegree connectivity.

    Args:
        n_stn: Number of STN neurons
        n_gpe: Number of GPe neurons
        n_gpi: Number of GPi neurons
        dt_ms: Timestep in ms
        seed: Random seed
    """

    # Neurons
    stn_state = create_stn_population(n_stn, heterogeneity=0.05, seed=seed)
    gpe_state = create_hh_population(n_gpe, cell_type='gpe', heterogeneity=0.1, seed=seed+1)
    gpi_state = create_hh_population(n_gpi, cell_type='gpi', heterogeneity=0.1, seed=seed+2)
    gpe_step_fn = create_vectorized_gpe_gpi(compile=True)
    gpi_step_fn = create_vectorized_gpe_gpi(compile=True)
    gpe_params = default_gpe_params()
    gpi_params = default_gpi_params()

    # ==========================================================================
    # SYNAPSES (fixed indegree, no conductance scaling)
    # ==========================================================================

    syn_cfg_stn_gpe = create_synapse_config(n_stn, n_gpe, K_STN_GPE, G_STN_GPE, 0.2, 5.0, 3.0, 0.0, dt_ms, seed+10)
    syn_state_stn_gpe = init_synapse_state(syn_cfg_stn_gpe)

    syn_cfg_gpe_stn = create_synapse_config(n_gpe, n_stn, K_GPE_STN, G_GPE_STN, 0.2, 8.0, 8.0, -70.0, dt_ms, seed+11)
    syn_state_gpe_stn = init_synapse_state(syn_cfg_gpe_stn)

    syn_cfg_stn_gpi = create_synapse_config(n_stn, n_gpi, K_STN_GPI, G_STN_GPI, 0.2, 5.0, 3.0, 0.0, dt_ms, seed+12)
    syn_state_stn_gpi = init_synapse_state(syn_cfg_stn_gpi)

    syn_cfg_gpe_gpi = create_synapse_config(n_gpe, n_gpi, K_GPE_GPI, G_GPE_GPI, 0.2, 5.0, 8.0, -70.0, dt_ms, seed+13)
    syn_state_gpe_gpi = init_synapse_state(syn_cfg_gpe_gpi)

    # Noise
    noise_cfg_stn, noise_state_stn = create_ou_for_population(n_stn, dt_ms, mu=1.8, seed=seed+20)
    noise_cfg_gpe, noise_state_gpe = create_ou_for_population(n_gpe, dt_ms, mu=0.0, seed=seed+21)
    noise_cfg_gpi, noise_state_gpi = create_ou_for_population(n_gpi, dt_ms, mu=0.0, seed=seed+22)

    state = {
        'stn': stn_state,
        'gpe': gpe_state,
        'gpi': gpi_state,
        'spikes_stn': jnp.zeros(n_stn, dtype=jnp.bool_),
        'spikes_gpe': jnp.zeros(n_gpe, dtype=jnp.bool_),
        'spikes_gpi': jnp.zeros(n_gpi, dtype=jnp.bool_),
        'synapses': {
            'stn_to_gpe': syn_state_stn_gpe,
            'gpe_to_stn': syn_state_gpe_stn,
            'stn_to_gpi': syn_state_stn_gpi,
            'gpe_to_gpi': syn_state_gpe_gpi
        },
        'noise': {
            'stn': noise_state_stn,
            'gpe': noise_state_gpe,
            'gpi': noise_state_gpi
        }
    }

    config = {
        'dt_ms': dt_ms,
        'populations': {'n_stn': n_stn, 'n_gpe': n_gpe, 'n_gpi': n_gpi},
        'synapses': {
            'stn_to_gpe': syn_cfg_stn_gpe,
            'gpe_to_stn': syn_cfg_gpe_stn,
            'stn_to_gpi': syn_cfg_stn_gpi,
            'gpe_to_gpi': syn_cfg_gpe_gpi
        },
        'noise': {
            'stn': noise_cfg_stn,
            'gpe': noise_cfg_gpe,
            'gpi': noise_cfg_gpi
        },
        'neuron_step_fns': {
            'stn': create_vectorized_stn(compile=True),
            'gpe': gpe_step_fn,
            'gpi': gpi_step_fn
        },
        'neuron_params': {
            'stn': default_stn_params(),
            'gpe': gpe_params,
            'gpi': gpi_params
        },
    }

    return state, config
