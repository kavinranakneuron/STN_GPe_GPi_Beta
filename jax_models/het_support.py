"""Per-neuron conductance heterogeneity (opt-in).

Provides het-aware vectorized step functions that map per-neuron conductance
arrays over the neuron axis while keeping every other parameter shared. The
per-neuron conductances are passed as a closure-bound, vmapped argument and
merged over the (shared) parameter dict inside each step, so the integrator's
call signatures are unchanged.

This module is imported and used ONLY when build_network_state is called with
het_sigma>0. The default homogeneous path (het_sigma=0.0) never touches any of
this and is byte-identical to the submitted-tag behaviour.

Heterogeneity model: g_i = g_base * (1 + het_sigma * N(0,1)) per neuron, drawn
from a deterministic seed, following the per-cell conductance variability used
in Hahn & McIntyre (2010).
"""
import jax
import jax.numpy as jnp
import numpy as np

from .stn_jax import stn_step
from .gpe_gpi_hh import gpe_gpi_step

# Conductances perturbed per-neuron.
STN_HET_KEYS = ('gNa', 'gK', 'gT', 'gCa', 'gAHP')
HH_HET_KEYS = ('g_Na', 'g_K', 'g_T', 'g_Ca', 'g_AHP')


def make_het_conductances(base_params, keys, n_neurons, het_sigma, seed):
    """Per-neuron conductances = base * (1 + het_sigma * N(0,1)), kept positive."""
    rng = np.random.default_rng(seed)
    conds = {}
    for k in keys:
        base = float(base_params[k])
        factor = 1.0 + het_sigma * rng.standard_normal(n_neurons)
        factor = np.maximum(factor, 0.05)  # keep conductances strictly positive
        conds[k] = jnp.asarray(base * factor, dtype=jnp.float32)
    return conds


def make_het_stn_step(het_conds, compile=True):
    """Drop-in replacement for create_vectorized_stn() with per-neuron conductances."""
    def step_one(state, params, dt, I_ext, I_syn, t, conds):
        p = {**params, **conds}
        return stn_step(state, p, dt, I_ext, I_syn, t)
    vstep = jax.vmap(step_one, in_axes=(0, None, None, 0, 0, None, 0))
    def wrapped(state, params, dt, I_ext, I_syn, t):
        return vstep(state, params, dt, I_ext, I_syn, t, het_conds)
    return jax.jit(wrapped) if compile else wrapped


def make_het_hh_step(het_conds, compile=True):
    """Drop-in replacement for create_vectorized_gpe_gpi() with per-neuron conductances."""
    def step_one(V, h, n, r, Ca, I_syn, I_noise, dt, params, conds):
        p = {**params, **conds}
        return gpe_gpi_step(V, h, n, r, Ca, I_syn, I_noise, dt, p)
    vstep = jax.vmap(step_one, in_axes=(0, 0, 0, 0, 0, 0, 0, None, None, 0))
    def wrapped(V, h, n, r, Ca, I_syn, I_noise, dt, params):
        return vstep(V, h, n, r, Ca, I_syn, I_noise, dt, params, het_conds)
    return jax.jit(wrapped) if compile else wrapped
