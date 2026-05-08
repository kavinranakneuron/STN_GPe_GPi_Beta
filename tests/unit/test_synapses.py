"""Unit tests for the double-exponential synapse module."""
from __future__ import annotations

import math

import jax.numpy as jnp

from bgnet.synapses import (
    ampa_params,
    gaba_a_params,
    init_state,
    peak_normalization,
    synapse_step,
    synaptic_current,
)


def _impulse_response(p, g_max=1.0, dt=0.025, T=80.0):
    """Run a single spike pulse of size g_max through the synapse and
    return the conductance trace g(t) on a uniform grid."""
    norm = peak_normalization(p.tau_rise_ms, p.tau_decay_ms)
    x, y = init_state(1)
    # impulse at t = 0
    x, y = synapse_step(x, y, jnp.array([g_max]), dt, p)
    g_t = []
    times = []
    for i in range(int(T / dt)):
        x, y = synapse_step(x, y, jnp.array([0.0]), dt, p)
        g = (x - y) / norm
        g_t.append(float(g[0]))
        times.append((i + 1) * dt)
    return times, g_t, norm


def test_ampa_peak_equals_g_max():
    p = ampa_params()
    g_max = 0.5
    times, g_t, _ = _impulse_response(p, g_max=g_max)
    peak = max(g_t)
    assert abs(peak - g_max) / g_max < 0.05, f"AMPA peak {peak}, expected {g_max}"


def test_ampa_time_to_peak_matches_analytical():
    p = ampa_params()
    times, g_t, _ = _impulse_response(p)
    t_peak_obs = times[g_t.index(max(g_t))]
    t_peak_ana = (p.tau_rise_ms * p.tau_decay_ms / (p.tau_decay_ms - p.tau_rise_ms)
                  * math.log(p.tau_decay_ms / p.tau_rise_ms))
    assert abs(t_peak_obs - t_peak_ana) < 0.1, (
        f"AMPA t_peak observed {t_peak_obs} ms vs analytical {t_peak_ana} ms"
    )


def test_gaba_a_peak_equals_g_max():
    p = gaba_a_params()
    g_max = 0.3
    _, g_t, _ = _impulse_response(p, g_max=g_max, T=120.0)
    assert abs(max(g_t) - g_max) / g_max < 0.05


def test_synaptic_current_sign_convention():
    """With V > E_syn the current is positive (hyperpolarizing in the
    membrane equation -I_syn convention)."""
    p = ampa_params()
    norm = peak_normalization(p.tau_rise_ms, p.tau_decay_ms)
    x = jnp.array([1.0])
    y = jnp.array([0.0])  # arbitrary positive g_syn
    V = jnp.array([10.0])  # above E_syn = 0
    I = synaptic_current(x, y, V, p, norm)
    assert float(I[0]) > 0, "I_syn should be positive when V > E_syn"

    pg = gaba_a_params()
    norm_g = peak_normalization(pg.tau_rise_ms, pg.tau_decay_ms)
    V = jnp.array([-58.0])  # above E_syn = -70
    I = synaptic_current(x, y, V, pg, norm_g)
    assert float(I[0]) > 0
