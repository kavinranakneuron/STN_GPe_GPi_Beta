"""Single-compartment subthalamic nucleus (STN) Hodgkin-Huxley neuron.

The model is a current-density formulation inspired by Gillies & Willshaw 2006,
adapted to a single compartment for whole-network simulation. Channels included:

    I_Na   fast sodium (spike upstroke)
    I_K    delayed-rectifier potassium (repolarization)
    I_L    leak
    I_T    T-type calcium (low-threshold burst conductance)
    I_CaH  high-voltage-activated calcium (instantaneous)
    I_AHP  calcium-activated potassium (afterhyperpolarization)
    I_H    hyperpolarization-activated cation (pacemaker)

Membrane equation:
    C_m dV/dt = -I_Na - I_K - I_L - I_T - I_CaH - I_AHP - I_H - I_syn + I_drive + I_noise

Units (current density convention, no scaling factors):
    conductance     mS/cm^2
    current         uA/cm^2
    voltage         mV
    time            ms
    calcium         uM
    capacitance     uF/cm^2

Spike detection: upward zero-crossing of V (V_now >= 0 mV and V_prev < 0 mV)
with a 2 ms refractory window.

Sources for parameter values:
    Gillies & Willshaw 2006, J Neurophysiol 95(4):2352-2365
    Otsuka et al. 2004, J Neurophysiol 92:255-264
    Terman et al. 2002, J Neurosci 22(7):2963-2976

The tonic drive I_drive is an optimization parameter (uA/cm^2). Its default
in this module is zero; the legacy code embedded a hard-coded I=42 here, which
the rebuild rejects. If you need spontaneous firing for a unit test, apply
explicit drive to the step function.

f-I curve (1 s simulations after 200 ms burn-in, no synaptic input, no noise),
characterized at module-build time:
    I_drive (uA/cm^2):    -2  0  2  5  10  15  20  30  42
    rate (Hz):             0  0  0  1   1   4   5   9  11
This is consistent with an Otsuka-style in-vitro STN: the bare neuron is
near-silent until ~5 uA/cm^2 of tonic drive, and reaches ~10 Hz only at
much larger drive than the optimizer's I_drive bound (+/- 5). In the network,
the missing drive is supplied by AMPA input from connected neurons and by
the OU process (mean and sigma both optimized).
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Parameters and state containers
# ---------------------------------------------------------------------------

class STNParams(NamedTuple):
    """Per-neuron STN parameters. Same values broadcast across the population
    in the simple homogeneous case; could be made heterogeneous by supplying
    arrays in fields where appropriate (vmap will then work over a different
    leading axis)."""
    # capacitance (uF/cm^2)
    C_m: float = 1.0
    # peak conductances (mS/cm^2) — Gillies & Willshaw 2006 lineage
    g_Na: float = 49.0
    g_K: float = 57.0
    g_L: float = 0.35
    g_T: float = 5.0
    g_CaH: float = 0.5
    g_AHP: float = 15.0
    g_H: float = 0.5
    # reversal potentials (mV)
    E_Na: float = 60.0
    E_K: float = -90.0
    E_Ca: float = 140.0
    E_L: float = -60.0
    E_H: float = -43.0
    # T-current activation (Vp_half is V_50, kp the slope factor in mV)
    Vp_half: float = -52.0
    kp: float = 6.2
    # H-current activation
    Vr_half: float = -74.0
    kr: float = 9.0
    tau_r_ms: float = 200.0
    # calcium handling
    alpha_Ca: float = 0.005   # uM per (uA/cm^2 * ms)
    tau_Ca_ms: float = 120.0
    k1: float = 15.0          # Michaelis constant (uM) for AHP gating by Ca
    # spike detection
    V_thresh: float = 0.0
    min_isi_ms: float = 2.0
    # numerical safety
    V_clip_lo: float = -100.0
    V_clip_hi: float = 60.0


class STNState(NamedTuple):
    """Per-neuron dynamic state. All fields are arrays of identical shape
    (typically (N,) for a population). last_spike_ms is initialized to a
    very negative number so that the first ISI check always passes."""
    V: jnp.ndarray
    n: jnp.ndarray             # K+ delayed-rectifier activation
    h: jnp.ndarray             # Na+ inactivation
    r: jnp.ndarray             # T-type Ca / H-current inactivation (shared)
    Ca: jnp.ndarray            # intracellular calcium (uM)
    last_spike_ms: jnp.ndarray


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

_EPS = 1e-9
_EXP_CLAMP = 50.0


def _safe_exp(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.exp(jnp.clip(x, -_EXP_CLAMP, _EXP_CLAMP))


def _safe_div(num: jnp.ndarray, den: jnp.ndarray) -> jnp.ndarray:
    """Avoid 0/0 in HH alpha/beta forms by replacing tiny denominators with
    a sign-preserving epsilon."""
    safe_den = jnp.where(jnp.abs(den) > _EPS, den, jnp.where(den >= 0, _EPS, -_EPS))
    return num / safe_den


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def initial_state(n_neurons: int, V_init: float = -58.0,
                  key: jax.Array | None = None,
                  heterogeneity: float = 0.05) -> STNState:
    """Build a population state. With heterogeneity > 0 and a key supplied,
    add small Gaussian noise to V and gating variables to break degeneracy."""
    V = jnp.full((n_neurons,), V_init)
    n = jnp.full((n_neurons,), 0.317)
    h = jnp.full((n_neurons,), 0.596)
    r = jnp.full((n_neurons,), 0.10)
    Ca = jnp.full((n_neurons,), 0.02)
    last = jnp.full((n_neurons,), -1.0e9)

    if key is not None and heterogeneity > 0.0:
        kV, kn, kh, kr_ = jax.random.split(key, 4)
        V = V + heterogeneity * jnp.abs(V_init) * jax.random.normal(kV, (n_neurons,))
        n = jnp.clip(n + heterogeneity * 0.317 * jax.random.normal(kn, (n_neurons,)), 0.0, 1.0)
        h = jnp.clip(h + heterogeneity * 0.596 * jax.random.normal(kh, (n_neurons,)), 0.0, 1.0)
        r = jnp.clip(r + heterogeneity * 0.10 * jax.random.normal(kr_, (n_neurons,)), 0.0, 1.0)

    return STNState(V=V, n=n, h=h, r=r, Ca=Ca, last_spike_ms=last)


# ---------------------------------------------------------------------------
# Gating kinetics
# ---------------------------------------------------------------------------

def _gating_steady(V: jnp.ndarray, p: STNParams):
    """Return (m_inf, n_inf, h_inf, a_inf, r_inf, s_inf, tau_n, tau_h, tau_r)
    in current-density / voltage-clamp form."""
    # Na+ activation (instantaneous m)
    am = 0.32 * _safe_div(V + 54.0, 1.0 - _safe_exp(-(V + 54.0) / 4.0))
    bm = 0.28 * _safe_div(V + 27.0, _safe_exp((V + 27.0) / 5.0) - 1.0)
    m_inf = am / (am + bm)

    # K+ delayed-rectifier
    an = 0.032 * _safe_div(V + 52.0, 1.0 - _safe_exp(-(V + 52.0) / 5.0))
    bn = 0.5 * _safe_exp(-(V + 57.0) / 40.0)
    n_inf = an / (an + bn)
    tau_n = (1.0 + 100.0 / (1.0 + _safe_exp((V + 80.0) / 26.0))) * 0.75

    # Na+ inactivation
    ah = 0.128 * _safe_exp(-(V + 50.0) / 18.0)
    bh = 4.0 / (1.0 + _safe_exp(-(V + 27.0) / 5.0))
    h_inf = ah / (ah + bh)
    tau_h = (1.0 + 500.0 / (1.0 + _safe_exp((V + 57.0) / 3.0))) * 0.75

    # T-current activation (instantaneous a)
    a_inf = 1.0 / (1.0 + _safe_exp(-(V - p.Vp_half) / p.kp))

    # T-current / H-current inactivation
    r_inf = 1.0 / (1.0 + _safe_exp((V - p.Vr_half) / p.kr))
    tau_r = jnp.maximum(p.tau_r_ms, 1e-3)

    # High-voltage Ca activation (instantaneous s)
    s_inf = 1.0 / (1.0 + _safe_exp(-(V + 39.0) / 8.0))

    return m_inf, n_inf, h_inf, a_inf, r_inf, s_inf, tau_n, tau_h, tau_r


def _ionic_currents(state: STNState, p: STNParams):
    """Return dict of ionic currents in uA/cm^2, plus s_inf for trace."""
    V = state.V
    m_inf, _, _, a_inf, _, s_inf, _, _, _ = _gating_steady(V, p)

    I_Na = p.g_Na * (m_inf ** 3) * state.h * (V - p.E_Na)
    I_K = p.g_K * (state.n ** 4) * (V - p.E_K)
    I_L = p.g_L * (V - p.E_L)
    I_T = p.g_T * (a_inf ** 3) * (state.r ** 2) * (V - p.E_Ca)
    I_CaH = p.g_CaH * (s_inf ** 3) * (V - p.E_Ca)
    I_AHP = p.g_AHP * (V - p.E_K) * state.Ca / jnp.maximum(state.Ca + p.k1, _EPS)
    I_H = p.g_H * state.r * (V - p.E_H)

    return I_Na, I_K, I_L, I_T, I_CaH, I_AHP, I_H


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

def stn_step(state: STNState, p: STNParams, dt: float,
             I_drive: jnp.ndarray, I_syn: jnp.ndarray, I_noise: jnp.ndarray,
             t_ms: float) -> tuple[STNState, jnp.ndarray]:
    """Forward-Euler update of one STN neuron (or a population, when arrays
    are 1-D and params are scalar). Returns (new_state, spiked_bool)."""
    V = state.V

    # Compute gating steady states / time constants at the current V
    m_inf, n_inf, h_inf, a_inf, r_inf, s_inf, tau_n, tau_h, tau_r = _gating_steady(V, p)

    # Update slow gates (forward Euler)
    n_new = jnp.clip(state.n + dt * (n_inf - state.n) / tau_n, 0.0, 1.0)
    h_new = jnp.clip(state.h + dt * (h_inf - state.h) / tau_h, 0.0, 1.0)
    r_new = jnp.clip(state.r + dt * (r_inf - state.r) / tau_r, 0.0, 1.0)

    # Ionic currents at the *current* state (m, s instantaneous)
    I_Na = p.g_Na * (m_inf ** 3) * state.h * (V - p.E_Na)
    I_K = p.g_K * (state.n ** 4) * (V - p.E_K)
    I_L = p.g_L * (V - p.E_L)
    I_T = p.g_T * (a_inf ** 3) * (state.r ** 2) * (V - p.E_Ca)
    I_CaH = p.g_CaH * (s_inf ** 3) * (V - p.E_Ca)
    I_AHP = p.g_AHP * (V - p.E_K) * state.Ca / jnp.maximum(state.Ca + p.k1, _EPS)
    I_H = p.g_H * state.r * (V - p.E_H)

    I_ion = I_Na + I_K + I_L + I_T + I_CaH + I_AHP + I_H

    # Membrane update
    dVdt = (-I_ion - I_syn + I_drive + I_noise) / p.C_m
    V_new = jnp.clip(V + dt * dVdt, p.V_clip_lo, p.V_clip_hi)

    # Calcium update (uM)
    I_Ca_total = I_T + I_CaH
    dCa = -p.alpha_Ca * I_Ca_total - state.Ca / jnp.maximum(p.tau_Ca_ms, 1e-3)
    Ca_new = jnp.maximum(state.Ca + dt * dCa, 0.0)

    # Spike: upward crossing of V_thresh, with 2 ms refractory
    crossed = (V_new >= p.V_thresh) & (V < p.V_thresh)
    isi_ok = (t_ms - state.last_spike_ms) >= p.min_isi_ms
    spiked = crossed & isi_ok
    last_new = jnp.where(spiked, t_ms, state.last_spike_ms)

    new_state = STNState(V=V_new, n=n_new, h=h_new, r=r_new,
                         Ca=Ca_new, last_spike_ms=last_new)
    return new_state, spiked


# Vectorized over neurons; scalar params broadcast.
stn_step_vmap = jax.vmap(
    stn_step,
    in_axes=(STNState(V=0, n=0, h=0, r=0, Ca=0, last_spike_ms=0),
             None, None, 0, 0, 0, None),
)
