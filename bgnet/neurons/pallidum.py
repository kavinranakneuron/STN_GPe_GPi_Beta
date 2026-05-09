"""Single-compartment Rubin-Terman pallidum (GPe and GPi) neuron.

A single step function serves both GPe and GPi; the only differences between
the two cell types are the parameter values (g_T, g_AHP, baseline I_app),
provided through PallidumParams.

Currents:
    I_Na   fast sodium (spike upstroke)
    I_K    delayed-rectifier potassium (repolarization)
    I_L    leak
    I_T    T-type calcium (low-threshold rebound)
    I_Ca   high-threshold calcium (instantaneous activation)
    I_AHP  calcium-activated potassium

Membrane equation (matches STN for unit-convention consistency):
    C_m dV/dt = -I_Na - I_K - I_L - I_T - I_Ca - I_AHP - I_syn + I_drive + I_noise + I_app

I_app is the cell-type baseline tonic from Rubin & Terman 2004; I_drive is
the optimizer's per-population tonic perturbation; I_noise is the OU output;
I_syn is computed by bgnet.synapses with the same I = g*(V-E) sign convention.

Spike detection: upward crossing of V_thresh = -20 mV (V_now >= -20 and
V_prev < -20). No refractory window — Rubin-Terman cells have a strong
intrinsic AHP that prevents double-counting.

Sources:
    Rubin & Terman 2004, J Comput Neurosci 16(3):211-235
    Terman et al. 2002, J Neurosci 22(7):2963-2976
    Ebert et al. 2014, Front Comput Neurosci 8:154 (parameter ranges)

Units (current density throughout): mS/cm^2, uA/cm^2, mV, ms, uM, uF/cm^2.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from bgnet.heterogeneity import HetMul, homogeneous_het


class PallidumParams(NamedTuple):
    """Parameters for one pallidum cell type (GPe or GPi).

    Use ``gpe_params()`` / ``gpi_params()`` to construct the canonical sets.
    """
    # capacitance (uF/cm^2)
    C_m: float = 1.0
    # peak conductances (mS/cm^2) — Rubin-Terman 2004 baseline
    g_Na: float = 120.0
    g_K: float = 30.0
    g_L: float = 0.1
    g_T: float = 0.5
    g_Ca: float = 0.15
    g_AHP: float = 30.0
    # reversal potentials (mV)
    E_Na: float = 55.0
    E_K: float = -80.0
    E_Ca: float = 120.0
    E_L: float = -65.0
    # calcium dynamics
    tau_Ca_ms: float = 20.0
    k_Ca: float = 0.002        # uM per (uA/cm^2 * ms)
    k_AHP: float = 10.0        # half-saturation Ca for AHP gating (uM)
    # baseline tonic (cell-type intrinsic, NOT optimizer-controlled)
    I_app: float = 1.5
    # spike detection
    V_thresh: float = -20.0
    # numerical safety
    V_clip_lo: float = -100.0
    V_clip_hi: float = 60.0


def gpe_params(**overrides) -> PallidumParams:
    """Canonical GPe cell parameters (Rubin & Terman 2004)."""
    return PallidumParams(
        g_T=0.5,
        g_AHP=30.0,
        I_app=1.5,
        **overrides,
    )


def gpi_params(**overrides) -> PallidumParams:
    """Canonical GPi cell parameters (Rubin & Terman 2004 + Ebert et al. 2014).

    GPi differs from GPe by reduced T-current, reduced AHP, and a higher
    intrinsic baseline (consistent with GPi's higher tonic firing rate).
    """
    return PallidumParams(
        g_T=0.3,
        g_AHP=20.0,
        I_app=2.0,
        **overrides,
    )


class PallidumState(NamedTuple):
    """Per-neuron dynamic state for a Rubin-Terman pallidum cell."""
    V: jnp.ndarray
    n: jnp.ndarray             # K+ delayed-rectifier activation
    h: jnp.ndarray             # Na+ inactivation
    r: jnp.ndarray             # T-type Ca inactivation (rebound)
    Ca: jnp.ndarray            # intracellular Ca (uM)
    last_spike_ms: jnp.ndarray


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

_EPS = 1e-9
_EXP_CLAMP = 50.0


def _safe_exp(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.exp(jnp.clip(x, -_EXP_CLAMP, _EXP_CLAMP))


def _safe_div(num: jnp.ndarray, den: jnp.ndarray) -> jnp.ndarray:
    safe_den = jnp.where(jnp.abs(den) > _EPS, den, jnp.where(den >= 0, _EPS, -_EPS))
    return num / safe_den


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def initial_state(n_neurons: int, V_init: float = -65.0,
                  key: jax.Array | None = None,
                  heterogeneity: float = 0.05) -> PallidumState:
    V = jnp.full((n_neurons,), V_init)
    n = jnp.full((n_neurons,), 0.05)
    h = jnp.full((n_neurons,), 0.85)
    r = jnp.full((n_neurons,), 0.20)
    Ca = jnp.full((n_neurons,), 0.0)
    last = jnp.full((n_neurons,), -1.0e9)

    if key is not None and heterogeneity > 0.0:
        kV, kn, kh, kr_ = jax.random.split(key, 4)
        V = V + heterogeneity * jnp.abs(V_init) * jax.random.normal(kV, (n_neurons,))
        n = jnp.clip(n + heterogeneity * 0.05 * jax.random.normal(kn, (n_neurons,)), 0.0, 1.0)
        h = jnp.clip(h + heterogeneity * 0.85 * jax.random.normal(kh, (n_neurons,)), 0.0, 1.0)
        r = jnp.clip(r + heterogeneity * 0.20 * jax.random.normal(kr_, (n_neurons,)), 0.0, 1.0)

    return PallidumState(V=V, n=n, h=h, r=r, Ca=Ca, last_spike_ms=last)


# ---------------------------------------------------------------------------
# Gating kinetics
# ---------------------------------------------------------------------------

def _gating(V: jnp.ndarray):
    """Return (m_inf, n_inf, h_inf, a_inf, r_inf, s_inf, tau_n, tau_h, tau_r)
    for the Rubin-Terman pallidum cell."""
    # Na+ activation (instantaneous m)
    am = 0.32 * _safe_div(V + 54.0, 1.0 - _safe_exp(-(V + 54.0) / 4.0))
    bm = 0.28 * _safe_div(V + 27.0, _safe_exp((V + 27.0) / 5.0) - 1.0)
    m_inf = am / (am + bm + _EPS)

    # K+ delayed-rectifier
    an = 0.032 * _safe_div(V + 52.0, 1.0 - _safe_exp(-(V + 52.0) / 5.0))
    bn = 0.5 * _safe_exp(-(V + 57.0) / 40.0)
    n_inf = an / (an + bn + _EPS)
    tau_n = 1.0 / (an + bn + _EPS)

    # Na+ inactivation
    ah = 0.128 * _safe_exp(-(V + 50.0) / 18.0)
    bh = 4.0 / (1.0 + _safe_exp(-(V + 27.0) / 5.0))
    h_inf = ah / (ah + bh + _EPS)
    tau_h = 1.0 / (ah + bh + _EPS)

    # T-current (low-threshold Ca) activation: instantaneous a
    a_inf = 1.0 / (1.0 + _safe_exp(-(V + 63.0) / 7.8))

    # T-current inactivation: r
    r_inf = 1.0 / (1.0 + _safe_exp((V + 67.0) / 2.0))
    tau_r = 30.0 + 150.0 / (1.0 + _safe_exp((V + 67.0) / 2.0))

    # High-threshold Ca activation: instantaneous s
    s_inf = 1.0 / (1.0 + _safe_exp(-(V + 35.0) / 2.0))

    return m_inf, n_inf, h_inf, a_inf, r_inf, s_inf, tau_n, tau_h, tau_r


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

def pallidum_step(state: PallidumState, p: PallidumParams, het: HetMul,
                  dt: float,
                  I_drive: jnp.ndarray, I_syn: jnp.ndarray, I_noise: jnp.ndarray,
                  t_ms: float) -> tuple[PallidumState, jnp.ndarray]:
    """Forward-Euler update of one pallidum neuron (GPe or GPi). For
    unit tests on a single neuron use ``homogeneous_het(1)``. Returns
    ``(new_state, spiked_bool)``."""
    V = state.V
    m_inf, n_inf, h_inf, a_inf, r_inf, s_inf, tau_n, tau_h, tau_r = _gating(V)

    # Slow gates
    n_new = jnp.clip(state.n + dt * (n_inf - state.n) / jnp.maximum(tau_n, 1e-3), 0.0, 1.0)
    h_new = jnp.clip(state.h + dt * (h_inf - state.h) / jnp.maximum(tau_h, 1e-3), 0.0, 1.0)
    r_new = jnp.clip(state.r + dt * (r_inf - state.r) / jnp.maximum(tau_r, 1e-3), 0.0, 1.0)

    # Currents (HH sign convention: I_X = g_X * (V - E_X), positive hyperpolarizing).
    # Per-neuron heterogeneity multipliers on g_L, g_Na, g_K.
    I_Na = p.g_Na * het.mul_g_Na * (m_inf ** 3) * state.h * (V - p.E_Na)
    I_K = p.g_K * het.mul_g_K * (state.n ** 4) * (V - p.E_K)
    I_L = p.g_L * het.mul_g_L * (V - p.E_L)
    I_T = p.g_T * (a_inf ** 3) * state.r * (V - p.E_Ca)
    I_Ca = p.g_Ca * (s_inf ** 2) * (V - p.E_Ca)
    ahp = state.Ca / (state.Ca + p.k_AHP + _EPS)
    I_AHP = p.g_AHP * ahp * (V - p.E_K)

    I_ion = I_Na + I_K + I_L + I_T + I_Ca + I_AHP

    # Membrane update
    dVdt = (-I_ion - I_syn + I_drive + I_noise + p.I_app) / p.C_m
    V_new = jnp.clip(V + dt * dVdt, p.V_clip_lo, p.V_clip_hi)

    # Calcium: pumped out of cell by the negative-going Ca currents
    dCa = -p.k_Ca * (I_T + I_Ca) - state.Ca / jnp.maximum(p.tau_Ca_ms, 1e-3)
    Ca_new = jnp.maximum(state.Ca + dt * dCa, 0.0)

    # Spike detection: upward crossing of -20 mV
    spiked = (V_new >= p.V_thresh) & (V < p.V_thresh)
    last_new = jnp.where(spiked, t_ms, state.last_spike_ms)

    new_state = PallidumState(V=V_new, n=n_new, h=h_new, r=r_new,
                              Ca=Ca_new, last_spike_ms=last_new)
    return new_state, spiked


pallidum_step_vmap = jax.vmap(
    pallidum_step,
    in_axes=(PallidumState(V=0, n=0, h=0, r=0, Ca=0, last_spike_ms=0),
             None,
             HetMul(mul_g_L=0, mul_g_Na=0, mul_g_K=0),
             None, 0, 0, 0, None),
)
