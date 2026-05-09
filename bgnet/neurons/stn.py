"""Single-compartment subthalamic nucleus (STN) Hodgkin-Huxley neuron.

Single-compartment Hodgkin-Huxley STN with the Terman-Rubin 2002 current
set (I_L, I_Na, I_K, I_AHP, I_Ca, I_T). Channel kinetics, time constants,
and reversal potentials follow the canonical ``episodic.ode`` distribution
of the paper. The high-threshold calcium current uses a linear instantaneous
gate (``I_Ca = g_Ca * sinf(V) * (V - E_Ca)``) rather than ``sinf^2``;
rationale and biophysical-mechanism analysis are in docs/stn_validation.md
and docs/stn_linear_form_diagnostics.md.

Pacemaking mechanism. At I_drive = 0 the cell fires tonically at ~10 Hz.
The pacing comes from a sub-threshold inward window current — partially
activated I_Ca (s_inf ≈ 0.09 at -54 mV under the linear form) plus the
Na-window current — overcoming a balanced leak and Ca-dependent AHP.
The T-current is fully armed but not engaged during tonic firing because
the cell never hyperpolarizes deeply enough between spikes for r to
de-inactivate (max r in the tonic regime is ~0.14, vs. ~0.91 reachable
under sustained hyperpolarization). On post-inhibitory release the
T-current produces a transient -77 µA/cm² spike and a 2-3 spike rebound
burst — the canonical RT-style rebound mechanism is intact and only
selectively engaged. This sub-threshold-pacemaker / rebound-on-demand
behavior is consistent with the experimental characterization of
autonomous STN firing (Bevan & Wilson 1999, J Neurosci 19:7617-7628;
Atherton & Bevan 2005, J Neurosci 25:8272-8281), which describe STN
autonomous firing as driven by persistent sodium and a small, sustained
calcium current at sub-threshold voltages, with low-threshold (T-type)
calcium recruited only for rebound bursts.

References:
    Terman D, Rubin JE, Yew AC, Wilson CJ (2002). Activity patterns in a
    model for the subthalamopallidal network of the basal ganglia.
    J Neurosci 22(7):2963-2976. ModelDB 182758 (XPP source episodic.ode).
    Bevan MD, Wilson CJ (1999). Mechanisms underlying spontaneous
    oscillation and rhythmic firing in rat subthalamic neurons.
    J Neurosci 19(17):7617-7628.
    Atherton JF, Bevan MD (2005). Ionic mechanisms underlying autonomous
    action potential generation in the somata and dendrites of GABAergic
    substantia nigra pars reticulata neurons in vitro.
    J Neurosci 25(36):8272-8281.

Channels:
    I_L    leak
    I_Na   fast sodium (m instantaneous, h slow)
    I_K    delayed-rectifier potassium (n slow)
    I_AHP  Ca-dependent K (afterhyperpolarization)
    I_Ca   high-threshold calcium (s instantaneous, linear:
           I_Ca = g_Ca * s_inf * (V - E_Ca))
    I_T    low-threshold T-type calcium (a instantaneous, r slow with binf
           transformation; r is de-inactivation, binf(r) is the effective
           inactivation that gates the current)

This model has NO I_H (hyperpolarization-activated cation) and NO
separate I_CaH; those were features of the previous Gillies-Willshaw-style
implementation that has been replaced. RT 2002 has no separate m gate —
m is instantaneous via minf(V).

Membrane equation (HH sign convention, current density throughout):
    C_m dV/dt = -(I_L + I_Na + I_K + I_AHP + I_Ca + I_T)
                - I_syn + I_drive + I_noise
with I_X = g_X * (V - E_X) (positive when hyperpolarizing).

Units (current density convention, no scaling factors):
    conductance     mS/cm^2
    current         uA/cm^2
    voltage         mV
    time            ms
    calcium         uM
    capacitance     uF/cm^2

Spike detection: upward zero-crossing of V (V_now >= 0 mV and V_prev < 0 mV)
with a 2 ms refractory window. The neuron module reports spikes per step;
the integrator is responsible for accumulating spike trains.

The tonic drive I_drive is an optimization parameter (uA/cm^2). Its default
in this module is zero. Unlike the previous GW-inspired implementation, the
RT 2002 STN fires ~10 Hz spontaneously at I_drive = 0, so the optimizer's
bounded search space ([-5, +5] uA/cm^2) gives it comfortable headroom over
the healthy 20 Hz target without resorting to large applied currents.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Parameters and state containers
# ---------------------------------------------------------------------------

class STNParams(NamedTuple):
    """Per-neuron Terman-Rubin 2002 STN parameters (episodic.ode ground truth).

    The ``theta``/``sigma`` convention follows the XPP source. For each
    Boltzmann-form steady state the embedded sign inside ``exp(...)`` matters
    and is preserved exactly: minf and sinf use ``-(V + theta)/sigma`` while
    hinf, ninf, ainf, rinf use ``(V - theta)/sigma`` with sigma sometimes
    negative. See module-level helper functions for the literal forms.
    """
    # Membrane capacitance (uF/cm^2)
    C_m: float = 1.0

    # Reversal potentials (mV)
    E_L: float = -60.0
    E_Na: float = 55.0
    E_K: float = -80.0
    E_Ca: float = 140.0

    # Maximum conductances (mS/cm^2) — episodic.ode
    g_L: float = 2.25
    g_Na: float = 37.5
    g_K: float = 45.0
    g_AHP: float = 9.0
    g_Ca: float = 0.5
    g_T: float = 0.5

    # Activation/inactivation parameters (mV)
    thetam: float = 30.0
    sigmam: float = 15.0
    thetah: float = -39.0
    sigmah: float = 3.1
    thetan: float = -32.0
    sigman: float = -8.0
    thetas: float = 39.0
    sigmas: float = 8.0
    thetaa: float = -63.0
    sigmaa: float = -7.8
    thetar: float = -67.0
    sigmar: float = 2.0
    thetab: float = 0.25
    sigmab: float = -0.07

    # Time constant parameters (ms; thresholds in mV)
    taun0: float = 1.0
    taun1: float = 100.0
    thn: float = 80.0
    sigmant: float = 26.0
    tauh0: float = 1.0
    tauh1: float = 500.0
    thh: float = 57.0
    sigmaht: float = 3.0
    taur0: float = 7.1
    taur1: float = 17.5
    thr_t: float = 68.0
    sigmart: float = 2.2

    # Calcium dynamics
    eps: float = 5e-5
    kca: float = 22.5
    k1: float = 15.0           # half-saturation Ca for AHP gating (uM)

    # Rate factors (RT 2002 phi multiplies the slow gate ODEs and Ca eqn)
    phi: float = 0.75          # for h, n, Ca
    phir: float = 0.5          # for r

    # Spike detection
    V_thresh: float = 0.0
    min_isi_ms: float = 2.0

    # Numerical safety
    V_clip_lo: float = -100.0
    V_clip_hi: float = 60.0


class STNState(NamedTuple):
    """Per-neuron dynamic state.

    RT 2002 dynamic variables: V, h, n, r, Ca. The m, s, and a gates are
    instantaneous functions of V and are not stored. last_spike_ms tracks
    the most recent spike time for refractory bookkeeping.
    """
    V: jnp.ndarray
    h: jnp.ndarray             # Na+ inactivation
    n: jnp.ndarray             # K+ delayed-rectifier activation
    r: jnp.ndarray             # T-current de-inactivation (binf(r) is the effective gate)
    Ca: jnp.ndarray            # intracellular calcium (uM)
    last_spike_ms: jnp.ndarray


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

_EXP_CLAMP = 50.0


def _safe_exp(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.exp(jnp.clip(x, -_EXP_CLAMP, _EXP_CLAMP))


# ---------------------------------------------------------------------------
# Steady-state activation/inactivation functions (RT 2002 episodic.ode forms)
# ---------------------------------------------------------------------------
#
# Public, JAX-compatible. Test code calls these for plotting f-I and rebound.
# The sign embedding inside each formula follows the XPP source verbatim;
# do not "simplify" by absorbing signs into sigma — sigma is sometimes
# negative on purpose.

def minf(V: jnp.ndarray, p: STNParams = STNParams()) -> jnp.ndarray:
    return 1.0 / (1.0 + _safe_exp(-(V + p.thetam) / p.sigmam))


def hinf(V: jnp.ndarray, p: STNParams = STNParams()) -> jnp.ndarray:
    return 1.0 / (1.0 + _safe_exp((V - p.thetah) / p.sigmah))


def ninf(V: jnp.ndarray, p: STNParams = STNParams()) -> jnp.ndarray:
    return 1.0 / (1.0 + _safe_exp((V - p.thetan) / p.sigman))


def sinf(V: jnp.ndarray, p: STNParams = STNParams()) -> jnp.ndarray:
    return 1.0 / (1.0 + _safe_exp(-(V + p.thetas) / p.sigmas))


def ainf(V: jnp.ndarray, p: STNParams = STNParams()) -> jnp.ndarray:
    return 1.0 / (1.0 + _safe_exp((V - p.thetaa) / p.sigmaa))


def rinf(V: jnp.ndarray, p: STNParams = STNParams()) -> jnp.ndarray:
    return 1.0 / (1.0 + _safe_exp((V - p.thetar) / p.sigmar))


def binf(r: jnp.ndarray, p: STNParams = STNParams()) -> jnp.ndarray:
    """T-current effective inactivation as a function of r.

    The constant subtraction is intentional: it forces binf(0) = 0 so that
    when r is fully inactivated the T-current vanishes. Do not simplify it
    away.
    """
    return (1.0 / (1.0 + _safe_exp((r - p.thetab) / p.sigmab))
            - 1.0 / (1.0 + _safe_exp(-p.thetab / p.sigmab)))


def taun(V: jnp.ndarray, p: STNParams = STNParams()) -> jnp.ndarray:
    return p.taun0 + p.taun1 / (1.0 + _safe_exp((V + p.thn) / p.sigmant))


def tauh(V: jnp.ndarray, p: STNParams = STNParams()) -> jnp.ndarray:
    return p.tauh0 + p.tauh1 / (1.0 + _safe_exp((V + p.thh) / p.sigmaht))


def taur(V: jnp.ndarray, p: STNParams = STNParams()) -> jnp.ndarray:
    return p.taur0 + p.taur1 / (1.0 + _safe_exp((V + p.thr_t) / p.sigmart))


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def initial_state(n_neurons: int, V_init: float = -65.0,
                  key: jax.Array | None = None,
                  heterogeneity: float = 0.05) -> STNState:
    """Build a population state at V_init. Gating variables are seeded at
    their steady-state values for V_init, and Ca is initialized to 0.1 uM
    (a reasonable resting concentration). With heterogeneity > 0 and a key
    supplied, add small Gaussian noise to V and gating variables to break
    degeneracy."""
    p = STNParams()
    # Steady-state gates at V_init (computed in plain Python for setup)
    h0 = 1.0 / (1.0 + math.exp((V_init - p.thetah) / p.sigmah))
    n0 = 1.0 / (1.0 + math.exp((V_init - p.thetan) / p.sigman))
    r0 = 1.0 / (1.0 + math.exp((V_init - p.thetar) / p.sigmar))
    Ca0 = 0.1

    V = jnp.full((n_neurons,), V_init)
    h = jnp.full((n_neurons,), h0)
    n = jnp.full((n_neurons,), n0)
    r = jnp.full((n_neurons,), r0)
    Ca = jnp.full((n_neurons,), Ca0)
    last = jnp.full((n_neurons,), -1.0e9)

    if key is not None and heterogeneity > 0.0:
        kV, kh, kn, kr_ = jax.random.split(key, 4)
        V = V + heterogeneity * abs(V_init) * jax.random.normal(kV, (n_neurons,))
        h = jnp.clip(h + heterogeneity * h0 * jax.random.normal(kh, (n_neurons,)), 0.0, 1.0)
        n = jnp.clip(n + heterogeneity * n0 * jax.random.normal(kn, (n_neurons,)), 0.0, 1.0)
        r = jnp.clip(r + heterogeneity * r0 * jax.random.normal(kr_, (n_neurons,)), 0.0, 1.0)

    return STNState(V=V, h=h, n=n, r=r, Ca=Ca, last_spike_ms=last)


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

def stn_step(state: STNState, p: STNParams, dt: float,
             I_drive: jnp.ndarray, I_syn: jnp.ndarray, I_noise: jnp.ndarray,
             t_ms: float) -> tuple[STNState, jnp.ndarray]:
    """Forward-Euler update of one STN neuron (or a population, when arrays
    are 1-D and params are scalar). Returns (new_state, spiked_bool)."""
    V = state.V

    # Steady states and time constants at the current V
    m_inf = minf(V, p)
    h_inf = hinf(V, p)
    n_inf = ninf(V, p)
    s_inf = sinf(V, p)
    a_inf = ainf(V, p)
    r_inf = rinf(V, p)
    b_r = binf(state.r, p)

    tau_h = tauh(V, p)
    tau_n = taun(V, p)
    tau_r = taur(V, p)

    # Ionic currents (HH sign convention, m and s and a are instantaneous)
    I_L = p.g_L * (V - p.E_L)
    I_Na = p.g_Na * (m_inf ** 3) * state.h * (V - p.E_Na)
    I_K = p.g_K * (state.n ** 4) * (V - p.E_K)
    I_AHP = p.g_AHP * (V - p.E_K) * state.Ca / (state.Ca + p.k1)
    # I_Ca uses a linear instantaneous gate (sinf^1). With sinf^1 the cell
    # fires tonically at ~10 Hz from a sub-threshold I_Ca + Na-window
    # pacemaker, consistent with experimental STN characterization
    # (Bevan & Wilson 1999, Atherton & Bevan 2005). The T-current rebound
    # mechanism is preserved and only selectively engaged. See the module
    # docstring and docs/stn_validation.md for the empirical/biophysical
    # justification.
    I_Ca = p.g_Ca * s_inf * (V - p.E_Ca)
    I_T = p.g_T * (a_inf ** 3) * (b_r ** 2) * (V - p.E_Ca)

    I_ion = I_L + I_Na + I_K + I_AHP + I_Ca + I_T

    # Membrane update
    dVdt = (-I_ion - I_syn + I_drive + I_noise) / p.C_m
    V_new = jnp.clip(V + dt * dVdt, p.V_clip_lo, p.V_clip_hi)

    # Slow gating variables (RT 2002: phi multiplies the rate)
    h_new = jnp.clip(state.h + dt * p.phi * (h_inf - state.h) / tau_h, 0.0, 1.0)
    n_new = jnp.clip(state.n + dt * p.phi * (n_inf - state.n) / tau_n, 0.0, 1.0)
    r_new = jnp.clip(state.r + dt * p.phir * (r_inf - state.r) / tau_r, 0.0, 1.0)

    # Calcium dynamics: phi * eps * (-I_Ca - I_T - kca * Ca)
    dCa = p.phi * p.eps * (-I_Ca - I_T - p.kca * state.Ca)
    Ca_new = jnp.maximum(state.Ca + dt * dCa, 0.0)

    # Spike: upward crossing of V_thresh, with min_isi_ms refractory
    crossed = (V_new >= p.V_thresh) & (V < p.V_thresh)
    isi_ok = (t_ms - state.last_spike_ms) >= p.min_isi_ms
    spiked = crossed & isi_ok
    last_new = jnp.where(spiked, t_ms, state.last_spike_ms)

    new_state = STNState(V=V_new, h=h_new, n=n_new, r=r_new,
                         Ca=Ca_new, last_spike_ms=last_new)
    return new_state, spiked


# Vectorized over neurons; scalar params broadcast.
stn_step_vmap = jax.vmap(
    stn_step,
    in_axes=(STNState(V=0, h=0, n=0, r=0, Ca=0, last_spike_ms=0),
             None, None, 0, 0, 0, None),
)
