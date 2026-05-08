"""Unit tests for the STN neuron model.

The bare STN model with Otsuka/Gillies-Willshaw-derived conductances is
near-silent at zero applied drive. We exercise the upper end of its f-I
curve where rate is large enough to be measurable in a short simulation
and where ~10% tolerance corresponds to >= 1 spike.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from bgnet.neurons.stn import STNParams, initial_state, stn_step


def _run_isolated(I_drive: float, T_ms: float, dt_ms: float = 0.025,
                  burn_ms: float = 200.0) -> float:
    """Run one isolated STN neuron with constant drive and return its
    post-burn-in firing rate in Hz."""
    p = STNParams()
    state0 = initial_state(1)
    n_steps = int(T_ms / dt_ms)

    @jax.jit
    def run():
        def body(carry, i):
            state, _ = carry
            t = i * dt_ms
            new_state, sp = stn_step(state, p, dt_ms,
                                     jnp.array([I_drive]), jnp.array([0.0]),
                                     jnp.array([0.0]), t)
            return (new_state, sp), sp
        (final, _), sps = jax.lax.scan(body, (state0, jnp.array([False])),
                                       jnp.arange(n_steps))
        return sps

    sps = run()
    burn_steps = int(burn_ms / dt_ms)
    spikes_after = jnp.sum(sps[burn_steps:])
    return float(spikes_after) / ((T_ms - burn_ms) / 1000.0)


def test_stn_silent_at_zero_drive():
    """Without applied current the bare STN model should not spike on the
    integrator timescale (consistent with in-vitro Otsuka 2004)."""
    rate = _run_isolated(0.0, T_ms=400.0)
    assert rate <= 1.0, f"expected near-silent at I=0, got {rate} Hz"


def test_stn_fires_at_strong_drive():
    """At a calibrated drive the model fires reliably and reproducibly.
    The drive level here is chosen from the f-I characterization in the
    module docstring (I=42 -> ~11 Hz), tightened by a 10 percent band."""
    rate = _run_isolated(42.0, T_ms=1200.0)
    expected = 11.0
    tol = 0.10 * expected
    assert abs(rate - expected) <= tol, (
        f"STN at I=42 uA/cm^2: expected {expected} Hz +/- {tol} Hz, got {rate} Hz"
    )


def test_stn_no_nan_under_load():
    """Even under heavy depolarizing drive, voltage and gating variables
    must remain bounded."""
    p = STNParams()
    state0 = initial_state(50, key=jax.random.PRNGKey(0))
    n_steps = 8000  # 200 ms

    @jax.jit
    def run():
        def body(carry, i):
            state, _ = carry
            t = i * 0.025
            from bgnet.neurons.stn import stn_step_vmap
            I = jnp.full((50,), 60.0)
            zero = jnp.zeros((50,))
            new_state, sp = stn_step_vmap(state, p, 0.025, I, zero, zero, t)
            return (new_state, sp), state.V
        (final, _), Vs = jax.lax.scan(body, (state0, jnp.zeros((50,), dtype=bool)),
                                      jnp.arange(n_steps))
        return final, Vs

    final, Vs = run()
    assert not bool(jnp.any(jnp.isnan(Vs))), "NaN voltages encountered"
    assert not bool(jnp.any(jnp.isnan(final.h))), "NaN h gate"
    assert not bool(jnp.any(jnp.isnan(final.Ca))), "NaN calcium"
