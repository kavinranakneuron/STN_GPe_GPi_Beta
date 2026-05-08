"""Unit tests for GPe and GPi Rubin-Terman neurons."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from bgnet.neurons.pallidum import gpe_params, gpi_params, initial_state, pallidum_step


def _run_isolated(p, I_drive: float, T_ms: float = 1200.0,
                  dt_ms: float = 0.025, burn_ms: float = 200.0) -> float:
    state0 = initial_state(1)
    n_steps = int(T_ms / dt_ms)

    @jax.jit
    def run():
        def body(carry, i):
            state, _ = carry
            new_state, sp = pallidum_step(state, p, dt_ms,
                                          jnp.array([I_drive]),
                                          jnp.array([0.0]),
                                          jnp.array([0.0]), i * dt_ms)
            return (new_state, sp), sp
        (final, _), sps = jax.lax.scan(body, (state0, jnp.array([False])),
                                       jnp.arange(n_steps))
        return sps

    sps = run()
    burn_steps = int(burn_ms / dt_ms)
    spikes_after = jnp.sum(sps[burn_steps:])
    return float(spikes_after) / ((T_ms - burn_ms) / 1000.0)


def test_gpe_baseline_rate_at_zero_drive():
    """Cell-type intrinsic baseline (I_app = 1.5) drives GPe to ~40 Hz
    at I_drive = 0, in line with the f-I characterization."""
    rate = _run_isolated(gpe_params(), I_drive=0.0)
    expected = 41.0
    tol = max(0.10 * expected, 2.0)
    assert abs(rate - expected) <= tol, f"GPe at I=0: expected {expected} Hz, got {rate}"


def test_gpe_rate_at_drive_2():
    """GPe healthy target ~65 Hz is reached near I_drive = 2 (f-I curve)."""
    rate = _run_isolated(gpe_params(), I_drive=2.0)
    expected = 66.0
    tol = max(0.10 * expected, 2.0)
    assert abs(rate - expected) <= tol, f"GPe at I=2: expected {expected} Hz, got {rate}"


def test_gpi_baseline_rate_at_zero_drive():
    rate = _run_isolated(gpi_params(), I_drive=0.0)
    expected = 71.0
    tol = max(0.10 * expected, 2.0)
    assert abs(rate - expected) <= tol, f"GPi at I=0: expected {expected} Hz, got {rate}"


def test_gpe_silent_at_strong_hyperpolarization():
    """A large negative drive must overwhelm I_app and silence the cell."""
    rate = _run_isolated(gpe_params(), I_drive=-5.0)
    assert rate <= 1.0, f"expected silent at I=-5, got {rate} Hz"


def test_gpi_no_nan():
    """No NaN even at strong positive drive."""
    p = gpi_params()
    state0 = initial_state(50, key=jax.random.PRNGKey(0))

    @jax.jit
    def run():
        from bgnet.neurons.pallidum import pallidum_step_vmap
        def body(carry, i):
            state, _ = carry
            I = jnp.full((50,), 5.0)
            zero = jnp.zeros((50,))
            new_state, sp = pallidum_step_vmap(state, p, 0.025, I, zero, zero, i * 0.025)
            return (new_state, sp), state.V
        (final, _), Vs = jax.lax.scan(body,
                                      (state0, jnp.zeros((50,), dtype=bool)),
                                      jnp.arange(8000))
        return final, Vs

    final, Vs = run()
    assert not bool(jnp.any(jnp.isnan(Vs)))
