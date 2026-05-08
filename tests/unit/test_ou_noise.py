"""Unit tests for the OU noise process."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from bgnet.noise import OUParams, init_state, ou_step


def test_stationary_mean_and_variance():
    """Long-run mean -> mu, variance -> sigma^2 within 5 percent across a
    population of 5000 cells over 5 s."""
    N = 5000
    T_ms = 5000.0
    dt = 0.025
    n_steps = int(T_ms / dt)
    p = OUParams(mu=2.0, sigma=1.5, tau_ms=5.0)
    I0 = init_state(N, p)
    key = jax.random.PRNGKey(0)

    @jax.jit
    def run(I, key):
        def body(carry, _):
            I, k = carry
            I_new, k_new = ou_step(I, k, dt, p)
            return (I_new, k_new), I_new
        (I_final, _), Is = jax.lax.scan(body, (I, key), jnp.arange(n_steps))
        return Is

    Is = run(I0, key)
    burn = int(50 / dt)
    Is_steady = Is[burn:]
    mean_obs = float(jnp.mean(Is_steady))
    std_obs = float(jnp.std(Is_steady))
    assert abs(mean_obs - p.mu) < 0.05 * p.mu, (
        f"OU mean: expected {p.mu}, got {mean_obs}"
    )
    assert abs(std_obs - p.sigma) < 0.05 * p.sigma, (
        f"OU std: expected {p.sigma}, got {std_obs}"
    )


def test_autocorr_at_lag_tau_is_one_over_e():
    """The autocorrelation at lag tau equals 1/e for an OU process."""
    N = 5000
    T_ms = 5000.0
    dt = 0.025
    n_steps = int(T_ms / dt)
    p = OUParams(mu=0.0, sigma=2.0, tau_ms=5.0)
    I0 = init_state(N, p)
    key = jax.random.PRNGKey(1)

    @jax.jit
    def run(I, key):
        def body(carry, _):
            I, k = carry
            I_new, k_new = ou_step(I, k, dt, p)
            return (I_new, k_new), I_new
        (I_final, _), Is = jax.lax.scan(body, (I, key), jnp.arange(n_steps))
        return Is

    Is = run(I0, key)[int(50 / dt):, 0]
    mean = jnp.mean(Is)
    var = jnp.var(Is)
    lag = int(p.tau_ms / dt)
    ac = jnp.mean((Is[:-lag] - mean) * (Is[lag:] - mean)) / var
    assert abs(float(ac) - jnp.exp(-1.0)) < 0.05, f"autocorr at tau: {float(ac)}"
