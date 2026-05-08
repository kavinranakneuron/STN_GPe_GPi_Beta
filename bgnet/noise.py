"""Ornstein-Uhlenbeck background-input noise.

Each neuron receives an independent OU process I_noise(t) with population-
level mean mu, standard deviation sigma, and correlation time tau. The
deterministic-plus-stochastic SDE is

    dI = -(I - mu)/tau dt + sigma * sqrt(2/tau) dW

so that the stationary distribution is N(mu, sigma^2) and the autocorrelation
time is tau. Per AGENTS.md step 5 we use Euler-Maruyama integration:

    I[t+dt] = I[t] + (-(I-mu)/tau) * dt + sigma * sqrt(2*dt/tau) * Z,
    Z ~ N(0, 1) (independent per neuron, per step)

Why OU over white noise or Poisson: a real cell receives many small
afferent spikes, and after passive low-pass membrane filtering the
resulting input current is approximately Gaussian with a finite
autocorrelation time set by the slowest of (membrane tau, synaptic tau,
afferent rate). White noise (tau -> 0) over-amplifies high frequencies
and produces artifacts in HH dynamics; Poisson with tau ~ ms produces
indistinguishable averages but is more expensive. tau = 5 ms is short
compared to most neuronal time constants but long enough to act as a
proper colored process. (Destexhe et al. 2001, Neuroscience 107:13-24;
Tuckwell 1988, Stochastic Processes in the Neurosciences.)

Both mu and sigma are optimizer-controlled parameters (per population);
tau is fixed at 5 ms throughout the rebuild.

Units: I is in uA/cm^2, tau in ms. mu and sigma share I's units.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class OUParams(NamedTuple):
    """Per-population OU parameters."""
    mu: float           # uA/cm^2
    sigma: float        # uA/cm^2
    tau_ms: float = 5.0


def init_state(n_neurons: int, p: OUParams) -> jnp.ndarray:
    """Initialize each neuron's OU value to the population mean."""
    return jnp.full((n_neurons,), p.mu)


def ou_step(I: jnp.ndarray, key: jax.Array, dt: float,
            p: OUParams) -> tuple[jnp.ndarray, jax.Array]:
    """One Euler-Maruyama step. Returns (new I, fresh key for the next step).

    The caller is expected to pass a fresh PRNG key each step (typically by
    splitting once outside the JIT'd loop body).
    """
    key, sub = jax.random.split(key)
    z = jax.random.normal(sub, I.shape)
    drift = -(I - p.mu) / p.tau_ms
    diffusion = p.sigma * jnp.sqrt(2.0 * dt / p.tau_ms)
    return I + dt * drift + diffusion * z, key
