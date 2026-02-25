"""
Sparse synaptic connectivity with fixed indegree.

Each postsynaptic neuron receives exactly `indegree` presynaptic inputs,
sampled uniformly without replacement. This preserves both the mean AND
variance of total synaptic input across network sizes (Gerstner et al.,
Neuronal Dynamics, Ch 12.3). In contrast, fixed connection probability
with conductance scaling only preserves the mean — variance scales as
1/sqrt(N), suppressing fluctuation-driven dynamics at larger network sizes.

Author: Kavin Nakkeeran, Johns Hopkins University
"""

import jax.numpy as jnp
import numpy as np
from jax import random
from typing import NamedTuple


class SynapseConfig(NamedTuple):
    n_pre: int
    n_post: int
    indegree: int
    g_max: float
    delay_ms: float
    tau_rise: float
    tau_decay: float
    E_syn: float
    dt_ms: float
    connections: jnp.ndarray  # Shape: (n_connections, 2) - [pre_idx, post_idx]
    weights: jnp.ndarray      # Shape: (n_connections,)


class SynapseState(NamedTuple):
    s: jnp.ndarray           # Gating variable (n_connections,)
    x: jnp.ndarray           # Rising phase (n_connections,)
    delay_buffer: jnp.ndarray # Shape: (n_pre, buffer_size)
    buffer_idx: int


def create_synapse_config(n_pre, n_post, indegree, g_max, delay_ms,
                         tau_rise, tau_decay, E_syn, dt_ms, seed=42):
    """
    Create sparse connectivity with fixed indegree.

    Each postsynaptic neuron receives exactly `indegree` inputs sampled
    uniformly without replacement from the presynaptic population.
    Total connections = n_post * min(indegree, n_pre).

    Args:
        n_pre: Number of presynaptic neurons
        n_post: Number of postsynaptic neurons
        indegree: Number of presynaptic inputs per postsynaptic neuron
        g_max: Maximum synaptic conductance
        delay_ms: Synaptic delay
        tau_rise: Rise time constant (ms)
        tau_decay: Decay time constant (ms)
        E_syn: Reversal potential (mV)
        dt_ms: Timestep (ms)
        seed: Random seed
    """
    rng = np.random.default_rng(seed)

    k = min(indegree, n_pre)  # Cap at n_pre if indegree exceeds it

    # For each postsynaptic neuron, sample k presynaptic indices without replacement
    pre_indices = np.array([rng.choice(n_pre, size=k, replace=False)
                            for _ in range(n_post)])  # (n_post, k)
    post_indices = np.repeat(np.arange(n_post), k)
    pre_indices = pre_indices.ravel()

    connections_np = np.stack([pre_indices, post_indices], axis=1)
    weights_np = np.full(len(connections_np), g_max, dtype=np.float32)

    connections = jnp.array(connections_np, dtype=jnp.int32)
    weights = jnp.array(weights_np, dtype=jnp.float32)

    return SynapseConfig(
        n_pre=n_pre, n_post=n_post,
        indegree=k, g_max=g_max,
        delay_ms=delay_ms, tau_rise=tau_rise, tau_decay=tau_decay,
        E_syn=E_syn, dt_ms=dt_ms,
        connections=connections, weights=weights
    )


def create_synapse_config_jax(n_pre, n_post, indegree, g_max, delay_ms,
                              tau_rise, tau_decay, E_syn, dt_ms, key):
    """
    Pure JAX version for GPU-accelerated connection generation.

    Deprecated: prefer create_synapse_config (NumPy-based) for deterministic
    connectivity. This version uses approximate sampling on GPU.
    """
    k = min(indegree, n_pre)
    n_total = n_post * k

    # Generate random pre indices on GPU (approximate: with replacement)
    key1, _ = random.split(key)
    pre_indices = random.randint(key1, (n_total,), 0, n_pre)
    post_indices = jnp.repeat(jnp.arange(n_post), k)

    connections = jnp.stack([pre_indices, post_indices], axis=1)
    weights = jnp.full(n_total, g_max, dtype=jnp.float32)

    return SynapseConfig(
        n_pre=n_pre, n_post=n_post,
        indegree=k, g_max=g_max,
        delay_ms=delay_ms, tau_rise=tau_rise, tau_decay=tau_decay,
        E_syn=E_syn, dt_ms=dt_ms,
        connections=connections, weights=weights
    )


def init_synapse_state(config: SynapseConfig) -> SynapseState:
    """Initialize synapse state"""
    n_connections = config.connections.shape[0]
    buffer_size = max(1, int(jnp.ceil(config.delay_ms / config.dt_ms)))

    return SynapseState(
        s=jnp.zeros(n_connections, dtype=jnp.float32),
        x=jnp.zeros(n_connections, dtype=jnp.float32),
        delay_buffer=jnp.zeros((config.n_pre, buffer_size), dtype=jnp.bool_),
        buffer_idx=0
    )


def synapse_step(state: SynapseState, config: SynapseConfig,
                 spikes_pre: jnp.ndarray, V_post: jnp.ndarray):
    """Update synapses - JAX compatible (unchanged from original)"""
    dt = config.dt_ms

    # Update delay buffer
    new_buffer = state.delay_buffer.at[:, state.buffer_idx].set(spikes_pre)
    next_idx = (state.buffer_idx + 1) % state.delay_buffer.shape[1]

    # Get delayed spikes
    delayed_spikes = new_buffer[:, next_idx]

    # For each connection, check if pre-synaptic neuron spiked
    pre_indices = config.connections[:, 0]
    post_indices = config.connections[:, 1]

    # Get spikes for each connection
    spikes_at_connections = delayed_spikes[pre_indices]

    # Update gating variables (vectorized)
    alpha_x = 1.0 / config.tau_rise
    alpha_s = 1.0 / config.tau_decay

    # x dynamics
    dx = -alpha_x * state.x
    new_x = state.x + dt * dx + spikes_at_connections.astype(jnp.float32)

    # s dynamics
    ds = -alpha_s * state.s + alpha_x * state.x
    new_s = state.s + dt * ds
    new_s = jnp.clip(new_s, 0.0, 1.0)

    # Compute currents for each connection
    V_at_connections = V_post[post_indices]
    I_at_connections = config.weights * new_s * (config.E_syn - V_at_connections)

    # Sum currents for each post-synaptic neuron
    I_syn = jnp.zeros(config.n_post, dtype=jnp.float32)
    I_syn = I_syn.at[post_indices].add(I_at_connections)

    new_state = SynapseState(
        s=new_s, x=new_x,
        delay_buffer=new_buffer,
        buffer_idx=next_idx
    )

    return new_state, I_syn
