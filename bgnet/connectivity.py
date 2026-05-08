"""Fixed-indegree sparse connectivity for bgnet.

For each postsynaptic neuron, exactly K presynaptic neurons are sampled
uniformly without replacement from the source population. The result is
stored in post-major form as an integer index array of shape (N_post, K),
which is the natural layout for the inner-loop gather operation

    pulse_post = g_max * spikes_pre[pre_idx]   # shape (N_post, K)
    pulse_post = jnp.sum(pulse_post, axis=1)   # shape (N_post,)

This is mathematically equivalent to a sparse COO ([post_idx, pre_idx]) but
avoids a scatter and is JIT-friendly.

Why fixed indegree (over Erdős-Rényi / fixed probability): in fixed-probability
schemes the input variance scales as 1/sqrt(N) and the dynamics drift as the
network is grown, so parameters tuned at one size do not transfer. Fixed
indegree preserves the per-neuron input distribution across sizes (Gerstner
et al. 2014, Neuronal Dynamics, Cambridge UP, Ch 12.3) and is what makes
the bgnet scaling validation honest.

Default per-pathway indegrees (AGENTS.md section 4.5):
    K_STN_to_GPe = 15
    K_GPe_to_STN = 14
    K_STN_to_GPi = 30
    K_GPe_to_GPi = 10
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


class Connectivity(NamedTuple):
    """Post-major adjacency for one pathway.

    Attributes
    ----------
    pre_idx : (N_post, K) int32
        Indices of the K presynaptic neurons feeding each postsynaptic neuron.
    K : int
        Indegree per postsynaptic neuron (constant across the population).
    """
    pre_idx: jnp.ndarray
    K: int


def build_connectivity(rng: np.random.Generator, n_pre: int, n_post: int,
                       K: int) -> Connectivity:
    """Sample fixed-indegree connectivity. K presynaptic partners per
    postsynaptic neuron, drawn uniformly without replacement.

    Build once at network setup time on the host (numpy), then converted to
    a jnp array for the inner loop.
    """
    if K > n_pre:
        raise ValueError(
            f"K={K} exceeds n_pre={n_pre}; cannot sample K distinct presynaptic neurons."
        )
    if K < 0:
        raise ValueError(f"K={K} must be non-negative.")
    pre_idx = np.empty((n_post, K), dtype=np.int32)
    for i in range(n_post):
        pre_idx[i] = rng.choice(n_pre, size=K, replace=False)
    return Connectivity(pre_idx=jnp.asarray(pre_idx), K=int(K))


def gather_input(spikes_pre: jnp.ndarray, conn: Connectivity,
                 g_max: float) -> jnp.ndarray:
    """Compute the postsynaptic conductance pulse arriving this step.

    Parameters
    ----------
    spikes_pre : (N_pre,) bool/float
        Presynaptic spikes (1 if spiking this step, 0 otherwise).
    conn : Connectivity
        Pathway connectivity (post-major).
    g_max : float
        Per-synapse peak conductance (mS/cm^2).

    Returns
    -------
    (N_post,) float — total g_max-weighted spike count arriving at each
    postsynaptic neuron.
    """
    return g_max * jnp.sum(spikes_pre[conn.pre_idx].astype(spikes_pre.dtype), axis=1)
