"""Per-neuron biophysical heterogeneity.

Identical-neuron populations synchronize too tightly: in the rebuild's
STN-GPe loop, even minimal STN drive locks the network into a ~20 Hz
rhythm regardless of whether the optimizer is targeting healthy or PD
behavior (see `docs/network_beta_sanity_check.md` and the Phase 3
Step 2-3 surfacing in conversation history). Per-neuron variability in
intrinsic conductances breaks the degeneracy and lets STN reach 20 Hz
without forcing the population to oscillate coherently — well-precedented
in the model lineage:

    Hahn & McIntyre 2010, J Comput Neurosci 28(3):425-441
    Kumaravelu et al. 2016, J Comput Neurosci 40(2):207-229

The heterogeneity here is applied multiplicatively to the three principal
intrinsic conductances (g_L, g_Na, g_K), with the same Gaussian SD
across populations. It is a fixed, build-time perturbation — sampled
once at network construction and held constant for the duration of a
simulation. It is not an optimization parameter.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class HetMul(NamedTuple):
    """Per-neuron multiplicative perturbations on intrinsic conductances.

    Each field is shape ``(n_neurons,)`` and floats around 1.0. When
    `heterogeneity_pct = 0.10`, each multiplier ~ N(1, 0.10).
    """
    mul_g_L: jnp.ndarray
    mul_g_Na: jnp.ndarray
    mul_g_K: jnp.ndarray


def homogeneous_het(n_neurons: int) -> HetMul:
    """All-1.0 multipliers — equivalent to no heterogeneity. Used by
    direct unit tests and by the network builder when
    ``heterogeneity_pct == 0``."""
    ones = jnp.ones((n_neurons,))
    return HetMul(mul_g_L=ones, mul_g_Na=ones, mul_g_K=ones)


def sampled_het(n_neurons: int, pct: float, key: jax.Array) -> HetMul:
    """Sample per-neuron multipliers ~ N(1, pct) for each of g_L, g_Na, g_K.

    Independent draws per channel and per neuron — three separate
    Gaussian samples per cell. Multipliers are clipped to [0.5, 1.5] to
    guarantee positive conductances even at extreme tail values; with
    pct = 0.10 a 5σ excursion would land at 1.5 anyway, so the clip is
    safety margin.
    """
    if pct <= 0.0:
        return homogeneous_het(n_neurons)
    k1, k2, k3 = jax.random.split(key, 3)
    return HetMul(
        mul_g_L=jnp.clip(1.0 + pct * jax.random.normal(k1, (n_neurons,)), 0.5, 1.5),
        mul_g_Na=jnp.clip(1.0 + pct * jax.random.normal(k2, (n_neurons,)), 0.5, 1.5),
        mul_g_K=jnp.clip(1.0 + pct * jax.random.normal(k3, (n_neurons,)), 0.5, 1.5),
    )
