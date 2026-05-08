"""Network builder: from a NetworkConfig to a runnable simulation.

Stitches together the neuron, synapse, noise, connectivity, and integrator
modules. Connectivity is built once at network setup time using the given
network_seed; the OU PRNG key is derived from a separate ou_seed so trials
can re-use connectivity while varying noise.

Population size convention is STN:GPe:GPi = 2:4:3 (AGENTS.md section 4.5),
with optimization default 100/200/150 = 450 neurons and validation default
10000/20000/15000 = 45000.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from bgnet.connectivity import build_connectivity
from bgnet.integrator import (CircularBuffer, NetworkState, StaticParams,
                              SynapseState, buffer_init, make_run)
from bgnet.neurons.pallidum import (PallidumState, gpe_params, gpi_params,
                                    initial_state as pallidum_init)
from bgnet.neurons.stn import (STNParams, STNState,
                               initial_state as stn_init)
from bgnet.noise import OUParams, init_state as ou_init
from bgnet.synapses import (ampa_params, gaba_a_params, peak_normalization)


@dataclass
class NetworkConfig:
    """Everything needed to build and run one simulation."""

    # population sizes (STN:GPe:GPi = 2:4:3)
    n_stn: int = 100
    n_gpe: int = 200
    n_gpi: int = 150

    # fixed-indegrees per pathway (AGENTS.md 4.5)
    K_stn_gpe: int = 15
    K_gpe_stn: int = 14
    K_stn_gpi: int = 30
    K_gpe_gpi: int = 10

    # synaptic delays in ms (AGENTS.md 4.6)
    delay_stn_gpe_ms: float = 5.0
    delay_stn_gpi_ms: float = 5.0
    delay_gpe_stn_ms: float = 8.0
    delay_gpe_gpi_ms: float = 5.0

    # numerical
    dt_ms: float = 0.025

    # trial parameters (defaults are unit values; the optimizer overrides these)
    g_stn_gpe: float = 0.05    # mS/cm^2
    g_stn_gpi: float = 0.05
    g_gpe_stn: float = 0.05
    g_gpe_gpi: float = 0.05
    I_drive_stn: float = 0.0   # uA/cm^2
    I_drive_gpe: float = 0.0
    I_drive_gpi: float = 0.0
    mu_stn: float = 0.0
    mu_gpe: float = 0.0
    mu_gpi: float = 0.0
    sigma_stn: float = 1.0
    sigma_gpe: float = 1.0
    sigma_gpi: float = 1.0

    # seeds
    network_seed: int = 42
    ou_seed: int = 1
    init_seed: int = 7


def _delay_steps(delay_ms: float, dt_ms: float) -> int:
    return max(1, int(math.ceil(delay_ms / dt_ms)))


def build(cfg: NetworkConfig) -> tuple[NetworkState, StaticParams]:
    """Build initial state and StaticParams. Connectivity is constructed
    here (numpy) and converted to jnp arrays."""
    # 1. Connectivity
    rng = np.random.default_rng(cfg.network_seed)
    conn_stn_gpe = build_connectivity(rng, cfg.n_stn, cfg.n_gpe, cfg.K_stn_gpe)
    conn_stn_gpi = build_connectivity(rng, cfg.n_stn, cfg.n_gpi, cfg.K_stn_gpi)
    conn_gpe_stn = build_connectivity(rng, cfg.n_gpe, cfg.n_stn, cfg.K_gpe_stn)
    conn_gpe_gpi = build_connectivity(rng, cfg.n_gpe, cfg.n_gpi, cfg.K_gpe_gpi)

    # 2. Synapse param objects (delay_ms is informational; integrator uses delay_steps)
    syn_stn_gpe_p = ampa_params(delay_ms=cfg.delay_stn_gpe_ms)
    syn_stn_gpi_p = ampa_params(delay_ms=cfg.delay_stn_gpi_ms)
    syn_gpe_stn_p = gaba_a_params(delay_ms=cfg.delay_gpe_stn_ms)
    syn_gpe_gpi_p = gaba_a_params(delay_ms=cfg.delay_gpe_gpi_ms)

    norm_ampa = peak_normalization(syn_stn_gpe_p.tau_rise_ms, syn_stn_gpe_p.tau_decay_ms)
    norm_gaba = peak_normalization(syn_gpe_stn_p.tau_rise_ms, syn_gpe_stn_p.tau_decay_ms)

    # 3. OU params per population
    ou_stn_p = OUParams(mu=cfg.mu_stn, sigma=cfg.sigma_stn)
    ou_gpe_p = OUParams(mu=cfg.mu_gpe, sigma=cfg.sigma_gpe)
    ou_gpi_p = OUParams(mu=cfg.mu_gpi, sigma=cfg.sigma_gpi)

    # 4. Cell-type params
    stn_p = STNParams()
    gpe_p = gpe_params()
    gpi_p = gpi_params()

    # 5. Delay steps and buffer sizes
    dsteps_stn_gpe = _delay_steps(cfg.delay_stn_gpe_ms, cfg.dt_ms)
    dsteps_stn_gpi = _delay_steps(cfg.delay_stn_gpi_ms, cfg.dt_ms)
    dsteps_gpe_stn = _delay_steps(cfg.delay_gpe_stn_ms, cfg.dt_ms)
    dsteps_gpe_gpi = _delay_steps(cfg.delay_gpe_gpi_ms, cfg.dt_ms)
    buf_stn_size = max(dsteps_stn_gpe, dsteps_stn_gpi) + 1
    buf_gpe_size = max(dsteps_gpe_stn, dsteps_gpe_gpi) + 1

    sp = StaticParams(
        stn_p=stn_p, gpe_p=gpe_p, gpi_p=gpi_p,
        syn_stn_gpe_p=syn_stn_gpe_p, syn_stn_gpi_p=syn_stn_gpi_p,
        syn_gpe_stn_p=syn_gpe_stn_p, syn_gpe_gpi_p=syn_gpe_gpi_p,
        ou_stn_p=ou_stn_p, ou_gpe_p=ou_gpe_p, ou_gpi_p=ou_gpi_p,
        g_stn_gpe=cfg.g_stn_gpe, g_stn_gpi=cfg.g_stn_gpi,
        g_gpe_stn=cfg.g_gpe_stn, g_gpe_gpi=cfg.g_gpe_gpi,
        conn_stn_gpe=conn_stn_gpe, conn_stn_gpi=conn_stn_gpi,
        conn_gpe_stn=conn_gpe_stn, conn_gpe_gpi=conn_gpe_gpi,
        norm_ampa=norm_ampa, norm_gaba=norm_gaba,
        delay_steps_stn_gpe=dsteps_stn_gpe,
        delay_steps_stn_gpi=dsteps_stn_gpi,
        delay_steps_gpe_stn=dsteps_gpe_stn,
        delay_steps_gpe_gpi=dsteps_gpe_gpi,
        I_drive_stn=cfg.I_drive_stn, I_drive_gpe=cfg.I_drive_gpe, I_drive_gpi=cfg.I_drive_gpi,
        dt_ms=cfg.dt_ms,
    )

    # 6. Initial state with heterogeneity
    key_init = jax.random.PRNGKey(cfg.init_seed)
    k1, k2, k3, k_ou = jax.random.split(key_init, 4)
    stn_state = stn_init(cfg.n_stn, key=k1)
    gpe_state = pallidum_init(cfg.n_gpe, key=k2)
    gpi_state = pallidum_init(cfg.n_gpi, key=k3)

    state0 = NetworkState(
        stn=stn_state, gpe=gpe_state, gpi=gpi_state,
        syn_stn_gpe=SynapseState(jnp.zeros((cfg.n_gpe,)), jnp.zeros((cfg.n_gpe,))),
        syn_stn_gpi=SynapseState(jnp.zeros((cfg.n_gpi,)), jnp.zeros((cfg.n_gpi,))),
        syn_gpe_stn=SynapseState(jnp.zeros((cfg.n_stn,)), jnp.zeros((cfg.n_stn,))),
        syn_gpe_gpi=SynapseState(jnp.zeros((cfg.n_gpi,)), jnp.zeros((cfg.n_gpi,))),
        ou_stn=ou_init(cfg.n_stn, ou_stn_p),
        ou_gpe=ou_init(cfg.n_gpe, ou_gpe_p),
        ou_gpi=ou_init(cfg.n_gpi, ou_gpi_p),
        key_ou=jax.random.PRNGKey(cfg.ou_seed),
        buf_stn_spikes=buffer_init(cfg.n_stn, buf_stn_size),
        buf_gpe_spikes=buffer_init(cfg.n_gpe, buf_gpe_size),
    )
    return state0, sp


_RUN_CACHE: dict[int, Callable] = {}


def get_run(n_steps: int) -> Callable:
    """Return a cached JIT'd run function for a given step count, building it
    on first request. Reusing the same run function across trials keeps
    JAX's tracing/compilation amortized: a fresh jax.jit wrapper would
    re-trace per call (~3 s overhead for our network) even when the XLA
    binary is cache-hit."""
    fn = _RUN_CACHE.get(n_steps)
    if fn is None:
        fn = make_run(n_steps)
        _RUN_CACHE[n_steps] = fn
    return fn


def simulate(cfg: NetworkConfig, duration_ms: float) -> dict:
    """Full convenience driver: build the network, run for ``duration_ms``,
    return a dict with the per-population spike arrays and the final state.
    n_steps is computed from duration / dt and used as a JIT-static constant.
    """
    state0, sp = build(cfg)
    n_steps = int(round(duration_ms / cfg.dt_ms))
    run = get_run(n_steps)
    final, outputs = run(state0, sp)
    return {
        "spikes_stn": outputs["sp_stn"],   # (n_steps, n_stn) bool
        "spikes_gpe": outputs["sp_gpe"],
        "spikes_gpi": outputs["sp_gpi"],
        "final_state": final,
        "n_steps": n_steps,
        "dt_ms": cfg.dt_ms,
    }
