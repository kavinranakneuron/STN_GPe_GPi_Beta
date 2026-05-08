"""Forward-Euler integrator for the STN-GPe-GPi network.

A single jax.lax.scan loop advances all three populations and all four
synaptic pathways through one full simulation. The function is JIT'd and
returns per-population spike arrays of shape (N_steps, N).

NO scaling factors. The membrane equation for every neuron is

    C_m dV/dt = -I_ion - I_syn + I_drive + I_noise (+ I_app for pallidum)

with I_X = g_X * (V - E_X) for every channel and synapse (positive when
hyperpolarizing). I_drive is the optimizer's tonic perturbation in
uA/cm^2; I_noise is the OU output. AGENTS.md operating principle 1
forbids inserting any factor like 0.005 here to "make the math work" —
if currents look wrong, fix the unit convention in the source module.

Spike delays are implemented with one circular buffer per source
population (length = ceil(max_delay_ms / dt) steps). Each pathway reads
from the buffer at the offset corresponding to its delay.

Per-trial random keys:
    key_ou_stn, key_ou_gpe, key_ou_gpi advance once per step inside the scan.
    Connectivity is built outside the scan (numpy) and passed in.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from bgnet.connectivity import Connectivity, gather_input
from bgnet.neurons.pallidum import (PallidumParams, PallidumState,
                                    pallidum_step_vmap)
from bgnet.neurons.stn import STNParams, STNState, stn_step_vmap
from bgnet.noise import OUParams, ou_step
from bgnet.synapses import (SynapseParams, peak_normalization,
                            synapse_step, synaptic_current)


# ---------------------------------------------------------------------------
# Aggregated state
# ---------------------------------------------------------------------------

class SynapseState(NamedTuple):
    x: jnp.ndarray   # (N_post,) decay-time variable
    y: jnp.ndarray   # (N_post,) rise-time variable


class CircularBuffer(NamedTuple):
    """Recent presynaptic spikes for one source population.
    ``data`` has shape (D, N) and ``head`` points to the next slot to write.
    Spikes at time t are stored at index (head - 1) mod D after writing."""
    data: jnp.ndarray
    head: jnp.ndarray   # int32 scalar


def buffer_init(n_pre: int, D: int) -> CircularBuffer:
    return CircularBuffer(
        data=jnp.zeros((D, n_pre), dtype=jnp.float32),
        head=jnp.int32(0),
    )


def buffer_write(buf: CircularBuffer, spikes: jnp.ndarray) -> CircularBuffer:
    """Write the current step's spikes at position ``head``, then advance head.
    Uses dynamic_update_slice (compiles to an in-place scatter) instead of
    .at[].set, which we found ~4x slower in the integrator hot loop."""
    row = spikes.astype(buf.data.dtype).reshape((1, -1))
    new_data = jax.lax.dynamic_update_slice(buf.data, row, (buf.head, jnp.int32(0)))
    new_head = (buf.head + 1) % buf.data.shape[0]
    return CircularBuffer(data=new_data, head=new_head)


def buffer_read(buf: CircularBuffer, delay_steps: int) -> jnp.ndarray:
    """Read spikes that fired ``delay_steps`` steps ago (relative to the
    most recent write)."""
    D = buf.data.shape[0]
    n = buf.data.shape[1]
    idx = (buf.head - delay_steps) % D
    sl = jax.lax.dynamic_slice(buf.data, (idx, jnp.int32(0)), (1, n))
    return sl[0]


class NetworkState(NamedTuple):
    stn: STNState
    gpe: PallidumState
    gpi: PallidumState
    syn_stn_gpe: SynapseState  # AMPA, post = GPe
    syn_stn_gpi: SynapseState  # AMPA, post = GPi
    syn_gpe_stn: SynapseState  # GABA_A, post = STN
    syn_gpe_gpi: SynapseState  # GABA_A, post = GPi
    ou_stn: jnp.ndarray
    ou_gpe: jnp.ndarray
    ou_gpi: jnp.ndarray
    key_ou: jax.Array
    buf_stn_spikes: CircularBuffer
    buf_gpe_spikes: CircularBuffer


class StaticParams(NamedTuple):
    """Everything constant across the scan (parameters, connectivity, derived
    delay-step counts and synapse normalizations)."""
    stn_p: STNParams
    gpe_p: PallidumParams
    gpi_p: PallidumParams
    syn_stn_gpe_p: SynapseParams
    syn_stn_gpi_p: SynapseParams
    syn_gpe_stn_p: SynapseParams
    syn_gpe_gpi_p: SynapseParams
    ou_stn_p: OUParams
    ou_gpe_p: OUParams
    ou_gpi_p: OUParams
    g_stn_gpe: float
    g_stn_gpi: float
    g_gpe_stn: float
    g_gpe_gpi: float
    conn_stn_gpe: Connectivity
    conn_stn_gpi: Connectivity
    conn_gpe_stn: Connectivity
    conn_gpe_gpi: Connectivity
    norm_ampa: float
    norm_gaba: float
    delay_steps_stn_gpe: int
    delay_steps_stn_gpi: int
    delay_steps_gpe_stn: int
    delay_steps_gpe_gpi: int
    I_drive_stn: float
    I_drive_gpe: float
    I_drive_gpi: float
    dt_ms: float


# ---------------------------------------------------------------------------
# Single step
# ---------------------------------------------------------------------------

def _step(state: NetworkState, t_ms: float, sp: StaticParams) -> tuple[NetworkState, dict]:
    """Advance the network one timestep. Returns the new state and a dict
    of per-step outputs (spikes per population)."""
    dt = sp.dt_ms

    # 1. Compute synaptic currents at the current V using current synapse state.
    I_syn_stn = synaptic_current(state.syn_gpe_stn.x, state.syn_gpe_stn.y,
                                 state.stn.V, sp.syn_gpe_stn_p, sp.norm_gaba)
    I_syn_gpe = synaptic_current(state.syn_stn_gpe.x, state.syn_stn_gpe.y,
                                 state.gpe.V, sp.syn_stn_gpe_p, sp.norm_ampa)
    I_syn_gpi = (synaptic_current(state.syn_stn_gpi.x, state.syn_stn_gpi.y,
                                  state.gpi.V, sp.syn_stn_gpi_p, sp.norm_ampa)
                 + synaptic_current(state.syn_gpe_gpi.x, state.syn_gpe_gpi.y,
                                    state.gpi.V, sp.syn_gpe_gpi_p, sp.norm_gaba))

    # 2. Advance the OU processes.
    I_noise_stn, key1 = ou_step(state.ou_stn, state.key_ou, dt, sp.ou_stn_p)
    I_noise_gpe, key2 = ou_step(state.ou_gpe, key1, dt, sp.ou_gpe_p)
    I_noise_gpi, key3 = ou_step(state.ou_gpi, key2, dt, sp.ou_gpi_p)

    # 3. Step neurons. Drives broadcast to vector via jnp.full_like? — use scalars
    #    directly: stn_step_vmap broadcasts because in_axes for I_drive=0 expects
    #    a 1-D array. Build per-neuron drives:
    n_stn = state.stn.V.shape[0]
    n_gpe = state.gpe.V.shape[0]
    n_gpi = state.gpi.V.shape[0]
    I_drive_stn_vec = jnp.full((n_stn,), sp.I_drive_stn)
    I_drive_gpe_vec = jnp.full((n_gpe,), sp.I_drive_gpe)
    I_drive_gpi_vec = jnp.full((n_gpi,), sp.I_drive_gpi)

    new_stn, sp_stn = stn_step_vmap(state.stn, sp.stn_p, dt,
                                    I_drive_stn_vec, I_syn_stn, I_noise_stn, t_ms)
    new_gpe, sp_gpe = pallidum_step_vmap(state.gpe, sp.gpe_p, dt,
                                         I_drive_gpe_vec, I_syn_gpe, I_noise_gpe, t_ms)
    new_gpi, sp_gpi = pallidum_step_vmap(state.gpi, sp.gpi_p, dt,
                                         I_drive_gpi_vec, I_syn_gpi, I_noise_gpi, t_ms)

    # 4. Push current-step spikes into the per-source circular buffers.
    new_buf_stn = buffer_write(state.buf_stn_spikes, sp_stn)
    new_buf_gpe = buffer_write(state.buf_gpe_spikes, sp_gpe)

    # 5. Pull delayed spikes for each pathway and convert to conductance pulses.
    arrived_stn_to_gpe = buffer_read(new_buf_stn, sp.delay_steps_stn_gpe)
    arrived_stn_to_gpi = buffer_read(new_buf_stn, sp.delay_steps_stn_gpi)
    arrived_gpe_to_stn = buffer_read(new_buf_gpe, sp.delay_steps_gpe_stn)
    arrived_gpe_to_gpi = buffer_read(new_buf_gpe, sp.delay_steps_gpe_gpi)

    pulse_stn_gpe = gather_input(arrived_stn_to_gpe, sp.conn_stn_gpe, sp.g_stn_gpe)
    pulse_stn_gpi = gather_input(arrived_stn_to_gpi, sp.conn_stn_gpi, sp.g_stn_gpi)
    pulse_gpe_stn = gather_input(arrived_gpe_to_stn, sp.conn_gpe_stn, sp.g_gpe_stn)
    pulse_gpe_gpi = gather_input(arrived_gpe_to_gpi, sp.conn_gpe_gpi, sp.g_gpe_gpi)

    # 6. Advance synapse states.
    x1, y1 = synapse_step(state.syn_stn_gpe.x, state.syn_stn_gpe.y,
                          pulse_stn_gpe, dt, sp.syn_stn_gpe_p)
    x2, y2 = synapse_step(state.syn_stn_gpi.x, state.syn_stn_gpi.y,
                          pulse_stn_gpi, dt, sp.syn_stn_gpi_p)
    x3, y3 = synapse_step(state.syn_gpe_stn.x, state.syn_gpe_stn.y,
                          pulse_gpe_stn, dt, sp.syn_gpe_stn_p)
    x4, y4 = synapse_step(state.syn_gpe_gpi.x, state.syn_gpe_gpi.y,
                          pulse_gpe_gpi, dt, sp.syn_gpe_gpi_p)

    new_state = NetworkState(
        stn=new_stn, gpe=new_gpe, gpi=new_gpi,
        syn_stn_gpe=SynapseState(x=x1, y=y1),
        syn_stn_gpi=SynapseState(x=x2, y=y2),
        syn_gpe_stn=SynapseState(x=x3, y=y3),
        syn_gpe_gpi=SynapseState(x=x4, y=y4),
        ou_stn=I_noise_stn, ou_gpe=I_noise_gpe, ou_gpi=I_noise_gpi,
        key_ou=key3,
        buf_stn_spikes=new_buf_stn,
        buf_gpe_spikes=new_buf_gpe,
    )
    return new_state, {"sp_stn": sp_stn, "sp_gpe": sp_gpe, "sp_gpi": sp_gpi}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def make_run(n_steps: int):
    """Build and return a JIT'd function that runs ``n_steps`` of the network.

    The closure captures n_steps (static for compilation) but accepts the
    initial state and StaticParams as runtime arguments. Returns
    ``(final_state, spikes_dict)`` where spikes are stacked along the leading
    time axis.
    """

    def run(state0: NetworkState, sp: StaticParams) -> tuple[NetworkState, dict]:
        def body(carry, i):
            t_ms = i * sp.dt_ms
            new_carry, out = _step(carry, t_ms, sp)
            return new_carry, out
        final, outputs = jax.lax.scan(body, state0, jnp.arange(n_steps))
        return final, outputs

    return jax.jit(run)
