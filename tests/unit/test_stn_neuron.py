"""Unit tests for the Terman-Rubin 2002 STN neuron model.

These tests exercise an isolated single STN cell (n_neurons=1, no synapses,
no other populations, no noise unless explicitly stated) at the canonical
``episodic.ode`` parameters. Numerical bounds come directly from
PHASE_1_5_INSTRUCTIONS.md §3.

References:
    Terman D, Rubin JE, Yew AC, Wilson CJ (2002). Activity patterns in a
    model for the subthalamopallidal network of the basal ganglia.
    J Neurosci 22(7):2963-2976.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from bgnet.neurons.stn import STNParams, initial_state, stn_step, stn_step_vmap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_isolated(I_drive: float, T_ms: float, dt_ms: float = 0.025,
                  burn_ms: float = 500.0) -> tuple[float, np.ndarray, np.ndarray]:
    """Run one isolated STN neuron with a constant drive.

    Returns ``(rate_Hz, V_trace, spike_trace)``. ``rate_Hz`` is computed over
    the post-burn window. Traces span the whole simulation."""
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
            return (new_state, sp), (new_state.V, sp)
        (final, _), (Vs, sps) = jax.lax.scan(body, (state0, jnp.array([False])),
                                             jnp.arange(n_steps))
        return Vs, sps

    Vs, sps = run()
    Vs = np.asarray(Vs).flatten()
    sps = np.asarray(sps).flatten()
    burn_steps = int(burn_ms / dt_ms)
    spikes_after = int(sps[burn_steps:].sum())
    rate = spikes_after / ((T_ms - burn_ms) / 1000.0)
    return rate, Vs, sps


def _run_step_drive(drive_schedule, dt_ms: float = 0.025
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one isolated STN neuron with a time-varying drive.

    ``drive_schedule`` is a callable ``t_ms -> I_drive`` that the JIT'd
    integrator will use to determine the drive at each step. Returns
    ``(V_trace, r_trace, spike_trace)``.
    """
    p = STNParams()
    state0 = initial_state(1)
    n_steps = drive_schedule.shape[0]

    @jax.jit
    def run():
        def body(carry, i):
            state, _ = carry
            t = i * dt_ms
            I = jnp.array([drive_schedule[i]])
            new_state, sp = stn_step(state, p, dt_ms, I,
                                     jnp.array([0.0]), jnp.array([0.0]), t)
            return (new_state, sp), (new_state.V, new_state.r, sp)
        (final, _), (Vs, rs, sps) = jax.lax.scan(body, (state0, jnp.array([False])),
                                                 jnp.arange(n_steps))
        return Vs, rs, sps

    Vs, rs, sps = run()
    return np.asarray(Vs).flatten(), np.asarray(rs).flatten(), np.asarray(sps).flatten()


# ---------------------------------------------------------------------------
# Test 1: spontaneous firing rate at zero drive
# ---------------------------------------------------------------------------

def test_stn_zero_drive_firing_rate():
    """Canonical RT 2002 STN fires ~10 Hz tonically with I_drive = 0.

    Allow 8-14 Hz to leave room for forward-Euler integration error.
    """
    rate, _, _ = _run_isolated(0.0, T_ms=1500.0)
    assert 8.0 <= rate <= 14.0, (
        f"expected 8-14 Hz tonic firing at I=0, got {rate} Hz"
    )


# ---------------------------------------------------------------------------
# Test 2: f-I monotonicity and finiteness
# ---------------------------------------------------------------------------

def test_stn_fi_monotonic_and_finite():
    """Sweep I_drive from -5 to 30 µA/cm². Voltages stay finite and bounded;
    firing rate is non-decreasing (with a 0.5 Hz tolerance for ISI quantization)."""
    drives = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    rates = []
    for d in drives:
        rate, V, _ = _run_isolated(d, T_ms=1500.0)
        assert np.all(np.isfinite(V)), f"non-finite voltage at I={d}"
        assert V.min() >= -100.0 and V.max() <= 60.0, (
            f"voltage out of clamp range at I={d}: [{V.min()}, {V.max()}]"
        )
        rates.append(rate)
    for i in range(1, len(rates)):
        assert rates[i] >= rates[i - 1] - 0.5, (
            f"f-I non-monotonic: I={drives[i - 1]}→{drives[i]}, "
            f"rate {rates[i - 1]}→{rates[i]}"
        )


# ---------------------------------------------------------------------------
# Test 3: post-inhibitory rebound burst
# ---------------------------------------------------------------------------

def test_stn_post_inhibitory_rebound():
    """Hold the cell hyperpolarized for 200 ms via a strong negative drive,
    then release. Expect at least 2 spikes in the next 100 ms (the T-current
    rebound burst). During the clamp, the de-inactivation gate r should
    reach >= 0.5."""
    dt_ms = 0.025
    n_clamp = int(200.0 / dt_ms)
    n_release = int(100.0 / dt_ms)
    drive = jnp.concatenate([
        jnp.full((n_clamp,), -30.0),
        jnp.full((n_release,), 0.0),
    ])
    V, r, sps = _run_step_drive(drive, dt_ms=dt_ms)

    assert r[:n_clamp].max() >= 0.5, (
        f"r failed to de-inactivate during hyperpolarization: max={r[:n_clamp].max()}"
    )
    spikes_after_release = int(sps[n_clamp:].sum())
    assert spikes_after_release >= 2, (
        f"post-inhibitory rebound produced {spikes_after_release} spikes, expected >=2"
    )


# ---------------------------------------------------------------------------
# Test 4: spike-frequency adaptation
# ---------------------------------------------------------------------------

def test_stn_spike_frequency_adaptation():
    """Apply a step drive of I=25 µA/cm² from t=100 ms onward (zero before).
    Compute ISIs in the suprathreshold window. The mean of the first 3 ISIs
    should be at least 10% shorter than the mean of the last 3 ISIs (calcium-
    AHP-mediated adaptation)."""
    dt_ms = 0.025
    n_pre = int(100.0 / dt_ms)
    n_post = int(500.0 / dt_ms)
    drive = jnp.concatenate([
        jnp.zeros((n_pre,)),
        jnp.full((n_post,), 25.0),
    ])
    V, r, sps = _run_step_drive(drive, dt_ms=dt_ms)
    times_ms = np.arange(drive.shape[0]) * dt_ms
    spike_times = times_ms[sps.astype(bool)]
    spike_times = spike_times[spike_times >= 100.0]
    assert len(spike_times) >= 6, (
        f"insufficient spikes ({len(spike_times)}) at I=25 to measure adaptation"
    )
    isis = np.diff(spike_times)
    early = isis[:3].mean()
    late = isis[-3:].mean()
    assert late > early * 1.10, (
        f"weak adaptation: early ISI {early:.2f} ms, late ISI {late:.2f} ms"
    )


# ---------------------------------------------------------------------------
# Test 5: dt sensitivity
# ---------------------------------------------------------------------------

def test_stn_dt_sensitivity():
    """Forward-Euler firing rate at I=10 µA/cm² should agree within 2 Hz
    between dt=0.025 and dt=0.0125. Noise is zero so the comparison is
    deterministic."""
    rate_coarse, _, _ = _run_isolated(10.0, T_ms=600.0, dt_ms=0.025, burn_ms=100.0)
    rate_fine, _, _ = _run_isolated(10.0, T_ms=600.0, dt_ms=0.0125, burn_ms=100.0)
    assert abs(rate_coarse - rate_fine) < 2.0, (
        f"dt sensitivity exceeds 2 Hz: dt=0.025→{rate_coarse}, dt=0.0125→{rate_fine}"
    )


# ---------------------------------------------------------------------------
# Hygiene: voltage and gating bounds
# ---------------------------------------------------------------------------

def test_stn_no_nan_and_bounded_under_load():
    """Even under heavy depolarizing drive over a population, voltage and
    gating variables remain bounded and free of NaN."""
    p = STNParams()
    state0 = initial_state(50, key=jax.random.PRNGKey(0))
    dt = 0.025
    n_steps = 8000  # 200 ms

    @jax.jit
    def run():
        def body(carry, i):
            state, _ = carry
            t = i * dt
            I = jnp.full((50,), 60.0)
            zero = jnp.zeros((50,))
            new_state, sp = stn_step_vmap(state, p, dt, I, zero, zero, t)
            return (new_state, sp), state.V
        (final, _), Vs = jax.lax.scan(body, (state0, jnp.zeros((50,), dtype=bool)),
                                      jnp.arange(n_steps))
        return final, Vs

    final, Vs = run()
    assert not bool(jnp.any(jnp.isnan(Vs))), "NaN voltages encountered"
    assert not bool(jnp.any(jnp.isnan(final.h))), "NaN h gate"
    assert not bool(jnp.any(jnp.isnan(final.n))), "NaN n gate"
    assert not bool(jnp.any(jnp.isnan(final.r))), "NaN r gate"
    assert not bool(jnp.any(jnp.isnan(final.Ca))), "NaN calcium"
    # Gating variables stay in [0, 1]; voltage in clamp range; Ca >= 0.
    for arr, name in [(final.h, "h"), (final.n, "n"), (final.r, "r")]:
        assert bool(jnp.all((arr >= 0.0) & (arr <= 1.0))), f"{name} out of [0,1]"
    assert bool(jnp.all(final.Ca >= 0.0)), "Ca went negative"
    assert bool(jnp.all((Vs >= -100.0) & (Vs <= 60.0))), "V out of clamp range"
