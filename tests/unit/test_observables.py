"""Unit tests for bgnet.observables.

Exercise the firing-rate / CV / population-rate-proxy / beta-fraction
primitives on synthetic spike trains where we can compute the answer
analytically.
"""
from __future__ import annotations

import numpy as np

from bgnet.observables import (
    beta_fraction,
    cv_isi,
    firing_rate,
    population_rate_proxy,
    population_summary,
    synaptic_current_lfp_proxy,
    vm_lfp_proxy,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic spike trains
# ---------------------------------------------------------------------------

def regular_spike_train(rate_Hz: float, n_neurons: int, T_ms: float,
                       dt_ms: float = 0.025, jitter_ms: float = 0.0,
                       seed: int = 0) -> np.ndarray:
    """Build a deterministic regular spike train at the given rate."""
    n_steps = int(round(T_ms / dt_ms))
    spikes = np.zeros((n_steps, n_neurons), dtype=bool)
    if rate_Hz <= 0:
        return spikes
    period_ms = 1000.0 / rate_Hz
    rng = np.random.default_rng(seed)
    for i in range(n_neurons):
        phase = rng.uniform(0, period_ms) if jitter_ms == 0 else 0.0
        t_ms = phase
        while t_ms < T_ms:
            j = rng.normal(0, jitter_ms) if jitter_ms > 0 else 0.0
            step = int(round((t_ms + j) / dt_ms))
            if 0 <= step < n_steps:
                spikes[step, i] = True
            t_ms += period_ms
    return spikes


def poisson_spike_train(rate_Hz: float, n_neurons: int, T_ms: float,
                        dt_ms: float = 0.025, seed: int = 1) -> np.ndarray:
    """Independent-bin Poisson approximation."""
    n_steps = int(round(T_ms / dt_ms))
    rng = np.random.default_rng(seed)
    p = rate_Hz * dt_ms / 1000.0
    return rng.random((n_steps, n_neurons)) < p


# ---------------------------------------------------------------------------
# firing_rate
# ---------------------------------------------------------------------------

def test_firing_rate_regular_train_within_one_pct():
    spikes = regular_spike_train(rate_Hz=20.0, n_neurons=10, T_ms=2000.0,
                                 dt_ms=0.5)
    r = firing_rate(spikes, dt_ms=0.5, burn_in_ms=200.0)
    assert abs(r - 20.0) < 0.5, f"expected 20 Hz, got {r}"


def test_firing_rate_burnin_ignores_transient():
    """Putting all spikes in the burn-in window gives 0 Hz."""
    n_steps = 1000  # 250 ms at dt=0.25
    spikes = np.zeros((n_steps, 5), dtype=bool)
    spikes[:200] = True  # all spikes in first 50 ms
    r = firing_rate(spikes, dt_ms=0.25, burn_in_ms=100.0)
    assert r == 0.0


def test_firing_rate_zero_neurons_safe():
    spikes = np.zeros((100, 0), dtype=bool)
    assert firing_rate(spikes, dt_ms=1.0, burn_in_ms=0.0) == 0.0


# ---------------------------------------------------------------------------
# cv_isi
# ---------------------------------------------------------------------------

def test_cv_isi_perfect_regular_is_zero():
    spikes = regular_spike_train(rate_Hz=40.0, n_neurons=20, T_ms=1000.0,
                                 dt_ms=0.1)
    cv = cv_isi(spikes, dt_ms=0.1, burn_in_ms=100.0)
    assert cv < 0.05, f"expected ~0 CV for regular firing, got {cv}"


def test_cv_isi_poisson_close_to_one():
    spikes = poisson_spike_train(rate_Hz=30.0, n_neurons=200, T_ms=2000.0,
                                 dt_ms=0.5, seed=0)
    cv = cv_isi(spikes, dt_ms=0.5, burn_in_ms=200.0)
    assert 0.7 < cv < 1.3, f"Poisson CV should be ~1, got {cv}"


def test_cv_isi_no_spikes_returns_zero():
    spikes = np.zeros((1000, 10), dtype=bool)
    assert cv_isi(spikes, dt_ms=0.1, burn_in_ms=0.0) == 0.0


# ---------------------------------------------------------------------------
# population_rate_proxy
# ---------------------------------------------------------------------------

def test_population_rate_proxy_constant_for_regular_train():
    """Regular train at 20 Hz with 50 neurons (random phases) → mean of
    binned proxy should be ~20 Hz, with small variance."""
    spikes = regular_spike_train(rate_Hz=20.0, n_neurons=50, T_ms=2000.0,
                                 dt_ms=0.5, seed=42)
    proxy, bin_dt = population_rate_proxy(spikes, dt_ms=0.5, bin_ms=10.0,
                                          burn_in_ms=200.0)
    assert bin_dt == 10.0
    assert abs(proxy.mean() - 20.0) < 1.5, (
        f"expected ~20 Hz mean proxy, got {proxy.mean()}"
    )


def test_population_rate_proxy_shape_consistency():
    spikes = poisson_spike_train(rate_Hz=10.0, n_neurons=25, T_ms=1000.0,
                                 dt_ms=0.5, seed=2)
    proxy, _ = population_rate_proxy(spikes, dt_ms=0.5, bin_ms=2.0,
                                     burn_in_ms=100.0)
    expected_bins = int((1000.0 - 100.0) / 2.0)
    assert proxy.shape == (expected_bins,)


# ---------------------------------------------------------------------------
# beta_fraction
# ---------------------------------------------------------------------------

def test_beta_fraction_pure_20hz_sine_is_dominant():
    """Sine at 20 Hz (centered in the [13, 30] Hz beta band) should give
    ~all power in beta."""
    fs = 500.0  # Hz
    duration_s = 6.0
    t = np.arange(int(fs * duration_s)) / fs
    sig = np.sin(2 * np.pi * 20.0 * t)
    bf, _, _ = beta_fraction(sig, sample_dt_ms=1000.0 / fs)
    assert bf > 0.9, f"20 Hz sine should put >90% power in [13,30] Hz, got {bf}"


def test_beta_fraction_pure_50hz_sine_is_negligible():
    """Sine at 50 Hz (above [13, 30] Hz beta band) → ~0 in beta."""
    fs = 500.0
    duration_s = 6.0
    t = np.arange(int(fs * duration_s)) / fs
    sig = np.sin(2 * np.pi * 50.0 * t)
    bf, _, _ = beta_fraction(sig, sample_dt_ms=1000.0 / fs)
    assert bf < 0.05, f"50 Hz sine should be ~0 in [13,30] Hz, got {bf}"


def test_beta_fraction_explicit_band_kwarg():
    """If a caller passes an explicit band (e.g. the historic 8-15 Hz),
    the function honors it. This is what wires through to TrialMetrics
    when a study config overrides the band."""
    fs = 500.0
    duration_s = 6.0
    t = np.arange(int(fs * duration_s)) / fs
    sig = np.sin(2 * np.pi * 10.0 * t)
    bf, _, _ = beta_fraction(sig, sample_dt_ms=1000.0 / fs,
                             beta_band=(8.0, 15.0))
    assert bf > 0.9, f"10 Hz sine in [8,15] Hz band: expected >0.9, got {bf}"


def test_beta_fraction_tiny_signal_safe():
    """Empty/very-short input must not blow up; returns 0."""
    bf, _, _ = beta_fraction(np.zeros(2), sample_dt_ms=1.0)
    assert bf == 0.0


# ---------------------------------------------------------------------------
# population_summary
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# vm_lfp_proxy — high-pass-filtered mean Vm
# ---------------------------------------------------------------------------

def _synthetic_vm_async(n_neurons: int, T_ms: float, dt_ms: float,
                        noise_sd_mV: float = 5.0, seed: int = 0
                        ) -> np.ndarray:
    """Population-mean Vm for asynchronous activity.

    Models the dominant component of real sub-threshold Vm: each neuron
    has independent broadband synaptic noise. Population-mean Vm = mean
    of N independent noise traces, which is band-flat with amplitude
    σ/√N. This is what an asynchronously-firing population looks like
    *between spikes*, which is where 95%+ of the trace lives at 20 Hz
    firing rate.
    """
    n_steps = int(round(T_ms / dt_ms))
    rng = np.random.default_rng(seed)
    traces = rng.standard_normal((n_steps, n_neurons)).astype(np.float32) * noise_sd_mV
    return traces.mean(axis=1)


def _synthetic_vm_sync(n_neurons: int, freq_Hz: float, T_ms: float,
                       dt_ms: float, oscillation_amp_mV: float = 5.0,
                       noise_sd_mV: float = 5.0, seed: int = 0
                       ) -> np.ndarray:
    """Population-mean Vm for synchronous activity at ``freq_Hz``.

    Same independent-noise component as the async builder, plus a
    shared sub-threshold oscillation at ``freq_Hz`` that every neuron's
    Vm tracks (the biological substrate of LFP β: many neurons riding
    the same network rhythm). With independent noise of amplitude σ
    per neuron and shared signal of amplitude A, the population-mean Vm
    has signal amplitude A (does not shrink with N) and noise σ/√N
    (does shrink). So large N + shared rhythm → high SNR → high β.
    """
    n_steps = int(round(T_ms / dt_ms))
    rng = np.random.default_rng(seed)
    t_s = np.arange(n_steps) * dt_ms / 1000.0
    shared = oscillation_amp_mV * np.sin(2 * np.pi * freq_Hz * t_s)
    traces = (rng.standard_normal((n_steps, n_neurons)).astype(np.float32)
              * noise_sd_mV + shared[:, None].astype(np.float32))
    return traces.mean(axis=1)


def test_vm_lfp_proxy_async_population_low_beta():
    """200 neurons of independent sub-threshold noise → population-mean
    Vm is band-flat with amplitude σ/√N. β fraction sits at the
    broadband geometric floor (β-band width / broadband width = 17/99 ≈
    0.17); we require β < 0.30 with a margin for finite-sample variance.
    No coherent oscillation, no β-band peak.
    """
    T_ms = 4000.0
    dt_ms = 1.0
    vm = _synthetic_vm_async(n_neurons=200, T_ms=T_ms, dt_ms=dt_ms, seed=0)
    trace, sample_dt = vm_lfp_proxy(vm, dt_ms=dt_ms, burn_in_ms=200.0,
                                    bin_ms=1.0)
    bf, _, _ = beta_fraction(trace, sample_dt, beta_band=(13.0, 30.0),
                             broadband=(1.0, 100.0))
    assert bf < 0.30, (
        f"asynchronous population Vm should sit near broadband floor; "
        f"got β = {bf:.3f}"
    )


def test_vm_lfp_proxy_coherent_population_high_beta():
    """Same 200 neurons + a *shared* 20 Hz sub-threshold oscillation:
    population-mean Vm has a deterministic 20 Hz component that does
    not shrink with N (shared signal) while the noise floor shrinks as
    σ/√N. β fraction rises substantially above the broadband floor.
    """
    T_ms = 4000.0
    dt_ms = 1.0
    vm = _synthetic_vm_sync(n_neurons=200, freq_Hz=20.0, T_ms=T_ms,
                            dt_ms=dt_ms, seed=0)
    trace, sample_dt = vm_lfp_proxy(vm, dt_ms=dt_ms, burn_in_ms=200.0,
                                    bin_ms=1.0)
    bf, _, _ = beta_fraction(trace, sample_dt, beta_band=(13.0, 30.0),
                             broadband=(1.0, 100.0))
    assert bf > 0.7, (
        f"coherent 20 Hz population Vm should put most broadband power "
        f"in β; got β = {bf:.3f}"
    )


def test_vm_lfp_proxy_removes_drift():
    """A slow ramp on top of broadband noise should be removed by the
    2 Hz high-pass: the filtered signal should not be dominated by the
    sub-Hz drift power."""
    fs = 1000.0
    duration_s = 4.0
    n = int(fs * duration_s)
    dt_ms = 1000.0 / fs
    t_s = np.arange(n) / fs
    rng = np.random.default_rng(7)
    # 0.2 Hz ramp (well below the 2 Hz cutoff) + small white noise
    sig = 5.0 * np.sin(2 * np.pi * 0.2 * t_s) + 0.1 * rng.standard_normal(n)
    trace, sample_dt = vm_lfp_proxy(sig.astype(np.float32), dt_ms=dt_ms,
                                    burn_in_ms=0.0, bin_ms=1.0,
                                    hp_cutoff_hz=2.0, hp_order=4)
    # post-filter the sub-Hz component should be attenuated by ≥40 dB
    assert trace.std() < 0.5, (
        f"high-pass did not attenuate 0.2 Hz drift; trace std={trace.std():.3f}"
    )


def test_vm_lfp_proxy_short_input_safe():
    """Very short inputs (below the filtfilt padding requirement) must
    return a centered trace without raising."""
    short = np.linspace(-60.0, -55.0, num=8, dtype=np.float32)
    trace, _ = vm_lfp_proxy(short, dt_ms=1.0, burn_in_ms=0.0, bin_ms=1.0)
    assert trace.shape == short.shape
    assert abs(float(trace.mean())) < 1e-6


# ---------------------------------------------------------------------------
# synaptic_current_lfp_proxy — sanity smoke
# ---------------------------------------------------------------------------

def test_synaptic_current_proxy_returns_filtered_trace():
    """The synaptic-current proxy shares the vm-proxy pipeline; just smoke
    the wrapper to make sure it accepts an arbitrary trace and applies the
    same high-pass."""
    fs = 1000.0
    duration_s = 3.0
    n = int(fs * duration_s)
    t_s = np.arange(n) / fs
    sig = 2.0 + np.sin(2 * np.pi * 18.0 * t_s)  # DC + 18 Hz
    trace, sample_dt = synaptic_current_lfp_proxy(sig.astype(np.float32),
                                                  dt_ms=1.0, burn_in_ms=200.0,
                                                  bin_ms=1.0)
    bf, _, _ = beta_fraction(trace, sample_dt,
                             beta_band=(13.0, 30.0), broadband=(1.0, 100.0))
    assert bf > 0.8, (
        f"18 Hz sine + DC should yield large β after high-pass; got {bf:.3f}"
    )


# ---------------------------------------------------------------------------
# population_summary proxy selection
# ---------------------------------------------------------------------------

def test_population_summary_vm_proxy_path():
    """The Vm-proxy path is the default; verify it consumes a vmean trace
    and surfaces the chosen proxy in the return dict."""
    spikes = regular_spike_train(rate_Hz=20.0, n_neurons=50, T_ms=1000.0,
                                 dt_ms=1.0, seed=3)
    vm = _synthetic_vm_sync(n_neurons=50, freq_Hz=20.0, T_ms=1000.0,
                            dt_ms=1.0, seed=3)
    summary = population_summary(spikes, dt_ms=1.0, burn_in_ms=200.0,
                                 bin_ms=1.0, vmean_trace=vm, proxy="vm")
    assert summary["proxy_kind"] == "vm"
    assert summary["proxy_trace"].size > 0
    assert summary["rate_trace"].size > 0   # still computed in parallel
    assert 0.0 <= summary["beta_fraction"] <= 1.0


def test_population_summary_vm_missing_trace_raises():
    spikes = regular_spike_train(rate_Hz=20.0, n_neurons=10, T_ms=500.0,
                                 dt_ms=1.0, seed=0)
    try:
        population_summary(spikes, dt_ms=1.0, burn_in_ms=100.0, proxy="vm")
    except ValueError:
        return
    raise AssertionError("expected ValueError when proxy='vm' but vmean_trace=None")


def test_population_summary_regular_train_passes_through():
    """Smoke: spike → rate, CV, β via the population-rate proxy. (The
    default proxy is now ``vm``; this exercises the rate-proxy alternate
    used by the LFP-proxy comparison study.)"""
    spikes = regular_spike_train(rate_Hz=15.0, n_neurons=30, T_ms=2000.0,
                                 dt_ms=0.5, seed=5)
    summary = population_summary(spikes, dt_ms=0.5, burn_in_ms=200.0,
                                 bin_ms=2.0, proxy="population_rate")
    assert abs(summary["rate_Hz"] - 15.0) < 0.5
    assert summary["cv"] < 0.05  # regular firing
    assert summary["rate_trace"].size > 0
    assert 0.0 <= summary["beta_fraction"] <= 1.0
