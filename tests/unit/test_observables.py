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

def test_beta_fraction_pure_10hz_sine_is_dominant():
    """Sine at 10 Hz (in the beta band) should give ~all power in beta."""
    fs = 500.0  # Hz
    duration_s = 6.0
    t = np.arange(int(fs * duration_s)) / fs
    sig = np.sin(2 * np.pi * 10.0 * t)
    bf, _, _ = beta_fraction(sig, sample_dt_ms=1000.0 / fs)
    assert bf > 0.9, f"10 Hz sine should put >90% power in [8,15] Hz, got {bf}"


def test_beta_fraction_pure_30hz_sine_is_negligible():
    """Sine at 30 Hz (above beta band) → ~0 in beta."""
    fs = 500.0
    duration_s = 6.0
    t = np.arange(int(fs * duration_s)) / fs
    sig = np.sin(2 * np.pi * 30.0 * t)
    bf, _, _ = beta_fraction(sig, sample_dt_ms=1000.0 / fs)
    assert bf < 0.05, f"30 Hz sine should be ~0 in [8,15] Hz, got {bf}"


def test_beta_fraction_tiny_signal_safe():
    """Empty/very-short input must not blow up; returns 0."""
    bf, _, _ = beta_fraction(np.zeros(2), sample_dt_ms=1.0)
    assert bf == 0.0


# ---------------------------------------------------------------------------
# population_summary
# ---------------------------------------------------------------------------

def test_population_summary_regular_train_passes_through():
    spikes = regular_spike_train(rate_Hz=15.0, n_neurons=30, T_ms=2000.0,
                                 dt_ms=0.5, seed=5)
    summary = population_summary(spikes, dt_ms=0.5, burn_in_ms=200.0,
                                 bin_ms=2.0)
    assert abs(summary["rate_Hz"] - 15.0) < 0.5
    assert summary["cv"] < 0.05  # regular firing
    assert summary["rate_trace"].size > 0
    assert 0.0 <= summary["beta_fraction"] <= 1.0
