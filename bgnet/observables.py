"""Spike-train and LFP-proxy observables for the BG network.

Functions here turn the integrator's output (per-step spike booleans,
voltage trajectories, synaptic-current trajectories) into the scalar
metrics the optimizer's objective and constraints consume.

Conventions
-----------
- Spike arrays are shape ``(n_steps, n_neurons)`` and dtype bool
  (``True`` at the step a neuron fired).
- ``dt_ms`` is the simulation timestep in ms.
- ``burn_in_ms`` discards the leading transient; everything else is the
  analysis window. Default is 100 ms (matches the optimization burn-in
  in AGENTS.md §4.7).
- LFP proxies are time series sampled on the simulation grid (or on a
  coarser bin grid for the firing-rate proxy).
- Beta fraction follows AGENTS.md §4.3: power in [13, 30] Hz divided by
  power in [1, 100] Hz, after DC removal, via Welch's method.

Why these primitives
--------------------
Three LFP proxies are computed (population firing rate, mean Vm,
synaptic-current sum) so a downstream "LFP proxy comparison" study
(scripts/11) can show that the beta-fraction conclusions are robust to
proxy choice. Population firing rate is the *primary* proxy used by
the optimizer's constraint, per `docs/rebuild_scope.md` §2.2.9.

Citations:
    # Welch 1967, IEEE Trans Audio Electroacoust 15:70-73
    # Mallet et al. 2008, J Neurosci 28(18):4795-4806 (population rate as LFP proxy)
"""
from __future__ import annotations

import numpy as np
from scipy import signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _post_burn_window(spikes: np.ndarray, dt_ms: float, burn_in_ms: float
                      ) -> tuple[np.ndarray, float]:
    """Slice out the post-burn-in part of a spike trajectory.

    Returns ``(spikes_post, duration_s)`` where ``duration_s`` is the
    analysed wall-clock duration. Accepts both bool and float arrays;
    callers downstream convert as needed.
    """
    spikes = np.asarray(spikes)
    burn_steps = int(round(burn_in_ms / dt_ms))
    spikes_post = spikes[burn_steps:]
    duration_s = spikes_post.shape[0] * dt_ms / 1000.0
    return spikes_post, duration_s


# ---------------------------------------------------------------------------
# Firing rate
# ---------------------------------------------------------------------------

def firing_rate(spikes: np.ndarray, dt_ms: float, burn_in_ms: float = 100.0
                ) -> float:
    """Mean per-neuron firing rate (Hz) over the post-burn-in window."""
    spikes_post, duration_s = _post_burn_window(spikes, dt_ms, burn_in_ms)
    n_neurons = spikes_post.shape[1]
    if duration_s <= 0 or n_neurons == 0:
        return 0.0
    return float(spikes_post.sum()) / n_neurons / duration_s


# ---------------------------------------------------------------------------
# Coefficient of variation of ISIs
# ---------------------------------------------------------------------------

def cv_isi(spikes: np.ndarray, dt_ms: float, burn_in_ms: float = 100.0,
           min_isis: int = 2) -> float:
    """Population-mean coefficient of variation of inter-spike intervals.

    For each neuron with at least ``min_isis`` ISIs in the post-burn-in
    window, compute std(ISI)/mean(ISI). Average across qualifying neurons.
    Returns 0.0 if no neuron has enough ISIs (caller handles).
    """
    spikes_post, _ = _post_burn_window(spikes, dt_ms, burn_in_ms)
    n_steps, n_neurons = spikes_post.shape
    cvs = []
    for i in range(n_neurons):
        idx = np.flatnonzero(spikes_post[:, i])
        if idx.size < min_isis + 1:
            continue
        isis = np.diff(idx) * dt_ms
        mean_isi = isis.mean()
        if mean_isi <= 0:
            continue
        cvs.append(isis.std() / mean_isi)
    if not cvs:
        return 0.0
    return float(np.mean(cvs))


# ---------------------------------------------------------------------------
# Population-rate LFP proxy (primary)
# ---------------------------------------------------------------------------

def population_rate_proxy(spikes: np.ndarray, dt_ms: float,
                          bin_ms: float = 1.0,
                          burn_in_ms: float = 100.0) -> tuple[np.ndarray, float]:
    """Binned population firing rate (Hz). Primary LFP proxy.

    Returns ``(rate_trace, bin_dt_ms)`` where ``rate_trace`` has shape
    ``(n_bins,)`` and gives the per-bin mean spike count per neuron,
    converted to Hz. ``bin_dt_ms`` is the bin width in ms (which is
    also the effective sampling interval of the proxy time-series).
    """
    spikes_post, _ = _post_burn_window(spikes, dt_ms, burn_in_ms)
    n_steps, n_neurons = spikes_post.shape
    if n_steps == 0 or n_neurons == 0:
        return np.zeros(0), bin_ms
    steps_per_bin = max(1, int(round(bin_ms / dt_ms)))
    n_bins = n_steps // steps_per_bin
    if n_bins == 0:
        return np.zeros(0), bin_ms
    trimmed = spikes_post[:n_bins * steps_per_bin]
    counts = trimmed.reshape(n_bins, steps_per_bin, n_neurons).sum(axis=(1, 2))
    bin_s = steps_per_bin * dt_ms / 1000.0
    rate_trace = counts.astype(np.float64) / n_neurons / bin_s
    return rate_trace, steps_per_bin * dt_ms


# ---------------------------------------------------------------------------
# Beta fraction via Welch
# ---------------------------------------------------------------------------

def beta_fraction(trace: np.ndarray, sample_dt_ms: float,
                  beta_band: tuple[float, float] = (13.0, 30.0),
                  broadband: tuple[float, float] = (1.0, 100.0)
                  ) -> tuple[float, np.ndarray, np.ndarray]:
    """Fraction of power in the beta band relative to broadband.

    Computes the PSD of ``trace`` via Welch's method (Hann window, 50%
    overlap, segment length ``min(N, 8192)``), removes DC by selecting
    only frequencies in ``broadband``, integrates power in ``beta_band``
    and divides by the broadband total.

    Returns ``(fraction, freqs_Hz, psd)`` so callers can plot the PSD
    if they want.

    Sampling rate: ``fs = 1000 / sample_dt_ms`` Hz.
    """
    trace = np.asarray(trace, dtype=np.float64)
    n = trace.shape[0]
    if n < 4:
        return 0.0, np.zeros(0), np.zeros(0)
    fs = 1000.0 / sample_dt_ms
    nperseg = min(n, 8192)
    freqs, psd = signal.welch(trace, fs=fs, window="hann",
                              nperseg=nperseg,
                              noverlap=nperseg // 2,
                              detrend="constant",
                              scaling="density")
    bb_lo, bb_hi = broadband
    b_lo, b_hi = beta_band
    bb_mask = (freqs >= bb_lo) & (freqs <= bb_hi)
    beta_mask = (freqs >= b_lo) & (freqs <= b_hi)
    bb_total = float(np.trapezoid(psd[bb_mask], freqs[bb_mask])) if bb_mask.any() else 0.0
    beta_total = float(np.trapezoid(psd[beta_mask], freqs[beta_mask])) if beta_mask.any() else 0.0
    if bb_total <= 0:
        return 0.0, freqs, psd
    return beta_total / bb_total, freqs, psd


# ---------------------------------------------------------------------------
# Convenience: per-population summary
# ---------------------------------------------------------------------------

def population_summary(spikes: np.ndarray, dt_ms: float,
                       burn_in_ms: float = 100.0,
                       bin_ms: float = 1.0,
                       beta_band: tuple[float, float] = (13.0, 30.0),
                       broadband: tuple[float, float] = (1.0, 100.0)
                       ) -> dict:
    """Compute rate, CV, population-rate proxy, and beta fraction in one pass."""
    rate = firing_rate(spikes, dt_ms, burn_in_ms)
    cv = cv_isi(spikes, dt_ms, burn_in_ms)
    proxy, bin_dt = population_rate_proxy(spikes, dt_ms, bin_ms, burn_in_ms)
    beta, freqs, psd = beta_fraction(proxy, bin_dt, beta_band, broadband)
    return {
        "rate_Hz": rate,
        "cv": cv,
        "beta_fraction": beta,
        "rate_trace": proxy,
        "trace_dt_ms": bin_dt,
        "psd_freqs": freqs,
        "psd": psd,
    }
