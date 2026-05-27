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
  coarser bin grid for the rate and Vm proxies).
- Beta fraction follows AGENTS.md §4.3: power in [13, 30] Hz divided by
  power in [1, 100] Hz, after DC removal, via Welch's method.

Three LFP proxies are exposed:
- ``vm_lfp_proxy`` (primary) — high-pass-filtered per-step mean Vm.
  Captures sub-threshold synchrony; asynchronous firing at a given rate
  has low β fraction even if that rate falls inside the β band.
  See ``docs/silent_stn_diagnostics.md`` for why population-rate was
  abandoned as primary.
- ``population_rate_proxy`` (alternate) — population spike rate. Kept for
  the LFP-proxy-comparison validation study.
- ``synaptic_current_lfp_proxy`` (alternate) — high-pass-filtered mean
  synaptic current. Kept as a second alternate.

Citations:
    # Welch 1967, IEEE Trans Audio Electroacoust 15:70-73
    # Mallet et al. 2008, J Neurosci 28(18):4795-4806 (mean Vm as LFP proxy)
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
# Mean Vm LFP proxy (primary) — high-pass filtered, 1 ms bins
# ---------------------------------------------------------------------------

def _bin_average_1d(trace: np.ndarray, dt_ms: float, bin_ms: float
                    ) -> tuple[np.ndarray, float]:
    """Average a per-step 1-D trace down into ``bin_ms``-wide bins.

    Returns ``(binned, bin_dt_ms)``. If the input is shorter than one
    bin, returns an empty array and the requested bin width.
    """
    steps_per_bin = max(1, int(round(bin_ms / dt_ms)))
    n_bins = trace.shape[0] // steps_per_bin
    if n_bins == 0:
        return np.zeros(0), steps_per_bin * dt_ms
    trimmed = trace[:n_bins * steps_per_bin]
    binned = trimmed.reshape(n_bins, steps_per_bin).mean(axis=1)
    return binned.astype(np.float64), steps_per_bin * dt_ms


def _highpass(trace: np.ndarray, sample_dt_ms: float,
              cutoff_hz: float, order: int) -> np.ndarray:
    """Zero-phase Butterworth high-pass via SOS + filtfilt.

    Removes the DC offset (subtract mean) before filtering to keep the
    filter in a well-conditioned regime. Returns the input unchanged if
    the trace is too short for filtfilt's padding requirements.
    """
    centered = trace - trace.mean()
    fs = 1000.0 / sample_dt_ms
    nyq = 0.5 * fs
    Wn = cutoff_hz / nyq
    if not (0.0 < Wn < 1.0):
        # Degenerate cutoff (e.g. > Nyquist or <= 0): return centered trace.
        return centered
    sos = signal.butter(order, Wn, btype="highpass", output="sos")
    # filtfilt padlen is 3 * len(sos[0]) per section by default; require a
    # safety margin so very short windows don't trip a ValueError.
    min_len = 3 * (2 * order) + 1
    if centered.shape[0] < min_len:
        return centered
    return signal.sosfiltfilt(sos, centered)


def vm_lfp_proxy(v_mean_trace: np.ndarray, dt_ms: float,
                 burn_in_ms: float = 100.0, bin_ms: float = 1.0,
                 hp_cutoff_hz: float = 2.0, hp_order: int = 4
                 ) -> tuple[np.ndarray, float]:
    """High-pass-filtered per-step mean Vm — primary LFP proxy.

    The integrator emits ``mean(V_pop)`` per step (a scalar). This function
    discards the burn-in transient, averages into ``bin_ms`` bins (default
    1 ms, matching the rate proxy), and applies a zero-phase Butterworth
    high-pass at ``hp_cutoff_hz`` (default 2 Hz, order 4) to remove drift.
    Returns ``(filtered_trace, sample_dt_ms)``.

    Why filter? Mean Vm drifts (sub-threshold network state varies on
    seconds timescale). Without high-pass the PSD has large DC and
    sub-1 Hz mass that contaminates the [1, 100] Hz broadband
    denominator of the β fraction. A 2 Hz cutoff is below the lower
    edge of any biologically meaningful band, including the lowest β
    edge at 13 Hz.
    """
    trace = np.asarray(v_mean_trace, dtype=np.float64)
    burn_steps = int(round(burn_in_ms / dt_ms))
    trace_post = trace[burn_steps:]
    if trace_post.size == 0:
        return np.zeros(0), bin_ms
    binned, sample_dt_ms = _bin_average_1d(trace_post, dt_ms, bin_ms)
    if binned.size == 0:
        return binned, sample_dt_ms
    filtered = _highpass(binned, sample_dt_ms, hp_cutoff_hz, hp_order)
    return filtered, sample_dt_ms


# ---------------------------------------------------------------------------
# Synaptic-current LFP proxy (alternate)
# ---------------------------------------------------------------------------

def synaptic_current_lfp_proxy(isyn_mean_trace: np.ndarray, dt_ms: float,
                               burn_in_ms: float = 100.0, bin_ms: float = 1.0,
                               hp_cutoff_hz: float = 2.0, hp_order: int = 4
                               ) -> tuple[np.ndarray, float]:
    """High-pass-filtered per-step mean synaptic current — alternate proxy.

    Same pipeline as ``vm_lfp_proxy`` but on the synaptic current the
    integrator records per step. Kept as a fallback for the LFP-proxy
    comparison study (see ``docs/silent_stn_diagnostics.md``).
    """
    trace = np.asarray(isyn_mean_trace, dtype=np.float64)
    burn_steps = int(round(burn_in_ms / dt_ms))
    trace_post = trace[burn_steps:]
    if trace_post.size == 0:
        return np.zeros(0), bin_ms
    binned, sample_dt_ms = _bin_average_1d(trace_post, dt_ms, bin_ms)
    if binned.size == 0:
        return binned, sample_dt_ms
    filtered = _highpass(binned, sample_dt_ms, hp_cutoff_hz, hp_order)
    return filtered, sample_dt_ms


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
                       broadband: tuple[float, float] = (1.0, 100.0),
                       vmean_trace: np.ndarray | None = None,
                       isyn_mean_trace: np.ndarray | None = None,
                       proxy: str = "vm",
                       hp_cutoff_hz: float = 2.0, hp_order: int = 4,
                       ) -> dict:
    """Compute rate, CV, the chosen LFP proxy, and β fraction in one pass.

    ``proxy`` selects which signal feeds the β-fraction computation:
    - ``"vm"`` (default) — uses ``vmean_trace``. Required when chosen.
    - ``"population_rate"`` — uses the spike-derived rate proxy.
    - ``"synaptic_current"`` — uses ``isyn_mean_trace``. Required when chosen.

    Diagnostics that want the rate trace alongside the chosen β can pass
    spikes and read ``rate_trace`` from the return regardless of which
    proxy fed the β fraction.
    """
    rate = firing_rate(spikes, dt_ms, burn_in_ms)
    cv = cv_isi(spikes, dt_ms, burn_in_ms)
    rate_trace, rate_bin_dt = population_rate_proxy(spikes, dt_ms, bin_ms, burn_in_ms)

    if proxy == "vm":
        if vmean_trace is None:
            raise ValueError("population_summary(proxy='vm') requires vmean_trace")
        trace, sample_dt = vm_lfp_proxy(vmean_trace, dt_ms, burn_in_ms,
                                        bin_ms, hp_cutoff_hz, hp_order)
    elif proxy == "population_rate":
        trace, sample_dt = rate_trace, rate_bin_dt
    elif proxy == "synaptic_current":
        if isyn_mean_trace is None:
            raise ValueError(
                "population_summary(proxy='synaptic_current') requires isyn_mean_trace")
        trace, sample_dt = synaptic_current_lfp_proxy(
            isyn_mean_trace, dt_ms, burn_in_ms, bin_ms,
            hp_cutoff_hz, hp_order)
    else:
        raise ValueError(
            f"unknown proxy {proxy!r}; expected 'vm', 'population_rate', "
            f"or 'synaptic_current'")

    beta, freqs, psd = beta_fraction(trace, sample_dt, beta_band, broadband)
    return {
        "rate_Hz": rate,
        "cv": cv,
        "beta_fraction": beta,
        "proxy_kind": proxy,
        "proxy_trace": trace,
        "trace_dt_ms": sample_dt,
        "psd_freqs": freqs,
        "psd": psd,
        # Always-computed rate proxy, useful as a sanity reference; the
        # diagnostic doc uses it to compare against the chosen proxy.
        "rate_trace": rate_trace,
        "rate_trace_dt_ms": rate_bin_dt,
    }
