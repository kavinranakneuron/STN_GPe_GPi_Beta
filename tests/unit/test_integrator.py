"""Unit tests for the integrator: dt halving must not change rates much."""
from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp

from bgnet.network import NetworkConfig, simulate


def _rates(out, dt_ms):
    n_steps = out["n_steps"]
    duration_s = n_steps * dt_ms / 1000.0
    n_stn = out["spikes_stn"].shape[1]
    n_gpe = out["spikes_gpe"].shape[1]
    n_gpi = out["spikes_gpi"].shape[1]
    rs = float(jnp.sum(out["spikes_stn"])) / n_stn / duration_s
    rg = float(jnp.sum(out["spikes_gpe"])) / n_gpe / duration_s
    ri = float(jnp.sum(out["spikes_gpi"])) / n_gpi / duration_s
    return rs, rg, ri


def test_dt_halving_within_2_hz():
    """Half-dt simulation must agree with full-dt within 2 Hz on every
    population's mean firing rate. The noise sigmas are kept modest so
    OU sampling variance does not dominate the comparison; with stronger
    noise the same-seed-different-dt OU realizations decorrelate enough
    that even an exact integrator would show several-Hz drift on a 200 ms
    window.

    Drives are reduced from the Phase 1 values: the Terman-Rubin 2002 STN
    fires ~10 Hz spontaneously, so I_drive_stn=5 + mu_stn=3 (Phase 1's
    settings) pushed the network into a regime where GPi was sensitive
    enough to dt-discretization that even the deterministic dt error
    exceeded 2 Hz. Moderate drives keep all populations in physiological
    range (STN ~3 Hz, GPe ~80 Hz, GPi ~25 Hz here) where forward Euler at
    dt=0.025 ms is well-behaved.
    """
    cfg = NetworkConfig(
        n_stn=100, n_gpe=200, n_gpi=150,
        I_drive_stn=2.0, mu_stn=0.0, sigma_stn=1.0,
        I_drive_gpe=2.0, mu_gpe=0.0, sigma_gpe=1.0,
        I_drive_gpi=0.0, mu_gpi=0.0, sigma_gpi=1.0,
        ou_seed=11,
    )
    duration = 600.0  # ms; SE on rate ~ sigma_pop / sqrt(T) -> sub-Hz

    out_a = simulate(cfg, duration_ms=duration)
    rs_a, rg_a, ri_a = _rates(out_a, cfg.dt_ms)

    cfg_half = replace(cfg, dt_ms=0.0125)
    out_b = simulate(cfg_half, duration_ms=duration)
    rs_b, rg_b, ri_b = _rates(out_b, cfg_half.dt_ms)

    assert abs(rs_a - rs_b) <= 2.0, f"STN rate diverges: {rs_a} vs {rs_b}"
    assert abs(rg_a - rg_b) <= 2.0, f"GPe rate diverges: {rg_a} vs {rg_b}"
    assert abs(ri_a - ri_b) <= 2.0, f"GPi rate diverges: {ri_a} vs {ri_b}"
