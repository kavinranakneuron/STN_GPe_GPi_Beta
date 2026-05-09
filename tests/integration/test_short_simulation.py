"""Phase 1 smoke tests for the integrated simulation.

Exit conditions for Phase 1 (AGENTS.md section 3 Phase 1):

  - Smoke test produces sensible output (non-zero firing in all populations,
    no NaN, consistent across two seeds with different OU realizations).
  - A 600 ms simulation at 450 neurons completes in under 5 seconds on the
    L4 GPU (perf measurement is in this file too, with a generous wall
    threshold so it passes on any GPU).
"""
from __future__ import annotations

import time
from dataclasses import replace

import jax.numpy as jnp

from bgnet.network import NetworkConfig, simulate


def _default_cfg() -> NetworkConfig:
    return NetworkConfig(
        n_stn=100, n_gpe=200, n_gpi=150,
        I_drive_stn=5.0, mu_stn=3.0, sigma_stn=2.0,
        I_drive_gpe=2.0, mu_gpe=1.0, sigma_gpe=2.0,
        I_drive_gpi=0.0, mu_gpi=0.0, sigma_gpi=2.0,
    )


def _rates(out, dt_ms: float):
    duration_s = out["n_steps"] * dt_ms / 1000.0
    rs = float(jnp.sum(out["spikes_stn"])) / out["spikes_stn"].shape[1] / duration_s
    rg = float(jnp.sum(out["spikes_gpe"])) / out["spikes_gpe"].shape[1] / duration_s
    ri = float(jnp.sum(out["spikes_gpi"])) / out["spikes_gpi"].shape[1] / duration_s
    return rs, rg, ri


def test_smoke_100ms_nonzero_firing_no_nan():
    """100 ms x 450 neurons, default-ish parameters: every population must
    spike, and no state variable may be NaN."""
    cfg = _default_cfg()
    out = simulate(cfg, duration_ms=100.0)
    rs, rg, ri = _rates(out, cfg.dt_ms)
    assert rs > 0, f"STN silent: {rs} Hz"
    assert rg > 0, f"GPe silent: {rg} Hz"
    assert ri > 0, f"GPi silent: {ri} Hz"

    final = out["final_state"]
    for label, arr in [("stn V", final.stn.V), ("gpe V", final.gpe.V),
                       ("gpi V", final.gpi.V), ("stn Ca", final.stn.Ca),
                       ("gpe Ca", final.gpe.Ca), ("gpi Ca", final.gpi.Ca)]:
        assert not bool(jnp.any(jnp.isnan(arr))), f"NaN in {label}"


def test_two_seed_consistency():
    """Two seeds sharing the same configuration produce different spike
    realizations (so OU is actually fresh per trial) but their population
    firing rates agree on a 200 ms window. With per-neuron heterogeneity
    enabled (heterogeneity_pct = 0.10 default) the GPi seed-to-seed
    sensitivity is a few Hz higher than in the strictly-homogeneous
    Phase-1 setup; the tolerance is loosened from 5 to 8 Hz to reflect
    that. Same heterogeneity_pct/het_seed across both runs ensures the
    cell-intrinsic perturbations are identical — only the OU realization
    differs."""
    cfg = _default_cfg()
    out_a = simulate(cfg, duration_ms=200.0)
    cfg_b = replace(cfg, ou_seed=cfg.ou_seed + 1)
    out_b = simulate(cfg_b, duration_ms=200.0)

    spikes_eq = bool(jnp.array_equal(out_a["spikes_stn"], out_b["spikes_stn"]))
    assert not spikes_eq, "different ou_seed gave identical STN spike trains"

    rs_a, rg_a, ri_a = _rates(out_a, cfg.dt_ms)
    rs_b, rg_b, ri_b = _rates(out_b, cfg.dt_ms)
    assert abs(rs_a - rs_b) < 8.0, f"STN: {rs_a} vs {rs_b}"
    assert abs(rg_a - rg_b) < 8.0, f"GPe: {rg_a} vs {rg_b}"
    assert abs(ri_a - ri_b) < 8.0, f"GPi: {ri_a} vs {ri_b}"


def test_perf_600ms_under_5s():
    """Phase 1 perf exit: warm 600 ms x 450 neurons must finish in under
    5 s on the L4 GPU. The first call below pays compilation cost, only
    the second is timed."""
    cfg = _default_cfg()
    out = simulate(cfg, duration_ms=600.0)   # warmup / compile
    out["spikes_stn"].block_until_ready()
    t0 = time.time()
    out = simulate(cfg, duration_ms=600.0)
    out["spikes_stn"].block_until_ready()
    elapsed = time.time() - t0
    assert elapsed < 5.0, f"600 ms simulation took {elapsed:.2f} s (Phase 1 budget 5 s)"
