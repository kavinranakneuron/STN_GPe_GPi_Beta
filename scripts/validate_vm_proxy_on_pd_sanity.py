"""Validate the mean-Vm LFP proxy on the hand-tuned PD strong-coupling
sanity config.

Decision rule (from configs/sanity_pd_strong_coupling.yaml):
- STN β > 0.15 → proxy works; proceed to healthy headline rerun.
- STN β in [0.05, 0.15] → surface for discussion.
- STN β < 0.05 → surface immediately; switch to synaptic-current proxy.

We use the *current* β band [13, 30] Hz (per the post-rebuild config),
not the historical [8, 15] Hz recorded in the sanity YAML's diagnostic
section. The sanity YAML's parameter list is what we honour for
coupling/drive values; only the band is updated to match the current
production config.

Outputs a markdown summary to stdout and writes
docs/figures/diagnostics/vm_proxy_validation_pd.png with all three
proxies' PSDs for the STN population so the comparison is visible.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from bgnet.network import NetworkConfig, simulate
from bgnet.observables import (
    beta_fraction,
    cv_isi,
    firing_rate,
    population_rate_proxy,
    synaptic_current_lfp_proxy,
    vm_lfp_proxy,
)

SANITY_YAML = ROOT / "configs" / "sanity_pd_strong_coupling.yaml"
FIG_OUT = ROOT / "docs" / "figures" / "diagnostics" / "vm_proxy_validation_pd.png"
FIG_OUT.parent.mkdir(parents=True, exist_ok=True)


def load_sanity_params() -> tuple[dict, dict]:
    raw = yaml.safe_load(SANITY_YAML.read_text())
    return raw["parameters"], raw["network"]


def main():
    params, net = load_sanity_params()
    cfg = NetworkConfig(
        n_stn=net["n_stn"], n_gpe=net["n_gpe"], n_gpi=net["n_gpi"],
        dt_ms=net["dt_ms"],
        g_stn_gpe=params["g_stn_gpe"], g_stn_gpi=params["g_stn_gpi"],
        g_gpe_stn=params["g_gpe_stn"], g_gpe_gpi=params["g_gpe_gpi"],
        I_drive_stn=params["I_drive_stn"], I_drive_gpe=params["I_drive_gpe"],
        I_drive_gpi=params["I_drive_gpi"],
        mu_stn=params["mu_stn"], mu_gpe=params["mu_gpe"], mu_gpi=params["mu_gpi"],
        sigma_stn=params["sigma_stn"], sigma_gpe=params["sigma_gpe"],
        sigma_gpi=params["sigma_gpi"],
        network_seed=net["network_seed"], ou_seed=net["ou_seed"],
        init_seed=net["init_seed"],
    )
    duration_ms = float(net["duration_ms"])
    burn_in_ms = float(net["burn_in_ms"])

    print(f"Building network at PD strong-coupling parameters from "
          f"{SANITY_YAML.relative_to(ROOT)} ...")
    sim = simulate(cfg, duration_ms)
    dt_ms = sim["dt_ms"]

    # Production β band — [13, 30] Hz (the value the rebuild has settled on,
    # per the post-Phase-2 update). The sanity YAML's own [8, 15] field is
    # legacy and not used here.
    beta_band = (13.0, 30.0)
    broadband = (1.0, 100.0)

    proxies: dict[str, dict] = {}
    for pop in ("stn", "gpe", "gpi"):
        sp = np.asarray(sim[f"spikes_{pop}"])
        rate = firing_rate(sp, dt_ms, burn_in_ms)
        cv = cv_isi(sp, dt_ms, burn_in_ms)
        # Vm proxy
        vm_trace = np.asarray(sim[f"vmean_{pop}"])
        vm_filt, vm_dt = vm_lfp_proxy(vm_trace, dt_ms, burn_in_ms, bin_ms=1.0)
        bf_vm, fr_vm, psd_vm = beta_fraction(
            vm_filt, vm_dt, beta_band=beta_band, broadband=broadband)
        # Population rate proxy
        rt, rt_dt = population_rate_proxy(sp, dt_ms, 1.0, burn_in_ms)
        bf_rt, fr_rt, psd_rt = beta_fraction(
            rt, rt_dt, beta_band=beta_band, broadband=broadband)
        # Synaptic current proxy
        isyn_trace = np.asarray(sim[f"isyn_mean_{pop}"])
        isyn_filt, isyn_dt = synaptic_current_lfp_proxy(
            isyn_trace, dt_ms, burn_in_ms, bin_ms=1.0)
        bf_isyn, fr_isyn, psd_isyn = beta_fraction(
            isyn_filt, isyn_dt, beta_band=beta_band, broadband=broadband)

        proxies[pop] = dict(rate_Hz=rate, cv=cv,
                            bf_vm=bf_vm, fr_vm=fr_vm, psd_vm=psd_vm,
                            bf_rate=bf_rt, fr_rate=fr_rt, psd_rate=psd_rt,
                            bf_isyn=bf_isyn, fr_isyn=fr_isyn, psd_isyn=psd_isyn)

    print()
    print("β band:", beta_band, "Hz; broadband:", broadband, "Hz")
    print(f"Duration: {duration_ms} ms, burn-in: {burn_in_ms} ms, "
          f"n=({cfg.n_stn},{cfg.n_gpe},{cfg.n_gpi})")
    print()
    print(f"{'pop':<5} {'rate':>7} {'cv':>6} "
          f"{'β_vm':>8} {'β_rate':>8} {'β_isyn':>8}")
    for pop in ("stn", "gpe", "gpi"):
        p = proxies[pop]
        print(f"{pop:<5} {p['rate_Hz']:7.2f} {p['cv']:6.3f} "
              f"{p['bf_vm']:8.4f} {p['bf_rate']:8.4f} {p['bf_isyn']:8.4f}")

    # PSD comparison figure for the STN population.
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0), sharex=True, sharey=False)
    pdata = proxies["stn"]
    for ax, (label, fr, psd, bf) in zip(axes, [
        ("Vm proxy (primary)", pdata["fr_vm"], pdata["psd_vm"], pdata["bf_vm"]),
        ("Population-rate proxy", pdata["fr_rate"], pdata["psd_rate"], pdata["bf_rate"]),
        ("Synaptic-current proxy", pdata["fr_isyn"], pdata["psd_isyn"], pdata["bf_isyn"]),
    ]):
        ax.loglog(fr, psd, color="C0", linewidth=0.8)
        ax.axvspan(*beta_band, color="C3", alpha=0.15, label="β band")
        ax.set_xlim(1.0, 100.0)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("PSD")
        ax.set_title(f"{label}\nβ = {bf:.3f}")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle(
        f"STN PSD under PD strong-coupling sanity parameters "
        f"(rate {pdata['rate_Hz']:.1f} Hz, CV {pdata['cv']:.2f})"
    )
    fig.tight_layout()
    fig.savefig(FIG_OUT, dpi=140)
    plt.close(fig)
    print(f"\nFigure: {FIG_OUT.relative_to(ROOT)}")

    bf_vm_stn = proxies["stn"]["bf_vm"]
    print()
    print(f"STN β (Vm proxy) = {bf_vm_stn:.4f}")
    if bf_vm_stn > 0.15:
        print("PROXY VALIDATION PASSED: Vm β > 0.15 at PD strong coupling.")
        print("Proceed to healthy headline rerun.")
        return 0
    elif bf_vm_stn >= 0.05:
        print("EQUIVOCAL: Vm β in [0.05, 0.15]. Surface for discussion.")
        return 1
    else:
        print("FAIL: Vm β < 0.05 even under PD strong coupling.")
        print("Surface; consider switching to synaptic-current proxy.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
