"""Why does the healthy optimizer converge to STN-silent?

Three diagnostics on the latest healthy run (heterogeneity_pct=0.10):

  1. Best-feasible parameter values and their position within bounds.
  2. Unforced state: best params with all four g_syn = 0 (1000 ms, 200 ms
     burn-in). Tells us whether GPe pacemaking alone explains the beta
     observed in the loop, or whether the beta is loop-driven.
  3. 2D feasible-region sweep over g_stn_gpe x g_gpe_stn (36 configs,
     600 ms each, 200 ms burn-in). Tells us whether feasibility exists
     anywhere in this slice, or is empty across the whole subspace.

Writes docs/silent_stn_diagnostics.md and PNGs in
docs/figures/diagnostics/. NOT committed; surface to user when done.
"""
from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bgnet.config import from_yaml
from bgnet.network import NetworkConfig, simulate
from bgnet.observables import population_summary

RUN_DIR = ROOT / "results" / "healthy" / "20260509_133717"
DOC = ROOT / "docs" / "silent_stn_diagnostics.md"
FIG = ROOT / "docs" / "figures" / "diagnostics"
FIG.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Diagnostic 1: parameter values vs bounds
# ---------------------------------------------------------------------------

PARAM_ORDER = [
    "g_stn_gpe", "g_stn_gpi", "g_gpe_stn", "g_gpe_gpi",
    "I_drive_stn", "I_drive_gpe", "I_drive_gpi",
    "mu_stn", "mu_gpe", "mu_gpi",
    "sigma_stn", "sigma_gpe", "sigma_gpi",
]

PARAM_UNITS = {
    "g_stn_gpe": "mS/cm²", "g_stn_gpi": "mS/cm²",
    "g_gpe_stn": "mS/cm²", "g_gpe_gpi": "mS/cm²",
    "I_drive_stn": "µA/cm²", "I_drive_gpe": "µA/cm²", "I_drive_gpi": "µA/cm²",
    "mu_stn": "µA/cm²", "mu_gpe": "µA/cm²", "mu_gpi": "µA/cm²",
    "sigma_stn": "µA/cm²", "sigma_gpe": "µA/cm²", "sigma_gpi": "µA/cm²",
}


def diagnostic_1(results: dict, study_cfg) -> tuple[str, list[str]]:
    """Return (markdown_section, list_of_pinned_names)."""
    bp = results["best_params"]
    bm = results["best_metrics"]
    pinned: list[str] = []

    rows = []
    for name in PARAM_ORDER:
        b = study_cfg.bounds.get(name)
        v = bp[name]
        rng = b.high - b.low
        frac = (v - b.low) / rng if rng > 0 else 0.0
        flag = ""
        if frac <= 0.05:
            flag = " ⚠️ PINNED LOWER"
            pinned.append(name)
        elif frac >= 0.95:
            flag = " ⚠️ PINNED UPPER"
            pinned.append(name)
        rows.append(
            f"| `{name}` | {v:+.4f} | [{b.low:+.3f}, {b.high:+.3f}] | "
            f"{frac*100:5.1f}% | {PARAM_UNITS[name]} |{flag}"
        )

    table = (
        "| Parameter | Value | Bound | Position | Unit | Note |\n"
        "|---|---:|:---:|---:|:---:|:---|\n"
        + "\n".join(rows)
    )

    # specific check: g_stn_gpe near upper bound?
    b_sg = study_cfg.bounds.get("g_stn_gpe")
    g_sg_val = bp["g_stn_gpe"]
    g_sg_frac = (g_sg_val - b_sg.low) / (b_sg.high - b_sg.low)
    g_sg_verdict = (
        f"g_stn_gpe = {g_sg_val:.4f} mS/cm² sits at {g_sg_frac*100:.1f}% of "
        f"its bound range [{b_sg.low}, {b_sg.high}]. "
        + ("**UPPER-PINNED** — the hypothesis that the optimizer chose strong "
           "STN→GPe coupling and silenced STN as the cheapest way to avoid "
           "loop-driven beta is **consistent with the data**." if g_sg_frac >= 0.95
           else ("**NEAR-UPPER** (within 25% of upper)." if g_sg_frac >= 0.75
                 else "**NOT pinned at the upper bound** — the "
                      "strong-STN→GPe-coupling hypothesis is **not supported** "
                      "by the parameter position alone."))
    )

    metrics_lines = (
        "Best-feasible metrics (from the optimizer's 300 ms analysis window):\n\n"
        f"- rate_stn = **{bm['rate_stn']:.2f} Hz**\n"
        f"- rate_gpe = **{bm['rate_gpe']:.2f} Hz**\n"
        f"- rate_gpi = **{bm['rate_gpi']:.2f} Hz**\n"
        f"- cv_stn  = **{bm['cv_stn']:.3f}**\n"
        f"- cv_gpe  = **{bm['cv_gpe']:.3f}**\n"
        f"- cv_gpi  = **{bm['cv_gpi']:.3f}**\n"
        f"- beta_stn = **{bm['beta_stn']:.4f}** "
        f"(STN is silent so its PSD is degenerate; constraint is trivially "
        f"satisfied)\n"
    )

    section = (
        "## Diagnostic 1 — Best-feasible parameter values vs bounds\n\n"
        f"{metrics_lines}\n"
        f"{table}\n\n"
        f"**g_stn_gpe verdict.** {g_sg_verdict}\n\n"
        + (f"**Pinned parameters (within 5% of bound):** "
           + ", ".join(f"`{n}`" for n in pinned) + ".\n"
           if pinned else "**No parameter is pinned within 5% of either bound.**\n")
    )
    return section, pinned


# ---------------------------------------------------------------------------
# Helper: build a NetworkConfig from study_cfg and a parameter dict
# ---------------------------------------------------------------------------

def _build_netcfg(study_cfg, params: dict, ou_seed: int = 1) -> NetworkConfig:
    return NetworkConfig(
        n_stn=study_cfg.n_stn, n_gpe=study_cfg.n_gpe, n_gpi=study_cfg.n_gpi,
        K_stn_gpe=study_cfg.K_stn_gpe, K_gpe_stn=study_cfg.K_gpe_stn,
        K_stn_gpi=study_cfg.K_stn_gpi, K_gpe_gpi=study_cfg.K_gpe_gpi,
        delay_stn_gpe_ms=study_cfg.delay_stn_gpe_ms,
        delay_stn_gpi_ms=study_cfg.delay_stn_gpi_ms,
        delay_gpe_stn_ms=study_cfg.delay_gpe_stn_ms,
        delay_gpe_gpi_ms=study_cfg.delay_gpe_gpi_ms,
        dt_ms=study_cfg.dt_ms,
        g_stn_gpe=params["g_stn_gpe"], g_stn_gpi=params["g_stn_gpi"],
        g_gpe_stn=params["g_gpe_stn"], g_gpe_gpi=params["g_gpe_gpi"],
        I_drive_stn=params["I_drive_stn"],
        I_drive_gpe=params["I_drive_gpe"],
        I_drive_gpi=params["I_drive_gpi"],
        mu_stn=params["mu_stn"], mu_gpe=params["mu_gpe"], mu_gpi=params["mu_gpi"],
        sigma_stn=params["sigma_stn"], sigma_gpe=params["sigma_gpe"],
        sigma_gpi=params["sigma_gpi"],
        network_seed=study_cfg.network_seed,
        ou_seed=ou_seed,
        init_seed=study_cfg.init_seed,
    )


# ---------------------------------------------------------------------------
# Diagnostic 2: unforced (all g_syn = 0)
# ---------------------------------------------------------------------------

def diagnostic_2(results: dict, study_cfg) -> str:
    bp = dict(results["best_params"])
    # Zero out all four synaptic conductances; keep drives, OU mu, OU sigma.
    bp["g_stn_gpe"] = 0.0
    bp["g_stn_gpi"] = 0.0
    bp["g_gpe_stn"] = 0.0
    bp["g_gpe_gpi"] = 0.0

    duration_ms = 1000.0
    burn_in_ms = 200.0
    netcfg = _build_netcfg(study_cfg, bp, ou_seed=study_cfg.ou_seed_base)

    print("[diag2] simulating unforced state (1000 ms, all g_syn = 0)...")
    sim = simulate(netcfg, duration_ms)

    summaries = {}
    for pop in ("stn", "gpe", "gpi"):
        sp = np.asarray(sim[f"spikes_{pop}"])
        s = population_summary(
            sp, dt_ms=sim["dt_ms"], burn_in_ms=burn_in_ms,
            beta_band=tuple(study_cfg.beta_band),
            broadband=tuple(study_cfg.broadband),
        )
        summaries[pop] = s
        # PSD figure
        fig, ax = plt.subplots(figsize=(6.0, 3.5))
        ax.loglog(s["psd_freqs"], s["psd"], color="C0", linewidth=0.8)
        ax.axvspan(study_cfg.beta_band[0], study_cfg.beta_band[1],
                   color="C3", alpha=0.15, label="β band")
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("PSD (Hz²/Hz)")
        ax.set_title(
            f"{pop.upper()} unforced PSD — rate {s['rate_Hz']:.1f} Hz, "
            f"β fraction {s['beta_fraction']:.3f}"
        )
        ax.set_xlim(1.0, 100.0)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = FIG / f"unforced_psd_{pop}.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"[diag2] wrote {out.relative_to(ROOT)}")

    rows = []
    for pop in ("stn", "gpe", "gpi"):
        s = summaries[pop]
        rows.append(
            f"| {pop.upper()} | {s['rate_Hz']:.2f} | {s['cv']:.3f} | "
            f"{s['beta_fraction']:.4f} |"
        )
    table = (
        "| Population | Rate (Hz) | CV | β fraction (13–30 Hz) |\n"
        "|:---:|---:|---:|---:|\n" + "\n".join(rows)
    )

    psd_block = "\n".join(
        f"![{pop.upper()} unforced PSD](figures/diagnostics/unforced_psd_{pop}.png)"
        for pop in ("stn", "gpe", "gpi")
    )

    # Interpretation block (data-driven)
    gpe_rate = summaries["gpe"]["rate_Hz"]
    gpe_beta = summaries["gpe"]["beta_fraction"]
    stn_rate = summaries["stn"]["rate_Hz"]
    stn_beta = summaries["stn"]["beta_fraction"]
    notes = []
    if gpe_beta > 0.05:
        notes.append(
            f"- **GPe shows β fraction = {gpe_beta:.3f} ABOVE the healthy "
            f"0.05 threshold even with all synaptic input removed.** "
            f"This means intrinsic GPe pacemaking + heterogeneity alone is "
            f"sufficient to put energy into the β band; the loop cannot "
            f"reduce that contribution.")
    else:
        notes.append(
            f"- GPe β fraction = {gpe_beta:.3f} is BELOW 0.05 in the unforced "
            f"state, so the loop must be producing the β observed in the "
            f"full-coupled runs.")
    if stn_rate < 1.0:
        notes.append(
            f"- **STN unforced rate = {stn_rate:.2f} Hz**: under the discovered "
            f"tonic + OU drive alone (no GPe inhibition), STN is "
            + ("nearly silent." if stn_rate < 1.0 else "low.") +
            " The discovered I_drive_stn, mu_stn, sigma_stn settings do not "
            "by themselves produce STN spiking.")
    else:
        notes.append(
            f"- STN unforced rate = {stn_rate:.2f} Hz — STN does spike under "
            f"the discovered tonic+OU drive alone; silencing in the loop must "
            f"come from GPe→STN inhibition.")

    section = (
        "## Diagnostic 2 — Unforced state (all g_syn = 0)\n\n"
        "Best-feasible parameters with the four synaptic conductances "
        "zeroed. Tonic drives, OU mean, OU sigma kept as discovered. "
        "1000 ms simulation, 200 ms burn-in, n=450 neurons, "
        "heterogeneity_pct=0.10.\n\n"
        f"{table}\n\n"
        f"{psd_block}\n\n"
        "**Read:**\n" + "\n".join(notes) + "\n"
    )

    return section


# ---------------------------------------------------------------------------
# Diagnostic 3: 2D g-sweep over (g_stn_gpe, g_gpe_stn)
# ---------------------------------------------------------------------------

def diagnostic_3(results: dict, study_cfg) -> str:
    bp = dict(results["best_params"])
    g_stn_gpe_grid = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    g_gpe_stn_grid = [0.005, 0.05, 0.10, 0.20, 0.30, 0.50]
    duration_ms = 600.0
    burn_in_ms = 200.0

    rate_stn = np.zeros((len(g_stn_gpe_grid), len(g_gpe_stn_grid)))
    beta_stn = np.zeros_like(rate_stn)
    feasible = np.zeros_like(rate_stn, dtype=bool)

    n_total = rate_stn.size
    k = 0
    for i, g_sg in enumerate(g_stn_gpe_grid):
        for j, g_gs in enumerate(g_gpe_stn_grid):
            k += 1
            params = dict(bp)
            params["g_stn_gpe"] = g_sg
            params["g_gpe_stn"] = g_gs
            netcfg = _build_netcfg(study_cfg, params,
                                   ou_seed=study_cfg.ou_seed_base + k)
            sim = simulate(netcfg, duration_ms)
            sp_stn = np.asarray(sim["spikes_stn"])
            s = population_summary(
                sp_stn, dt_ms=sim["dt_ms"], burn_in_ms=burn_in_ms,
                beta_band=tuple(study_cfg.beta_band),
                broadband=tuple(study_cfg.broadband),
            )
            r = s["rate_Hz"]
            b = s["beta_fraction"]
            rate_stn[i, j] = r
            beta_stn[i, j] = b
            feasible[i, j] = (r >= 15.0) and (b < 0.05)
            print(f"[diag3] {k:2d}/{n_total} g_sg={g_sg:.3f} g_gs={g_gs:.3f} "
                  f"rate_stn={r:6.2f} beta_stn={b:.4f} feasible={feasible[i,j]}")

    # heatmap figure
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    extent = [-0.5, len(g_gpe_stn_grid) - 0.5, -0.5, len(g_stn_gpe_grid) - 0.5]

    im0 = axes[0].imshow(rate_stn, origin="lower", aspect="auto",
                         cmap="viridis", vmin=0.0)
    axes[0].set_xticks(range(len(g_gpe_stn_grid)))
    axes[0].set_xticklabels([f"{g:.3f}" for g in g_gpe_stn_grid], rotation=45)
    axes[0].set_yticks(range(len(g_stn_gpe_grid)))
    axes[0].set_yticklabels([f"{g:.2f}" for g in g_stn_gpe_grid])
    axes[0].set_xlabel("g_gpe_stn (mS/cm²)")
    axes[0].set_ylabel("g_stn_gpe (mS/cm²)")
    axes[0].set_title("STN firing rate (Hz)")
    cb0 = plt.colorbar(im0, ax=axes[0])
    cb0.set_label("Hz")
    # annotate cells with rate
    for i in range(len(g_stn_gpe_grid)):
        for j in range(len(g_gpe_stn_grid)):
            axes[0].text(j, i, f"{rate_stn[i,j]:.1f}", ha="center", va="center",
                         color="white" if rate_stn[i,j] < rate_stn.max()*0.6 else "black",
                         fontsize=7)
    # overlay feasibility
    for i in range(len(g_stn_gpe_grid)):
        for j in range(len(g_gpe_stn_grid)):
            if feasible[i, j]:
                axes[0].plot(j, i, marker="o", markersize=15, mfc="none",
                             mec="red", mew=2)

    im1 = axes[1].imshow(beta_stn, origin="lower", aspect="auto",
                         cmap="magma", vmin=0.0, vmax=max(0.2, beta_stn.max()))
    axes[1].set_xticks(range(len(g_gpe_stn_grid)))
    axes[1].set_xticklabels([f"{g:.3f}" for g in g_gpe_stn_grid], rotation=45)
    axes[1].set_yticks(range(len(g_stn_gpe_grid)))
    axes[1].set_yticklabels([f"{g:.2f}" for g in g_stn_gpe_grid])
    axes[1].set_xlabel("g_gpe_stn (mS/cm²)")
    axes[1].set_ylabel("g_stn_gpe (mS/cm²)")
    axes[1].set_title("STN β fraction (13–30 Hz)")
    cb1 = plt.colorbar(im1, ax=axes[1])
    cb1.set_label("β fraction")
    for i in range(len(g_stn_gpe_grid)):
        for j in range(len(g_gpe_stn_grid)):
            axes[1].text(j, i, f"{beta_stn[i,j]:.3f}", ha="center", va="center",
                         color="white" if beta_stn[i,j] < 0.10 else "black",
                         fontsize=7)
            if feasible[i, j]:
                axes[1].plot(j, i, marker="o", markersize=15, mfc="none",
                             mec="cyan", mew=2)

    fig.suptitle(
        "Healthy 2D feasible region — circles = feasible "
        "(STN ≥ 15 Hz AND β < 0.05). Other 11 params fixed at best-feasible."
    )
    fig.tight_layout()
    out = FIG / "feasible_region_2d.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[diag3] wrote {out.relative_to(ROOT)}")

    n_feasible = int(feasible.sum())
    if n_feasible == 0:
        feas_text = (
            "**No cell in the 6×6 slice is feasible.** Across two orders of "
            "magnitude in both `g_stn_gpe` and `g_gpe_stn`, with the other 11 "
            "parameters fixed at their best-feasible values, the optimizer "
            "cannot simultaneously achieve STN ≥ 15 Hz AND β < 0.05. The "
            "constraint geometry is incompatible in this 2D slice."
        )
    else:
        feas_cells = []
        for i, gsg in enumerate(g_stn_gpe_grid):
            for j, ggs in enumerate(g_gpe_stn_grid):
                if feasible[i, j]:
                    feas_cells.append(
                        f"(g_stn_gpe={gsg:.3f}, g_gpe_stn={ggs:.3f}): "
                        f"rate={rate_stn[i,j]:.1f} Hz, β={beta_stn[i,j]:.3f}"
                    )
        feas_text = (
            f"**{n_feasible}/{rate_stn.size} cells feasible** "
            f"(STN ≥ 15 Hz AND β < 0.05):\n\n"
            + "\n".join(f"- {c}" for c in feas_cells)
        )

    # text tables for clarity
    def _fmt_grid(arr, title, fmt="{:.2f}"):
        head = "| g_stn_gpe \\ g_gpe_stn | " + " | ".join(
            f"{g:.3f}" for g in g_gpe_stn_grid) + " |"
        sep = "|" + "---|" * (len(g_gpe_stn_grid) + 1)
        rows = []
        for i, gsg in enumerate(g_stn_gpe_grid):
            cells = " | ".join(fmt.format(arr[i, j])
                               for j in range(len(g_gpe_stn_grid)))
            rows.append(f"| **{gsg:.2f}** | {cells} |")
        return f"**{title}**\n\n{head}\n{sep}\n" + "\n".join(rows)

    section = (
        "## Diagnostic 3 — 2D feasible-region sweep\n\n"
        "Sweep over `g_stn_gpe ∈ {0.05, 0.10, 0.20, 0.30, 0.40, 0.50}` × "
        "`g_gpe_stn ∈ {0.005, 0.05, 0.10, 0.20, 0.30, 0.50}` = 36 "
        "configurations. All other 11 parameters held fixed at the "
        "best-feasible values from Diagnostic 1. 600 ms simulation, "
        "200 ms burn-in, n=450, heterogeneity_pct=0.10, ou_seed varied "
        "per cell.\n\n"
        "![2D feasible region](figures/diagnostics/feasible_region_2d.png)\n\n"
        + _fmt_grid(rate_stn, "STN firing rate (Hz)") + "\n\n"
        + _fmt_grid(beta_stn, "STN β fraction (13–30 Hz)", fmt="{:.3f}")
        + "\n\n"
        + feas_text + "\n"
    )

    return section, dict(
        rate_stn=rate_stn, beta_stn=beta_stn, feasible=feasible,
        g_stn_gpe_grid=g_stn_gpe_grid, g_gpe_stn_grid=g_gpe_stn_grid,
    )


# ---------------------------------------------------------------------------
# Final interpretation
# ---------------------------------------------------------------------------

def interpretation(results: dict, study_cfg, pinned: list[str],
                   d3: dict) -> str:
    bp = results["best_params"]
    bm = results["best_metrics"]
    b_sg = study_cfg.bounds.get("g_stn_gpe")
    g_sg_frac = ((bp["g_stn_gpe"] - b_sg.low) /
                 (b_sg.high - b_sg.low))

    notes: list[str] = []

    # Diag 1 read
    if g_sg_frac >= 0.95:
        notes.append(
            "- **Mechanism 1 (strong STN→GPe coupling, Diag 1): SUPPORTED.** "
            "g_stn_gpe is pinned at the upper bound — the optimizer drove it "
            "up and silenced STN to avoid loop entrainment.")
    else:
        notes.append(
            f"- **Mechanism 1 (strong STN→GPe coupling, Diag 1): NOT "
            f"SUPPORTED.** g_stn_gpe = {bp['g_stn_gpe']:.3f} mS/cm² sits at "
            f"{g_sg_frac*100:.1f}% of its bound range, not near the upper "
            f"bound. The cause is not optimizer-chosen strong forward "
            f"coupling.")

    # Diag 3 read
    n_feas = int(d3["feasible"].sum())
    if n_feas == 0:
        notes.append(
            "- **Mechanism 3 (empty 2D feasible region, Diag 3): SUPPORTED.** "
            "No cell in the 6×6 (g_stn_gpe, g_gpe_stn) slice with the other 11 "
            "params fixed at best-feasible simultaneously achieves STN ≥ 15 Hz "
            "AND β < 0.05. Within this 2D slice, the constraint is "
            "infeasible.")
    else:
        notes.append(
            f"- **Mechanism 3 (empty 2D feasible region, Diag 3): NOT "
            f"SUPPORTED.** {n_feas} cells in the 6×6 slice are feasible. "
            f"A live optimizer could in principle reach those operating "
            f"points, so silent-STN is the optimizer's *choice*, not a "
            f"hard constraint violation.")

    notes.append(
        "- **Mechanism 2 (intrinsic GPe-driven β, Diag 2):** see the GPe "
        "row of the Diagnostic 2 table and the GPe unforced PSD figure. "
        "If GPe β is already > 0.05 without input, GPe pacemaking is a "
        "primary β source — but note the constraint is on STN β, not GPe β, "
        "so an intrinsic GPe oscillation is only a problem if it entrains "
        "STN through GPe→STN. Diag 3 tests whether breaking that entrainment "
        "is feasible in g-space.")

    return (
        "## Interpretation — which mechanism produces silent-STN?\n\n"
        "Three candidate mechanisms and what each diagnostic says about them:\n\n"
        + "\n".join(notes) + "\n"
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    results = pickle.load((RUN_DIR / "results.pkl").open("rb"))
    study_cfg = from_yaml(RUN_DIR / "config.yaml")

    header = (
        "# Silent-STN diagnostics — why does the healthy optimizer give "
        "rate_STN = 0?\n\n"
        f"Run analyzed: `{RUN_DIR.relative_to(ROOT)}` "
        f"(heterogeneity_pct = 0.10, 1500 trials, best-feasible flag = "
        f"{results['best_feasible_flag']}).\n\n"
        f"Best-feasible loss = **{results['best_loss_feasible']:.4f}** "
        f"out of {results['n_complete']} complete trials "
        f"({results['n_feasible']} feasible / {results['n_infeasible']} "
        f"infeasible).\n\n"
        "**Parameters fixed for Diagnostics 2 and 3.** Best-feasible "
        "parameter vector from the run above. Diagnostic 2 then zeroes the "
        "four synaptic conductances; Diagnostic 3 sweeps two of them and "
        "holds the rest.\n\n"
        "Simulation parameters across all diagnostics: n_stn=100, n_gpe=200, "
        "n_gpi=150 (450 total); heterogeneity_pct=0.10; dt=0.025 ms; "
        f"β band = [{study_cfg.beta_band[0]}, {study_cfg.beta_band[1]}] Hz; "
        f"broadband = [{study_cfg.broadband[0]}, {study_cfg.broadband[1]}] Hz; "
        "Welch PSD on 1 ms-binned population rate.\n"
    )

    sec1, pinned = diagnostic_1(results, study_cfg)
    sec2 = diagnostic_2(results, study_cfg)
    sec3, d3 = diagnostic_3(results, study_cfg)
    sec_interp = interpretation(results, study_cfg, pinned, d3)

    DOC.write_text("\n".join([header, sec1, sec2, sec3, sec_interp]))
    print(f"[diagnostics] wrote {DOC.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
