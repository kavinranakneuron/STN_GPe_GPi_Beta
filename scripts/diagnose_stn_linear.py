"""Linear-I_Ca STN biophysical diagnostics.

Runs the diagnostic battery requested for the Phase 1.5 STN swap. Writes
docs/stn_linear_form_diagnostics.md and figures into
docs/figures/diagnostics/. NOT committed; surface to user after running.

Run:
    python scripts/diagnose_stn_linear.py
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from bgnet.neurons.stn import (STNParams, STNState, initial_state, stn_step,
                               minf, hinf, ninf, sinf, ainf, rinf, binf,
                               taun, tauh, taur)
from bgnet.noise import OUParams, ou_step, init_state as ou_init

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "stn_linear_form_diagnostics.md"
FIG = ROOT / "docs" / "figures" / "diagnostics"
FIG.mkdir(parents=True, exist_ok=True)

DT = 0.025


# ---------------------------------------------------------------------------
# Simulation primitives
# ---------------------------------------------------------------------------

_RUN_CACHE: dict[tuple, callable] = {}


def _get_runner(n_steps: int, dt_ms: float, use_noise: bool):
    key = (n_steps, dt_ms, use_noise)
    if key in _RUN_CACHE:
        return _RUN_CACHE[key]
    p = STNParams()

    def run(state0, ou0, key0, I_drive_scalar, ou_p):
        def body(carry, i):
            s, ou, k, _ = carry
            t = i * dt_ms
            if use_noise:
                I_noise, k2 = ou_step(ou, k, dt_ms, ou_p)
            else:
                I_noise = jnp.zeros((1,))
                k2 = k
            ns, sp = stn_step(s, p, dt_ms,
                              jnp.array([I_drive_scalar]),
                              jnp.array([0.0]), I_noise, t)
            return (ns, I_noise, k2, sp), (ns.V, ns.h, ns.n, ns.r, ns.Ca,
                                           I_noise, sp)
        (final, *_), tr = jax.lax.scan(body,
                                       (state0, ou0, key0, jnp.array([False])),
                                       jnp.arange(n_steps))
        return tr

    fn = jax.jit(run)
    _RUN_CACHE[key] = fn
    return fn


def run_isolated(I_drive: float, T_ms: float = 1500.0, dt_ms: float = DT,
                 sigma: float = 0.0, seed: int = 0):
    n_steps = int(T_ms / dt_ms)
    use_noise = sigma > 0.0
    runner = _get_runner(n_steps, dt_ms, use_noise)
    state0 = initial_state(1)
    ou_p = OUParams(mu=0.0, sigma=float(sigma))
    ou0 = ou_init(1, ou_p)
    key0 = jax.random.PRNGKey(int(seed))
    V, h, n, r, Ca, I_noise, sps = runner(state0, ou0, key0,
                                          float(I_drive), ou_p)
    return tuple(np.asarray(x).flatten() for x in (V, h, n, r, Ca, I_noise, sps))


def run_step_drive(drive_arr: np.ndarray, dt_ms: float = DT):
    p = STNParams()
    state0 = initial_state(1)
    n_steps = drive_arr.shape[0]
    drive_jnp = jnp.asarray(drive_arr)

    @jax.jit
    def run():
        def body(carry, i):
            s, _ = carry
            t = i * dt_ms
            I = jnp.array([drive_jnp[i]])
            ns, sp = stn_step(s, p, dt_ms, I, jnp.array([0.0]),
                              jnp.array([0.0]), t)
            return (ns, sp), (ns.V, ns.h, ns.n, ns.r, ns.Ca, sp)
        (final, _), tr = jax.lax.scan(body, (state0, jnp.array([False])),
                                      jnp.arange(n_steps))
        return tr

    return tuple(np.asarray(x).flatten() for x in run())


# ---------------------------------------------------------------------------
# Current decomposition
# ---------------------------------------------------------------------------

def currents(V: np.ndarray, h: np.ndarray, n: np.ndarray, r: np.ndarray,
             Ca: np.ndarray, p: STNParams = STNParams()):
    m_ = np.asarray(minf(V, p))
    s_ = np.asarray(sinf(V, p))
    a_ = np.asarray(ainf(V, p))
    b_ = np.asarray(binf(r, p))
    I_L = p.g_L * (V - p.E_L)
    I_Na = p.g_Na * (m_ ** 3) * h * (V - p.E_Na)
    I_K = p.g_K * (n ** 4) * (V - p.E_K)
    I_AHP = p.g_AHP * (V - p.E_K) * Ca / (Ca + p.k1)
    I_Ca = p.g_Ca * s_ * (V - p.E_Ca)            # linear form
    I_T = p.g_T * (a_ ** 3) * (b_ ** 2) * (V - p.E_Ca)
    return dict(I_L=I_L, I_Na=I_Na, I_K=I_K, I_AHP=I_AHP, I_Ca=I_Ca, I_T=I_T,
                m_inf=m_, s_inf=s_, a_inf=a_, b_r=b_)


def stat_block(name: str, x: np.ndarray) -> str:
    return (f"| {name} | {x.mean():.4f} | {x.std():.4f} | "
            f"{x.min():.4f} | {x.max():.4f} |")


# ---------------------------------------------------------------------------
# Diagnostic battery
# ---------------------------------------------------------------------------

def main():
    p = STNParams()
    sections: list[str] = []

    sections.append("# STN linear-I_Ca diagnostics\n")
    sections.append(
        "Comprehensive biophysical check of the `bgnet/neurons/stn.py` "
        "module under the linear `I_Ca = g_Ca * sinf(V) * (V - E_Ca)` form. "
        "All single-cell, no synapses, dt = 0.025 ms.\n"
    )
    sections.append(
        "**Question:** is the 10 Hz spontaneous firing biophysically clean, "
        "or does the linear form produce 10 Hz via a non-canonical mechanism?\n"
    )

    # ---- Spontaneous firing trace, used by 1-7 ----
    V, h, n, r, Ca, _, sps = run_isolated(I_drive=0.0, T_ms=1500.0, sigma=0.0)
    times = np.arange(V.shape[0]) * DT
    mask = (times >= 500.0) & (times < 1500.0)
    Vw, hw, nw, rw, Caw, spsw = V[mask], h[mask], n[mask], r[mask], Ca[mask], sps[mask]
    cur = currents(Vw, hw, nw, rw, Caw, p)
    n_spikes = int(spsw.sum())
    duration_s = (1500.0 - 500.0) / 1000.0
    rate = n_spikes / duration_s
    sections.append(
        f"## Spontaneous trace at I_drive = 0\n\n"
        f"1500 ms simulation, drop first 500 ms transient, analyse {mask.sum()} "
        f"samples spanning t ∈ [500, 1500] ms. Spikes in window: **{n_spikes}**, "
        f"firing rate **{rate:.2f} Hz**.\n"
    )

    # ---- 1. Resting Ca ----
    Ca_mean = float(Caw.mean())
    flag_Ca = " ⚠️ ELEVATED (>0.5)" if Ca_mean > 0.5 else ""
    sections.append(
        f"## 1. Resting Ca\n\n"
        f"- Mean Ca over post-transient window: **{Ca_mean:.4f} µM**{flag_Ca}\n"
        f"- SD: {Caw.std():.4f}, range [{Caw.min():.4f}, {Caw.max():.4f}]\n"
        f"- RT 2002 baseline Ca is typically ~0.05–0.2 µM. Observed value is "
        f"{'within' if 0.05 <= Ca_mean <= 0.5 else 'above'} that range.\n"
    )

    # ---- 2. Resting I_AHP ----
    iahp_mean = float(cur["I_AHP"].mean())
    sections.append(
        f"## 2. Resting I_AHP\n\n"
        f"- Mean I_AHP over post-transient window: **{iahp_mean:.4f} µA/cm²**\n"
        f"- SD: {cur['I_AHP'].std():.4f}, range "
        f"[{cur['I_AHP'].min():.4f}, {cur['I_AHP'].max():.4f}]\n"
        f"- Sub-threshold (V < -40 mV) mean: "
        f"{cur['I_AHP'][Vw < -40.0].mean():.4f} µA/cm²\n"
    )

    # ---- 3. Resting I_Ca sub-threshold ----
    sub = Vw < -40.0
    iCa_sub_mean = float(cur["I_Ca"][sub].mean())
    sections.append(
        f"## 3. Resting I_Ca (sub-threshold)\n\n"
        f"- Sub-threshold (V < -40 mV) frac of window: {sub.mean():.3f}\n"
        f"- Sub-threshold I_Ca mean: **{iCa_sub_mean:.4f} µA/cm²**\n"
        f"- Sub-threshold I_Ca SD: {cur['I_Ca'][sub].std():.4f}\n"
        f"- Supra-threshold (V > -20 mV) I_Ca mean: "
        f"{cur['I_Ca'][Vw > -20.0].mean():.2f} µA/cm²\n"
        f"- Sub-threshold sinf(V) mean: {cur['s_inf'][sub].mean():.4f} "
        f"(linear form: this is the Ca activation factor directly).\n"
    )

    # ---- 4. r dynamics ----
    sections.append(
        "## 4. r (T-current de-inactivation) dynamics\n\n"
        "| Variable | mean | sd | min | max |\n"
        "|---|---:|---:|---:|---:|\n"
        f"{stat_block('r', rw)}\n\n"
        f"RT expectation: r oscillates ~0.05–0.5 in a tonic firer.\n"
    )

    # ---- 5. binf(r) dynamics ----
    sections.append(
        "## 5. binf(r) dynamics\n\n"
        "| Variable | mean | sd | min | max |\n"
        "|---|---:|---:|---:|---:|\n"
        f"{stat_block('binf(r)', cur['b_r'])}\n\n"
        "RT expectation: small at rest, briefly large during/after spikes.\n"
    )

    # ---- 6. I_T contribution ----
    iT_trace_mask = (times >= 500.0) & (times < 700.0)
    iT_trace = currents(V[iT_trace_mask], h[iT_trace_mask], n[iT_trace_mask],
                        r[iT_trace_mask], Ca[iT_trace_mask], p)["I_T"]
    fig6, ax = plt.subplots(figsize=(7.0, 3.5))
    ax.plot(times[iT_trace_mask], iT_trace, color="C3", linewidth=0.8)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel(r"$I_T$ ($\mu$A/cm$^2$)")
    ax.set_title("I_T(t), 200 ms post-transient (t = 500–700 ms)")
    ax.grid(alpha=0.3)
    fig6.tight_layout()
    fig6.savefig(FIG / "i_t_trace.png", dpi=140)
    plt.close(fig6)
    sections.append(
        f"## 6. I_T contribution\n\n"
        f"- Mean I_T over post-transient window: **{float(cur['I_T'].mean()):.4f} µA/cm²**\n"
        f"- SD: {float(cur['I_T'].std()):.4f}, range "
        f"[{float(cur['I_T'].min()):.4f}, {float(cur['I_T'].max()):.4f}]\n"
        f"- Time-resolved trace: ![I_T trace](figures/diagnostics/i_t_trace.png)\n"
    )

    # ---- 7. Inter-channel current balance at a sub-threshold point ~200 ms post-transient ----
    # times[mask] starts at 500 ms. "200 ms post-transient" -> absolute t ≈ 700 ms.
    sub_idx = np.where(Vw < -40.0)[0]
    target_relative = 200.0  # ms after start of window
    pick = sub_idx[np.argmin(np.abs(times[mask][sub_idx] - (500.0 + target_relative)))]
    pick_t = float(times[mask][pick])
    cV = float(Vw[pick])
    sums = (cur["I_L"][pick] + cur["I_Na"][pick] + cur["I_K"][pick]
            + cur["I_AHP"][pick] + cur["I_Ca"][pick] + cur["I_T"][pick])
    sections.append(
        f"## 7. Inter-channel current balance (sub-threshold point)\n\n"
        f"At t = {pick_t:.2f} ms (i.e. {pick_t - 500.0:.0f} ms post-transient), "
        f"V = {cV:.2f} mV, Ca = {Caw[pick]:.4f} µM, r = {rw[pick]:.4f}.\n\n"
        "| Current | µA/cm² | Direction |\n"
        "|---|---:|---|\n"
        f"| I_L | {float(cur['I_L'][pick]):+.3f} | "
        f"{'depolarizing' if cur['I_L'][pick] < 0 else 'hyperpolarizing'} |\n"
        f"| I_Na | {float(cur['I_Na'][pick]):+.3f} | "
        f"{'depolarizing' if cur['I_Na'][pick] < 0 else 'hyperpolarizing'} |\n"
        f"| I_K | {float(cur['I_K'][pick]):+.3f} | "
        f"{'depolarizing' if cur['I_K'][pick] < 0 else 'hyperpolarizing'} |\n"
        f"| I_AHP | {float(cur['I_AHP'][pick]):+.3f} | "
        f"{'depolarizing' if cur['I_AHP'][pick] < 0 else 'hyperpolarizing'} |\n"
        f"| I_Ca | {float(cur['I_Ca'][pick]):+.3f} | "
        f"{'depolarizing' if cur['I_Ca'][pick] < 0 else 'hyperpolarizing'} |\n"
        f"| I_T | {float(cur['I_T'][pick]):+.3f} | "
        f"{'depolarizing' if cur['I_T'][pick] < 0 else 'hyperpolarizing'} |\n"
        f"| **Sum** | {float(sums):+.3f} | net |\n\n"
        f"(Convention: I = g·(V - E), so negative I is depolarizing in "
        f"`dV/dt = -I_ion/C_m`.)\n"
    )

    # ---- 8. Phase plane V(t), Ca(t) ----
    pp_mask = (times >= 800.0) & (times < 1000.0)
    fig8, (ax_v, ax_c) = plt.subplots(2, 1, figsize=(7.0, 4.5), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]})
    ax_v.plot(times[pp_mask], V[pp_mask], color="C0", linewidth=0.8)
    ax_v.set_ylabel("V (mV)")
    ax_v.set_title("Steady-state firing trace, t = 800–1000 ms")
    ax_v.grid(alpha=0.3)
    ax_c.plot(times[pp_mask], Ca[pp_mask], color="C2", linewidth=0.8)
    ax_c.set_xlabel("time (ms)")
    ax_c.set_ylabel(r"Ca ($\mu$M)")
    ax_c.grid(alpha=0.3)
    fig8.tight_layout()
    fig8.savefig(FIG / "phase_plane_v_ca.png", dpi=140)
    plt.close(fig8)
    spike_times_pp = times[pp_mask][sps[pp_mask].astype(bool)]
    isis = np.diff(spike_times_pp)
    sections.append(
        f"## 8. Phase-plane: V(t) and Ca(t), 200 ms steady-state\n\n"
        f"![V and Ca trace](figures/diagnostics/phase_plane_v_ca.png)\n\n"
        f"- Spike times in window: {[f'{x:.1f}' for x in spike_times_pp]}\n"
        f"- ISIs (ms): {[f'{x:.2f}' for x in isis]}\n"
        f"- Ca trough/peak in window: "
        f"[{Ca[pp_mask].min():.4f}, {Ca[pp_mask].max():.4f}] µM "
        f"(rising during spikes, decaying between is the expected pattern; "
        f"flag if monotonic drift).\n"
    )

    # ---- 9. Rebound mechanism trace ----
    n_clamp = int(200.0 / DT)
    n_release = int(100.0 / DT)
    drive_rb = np.concatenate([np.full(n_clamp, -30.0), np.full(n_release, 0.0)])
    Vr, hr, nr, rr, Car, sps_r = run_step_drive(drive_rb)
    times_r = np.arange(drive_rb.shape[0]) * DT
    cur_r = currents(Vr, hr, nr, rr, Car, p)
    r_max_clamp = float(rr[:n_clamp].max())
    iT_post_release = cur_r["I_T"][n_clamp:n_clamp + int(50.0 / DT)]
    iT_post_release_min = float(iT_post_release.min())  # most-depolarizing
    spikes_after = int(sps_r[n_clamp:].sum())

    fig9, (ax_v, ax_r, ax_it) = plt.subplots(3, 1, figsize=(7.0, 6.0), sharex=True,
                                             gridspec_kw={"height_ratios": [2, 1, 1]})
    ax_v.plot(times_r, Vr, color="C0", linewidth=0.8)
    ax_v.axvspan(0, 200, color="grey", alpha=0.15, label="-30 µA/cm² clamp")
    ax_v.set_ylabel("V (mV)")
    ax_v.set_title("Rebound trace: V, r, I_T")
    ax_v.legend(fontsize=8, loc="upper right")
    ax_v.grid(alpha=0.3)
    ax_r.plot(times_r, rr, color="C1", linewidth=0.8)
    ax_r.axhline(0.5, linestyle="--", color="C3", linewidth=0.6, label="r = 0.5")
    ax_r.axvspan(0, 200, color="grey", alpha=0.15)
    ax_r.set_ylabel("r")
    ax_r.legend(fontsize=8, loc="upper right")
    ax_r.grid(alpha=0.3)
    ax_it.plot(times_r, cur_r["I_T"], color="C3", linewidth=0.8)
    ax_it.axvspan(0, 200, color="grey", alpha=0.15)
    ax_it.set_xlabel("time (ms)")
    ax_it.set_ylabel(r"$I_T$ ($\mu$A/cm$^2$)")
    ax_it.grid(alpha=0.3)
    fig9.tight_layout()
    fig9.savefig(FIG / "rebound_mechanism.png", dpi=140)
    plt.close(fig9)
    sections.append(
        f"## 9. Rebound mechanism\n\n"
        f"- r max during -30 µA/cm² clamp: **{r_max_clamp:.3f}** "
        f"({'OK (>0.5)' if r_max_clamp > 0.5 else 'TOO LOW (<0.5)'})\n"
        f"- I_T most-depolarizing value in 50 ms post-release: "
        f"**{iT_post_release_min:.2f} µA/cm²** (transient peak)\n"
        f"- Spikes in 100 ms post-release: **{spikes_after}**\n"
        f"![rebound mechanism](figures/diagnostics/rebound_mechanism.png)\n"
    )

    # ---- 10. f-I family with sigma=0 and sigma=1.5 ----
    drives = [-5, 0, 5, 10, 15, 20, 25, 30]
    rows = []
    for d in drives:
        # sigma=0 (deterministic)
        V0, *_, sps0 = run_isolated(I_drive=float(d), T_ms=1500.0, sigma=0.0)
        rate0 = sps0[int(500.0 / DT):].sum() / 1.0  # 1000 ms post-burn
        # sigma=1.5, average over 5 seeds
        rates_n = []
        for seed in range(5):
            _, _, _, _, _, _, sps_n = run_isolated(
                I_drive=float(d), T_ms=1500.0, sigma=1.5, seed=seed)
            rates_n.append(sps_n[int(500.0 / DT):].sum() / 1.0)
        rate_n_mean = float(np.mean(rates_n))
        rate_n_sd = float(np.std(rates_n))
        rows.append((d, float(rate0), rate_n_mean, rate_n_sd))

    fi_rows = "\n".join(
        f"| {d} | {r0:.2f} | {rn:.2f} | {rsd:.2f} |"
        for (d, r0, rn, rsd) in rows
    )
    sections.append(
        "## 10. Drive-dependent f-I family\n\n"
        "| I_drive (µA/cm²) | sigma=0 rate (Hz) | sigma=1.5 mean (Hz) | sigma=1.5 sd (Hz) |\n"
        "|---:|---:|---:|---:|\n"
        f"{fi_rows}\n"
    )

    # ---- Summary header ----
    summary_lines = [
        "## Summary read",
        "",
        "Brief mechanism check:",
        "",
        f"- Resting Ca: **{Ca_mean:.3f} µM** "
        f"({'within RT range' if 0.05 <= Ca_mean <= 0.5 else 'OUTSIDE RT 0.05-0.5 range'}).",
        f"- Sub-threshold I_Ca: **{iCa_sub_mean:+.2f} µA/cm²** "
        f"(this is the candidate pacemaker if large; chronic if always negative).",
        f"- r oscillates over **[{rw.min():.3f}, {rw.max():.3f}]** "
        f"(RT expects ~0.05-0.5).",
        f"- I_T over the firing cycle: **mean {float(cur['I_T'].mean()):.3f} µA/cm²** "
        f"({'transient' if abs(cur['I_T'].mean()) < 0.5 else 'chronic'}).",
        f"- Rebound: r reaches {r_max_clamp:.2f} during clamp, "
        f"I_T peaks at {iT_post_release_min:.2f} µA/cm² post-release, "
        f"{spikes_after} rebound spikes.",
        "",
        "Read this document end-to-end before deciding on commit.",
        "",
    ]
    sections.insert(2, "\n".join(summary_lines))

    DOC.write_text("\n".join(sections))
    print(f"[diagnostics] wrote {DOC.relative_to(ROOT)}")
    print(f"[diagnostics] figures in {FIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
