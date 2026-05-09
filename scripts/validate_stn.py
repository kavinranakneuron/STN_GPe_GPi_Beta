"""End-to-end validation of the Terman-Rubin 2002 STN module.

Regenerates every figure and number in ``docs/stn_validation.md`` from
scratch — no hand-edited intermediate data. Run from the project root:

    python scripts/validate_stn.py

Outputs:
    docs/figures/stn_fi_curve.png
    docs/figures/stn_fi_comparison.png
    docs/figures/stn_rebound.png
    docs/figures/stn_adaptation.png

Prints the validation summary table to stdout. The script imports the same
``bgnet.neurons.stn`` module used by tests, so the numbers cannot drift
silently away from the test bounds.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from bgnet.heterogeneity import homogeneous_het
from bgnet.neurons.stn import STNParams, initial_state, stn_step

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _run_constant(I_drive: float, T_ms: float, dt_ms: float = 0.025,
                  burn_ms: float = 500.0):
    p = STNParams()
    state0 = initial_state(1)
    het = homogeneous_het(1)
    n_steps = int(T_ms / dt_ms)

    @jax.jit
    def run():
        def body(carry, i):
            state, _ = carry
            t = i * dt_ms
            new_state, sp = stn_step(state, p, het, dt_ms,
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
    rate = sps[burn_steps:].sum() / ((T_ms - burn_ms) / 1000.0)
    return rate, Vs, sps


def _run_drive_schedule(drive_arr: np.ndarray, dt_ms: float = 0.025):
    p = STNParams()
    state0 = initial_state(1)
    het = homogeneous_het(1)
    n_steps = int(drive_arr.shape[0])
    drive_jnp = jnp.asarray(drive_arr)

    @jax.jit
    def run():
        def body(carry, i):
            state, _ = carry
            t = i * dt_ms
            I = jnp.array([drive_jnp[i]])
            new_state, sp = stn_step(state, p, het, dt_ms, I,
                                     jnp.array([0.0]), jnp.array([0.0]), t)
            return (new_state, sp), (new_state.V, new_state.r, new_state.Ca, sp)
        (final, _), tr = jax.lax.scan(body, (state0, jnp.array([False])),
                                      jnp.arange(n_steps))
        return tr

    Vs, rs, Cas, sps = (np.asarray(x).flatten() for x in run())
    return Vs, rs, Cas, sps


# ---------------------------------------------------------------------------
# Test 1 + Test 2: f-I curve
# ---------------------------------------------------------------------------

def fi_curve():
    drives = np.array([-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    rates = []
    finite = True
    bounded = True
    for d in drives:
        rate, V, _ = _run_constant(float(d), T_ms=1500.0)
        if not np.all(np.isfinite(V)):
            finite = False
        if V.min() < -100.0 or V.max() > 60.0:
            bounded = False
        rates.append(float(rate))
    rates = np.array(rates)
    monotonic = bool(np.all(np.diff(rates) >= -0.5))
    return drives, rates, finite, bounded, monotonic


def plot_fi(drives, rates, path: Path):
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(drives, rates, marker="o", color="C0", linewidth=2)
    # Annotate the spontaneous-rate point
    idx0 = np.where(drives == 0.0)[0][0]
    ax.annotate(f"spontaneous: {rates[idx0]:.1f} Hz",
                xy=(0.0, rates[idx0]),
                xytext=(7, rates[idx0] + 8),
                arrowprops=dict(arrowstyle="->", color="grey"))
    ax.axhline(20.0, linestyle="--", color="C2", alpha=0.7,
               label="healthy STN target (20 Hz)")
    ax.axhline(27.0, linestyle="--", color="C3", alpha=0.7,
               label="PD STN target (27 Hz)")
    ax.set_xlabel(r"$I_{\mathrm{drive}}$ ($\mu$A/cm$^2$)")
    ax.set_ylabel("firing rate (Hz)")
    ax.set_title("Terman-Rubin 2002 STN f-I curve\n"
                 "(1500 ms, 500 ms transient dropped, single isolated cell)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_fi_comparison(rt_drives, rt_rates, path: Path):
    """Side-by-side f-I with the GW-inspired curve from docs/phase1_review.md
    (Phase 1 measured: ~0 Hz at I=0, ~11 Hz at I=42)."""
    gw_drives = np.array([-2, 0, 2, 5, 10, 15, 20, 30, 42])
    gw_rates = np.array([0, 0, 0, 1, 1, 4, 5, 9, 11])
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(rt_drives, rt_rates, marker="o", color="C0", linewidth=2,
            label="Terman-Rubin 2002 (rebuilt)")
    ax.plot(gw_drives, gw_rates, marker="s", color="C1", linewidth=2,
            label="Gillies-Willshaw-inspired (Phase 1)")
    ax.axhline(20.0, linestyle="--", color="C2", alpha=0.7,
               label="healthy target (20 Hz)")
    ax.axvspan(-5.0, 5.0, color="grey", alpha=0.15,
               label=r"optimizer $I_{\mathrm{drive}}$ bound")
    ax.set_xlabel(r"$I_{\mathrm{drive}}$ ($\mu$A/cm$^2$)")
    ax.set_ylabel("firing rate (Hz)")
    ax.set_title("STN f-I: RT 2002 vs. GW-inspired\n"
                 "RT reaches 20 Hz inside the optimizer's bound; GW does not")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Test 3: post-inhibitory rebound
# ---------------------------------------------------------------------------

def rebound_test():
    dt_ms = 0.025
    n_clamp = int(200.0 / dt_ms)
    n_release = int(100.0 / dt_ms)
    drive = np.concatenate([np.full(n_clamp, -30.0), np.full(n_release, 0.0)])
    V, r, Ca, sps = _run_drive_schedule(drive, dt_ms=dt_ms)
    spikes_after = int(sps[n_clamp:].sum())
    return drive, V, r, sps, spikes_after, dt_ms


def plot_rebound(drive, V, r, sps, dt_ms, path: Path):
    times = np.arange(drive.shape[0]) * dt_ms
    fig, (ax_v, ax_r) = plt.subplots(2, 1, figsize=(6.0, 5.0), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})
    ax_v.plot(times, V, color="C0", linewidth=1.0)
    spike_times = times[sps.astype(bool)]
    for st in spike_times:
        ax_v.axvline(st, color="C3", alpha=0.4, linewidth=0.8)
    ax_v.axvspan(0, 200, color="grey", alpha=0.15, label="hyperpolarizing drive")
    ax_v.set_ylabel("V (mV)")
    ax_v.set_title("Post-inhibitory rebound burst (T-current mediated)\n"
                   "200 ms hyperpolarization → release → rebound spikes")
    ax_v.legend(fontsize=8, loc="upper right")
    ax_v.grid(alpha=0.3)
    ax_r.plot(times, r, color="C1", linewidth=1.0)
    ax_r.axvspan(0, 200, color="grey", alpha=0.15)
    ax_r.set_xlabel("time (ms)")
    ax_r.set_ylabel("r (T-deinact.)")
    ax_r.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Test 4: spike-frequency adaptation
# ---------------------------------------------------------------------------

def adaptation_test():
    dt_ms = 0.025
    n_pre = int(100.0 / dt_ms)
    n_post = int(500.0 / dt_ms)
    drive = np.concatenate([np.zeros(n_pre), np.full(n_post, 25.0)])
    V, r, Ca, sps = _run_drive_schedule(drive, dt_ms=dt_ms)
    times_ms = np.arange(drive.shape[0]) * dt_ms
    spike_times = times_ms[sps.astype(bool)]
    spike_times_drive = spike_times[spike_times >= 100.0]
    isis = np.diff(spike_times_drive)
    early = float(isis[:3].mean())
    late = float(isis[-3:].mean())
    return drive, V, Ca, sps, spike_times_drive, isis, early, late, dt_ms


def plot_adaptation(drive, V, Ca, sps, spike_times, isis, dt_ms, path: Path):
    times = np.arange(drive.shape[0]) * dt_ms
    fig, (ax_v, ax_c, ax_isi) = plt.subplots(3, 1, figsize=(6.0, 6.5),
                                             gridspec_kw={"height_ratios": [2, 1, 2]})
    ax_v.plot(times, V, color="C0", linewidth=0.8)
    ax_v.axvline(100.0, color="grey", linestyle="--",
                 label="step onset (I=25 µA/cm²)")
    ax_v.set_ylabel("V (mV)")
    ax_v.set_title("Spike-frequency adaptation under sustained drive\n"
                   "Calcium-AHP slows firing as Ca accumulates")
    ax_v.legend(fontsize=8, loc="upper right")
    ax_v.grid(alpha=0.3)

    ax_c.plot(times, Ca, color="C2", linewidth=0.8)
    ax_c.axvline(100.0, color="grey", linestyle="--")
    ax_c.set_ylabel(r"Ca ($\mu$M)")
    ax_c.grid(alpha=0.3)

    ax_isi.plot(np.arange(1, len(isis) + 1), isis, marker="o", color="C3")
    ax_isi.set_xlabel("ISI index (post-step onset)")
    ax_isi.set_ylabel("ISI (ms)")
    ax_isi.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Test 5: dt sensitivity
# ---------------------------------------------------------------------------

def dt_sensitivity():
    rate_coarse, _, _ = _run_constant(10.0, T_ms=600.0, dt_ms=0.025, burn_ms=100.0)
    rate_fine, _, _ = _run_constant(10.0, T_ms=600.0, dt_ms=0.0125, burn_ms=100.0)
    return float(rate_coarse), float(rate_fine)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    print("[validate_stn] f-I curve...")
    drives, rates, finite, bounded, monotonic = fi_curve()
    plot_fi(drives, rates, FIG_DIR / "stn_fi_curve.png")
    plot_fi_comparison(drives, rates, FIG_DIR / "stn_fi_comparison.png")

    spontaneous_rate = float(rates[np.where(drives == 0.0)[0][0]])

    print("[validate_stn] post-inhibitory rebound...")
    drive_r, V_r, r_r, sps_r, spikes_after, dt_r = rebound_test()
    plot_rebound(drive_r, V_r, r_r, sps_r, dt_r, FIG_DIR / "stn_rebound.png")

    print("[validate_stn] spike-frequency adaptation...")
    drive_a, V_a, Ca_a, sps_a, st_a, isis_a, early, late = adaptation_test()[:8]
    dt_a = adaptation_test()[8]
    plot_adaptation(drive_a, V_a, Ca_a, sps_a, st_a, isis_a, dt_a,
                    FIG_DIR / "stn_adaptation.png")
    adaptation_pct = 100.0 * (late - early) / early

    print("[validate_stn] dt sensitivity...")
    rate_coarse, rate_fine = dt_sensitivity()
    dt_diff = abs(rate_coarse - rate_fine)

    print("\n=== Validation summary ===")
    print(f"{'Test':40s}  {'Status':6s}  {'Expected':14s}  Observed")
    print("-" * 80)

    def report(name, status_ok, expected, observed):
        status = "PASS" if status_ok else "FAIL"
        print(f"{name:40s}  {status:6s}  {expected:14s}  {observed}")

    report("Zero-drive firing rate",
           8.0 <= spontaneous_rate <= 14.0, "8-14 Hz",
           f"{spontaneous_rate:.2f} Hz")
    report("f-I monotonic", monotonic, "monotonic",
           "yes" if monotonic else "no")
    report("No NaN/inf at extremes", finite, "finite",
           "finite" if finite else "non-finite")
    report("V in clamp range", bounded, "[-100,60] mV",
           "in range" if bounded else "out of range")
    report("Post-inhibitory rebound",
           spikes_after >= 2, ">=2 spikes", f"{spikes_after} spikes")
    report("Adaptation (late vs early ISI)",
           late > early * 1.10, ">=10% incr",
           f"{adaptation_pct:.1f}%")
    report("dt sensitivity (|Δrate|)", dt_diff < 2.0, "<2 Hz",
           f"{dt_diff:.2f} Hz")

    print("\nf-I numerical results:")
    for d, r in zip(drives, rates):
        print(f"  I_drive={d:>5.1f} µA/cm² -> {r:.2f} Hz")

    print(f"\ndt sensitivity table (I_drive=10 µA/cm², 600 ms, 100 ms transient):")
    print(f"  dt=0.025  ms -> {rate_coarse:.2f} Hz")
    print(f"  dt=0.0125 ms -> {rate_fine:.2f} Hz")
    print(f"  |Δrate|        -> {dt_diff:.2f} Hz")

    print("\n[validate_stn] figures written to docs/figures/")


if __name__ == "__main__":
    main()
