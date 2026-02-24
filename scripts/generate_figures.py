"""
Publication Figure Generation for CBGTC HH Network
Author: Kavin Nakkeeran, Johns Hopkins University

Uses STN beta as the primary beta metric.
All simulations at 45000 neurons (10000/20000/15000).
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
import pickle
import json
import jax
import jax.numpy as jnp

from jax_models.network_builder import build_network_state
from optimization.sim_jax import apply_params_to_config
from optimization.metrics_jax import compute_all_metrics, compute_beta_fraction_all
from jax_models.integrator import network_step
from jax import lax

print(f"JAX devices: {jax.devices()}")

# =============================================================================
# SETUP
# =============================================================================

N_STN, N_GPE, N_GPI = 10000, 20000, 15000
DT_MS = 0.025
N_STEPS = 24000  # 600ms
BURN_STEPS = 4000

print(f"Building {N_STN + N_GPE + N_GPI}-neuron network...")
state, config = build_network_state(N_STN, N_GPE, N_GPI, DT_MS)

def create_simulator(base_config, n_steps):
    @jax.jit
    def simulate(trial_params, init_state):
        config = apply_params_to_config(trial_params, base_config)
        syn_configs = dict(config['synapses'])
        for syn_name, mult_name in [
            ('stn_to_gpe', 'g_stn_gpe_mult'),
            ('gpe_to_stn', 'g_gpe_stn_mult'),
            ('stn_to_gpi', 'g_stn_gpi_mult'),
            ('gpe_to_gpi', 'g_gpe_gpi_mult'),
        ]:
            old_cfg = syn_configs[syn_name]
            new_weights = old_cfg.weights * trial_params.get(mult_name, 1.0)
            syn_configs[syn_name] = old_cfg._replace(weights=new_weights)
        config['synapses'] = syn_configs
        def step_fn(carry_state, t_idx):
            t_ms = t_idx * config['dt_ms']
            new_state, obs = network_step(carry_state, config, t_ms)
            return new_state, obs
        _, obs_history = lax.scan(step_fn, init=init_state, xs=jnp.arange(n_steps))
        return obs_history
    return simulate

simulator = create_simulator(config, N_STEPS)

# ---------------------------------------------------------------------------
# PARAMETERS (from 450n optimization with fixed-indegree)
# ---------------------------------------------------------------------------
healthy_params = {
    'ISTN': 132.235, 'I_gpe': 3.039, 'I_gpi': 2.209,
    'noise_stn_sigma': 3.362, 'noise_gpe_sigma': 33.417, 'noise_gpi_sigma': 67.284,
    'g_stn_gpe_mult': 1.866, 'g_gpe_stn_mult': 0.999,
    'g_stn_gpi_mult': 1.834, 'g_gpe_gpi_mult': 0.687,
}

pd_params = {
    'ISTN': 70.048, 'I_gpe': 1.663, 'I_gpi': 1.670,
    'noise_stn_sigma': 1.971, 'noise_gpe_sigma': 66.383, 'noise_gpi_sigma': 96.429,
    'g_stn_gpe_mult': 4.132, 'g_gpe_stn_mult': 0.114,
    'g_stn_gpi_mult': 1.943, 'g_gpe_gpi_mult': 0.996,
}

HEALTHY_MULTS = {
    'stn_gpe': healthy_params['g_stn_gpe_mult'],
    'gpe_stn': healthy_params['g_gpe_stn_mult'],
    'stn_gpi': healthy_params['g_stn_gpi_mult'],
    'gpe_gpi': healthy_params['g_gpe_gpi_mult'],
}
PD_MULTS = {
    'stn_gpe': pd_params['g_stn_gpe_mult'],
    'gpe_stn': pd_params['g_gpe_stn_mult'],
    'stn_gpi': pd_params['g_stn_gpi_mult'],
    'gpe_gpi': pd_params['g_gpe_gpi_mult'],
}

# Run simulations
print("Running healthy simulation...")
obs_h = simulator(healthy_params, state)
obs_h['V_stn'].block_until_ready()
print("  Done.")

print("Running PD simulation...")
obs_pd = simulator(pd_params, state)
obs_pd['V_stn'].block_until_ready()
print("  Done.")

# Compute metrics
metrics_h = compute_all_metrics(obs_h, DT_MS, burn_steps=BURN_STEPS)
metrics_pd = compute_all_metrics(obs_pd, DT_MS, burn_steps=BURN_STEPS)
beta_h = compute_beta_fraction_all(obs_h, DT_MS, burn_steps=BURN_STEPS)
beta_pd = compute_beta_fraction_all(obs_pd, DT_MS, burn_steps=BURN_STEPS)

print(f"\nHealthy: STN={metrics_h['firing_rates']['stn']:.1f} GPe={metrics_h['firing_rates']['gpe']:.1f} "
      f"GPi={metrics_h['firing_rates']['gpi']:.1f} STN beta={beta_h['stn']*100:.1f}%")
print(f"PD:      STN={metrics_pd['firing_rates']['stn']:.1f} GPe={metrics_pd['firing_rates']['gpe']:.1f} "
      f"GPi={metrics_pd['firing_rates']['gpi']:.1f} STN beta={beta_pd['stn']*100:.1f}%")

print("\nGenerating figures...")

# =============================================================================
# STYLE
# =============================================================================
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
})

HEALTHY_COLOR = '#2166AC'
PD_COLOR = '#B2182B'
BETA_BAND_COLOR = '#FFA500'
STN_COLOR = '#E74C3C'
GPE_COLOR = '#3498DB'
GPI_COLOR = '#2ECC71'

# =============================================================================
# HELPERS
# =============================================================================

def get_spike_times(spikes, dt_ms, burn_steps=4000):
    spikes_valid = np.array(spikes[burn_steps:])
    times, neurons = [], []
    for t_idx in range(spikes_valid.shape[0]):
        spike_neurons = np.where(spikes_valid[t_idx])[0]
        for n in spike_neurons:
            times.append(t_idx * dt_ms)
            neurons.append(n)
    return np.array(times), np.array(neurons)

def compute_psd_welch(V_trace, dt_ms, burn_steps=4000):
    valid_V = np.array(V_trace[burn_steps:])
    lfp = np.mean(valid_V, axis=1)
    fs = 1000.0 / dt_ms
    nperseg = min(len(lfp), int(fs * 0.5))
    freqs, psd = welch(lfp, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    return freqs, psd

# =============================================================================
# FIGURE 1: Raster Plots (Healthy vs PD) — subsample to 100 neurons
# =============================================================================

fig1, axes = plt.subplots(2, 3, figsize=(14, 7))

populations = ['stn', 'gpe', 'gpi']
pop_labels = ['STN', 'GPe', 'GPi']
pop_colors = [STN_COLOR, GPE_COLOR, GPI_COLOR]
n_neurons = [N_STN, N_GPE, N_GPI]
t_win = (0, 250)
N_DISPLAY = 100  # subsample for display

for col, (pop, label, color, n_n) in enumerate(zip(populations, pop_labels, pop_colors, n_neurons)):
    subsample = np.linspace(0, n_n-1, min(N_DISPLAY, n_n), dtype=int)
    for row, (obs, cond, m) in enumerate([
        (obs_h, 'Healthy', metrics_h),
        (obs_pd, 'PD', metrics_pd),
    ]):
        times, neurons = get_spike_times(obs[f'spikes_{pop}'], DT_MS)
        mask_t = (times >= t_win[0]) & (times <= t_win[1])
        mask_n = np.isin(neurons, subsample)

        # Remap neuron indices to display indices for clean y-axis
        neuron_map = {orig: disp for disp, orig in enumerate(subsample)}
        display_neurons = np.array([neuron_map.get(n, -1) for n in neurons])
        valid = mask_t & mask_n & (display_neurons >= 0)

        axes[row, col].scatter(times[valid], display_neurons[valid],
                               s=0.3, c=color, alpha=0.6, rasterized=True)
        axes[row, col].set_xlim(*t_win)
        axes[row, col].set_ylim(0, N_DISPLAY)
        axes[row, col].set_title(f'{label} -- {cond} ({m["firing_rates"][pop]:.1f} Hz)', fontsize=10)
        if col == 0:
            axes[row, col].set_ylabel('Neuron #')
        if row == 1:
            axes[row, col].set_xlabel('Time (ms)')

plt.tight_layout()
plt.savefig('results/figures/fig1_raster_plots.png', dpi=300, bbox_inches='tight')
plt.savefig('results/figures/fig1_raster_plots.pdf', bbox_inches='tight')
plt.close()
print("  Fig 1: Raster plots")

# =============================================================================
# FIGURE 2: Power Spectra — STN primary, GPe/GPi also shown
# =============================================================================

fig2, axes = plt.subplots(1, 3, figsize=(14, 4))
psd_labels = ['STN (primary)', 'GPe', 'GPi']

for col, (pop, label) in enumerate(zip(populations, psd_labels)):
    freqs_h, psd_h = compute_psd_welch(obs_h[f'V_{pop}'], DT_MS)
    freqs_pd, psd_pd = compute_psd_welch(obs_pd[f'V_{pop}'], DT_MS)
    mask = freqs_h <= 60

    axes[col].semilogy(freqs_h[mask], psd_h[mask], color=HEALTHY_COLOR, linewidth=1.5, label='Healthy')
    axes[col].semilogy(freqs_pd[mask], psd_pd[mask], color=PD_COLOR, linewidth=1.5, label='PD')
    axes[col].axvspan(13, 30, alpha=0.15, color=BETA_BAND_COLOR, label='Beta band')
    axes[col].set_xlabel('Frequency (Hz)')
    axes[col].set_ylabel('Power (mV^2/Hz)')
    axes[col].set_title(f'{label}  (beta: {beta_h[pop]*100:.1f}% -> {beta_pd[pop]*100:.1f}%)')
    axes[col].legend(fontsize=8)
    axes[col].set_xlim(0, 60)
    axes[col].grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('results/figures/fig2_power_spectra.png', dpi=300, bbox_inches='tight')
plt.savefig('results/figures/fig2_power_spectra.pdf', bbox_inches='tight')
plt.close()
print("  Fig 2: Power spectra")

# =============================================================================
# FIGURE 3: Firing Rates + STN Beta Bar Charts
# =============================================================================

fig3, axes = plt.subplots(1, 2, figsize=(11, 5))

x = np.arange(3)
w = 0.32

rates_h = [metrics_h['firing_rates'][p] for p in populations]
rates_pd = [metrics_pd['firing_rates'][p] for p in populations]

axes[0].bar(x - w/2, rates_h, w, label='Healthy', color=HEALTHY_COLOR, edgecolor='black', linewidth=0.5)
axes[0].bar(x + w/2, rates_pd, w, label='PD', color=PD_COLOR, edgecolor='black', linewidth=0.5)
for i, (vh, vpd) in enumerate(zip(rates_h, rates_pd)):
    axes[0].text(i - w/2, vh + 1, f'{vh:.1f}', ha='center', va='bottom', fontsize=8)
    axes[0].text(i + w/2, vpd + 1, f'{vpd:.1f}', ha='center', va='bottom', fontsize=8)
axes[0].set_ylabel('Firing Rate (Hz)')
axes[0].set_xticks(x)
axes[0].set_xticklabels(pop_labels)
axes[0].legend()
axes[0].set_title('A. Firing Rates')

# STN beta (primary metric)
x_beta = np.arange(1)
bh_stn = beta_h['stn'] * 100
bpd_stn = beta_pd['stn'] * 100

axes[1].bar(x_beta - w/2, [bh_stn], w, label='Healthy', color=HEALTHY_COLOR, edgecolor='black', linewidth=0.5)
axes[1].bar(x_beta + w/2, [bpd_stn], w, label='PD', color=PD_COLOR, edgecolor='black', linewidth=0.5)
axes[1].text(-w/2, bh_stn + 0.3, f'{bh_stn:.1f}%', ha='center', va='bottom', fontsize=8)
axes[1].text(w/2, bpd_stn + 0.3, f'{bpd_stn:.1f}%', ha='center', va='bottom', fontsize=8)
axes[1].set_ylabel('STN Beta Power (% of total)')
axes[1].set_xticks(x_beta)
axes[1].set_xticklabels(['STN'])
axes[1].legend()
axes[1].set_title('B. STN Beta Band Power (13-30 Hz)')

plt.tight_layout()
plt.savefig('results/figures/fig3_firing_rates_beta.png', dpi=300, bbox_inches='tight')
plt.savefig('results/figures/fig3_firing_rates_beta.pdf', bbox_inches='tight')
plt.close()
print("  Fig 3: Firing rates + STN beta")

# =============================================================================
# FIGURE 4: Network Schematic -- Healthy and PD multipliers with ratios
# =============================================================================

fig4, axes = plt.subplots(1, 2, figsize=(13, 6))

def draw_network(ax, title, mults, highlight_key=None):
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold')

    pos = {'STN': (0, 1.2), 'GPe': (-1.3, -0.3), 'GPi': (1.3, -0.3)}
    cols = {'STN': STN_COLOR, 'GPe': GPE_COLOR, 'GPi': GPI_COLOR}

    for name, (cx, cy) in pos.items():
        circle = plt.Circle((cx, cy), 0.45, color=cols[name], ec='black', linewidth=2, zorder=10)
        ax.add_patch(circle)
        ax.text(cx, cy, name, ha='center', va='center', fontsize=13, fontweight='bold', color='white', zorder=11)

    conns = [
        ('STN', 'GPe', 'stn_gpe', 'exc'),
        ('GPe', 'STN', 'gpe_stn', 'inh'),
        ('STN', 'GPi', 'stn_gpi', 'exc'),
        ('GPe', 'GPi', 'gpe_gpi', 'inh'),
    ]

    for src, tgt, key, ctype in conns:
        x1, y1 = pos[src]
        x2, y2 = pos[tgt]
        dx, dy = x2 - x1, y2 - y1
        dist = np.sqrt(dx**2 + dy**2)
        ux, uy = dx/dist, dy/dist

        perp_x, perp_y = -uy * 0.08, ux * 0.08
        if key == 'gpe_stn':
            perp_x, perp_y = uy * 0.08, -ux * 0.08

        x1a = x1 + ux * 0.50 + perp_x
        y1a = y1 + uy * 0.50 + perp_y
        x2a = x2 - ux * 0.50 + perp_x
        y2a = y2 - uy * 0.50 + perp_y

        mult = mults[key]
        color = '#2CA02C' if ctype == 'exc' else '#D62728'
        lw = 1.5 + mult * 1.0

        ax.annotate('', xy=(x2a, y2a), xytext=(x1a, y1a),
                     arrowprops=dict(arrowstyle='->', color=color, lw=lw, mutation_scale=15))

        mid_x = (x1a + x2a) / 2
        mid_y = (y1a + y2a) / 2

        fontweight = 'bold' if key == highlight_key else 'normal'
        bbox_color = '#FFFF99' if key == highlight_key else 'white'
        ax.text(mid_x + perp_x * 3, mid_y + perp_y * 3, f'{mult:.2f}x',
                fontsize=10, ha='center', va='center', fontweight=fontweight,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=bbox_color, edgecolor='gray', alpha=0.9))

    ax.plot([], [], color='#2CA02C', linewidth=3, label='Excitatory')
    ax.plot([], [], color='#D62728', linewidth=3, label='Inhibitory')
    ax.legend(loc='lower center', fontsize=9, ncol=2)

draw_network(axes[0], 'A. Healthy', HEALTHY_MULTS)
draw_network(axes[1], 'B. Parkinsonian', PD_MULTS, highlight_key='gpe_stn')

# PD/Healthy ratios annotation
ratios = {
    'stn_gpe': PD_MULTS['stn_gpe'] / HEALTHY_MULTS['stn_gpe'],
    'gpe_stn': PD_MULTS['gpe_stn'] / HEALTHY_MULTS['gpe_stn'],
    'stn_gpi': PD_MULTS['stn_gpi'] / HEALTHY_MULTS['stn_gpi'],
    'gpe_gpi': PD_MULTS['gpe_gpi'] / HEALTHY_MULTS['gpe_gpi'],
}
axes[1].text(0, -1.8,
    f'PD/Healthy ratios:\n'
    f'STN->GPe: {ratios["stn_gpe"]:.2f}x  |  GPe->STN: {ratios["gpe_stn"]:.2f}x  |  '
    f'STN->GPi: {ratios["stn_gpi"]:.2f}x  |  GPe->GPi: {ratios["gpe_gpi"]:.2f}x',
    ha='center', va='center', fontsize=9,
    bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', alpha=0.9))

plt.tight_layout()
plt.savefig('results/figures/fig4_network_schematic.png', dpi=300, bbox_inches='tight')
plt.savefig('results/figures/fig4_network_schematic.pdf', bbox_inches='tight')
plt.close()
print("  Fig 4: Network schematic")

# =============================================================================
# FIGURE 5: STN LFP Traces (primary beta metric)
# =============================================================================

fig5, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

lfp_h = np.mean(np.array(obs_h['V_stn'][BURN_STEPS:]), axis=1)
lfp_pd = np.mean(np.array(obs_pd['V_stn'][BURN_STEPS:]), axis=1)
t = np.arange(len(lfp_h)) * DT_MS
t_win_lfp = (50, 350)
mask = (t >= t_win_lfp[0]) & (t <= t_win_lfp[1])

axes[0].plot(t[mask], lfp_h[mask], color=HEALTHY_COLOR, linewidth=0.6)
axes[0].set_ylabel('LFP (mV)')
axes[0].set_title(f'Healthy STN -- Beta: {beta_h["stn"]*100:.1f}%')
axes[0].grid(True, alpha=0.2)

axes[1].plot(t[mask], lfp_pd[mask], color=PD_COLOR, linewidth=0.6)
axes[1].set_xlabel('Time (ms)')
axes[1].set_ylabel('LFP (mV)')
axes[1].set_title(f'Parkinsonian STN -- Beta: {beta_pd["stn"]*100:.1f}%')
axes[1].grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('results/figures/fig5_lfp_traces.png', dpi=300, bbox_inches='tight')
plt.savefig('results/figures/fig5_lfp_traces.pdf', bbox_inches='tight')
plt.close()
print("  Fig 5: STN LFP traces")

# =============================================================================
# FIGURE 8: Scaling Invariance (from run_scaling_study.py results)
# =============================================================================

try:
    with open('results/validation/scaling_study.pkl', 'rb') as f:
        scaling_data = pickle.load(f)

    sc_results = scaling_data['results']
    sc_sizes = scaling_data['config']['sizes']

    # Collect data for plotting
    size_labels = []
    size_totals = []
    h_rates_stn, h_rates_gpe, h_rates_gpi = [], [], []
    pd_rates_stn, pd_rates_gpe, pd_rates_gpi = [], [], []
    h_beta_stn, pd_beta_stn = [], []
    h_wall, pd_wall = [], []

    for n_stn_s, n_gpe_s, n_gpi_s, label in sc_sizes:
        res = sc_results.get(label)
        if res is None or 'error' in res:
            continue
        size_labels.append(label)
        size_totals.append(n_stn_s + n_gpe_s + n_gpi_s)
        h = res['healthy']
        p = res['pd']
        h_rates_stn.append(h['firing_rates']['stn'])
        h_rates_gpe.append(h['firing_rates']['gpe'])
        h_rates_gpi.append(h['firing_rates']['gpi'])
        pd_rates_stn.append(p['firing_rates']['stn'])
        pd_rates_gpe.append(p['firing_rates']['gpe'])
        pd_rates_gpi.append(p['firing_rates']['gpi'])
        h_beta_stn.append(h['beta']['stn'] * 100)
        pd_beta_stn.append(p['beta']['stn'] * 100)
        h_wall.append(h['run_time_s'])
        pd_wall.append(p['run_time_s'])

    fig8, axes = plt.subplots(1, 3, figsize=(15, 5))

    x_pos = np.arange(len(size_totals))

    # Panel A: Firing rates vs size
    axes[0].plot(x_pos, h_rates_stn, 'o-', color=STN_COLOR, label='STN (H)', linewidth=1.5)
    axes[0].plot(x_pos, h_rates_gpe, 's-', color=GPE_COLOR, label='GPe (H)', linewidth=1.5)
    axes[0].plot(x_pos, h_rates_gpi, '^-', color=GPI_COLOR, label='GPi (H)', linewidth=1.5)
    axes[0].plot(x_pos, pd_rates_stn, 'o--', color=STN_COLOR, label='STN (PD)', linewidth=1.5, alpha=0.6)
    axes[0].plot(x_pos, pd_rates_gpe, 's--', color=GPE_COLOR, label='GPe (PD)', linewidth=1.5, alpha=0.6)
    axes[0].plot(x_pos, pd_rates_gpi, '^--', color=GPI_COLOR, label='GPi (PD)', linewidth=1.5, alpha=0.6)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(size_labels, rotation=30)
    axes[0].set_ylabel('Firing Rate (Hz)')
    axes[0].set_title('A. Firing Rates vs Network Size')
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.2)

    # Panel B: STN beta vs size
    axes[1].plot(x_pos, h_beta_stn, 'o-', color=HEALTHY_COLOR, label='Healthy', linewidth=2)
    axes[1].plot(x_pos, pd_beta_stn, 's-', color=PD_COLOR, label='PD', linewidth=2)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(size_labels, rotation=30)
    axes[1].set_ylabel('STN Beta Power (%)')
    axes[1].set_title('B. STN Beta vs Network Size')
    axes[1].legend()
    axes[1].grid(True, alpha=0.2)

    # Panel C: Wall clock time vs size
    axes[2].plot(x_pos, h_wall, 'o-', color=HEALTHY_COLOR, label='Healthy', linewidth=2)
    axes[2].plot(x_pos, pd_wall, 's-', color=PD_COLOR, label='PD', linewidth=2)
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(size_labels, rotation=30)
    axes[2].set_ylabel('Wall Clock Time (s)')
    axes[2].set_title('C. Simulation Time vs Network Size')
    axes[2].legend()
    axes[2].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('results/figures/fig8_scaling_invariance.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/figures/fig8_scaling_invariance.pdf', bbox_inches='tight')
    plt.close()
    print("  Fig 8: Scaling invariance")

except FileNotFoundError:
    print("  Fig 8: SKIPPED (results/validation/scaling_study.pkl not found)")

# =============================================================================
# SUPPLEMENTARY: Sampler Comparison Convergence
# =============================================================================

try:
    with open('results/validation/sampler_comparison.pkl', 'rb') as f:
        sc = pickle.load(f)

    fig_s1, ax = plt.subplots(figsize=(8, 5))
    sampler_colors = {'CMA-ES': '#2166AC', 'TPE': '#D6604D', 'Random': '#999999'}

    for sampler in ['CMA-ES', 'TPE', 'Random']:
        curves = sc['results'][sampler]['convergence_curves']
        mean_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)
        trials = np.arange(1, len(mean_curve) + 1)

        ax.plot(trials, mean_curve, color=sampler_colors[sampler], linewidth=2,
                label=f'{sampler} ({mean_curve[-1]:.3f})')
        ax.fill_between(trials, mean_curve - std_curve, mean_curve + std_curve,
                         color=sampler_colors[sampler], alpha=0.15)

    ax.set_xlabel('Trial')
    ax.set_ylabel('Best Loss')
    ax.set_title('Supplementary: Sampler Convergence (5 seeds, 500 trials)')
    ax.legend()
    ax.grid(True, alpha=0.2)
    ax.set_xlim(1, 500)

    plt.tight_layout()
    plt.savefig('results/figures/supp_sampler_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/figures/supp_sampler_comparison.pdf', bbox_inches='tight')
    plt.close()
    print("  Supp: Sampler comparison")
except FileNotFoundError:
    print("  Supp: Sampler comparison SKIPPED (file not found)")

# =============================================================================
# SUPPLEMENTARY: Timestep Sensitivity
# =============================================================================

try:
    with open('results/validation/timestep_sensitivity.pkl', 'rb') as f:
        ts = pickle.load(f)

    fig_s2, axes = plt.subplots(1, 2, figsize=(11, 4))
    ts_results = ts['results']
    dts = sorted(ts_results.keys())

    for idx, condition in enumerate(['healthy', 'pd']):
        stn_rates = [ts_results[dt][condition]['firing_rates']['stn'] for dt in dts]
        gpe_rates = [ts_results[dt][condition]['firing_rates']['gpe'] for dt in dts]
        gpi_rates = [ts_results[dt][condition]['firing_rates']['gpi'] for dt in dts]
        stn_betas = [ts_results[dt][condition]['beta_fraction']['stn'] * 100 for dt in dts]

        ax2 = axes[idx].twinx()

        axes[idx].plot(dts, stn_rates, 'o-', color=STN_COLOR, label='STN rate')
        axes[idx].plot(dts, gpe_rates, 's-', color=GPE_COLOR, label='GPe rate')
        axes[idx].plot(dts, gpi_rates, '^-', color=GPI_COLOR, label='GPi rate')
        axes[idx].set_xlabel('Timestep (ms)')
        axes[idx].set_ylabel('Firing Rate (Hz)')
        axes[idx].set_title(f'{condition.capitalize()}')
        axes[idx].set_xscale('log')
        axes[idx].legend(loc='upper left', fontsize=8)
        axes[idx].grid(True, alpha=0.2)

        ax2.plot(dts, stn_betas, 'D--', color='orange', label='STN beta %')
        ax2.set_ylabel('STN Beta %')
        ax2.legend(loc='upper right', fontsize=8)

    plt.suptitle('Supplementary: Timestep Sensitivity', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/figures/supp_timestep_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/figures/supp_timestep_sensitivity.pdf', bbox_inches='tight')
    plt.close()
    print("  Supp: Timestep sensitivity")
except FileNotFoundError:
    print("  Supp: Timestep sensitivity SKIPPED (file not found)")

# =============================================================================
# SUPPLEMENTARY: Performance Scaling
# =============================================================================

try:
    with open('results/benchmarks/performance_table.pkl', 'rb') as f:
        perf = pickle.load(f)

    fig_s3, ax = plt.subplots(figsize=(9, 5))

    jax_results_valid = [r for r in perf['jax_results'] if 'error' not in r]
    jax_neurons = [r['n_total'] for r in jax_results_valid]
    jax_times = [r['median_s'] for r in jax_results_valid]
    numpy_time = perf['numpy_baseline']['extrapolated_time_s']
    numpy_neurons = perf['numpy_baseline']['n_total']

    ax.bar([0], [numpy_time], width=0.5, color='#999999', edgecolor='black', label=f'NumPy ({numpy_neurons}n)')
    for i, (n, t) in enumerate(zip(jax_neurons, jax_times)):
        ax.bar([i + 1], [t], width=0.5, color='#2166AC', edgecolor='black',
               label=f'JAX ({n}n)')
        ax.text(i + 1, t + max(jax_times)*0.03, f'{t:.1f}s', ha='center', fontsize=9)

    ax.text(0, numpy_time + max(jax_times)*0.03, f'{numpy_time:.0f}s', ha='center', fontsize=9)
    ax.set_ylabel('Wall Time (seconds)')
    if jax_times:
        speedup_450 = numpy_time / jax_times[0]
        ax.set_title(f'Supplementary: Performance -- {speedup_450:.0f}x Speedup (JAX vs NumPy at {numpy_neurons}n)')
    else:
        ax.set_title('Supplementary: Performance')
    ax.set_xticks(range(len(jax_neurons) + 1))
    ax.set_xticklabels([f'NumPy\n{numpy_neurons}n'] + [f'JAX\n{n}n' for n in jax_neurons])
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig('results/figures/supp_performance.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/figures/supp_performance.pdf', bbox_inches='tight')
    plt.close()
    print("  Supp: Performance scaling")
except FileNotFoundError:
    print("  Supp: Performance scaling SKIPPED (file not found)")

# =============================================================================
# DONE
# =============================================================================
print("\n" + "=" * 60)
print("ALL FIGURES GENERATED")
print("=" * 60)
print("Main figures:")
print("  fig1_raster_plots          -- Raster: healthy vs PD (45000n, 100 subsampled)")
print("  fig2_power_spectra         -- Welch PSD: STN primary, GPe/GPi also shown")
print("  fig3_firing_rates_beta     -- Bar charts: rates + STN beta")
print("  fig4_network_schematic     -- Synaptic multipliers with PD/healthy ratios")
print("  fig5_lfp_traces            -- STN LFP traces (primary beta metric)")
print("  fig6_statistical_validation -- (generated by run_statistical_validation.py)")
print("  fig7_dbs_effect             -- (generated by run_dbs_simulation.py)")
print("  fig8_scaling_invariance     -- Rates, STN beta, wall time vs network size")
print("Supplementary:")
print("  supp_sampler_comparison    -- CMA-ES vs TPE vs Random")
print("  supp_timestep_sensitivity  -- dt convergence (STN beta primary)")
print("  supp_performance           -- JAX vs NumPy speedup")
