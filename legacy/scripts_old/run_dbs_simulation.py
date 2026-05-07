"""
DBS Simulation - Informational Lesion Model
Author: Kavin Nakkeeran, Johns Hopkins University

DBS modeled as increased ISTN and further reduced GPe->STN coupling,
representing disruption of the STN-GPe feedback loop.

Uses STN beta as the primary beta metric.
"""

import sys
sys.path.insert(0, '.')

import gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import time
import pickle
from scipy.signal import welch as scipy_welch

from jax_models.network_builder import build_network_state
from optimization.sim_jax import apply_params_to_config
from optimization.metrics_jax import compute_all_metrics, compute_beta_fraction_all
from jax_models.integrator import network_step
from jax import lax

print(f"JAX devices: {jax.devices()}")

# =============================================================================
# PARAMETERS
# =============================================================================

N_STN, N_GPE, N_GPI = 10000, 20000, 15000
DT_MS = 0.025
N_STEPS = 24000  # 600ms
BURN_STEPS = 4000

DBS_FREQ = 130.0  # Hz (clinical standard)

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

# DBS effect: increase STN drive and further reduce GPe->STN coupling
pd_dbs_params = {
    **pd_params,
    'ISTN': 150.0,
    'g_gpe_stn_mult': 0.05,
}

# =============================================================================
# BUILD NETWORK
# =============================================================================

print(f"Building {N_STN + N_GPE + N_GPI}-neuron network...")
state, config = build_network_state(N_STN, N_GPE, N_GPI, DT_MS, seed=42)

# =============================================================================
# SIMULATOR WITH SYNAPTIC MULTIPLIERS
# =============================================================================

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

MAX_NEURONS_PSD = 500

def compute_psd(V_trace, dt_ms, burn_steps=4000, max_neurons=MAX_NEURONS_PSD):
    """Compute PSD from voltage trace, subsampling neurons to avoid OOM."""
    valid_V = V_trace[burn_steps:]
    n_neurons = valid_V.shape[1]
    if n_neurons > max_neurons:
        rng = np.random.default_rng(0)
        indices = rng.choice(n_neurons, size=max_neurons, replace=False)
        indices.sort()
        valid_V = valid_V[:, indices]
    lfp = np.array(jnp.mean(valid_V, axis=1))
    fs = 1000.0 / dt_ms
    nperseg = min(len(lfp), int(fs * 0.5))
    freqs, psd = scipy_welch(lfp, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    return freqs, psd

def extract_psds(obs, dt_ms, burn_steps):
    """Extract PSDs for all populations as numpy arrays, then data can be freed."""
    psds = {}
    for pop in ['stn', 'gpe', 'gpi']:
        psds[pop] = compute_psd(obs[f'V_{pop}'], dt_ms, burn_steps)
    return psds

# Warm up
print("Warming up JIT...")
obs = simulator(healthy_params, state)
obs['V_stn'].block_until_ready()
print("JIT ready!\n")

# =============================================================================
# RUN SIMULATIONS: Healthy, PD, PD+DBS
# =============================================================================

print("=" * 60)
print("DBS SIMULATION")
print("=" * 60)

# Healthy
print("\nRunning Healthy...")
t0 = time.time()
obs_h = simulator(healthy_params, state)
obs_h['V_stn'].block_until_ready()
print(f"  Time: {time.time()-t0:.1f}s")

metrics_h = compute_all_metrics(obs_h, DT_MS, burn_steps=BURN_STEPS)
beta_h = compute_beta_fraction_all(obs_h, DT_MS, burn_steps=BURN_STEPS)
psd_h = extract_psds(obs_h, DT_MS, BURN_STEPS)

del obs_h
gc.collect()
jax.clear_caches()

# PD without DBS
print("\nRunning PD (no DBS)...")
t0 = time.time()
obs_pd = simulator(pd_params, state)
obs_pd['V_stn'].block_until_ready()
print(f"  Time: {time.time()-t0:.1f}s")

metrics_pd = compute_all_metrics(obs_pd, DT_MS, burn_steps=BURN_STEPS)
beta_pd = compute_beta_fraction_all(obs_pd, DT_MS, burn_steps=BURN_STEPS)
psd_pd = extract_psds(obs_pd, DT_MS, BURN_STEPS)

del obs_pd
gc.collect()
jax.clear_caches()

# PD with DBS
print("\nRunning PD + DBS...")
t0 = time.time()
obs_dbs = simulator(pd_dbs_params, state)
obs_dbs['V_stn'].block_until_ready()
print(f"  Time: {time.time()-t0:.1f}s")

metrics_dbs = compute_all_metrics(obs_dbs, DT_MS, burn_steps=BURN_STEPS)
beta_dbs = compute_beta_fraction_all(obs_dbs, DT_MS, burn_steps=BURN_STEPS)
psd_dbs = extract_psds(obs_dbs, DT_MS, BURN_STEPS)

del obs_dbs
gc.collect()
jax.clear_caches()

# =============================================================================
# RESULTS
# =============================================================================

print("\n" + "=" * 70)
print("RESULTS: DBS EFFECT ON PARKINSONIAN NETWORK")
print("=" * 70)
print(f"{'Metric':<20} {'Healthy':>12} {'PD (OFF)':>12} {'PD+DBS (ON)':>15} {'DBS Change':>12}")
print("-" * 70)

for pop in ['stn', 'gpe', 'gpi']:
    r_h = metrics_h['firing_rates'][pop]
    r_pd = metrics_pd['firing_rates'][pop]
    r_dbs = metrics_dbs['firing_rates'][pop]
    pct = ((r_dbs / r_pd) - 1) * 100 if r_pd > 0 else 0
    print(f"{pop.upper()+' Rate (Hz)':<20} {r_h:>12.1f} {r_pd:>12.1f} {r_dbs:>15.1f} {pct:>+10.0f}%")

print("-" * 70)
print(">>> PRIMARY BETA METRIC (STN):")
print(f"{'STN Beta (%)':<20} {beta_h['stn']*100:>12.1f} {beta_pd['stn']*100:>12.1f} "
      f"{beta_dbs['stn']*100:>15.1f} {beta_dbs['stn']*100 - beta_pd['stn']*100:>+10.1f}pp")
print("    (supplementary):")
print(f"{'GPe Beta (%)':<20} {beta_h['gpe']*100:>12.1f} {beta_pd['gpe']*100:>12.1f} "
      f"{beta_dbs['gpe']*100:>15.1f} {beta_dbs['gpe']*100 - beta_pd['gpe']*100:>+10.1f}pp")
print(f"{'GPi Beta (%)':<20} {beta_h['gpi']*100:>12.1f} {beta_pd['gpi']*100:>12.1f} "
      f"{beta_dbs['gpi']*100:>15.1f} {beta_dbs['gpi']*100 - beta_pd['gpi']*100:>+10.1f}pp")

if beta_pd['stn'] > 0:
    suppression = (beta_pd['stn'] - beta_dbs['stn']) / beta_pd['stn'] * 100
    print(f"\nSTN beta suppression by DBS: {suppression:.0f}%")

# =============================================================================
# FIGURE: DBS Effects
# =============================================================================

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.linewidth': 1.2,
})

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle(f'Figure 7: DBS Suppresses Beta Oscillations ({N_STN+N_GPE+N_GPI} neurons)',
             fontsize=14, fontweight='bold')

# Row 1: Power Spectra (STN primary, GPe/GPi supplementary)
populations = ['stn', 'gpe', 'gpi']
pop_labels = ['STN (primary)', 'GPe', 'GPi']

for col, (pop, label) in enumerate(zip(populations, pop_labels)):
    freqs_h, psd_h_pop = psd_h[pop]
    freqs_pd, psd_pd_pop = psd_pd[pop]
    freqs_dbs, psd_dbs_pop = psd_dbs[pop]
    mask = freqs_pd <= 50

    axes[0, col].semilogy(freqs_h[mask], psd_h_pop[mask], 'b-',
                          label='Healthy', linewidth=1, alpha=0.5)
    axes[0, col].semilogy(freqs_pd[mask], psd_pd_pop[mask], 'r-',
                          label='PD (DBS OFF)', linewidth=1.5)
    axes[0, col].semilogy(freqs_dbs[mask], psd_dbs_pop[mask], 'g-',
                          label='PD + DBS (ON)', linewidth=1.5)
    axes[0, col].axvspan(13, 30, alpha=0.2, color='orange')
    axes[0, col].set_xlabel('Frequency (Hz)')
    axes[0, col].set_ylabel('Power (mV^2/Hz)')
    axes[0, col].set_title(f'{label}\nBeta: {beta_pd[pop]*100:.1f}% -> {beta_dbs[pop]*100:.1f}%')
    axes[0, col].legend(fontsize=8)
    axes[0, col].set_xlim(0, 50)
    axes[0, col].grid(True, alpha=0.3)

# Row 2: Bar charts
x = np.arange(3)
width = 0.25

# Firing rates
rates_h = [metrics_h['firing_rates'][p] for p in populations]
rates_pd = [metrics_pd['firing_rates'][p] for p in populations]
rates_dbs = [metrics_dbs['firing_rates'][p] for p in populations]

axes[1, 0].bar(x - width, rates_h, width, label='Healthy', color='steelblue', edgecolor='black')
axes[1, 0].bar(x, rates_pd, width, label='PD (OFF)', color='firebrick', edgecolor='black')
axes[1, 0].bar(x + width, rates_dbs, width, label='PD + DBS', color='forestgreen', edgecolor='black')
axes[1, 0].set_ylabel('Firing Rate (Hz)')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(['STN', 'GPe', 'GPi'])
axes[1, 0].legend(fontsize=8)
axes[1, 0].set_title('A. Firing Rates')

# STN Beta (primary metric)
stn_betas = [beta_h['stn']*100, beta_pd['stn']*100, beta_dbs['stn']*100]
bar_colors = ['steelblue', 'firebrick', 'forestgreen']
bar_labels = ['Healthy', 'PD (OFF)', 'PD + DBS']
x_beta = np.arange(3)

axes[1, 1].bar(x_beta, stn_betas, 0.5, color=bar_colors, edgecolor='black')
axes[1, 1].set_ylabel('STN Beta Power (%)')
axes[1, 1].set_xticks(x_beta)
axes[1, 1].set_xticklabels(bar_labels, fontsize=9)
axes[1, 1].set_title('B. STN Beta Band Power')
for i, v in enumerate(stn_betas):
    axes[1, 1].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=9)

# Summary text
axes[1, 2].axis('off')
summary_text = (
    f"DBS MECHANISM (Informational Lesion)\n\n"
    f"DBS Parameters:\n"
    f"  ISTN: {pd_params['ISTN']:.0f} -> {pd_dbs_params['ISTN']:.0f}\n"
    f"  g_gpe_stn_mult: {pd_params['g_gpe_stn_mult']:.3f} -> {pd_dbs_params['g_gpe_stn_mult']:.3f}\n\n"
    f"Key Results (STN beta):\n"
    f"  Healthy:  {beta_h['stn']*100:.1f}%\n"
    f"  PD (OFF): {beta_pd['stn']*100:.1f}%\n"
    f"  PD + DBS: {beta_dbs['stn']*100:.1f}%\n"
)
if beta_pd['stn'] > 0:
    summary_text += f"  Suppression: {suppression:.0f}%\n"
summary_text += (
    f"\nInterpretation:\n"
    f"  DBS disrupts pathological synchrony\n"
    f"  in the STN-GPe feedback loop."
)
axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('results/figures/fig7_dbs_effect.png', dpi=300, bbox_inches='tight')
plt.savefig('results/figures/fig7_dbs_effect.pdf', bbox_inches='tight')
print("\nSaved: results/figures/fig7_dbs_effect.png/pdf")

# =============================================================================
# SAVE
# =============================================================================

save_data = {
    'healthy': {'metrics': metrics_h, 'beta': {k: float(v) for k, v in beta_h.items()}},
    'pd': {'metrics': metrics_pd, 'beta': {k: float(v) for k, v in beta_pd.items()}},
    'dbs': {'metrics': metrics_dbs, 'beta': {k: float(v) for k, v in beta_dbs.items()}},
    'healthy_params': healthy_params,
    'pd_params': pd_params,
    'dbs_params': pd_dbs_params,
    'network_size': (N_STN, N_GPE, N_GPI),
}

with open('results/simulations/dbs_results.pkl', 'wb') as f:
    pickle.dump(save_data, f)
print("Saved: results/simulations/dbs_results.pkl")

print("\nDBS simulation complete!")
