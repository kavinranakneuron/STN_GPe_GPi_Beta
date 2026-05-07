# observables.py
import jax.numpy as jnp


def compute_firing_rates(spikes_dict, dt_ms, burn_steps=0):
    """
    Compute mean firing rates for each population.
    
    Args:
        spikes_dict: {'stn': (n_steps, n_neurons), 'gpe': ..., 'gpi': ...}
        dt_ms: Timestep in ms
        burn_steps: Number of initial steps to discard
        
    Returns:
        rates: {'stn': Hz, 'gpe': Hz, 'gpi': Hz}
    """
    rates = {}
    for pop_name, spike_array in spikes_dict.items():
        trimmed = spike_array[burn_steps:]
        n_steps, n_neurons = trimmed.shape
        total_time_sec = (n_steps * dt_ms) / 1000.0
        total_spikes = jnp.sum(trimmed)
        rates[pop_name] = (total_spikes / n_neurons) / total_time_sec
    return rates


def compute_beta_fraction(V_trace, dt_ms, burn_steps=0, beta_range=(13, 30), broadband_range=(1, 100)):
    """
    Compute fractional beta power: beta-band power / total broadband power.
    
    Args:
        V_trace: (n_steps, n_neurons) voltage traces
        dt_ms: Timestep in ms
        burn_steps: Number of initial steps to discard
        beta_range: (low, high) Hz for beta band
        broadband_range: (low, high) Hz for normalization
        
    Returns:
        beta_fraction: Scalar in [0, 1] (multiply by 100 for percentage)
    """
    trimmed = V_trace[burn_steps:]
    lfp = jnp.mean(trimmed, axis=1)
    
    fft_vals = jnp.fft.rfft(lfp)
    freqs = jnp.fft.rfftfreq(len(lfp), d=dt_ms / 1000.0)
    psd = jnp.abs(fft_vals) ** 2
    
    idx_beta = (freqs >= beta_range[0]) & (freqs <= beta_range[1])
    idx_total = (freqs >= broadband_range[0]) & (freqs <= broadband_range[1])
    
    beta_power = jnp.sum(psd[idx_beta])
    total_power = jnp.sum(psd[idx_total])
    
    return beta_power / jnp.maximum(total_power, 1e-12)


def compute_mean_voltage(V_trace, burn_steps=0):
    """Mean voltage across time and neurons (after burn-in)."""
    return jnp.mean(V_trace[burn_steps:])


def compute_all_metrics(observables_dict, dt_ms, burn_steps=4000):
    """
    Compute all metrics at once with burn-in trimming.
    
    Args:
        observables_dict: Output from simulation with keys:
            'V_stn', 'V_gpe', 'V_gpi', 'spikes_stn', 'spikes_gpe', 'spikes_gpi'
        dt_ms: Timestep in ms
        burn_steps: Number of initial steps to discard (default 4000 = 100ms at dt=0.025ms)
        
    Returns:
        metrics: Dict with firing_rates, beta_fraction, mean_V
    """
    spikes = {
        'stn': observables_dict['spikes_stn'],
        'gpe': observables_dict['spikes_gpe'],
        'gpi': observables_dict['spikes_gpi'],
    }
    firing_rates = compute_firing_rates(spikes, dt_ms, burn_steps=burn_steps)
    
    beta_fraction = {}
    mean_V = {}
    for pop in ['stn', 'gpe', 'gpi']:
        V = observables_dict[f'V_{pop}']
        beta_fraction[pop] = compute_beta_fraction(V, dt_ms, burn_steps=burn_steps)
        mean_V[pop] = compute_mean_voltage(V, burn_steps=burn_steps)
    
    return {
        'firing_rates': firing_rates,
        'beta_fraction': beta_fraction,
        'mean_V': mean_V,
    }
