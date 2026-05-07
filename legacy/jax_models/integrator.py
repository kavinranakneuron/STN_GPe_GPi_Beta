"""
Network integrator - steps all populations and synapses.
Uses Rubin-Terman HH neurons for GPe/GPi, Gillies-Willshaw HH for STN.
"""

import jax.numpy as jnp
from typing import Dict, Tuple
from .synapses_jax import synapse_step
from .noise_jax import ou_step


def network_step(state: Dict, config: Dict, t_ms: float) -> Tuple[Dict, Dict]:
    """
    Single network timestep.
    """
    dt = config['dt_ms']

    # Scale factors for HH units (μA/cm²)
    syn_scale = 0.005
    noise_scale = 0.01
    
    # Get step functions
    stn_step = config['neuron_step_fns']['stn']
    gpe_step = config['neuron_step_fns']['gpe']
    gpi_step = config['neuron_step_fns']['gpi']
    
    # Get parameters
    stn_params = config['neuron_params']['stn']
    gpe_params = config['neuron_params']['gpe']
    gpi_params = config['neuron_params']['gpi']
    
    # =========================================================================
    # COMPUTE SYNAPTIC CURRENTS
    # =========================================================================
    
    # STN receives from GPe (inhibitory)
    syn_state_gpe_stn, I_syn_stn_raw = synapse_step(
        state['synapses']['gpe_to_stn'],
        config['synapses']['gpe_to_stn'],
        state['spikes_gpe'],
        state['stn']['V']
    )
    I_syn_stn = I_syn_stn_raw  # STN already uses HH, synapses tuned for it
    
    # GPe receives from STN (excitatory)
    syn_state_stn_gpe, I_syn_gpe_raw = synapse_step(
        state['synapses']['stn_to_gpe'],
        config['synapses']['stn_to_gpe'],
        state['spikes_stn'],
        state['gpe']['V']
    )
    I_syn_gpe = I_syn_gpe_raw * syn_scale
    
    # GPi receives from STN (excitatory) and GPe (inhibitory)
    syn_state_stn_gpi, I_syn_gpi_from_stn_raw = synapse_step(
        state['synapses']['stn_to_gpi'],
        config['synapses']['stn_to_gpi'],
        state['spikes_stn'],
        state['gpi']['V']
    )
    
    syn_state_gpe_gpi, I_syn_gpi_from_gpe_raw = synapse_step(
        state['synapses']['gpe_to_gpi'],
        config['synapses']['gpe_to_gpi'],
        state['spikes_gpe'],
        state['gpi']['V']
    )
    
    I_syn_gpi = (I_syn_gpi_from_stn_raw + I_syn_gpi_from_gpe_raw) * syn_scale
    
    # =========================================================================
    # COMPUTE NOISE CURRENTS
    # =========================================================================
    
    noise_state_stn, I_noise_stn = ou_step(
        state['noise']['stn'], config['noise']['stn']
    )
    noise_state_gpe, I_noise_gpe_raw = ou_step(
        state['noise']['gpe'], config['noise']['gpe']
    )
    noise_state_gpi, I_noise_gpi_raw = ou_step(
        state['noise']['gpi'], config['noise']['gpi']
    )
    
    # Scale noise for HH
    I_noise_gpe = I_noise_gpe_raw * noise_scale
    I_noise_gpi = I_noise_gpi_raw * noise_scale
    
    # =========================================================================
    # STEP STN (Different signature: state_dict, params, dt, I_ext, I_syn, t_ms)
    # =========================================================================
    
    new_stn_state, (V_stn, spikes_stn) = stn_step(
        state['stn'],
        stn_params,
        dt,
        I_noise_stn,  # I_ext = noise
        I_syn_stn,    # I_syn
        t_ms
    )
    
    # =========================================================================
    # STEP GPe (Rubin-Terman HH)
    # =========================================================================

    V_gpe, h_gpe, n_gpe, r_gpe, Ca_gpe, spikes_gpe = gpe_step(
        state['gpe']['V'],
        state['gpe']['h'],
        state['gpe']['n'],
        state['gpe']['r'],
        state['gpe']['Ca'],
        I_syn_gpe,
        I_noise_gpe,
        dt,
        gpe_params
    )
    new_gpe_state = {
        'V': V_gpe, 'h': h_gpe, 'n': n_gpe, 'r': r_gpe, 'Ca': Ca_gpe,
        'refractory': state['gpe']['refractory']
    }

    # =========================================================================
    # STEP GPi (Rubin-Terman HH)
    # =========================================================================

    V_gpi, h_gpi, n_gpi, r_gpi, Ca_gpi, spikes_gpi = gpi_step(
        state['gpi']['V'],
        state['gpi']['h'],
        state['gpi']['n'],
        state['gpi']['r'],
        state['gpi']['Ca'],
        I_syn_gpi,
        I_noise_gpi,
        dt,
        gpi_params
    )
    new_gpi_state = {
        'V': V_gpi, 'h': h_gpi, 'n': n_gpi, 'r': r_gpi, 'Ca': Ca_gpi,
        'refractory': state['gpi']['refractory']
    }
    
    # =========================================================================
    # ASSEMBLE NEW STATE
    # =========================================================================
    
    new_state = {
        'stn': new_stn_state,
        'gpe': new_gpe_state,
        'gpi': new_gpi_state,
        'spikes_stn': spikes_stn,
        'spikes_gpe': spikes_gpe,
        'spikes_gpi': spikes_gpi,
        'synapses': {
            'gpe_to_stn': syn_state_gpe_stn,
            'stn_to_gpe': syn_state_stn_gpe,
            'stn_to_gpi': syn_state_stn_gpi,
            'gpe_to_gpi': syn_state_gpe_gpi,
        },
        'noise': {
            'stn': noise_state_stn,
            'gpe': noise_state_gpe,
            'gpi': noise_state_gpi,
        }
    }
    
    observables = {
        'V_stn': V_stn,
        'V_gpe': V_gpe,
        'V_gpi': V_gpi,
        'spikes_stn': spikes_stn,
        'spikes_gpe': spikes_gpe,
        'spikes_gpi': spikes_gpi,
    }
    
    return new_state, observables
