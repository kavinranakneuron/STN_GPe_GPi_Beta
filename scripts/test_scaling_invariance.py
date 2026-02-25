#!/usr/bin/env python3
"""
Verify that fixed-indegree connectivity is constant across network sizes.

Builds networks at 450, 1800, and 4500 neurons and checks that each
postsynaptic neuron receives exactly the expected number of presynaptic
inputs (indegree), regardless of total population size.

Does NOT run any simulations — just checks connectivity structure.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp
from jax_models.network_builder import (
    build_network_state,
    K_STN_GPE, K_GPE_STN, K_STN_GPI, K_GPE_GPI,
)

SIZES = [
    (100, 200, 150, "450n (optimization)"),
    (400, 800, 600, "1800n (publication)"),
    (1000, 2000, 1500, "4500n (stress test)"),
]

EXPECTED = {
    'stn_to_gpe': K_STN_GPE,
    'gpe_to_stn': K_GPE_STN,
    'stn_to_gpi': K_STN_GPI,
    'gpe_to_gpi': K_GPE_GPI,
}

print("=" * 70)
print("Fixed-Indegree Scaling Invariance Test")
print("=" * 70)

all_passed = True

for n_stn, n_gpe, n_gpi, label in SIZES:
    print(f"\n--- {label}: {n_stn} STN / {n_gpe} GPe / {n_gpi} GPi ---")
    state, config = build_network_state(n_stn, n_gpe, n_gpi, dt_ms=0.025)

    for syn_name, expected_k in EXPECTED.items():
        syn_cfg = config['synapses'][syn_name]
        n_connections = syn_cfg.connections.shape[0]
        n_post = syn_cfg.n_post

        # Verify total connections = n_post * indegree
        expected_total = n_post * expected_k
        total_ok = (n_connections == expected_total)

        # Verify each postsynaptic neuron has exactly K inputs
        post_indices = syn_cfg.connections[:, 1]
        counts = jnp.zeros(n_post, dtype=jnp.int32)
        counts = counts.at[post_indices].add(1)
        uniform_ok = bool(jnp.all(counts == expected_k))

        status = "PASS" if (total_ok and uniform_ok) else "FAIL"
        if status == "FAIL":
            all_passed = False

        print(f"  {syn_name}: K={syn_cfg.indegree}, "
              f"total_conn={n_connections} (expected {expected_total}), "
              f"uniform={uniform_ok} [{status}]")

print("\n" + "=" * 70)
if all_passed:
    print("ALL CHECKS PASSED — indegree is constant across all network sizes.")
else:
    print("SOME CHECKS FAILED — see above for details.")
print("=" * 70)
