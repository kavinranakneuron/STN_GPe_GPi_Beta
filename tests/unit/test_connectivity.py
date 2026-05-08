"""Unit tests for fixed-indegree connectivity."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from bgnet.connectivity import build_connectivity, gather_input


def _check_indegree(n_pre, n_post, K):
    rng = np.random.default_rng(0)
    c = build_connectivity(rng, n_pre, n_post, K)
    arr = np.asarray(c.pre_idx)
    assert arr.shape == (n_post, K)
    # every row has exactly K distinct entries within [0, n_pre)
    for i in range(n_post):
        row = arr[i].tolist()
        assert len(set(row)) == K, f"row {i} has duplicates"
        assert min(row) >= 0 and max(row) < n_pre


def test_indegree_at_optimization_size():
    # 450 neurons total: 100 STN, 200 GPe, 150 GPi
    _check_indegree(100, 200, 15)   # STN -> GPe
    _check_indegree(200, 100, 14)   # GPe -> STN
    _check_indegree(100, 150, 30)   # STN -> GPi
    _check_indegree(200, 150, 10)   # GPe -> GPi


def test_indegree_at_intermediate_size():
    # 4500 neurons total
    _check_indegree(1000, 2000, 15)
    _check_indegree(2000, 1000, 14)
    _check_indegree(1000, 1500, 30)
    _check_indegree(2000, 1500, 10)


def test_indegree_at_validation_size():
    # 45000 neurons total
    _check_indegree(10000, 20000, 15)
    _check_indegree(20000, 10000, 14)
    _check_indegree(10000, 15000, 30)
    _check_indegree(20000, 15000, 10)


def test_gather_input_counts_correctly():
    rng = np.random.default_rng(2)
    c = build_connectivity(rng, n_pre=20, n_post=10, K=4)
    spikes = jnp.zeros((20,))
    spikes = spikes.at[3].set(1.0)
    spikes = spikes.at[7].set(1.0)
    out = gather_input(spikes, c, g_max=0.25)
    arr = np.asarray(c.pre_idx)
    expected = np.array([0.25 * sum(idx in (3, 7) for idx in arr[i]) for i in range(10)])
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-6)
