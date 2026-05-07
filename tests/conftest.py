"""Shared pytest configuration.

Forces JAX onto x64 floats for tests; numerical tolerances assume float64.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
