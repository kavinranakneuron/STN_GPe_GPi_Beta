"""Unit tests for bgnet.optimize helpers (no simulator in the loop).

The full smoke-test that actually runs the simulator + Optuna lives in
tests/integration/test_optimize_smoke.py — it pays JIT compile cost and
is quarantined to that file.
"""
from __future__ import annotations

from bgnet.config import Bound, StudyConfig
from bgnet.optimize import _build_network_cfg, _initial_mean, _read_constraint
from optuna.trial import FrozenTrial, TrialState
from datetime import datetime


def _cfg() -> StudyConfig:
    return StudyConfig(name="unit", condition="healthy")


def test_initial_mean_is_midpoints():
    cfg = _cfg()
    x0 = _initial_mean(cfg)
    assert x0["g_stn_gpe"] == cfg.bounds.g_stn_gpe.midpoint()
    assert x0["I_drive_stn"] == 0.0  # bounds [-5, 5]
    assert x0["sigma_stn"] == cfg.bounds.sigma_stn.midpoint()
    assert set(x0) == set(cfg.bounds.names())


def test_build_network_cfg_translates_params():
    cfg = _cfg()
    params = {name: cfg.bounds.get(name).midpoint() for name in cfg.bounds.names()}
    net = _build_network_cfg(cfg, params, ou_seed=99)
    assert net.n_stn == cfg.n_stn
    assert net.dt_ms == cfg.dt_ms
    assert net.g_stn_gpe == params["g_stn_gpe"]
    assert net.mu_gpi == params["mu_gpi"]
    assert net.ou_seed == 99
    assert net.network_seed == cfg.network_seed


def _frozen_trial_with_attrs(attrs: dict) -> FrozenTrial:
    return FrozenTrial(
        number=0, state=TrialState.COMPLETE, value=0.0,
        datetime_start=datetime.now(), datetime_complete=datetime.now(),
        params={}, distributions={}, user_attrs=attrs, system_attrs={},
        intermediate_values={}, trial_id=0,
    )


def test_read_constraint_returns_user_attr():
    trial = _frozen_trial_with_attrs({"constraint": (0.03,)})
    assert _read_constraint(trial) == (0.03,)


def test_read_constraint_unset_is_hard_violation():
    trial = _frozen_trial_with_attrs({})
    out = _read_constraint(trial)
    assert out[0] > 1.0   # treated as infeasible
